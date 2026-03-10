"""
Enhanced edge_path_patching: 支持多通道 (q,k,v) & 联合 (qkv) path patching.

用法示例：
单通道 (默认 q)：
python edge_path_patching.py \
  --collapsed-graph results/ioi_gpt2/graph_collapsed.json \
  --output results/ioi_gpt2/patch_q.json \
  --channels q \
  --dataset-size 64 --prompt-type mixed --device cuda

同时输出 q/k/v 三种：
python edge_path_patching.py \
  --collapsed-graph results/ioi_gpt2/graph_collapsed.json \
  --output results/ioi_gpt2/patch_qkv_sep.json \
  --channels q k v --dataset-size 64 --device cuda

再加联合 (qkv)：
python edge_path_patching.py \
  --collapsed-graph results/ioi_gpt2/graph_collapsed.json \
  --output results/ioi_gpt2/patch_qkv_joint.json \
  --channels q k v --joint-patch --dataset-size 64 --device cuda

含 logits 边（如果 collapsed 图含 aX.hY->logits）自动处理。
"""

from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal
from collections import defaultdict

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from ioi_dataset import IOIDataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Dataclasses ---------------- #

@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int
    @classmethod
    def from_name(cls, name: str) -> "HeadRef":
        # name 形如 a9.h6
        l, h = name.split(".")
        return cls(layer=int(l[1:]), head=int(h[1:]))
    def as_name(self) -> str:
        return f"a{self.layer}.h{self.head}"

@dataclass(frozen=True)
class SenderRef:
    kind: Literal["head", "input"]
    head: Optional[HeadRef] = None

    @classmethod
    def from_name(cls, name: str) -> "SenderRef":
        if name == "input":
            return cls(kind="input", head=None)
        return cls(kind="head", head=HeadRef.from_name(name))

    def as_name(self) -> str:
        if self.kind == "input":
            return "input"
        assert self.head is not None
        return self.head.as_name()

@dataclass(frozen=True)
class EdgeSpec:
    src: SenderRef
    target_kind: Literal["attention","logits"]
    dst: Optional[HeadRef] = None              # 新增字段
    receiver_input: Optional[str] = None       # q / k / v / qkv / None
    def label(self) -> str:
        if self.target_kind == "attention":
            # 保护性判断
            dst_name = self.dst.as_name() if self.dst is not None else "UNKNOWN"
            return f"{self.src.as_name()}->{dst_name}<{self.receiver_input}>"
        return f"{self.src.as_name()}->logits"

# --------------- Loading collapsed graph --------------- #

def load_collapsed_edges(path: Path) -> List[Tuple[str,str]]:
    data = json.loads(path.read_text())
    edges = data.get("edge_list")
    if not isinstance(edges, list):
        raise ValueError("collapsed graph 缺少 edge_list 或格式错误")
    return [(a,b) for a,b in edges]

def load_edge_specs(
    collapsed_graph: Path,
    channels: Sequence[str],
    full_graph: Optional[Path]=None,
    joint_patch: bool=False
) -> List[EdgeSpec]:
    valid = {"q","k","v"}
    for c in channels:
        if c not in valid:
            raise ValueError(f"非法通道 {c}")
    edges = load_collapsed_edges(collapsed_graph)
    # 若提供 full_graph（含 qkv 标注），按其过滤
    qkv_lookup: Dict[Tuple[str,str], List[str]] = defaultdict(list)
    if full_graph:
        raw = json.loads(full_graph.read_text()).get("edges", {})
        for info in raw.values():
            if not info.get("in_graph", False): continue
            parent = info.get("parent")
            child = info.get("child")
            qkv = info.get("qkv")
            if parent and child and qkv in valid:
                qkv_lookup[(parent, child)].append(qkv)

    specs: List[EdgeSpec] = []
    for src, dst in edges:
        # 支持 input->head / input->logits
        if not (src == "input" or src.startswith("a")):
            continue
        if dst == "logits":
            specs.append(EdgeSpec(src=SenderRef.from_name(src), target_kind="logits"))
        elif dst.startswith("a"):
            # 确定该边要生成哪些通道
            use_channels = qkv_lookup.get((src,dst)) or list(channels)
            for ch in use_channels:
                specs.append(EdgeSpec(
                    src=SenderRef.from_name(src),
                    dst=HeadRef.from_name(dst),
                    target_kind="attention",
                    receiver_input=ch
                ))
            if joint_patch and len(use_channels) > 1:
                # 添加联合 spec（qkv）
                specs.append(EdgeSpec(
                    src=SenderRef.from_name(src),
                    dst=HeadRef.from_name(dst),
                    target_kind="attention",
                    receiver_input="qkv"
                ))
    return specs

# --------------- Metric ---------------- #

def logits_to_logit_diff(logits: t.Tensor, dataset: IOIDataset, per_prompt=False) -> t.Tensor:
    # logits[:, pos] 预测 input[pos]；数据集中 word_idx["end"] 预期指向预测下一个名字位置之前的 token
    pos = dataset.word_idx["end"]
    io_logits = logits[range(logits.size(0)), pos, dataset.io_tokenIDs]
    s_logits  = logits[range(logits.size(0)), pos, dataset.s_tokenIDs]
    diff = io_logits - s_logits
    return diff if per_prompt else diff.mean()

# --------------- Patch primitives --------------- #

def patch_or_freeze_head_vectors(
    head_out: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int,int]
):
    # 将所有头恢复为 clean，再替换一个 sender 头为 corrupted
    head_out[...] = orig_cache[hook.name][...]
    if hook.layer() == head_to_patch[0]:
        head_out[:,:,head_to_patch[1]] = new_cache[hook.name][:,:,head_to_patch[1]]
    return head_out

def freeze_all_head_vectors(
    head_out: t.Tensor,
    hook: HookPoint,
    orig_cache: ActivationCache,
):
    head_out[...] = orig_cache[hook.name][...]
    return head_out

def patch_resid_from_cache(
    resid: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
):
    resid[...] = new_cache[hook.name][...]
    return resid

def patch_receiver_inputs(
    activation: t.Tensor,
    hook: HookPoint,
    recv_cache: ActivationCache,
    receiver_head: HeadRef,
    channels: Sequence[str]
):
    # activation: [batch, seq, n_heads, d_head] (对应 q/k/v 钩子之一)
    if hook.layer() != receiver_head.layer: return activation
    # 把指定接收头的通道整体替换为 recv_cache 中的值
    activation[:,:,receiver_head.head] = recv_cache[hook.name][:,:,receiver_head.head]
    return activation

# --------------- Core path patch: single channel --------------- #

def path_patch_to_head_single_channel(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_input: str,
    orig_dataset: IOIDataset,
    new_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    """
    返回矩阵 [receiver_layer, n_heads]：每个 sender (layer<head) 的 delta。
    """
    receiver_layer = receiver_head.layer
    z_filter = lambda name: name.endswith("z")
    recv_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    recv_filter = lambda name: name == recv_hook_name
    results = t.zeros(receiver_layer, model.cfg.n_heads, device="cpu")
    for sender_layer in range(receiver_layer):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            def z_hook(tensor, hook):   # 原: z_hook(tensor, hk)
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
                )
            model.add_hook(z_filter, z_hook, level=1)
            with t.no_grad():
                _, recv_cache = model.run_with_cache(
                    orig_dataset.toks,
                    names_filter=recv_filter,
                    return_type=None
                )
            model.reset_hooks()
            def recv_hook_fn(tensor, hook):  # 原: recv_hook_fn(tensor, hk)
                return patch_receiver_inputs(tensor, hook, recv_cache, receiver_head, [receiver_input])
            with t.no_grad():
                patched_logits = model.run_with_hooks(
                    orig_dataset.toks,
                    fwd_hooks=[(recv_filter, recv_hook_fn)],
                    return_type="logits"
                )
            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta
    model.reset_hooks()
    return results

def path_patch_sender_to_head_single_channel(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_input: str,
    sender_layer: int,
    sender_head: int,
    orig_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> float:
    """
    Single sender-head version of path_patch_to_head_single_channel.
    Returns delta_logit_diff (patched - baseline).
    """
    z_filter = lambda name: name.endswith("z")
    recv_hook_name = utils.get_act_name(receiver_input, receiver_head.layer)
    recv_filter = lambda name: name == recv_hook_name

    model.reset_hooks()
    model.add_hook(
        z_filter,
        lambda tensor, hook: patch_or_freeze_head_vectors(
            tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
        ),
        level=1,
    )
    with t.no_grad():
        _, recv_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=recv_filter,
            return_type=None,
        )
    model.reset_hooks()
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks=[(recv_filter, lambda tensor, hook: patch_receiver_inputs(tensor, hook, recv_cache, receiver_head, [receiver_input]))],
            return_type="logits",
        )
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline

# --------------- Core path patch: joint channels --------------- #

def path_patch_to_head_joint(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_channels: Sequence[str],
    orig_dataset: IOIDataset,
    new_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    """
    同时替换 receiver_channels (q,k,v) —— 联合效果。
    返回矩阵 [receiver_layer, n_heads]
    """
    receiver_layer = receiver_head.layer
    z_filter = lambda name: name.endswith("z")
    recv_hook_names = {utils.get_act_name(ch, receiver_layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names
    results = t.zeros(receiver_layer, model.cfg.n_heads, device="cpu")
    for sender_layer in range(receiver_layer):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            def z_hook(tensor, hook):  # 原 hk
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
                )
            model.add_hook(z_filter, z_hook, level=1)
            with t.no_grad():
                _, recv_cache = model.run_with_cache(
                    orig_dataset.toks,
                    names_filter=recv_filter,
                    return_type=None
                )
            model.reset_hooks()
            def mk_recv_hook(hname: str):
                def fn(tensor, hook):   # 原 hk
                    if hook.name == hname:
                        tensor[:,:,receiver_head.head] = recv_cache[hname][:,:,receiver_head.head]
                    return tensor
                return fn
            fwd_hooks = [(lambda n, h=hname: n==h, mk_recv_hook(hname)) for hname in recv_hook_names]
            with t.no_grad():
                patched_logits = model.run_with_hooks(
                    orig_dataset.toks,
                    fwd_hooks=fwd_hooks,
                    return_type="logits"
                )
            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta
    model.reset_hooks()
    return results

def path_patch_sender_to_head_joint(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_channels: Sequence[str],
    sender_layer: int,
    sender_head: int,
    orig_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> float:
    """
    Single sender-head version of path_patch_to_head_joint.
    Returns delta_logit_diff (patched - baseline).
    """
    z_filter = lambda name: name.endswith("z")
    receiver_layer = receiver_head.layer
    recv_hook_names = {utils.get_act_name(ch, receiver_layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names

    model.reset_hooks()
    model.add_hook(
        z_filter,
        lambda tensor, hook: patch_or_freeze_head_vectors(
            tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
        ),
        level=1,
    )
    with t.no_grad():
        _, recv_cache = model.run_with_cache(
            orig_dataset.toks,
            names_filter=recv_filter,
            return_type=None,
        )
    model.reset_hooks()

    def mk_recv_hook(hname: str):
        def fn(tensor, hook):
            if hook.name == hname:
                tensor[:, :, receiver_head.head] = recv_cache[hname][:, :, receiver_head.head]
            return tensor

        return fn

    fwd_hooks = [(lambda n, h=hname: n == h, mk_recv_hook(hname)) for hname in recv_hook_names]
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks=fwd_hooks,
            return_type="logits",
        )
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline

def path_patch_input_to_head_single_channel(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_input: str,
    orig_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_resid_cache: ActivationCache,
    baseline: float,
) -> float:
    """
    sender='input' → receiver head input (q/k/v).
    用 corrupted prompts 的 layer0 resid_pre 替换 clean 的 layer0 resid_pre，同时冻结所有 head 的 z 为 clean，
    再按 path patching 算 receiver node 并注入，返回 delta_logit_diff。
    """
    receiver_layer = receiver_head.layer
    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name
    recv_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    recv_filter = lambda name: name == recv_hook_name
    z_filter = lambda name: name.endswith("z")

    model.reset_hooks()
    model.add_hook(resid0_filter, lambda x, hook: patch_resid_from_cache(x, hook, new_resid_cache), level=0)
    model.add_hook(z_filter, lambda x, hook: freeze_all_head_vectors(x, hook, orig_cache), level=1)
    with t.no_grad():
        _, recv_cache = model.run_with_cache(orig_dataset.toks, names_filter=recv_filter, return_type=None)
    model.reset_hooks()
    def recv_hook_fn(tensor, hook):
        return patch_receiver_inputs(tensor, hook, recv_cache, receiver_head, [receiver_input])
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks=[(recv_filter, recv_hook_fn)],
            return_type="logits",
        )
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline


def _cache_receiver_input_under_input_patch_single(
    model: HookedTransformer,
    receiver_layer: int,
    receiver_input: str,
    orig_dataset: IOIDataset,
    orig_z_cache: ActivationCache,
    new_resid0_cache: ActivationCache,
) -> ActivationCache:
    """
    Returns a cache containing the receiver input activation (q/k/v) at receiver_layer for all heads,
    under the 'input' sender patch (resid_pre@0 from corrupted) while freezing all head z to clean.
    """
    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name
    recv_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    recv_filter = lambda name: name == recv_hook_name
    z_filter = lambda name: name.endswith("z")

    model.reset_hooks()
    model.add_hook(resid0_filter, lambda x, hook: patch_resid_from_cache(x, hook, new_resid0_cache), level=0)
    model.add_hook(z_filter, lambda x, hook: freeze_all_head_vectors(x, hook, orig_z_cache), level=1)
    with t.no_grad():
        _, recv_cache = model.run_with_cache(orig_dataset.toks, names_filter=recv_filter, return_type=None)
    model.reset_hooks()
    return recv_cache


def _cache_receiver_input_under_input_patch_joint(
    model: HookedTransformer,
    receiver_layer: int,
    receiver_channels: Sequence[str],
    orig_dataset: IOIDataset,
    orig_z_cache: ActivationCache,
    new_resid0_cache: ActivationCache,
) -> ActivationCache:
    """
    Joint version: caches multiple receiver inputs (q/k/v) at receiver_layer for all heads,
    under the input patch + z-freeze setting.
    """
    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name
    recv_hook_names = {utils.get_act_name(ch, receiver_layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names
    z_filter = lambda name: name.endswith("z")

    model.reset_hooks()
    model.add_hook(resid0_filter, lambda x, hook: patch_resid_from_cache(x, hook, new_resid0_cache), level=0)
    model.add_hook(z_filter, lambda x, hook: freeze_all_head_vectors(x, hook, orig_z_cache), level=1)
    with t.no_grad():
        _, recv_cache = model.run_with_cache(orig_dataset.toks, names_filter=recv_filter, return_type=None)
    model.reset_hooks()
    return recv_cache


def _delta_from_cached_receiver_input_single(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_input: str,
    orig_dataset: IOIDataset,
    recv_cache: ActivationCache,
    baseline: float,
) -> float:
    recv_hook_name = utils.get_act_name(receiver_input, receiver_head.layer)
    recv_filter = lambda name: name == recv_hook_name

    def recv_hook_fn(tensor, hook):
        return patch_receiver_inputs(tensor, hook, recv_cache, receiver_head, [receiver_input])

    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks=[(recv_filter, recv_hook_fn)],
            return_type="logits",
        )
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline


def _delta_from_cached_receiver_input_joint(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_channels: Sequence[str],
    orig_dataset: IOIDataset,
    recv_cache: ActivationCache,
    baseline: float,
) -> float:
    recv_hook_names = {utils.get_act_name(ch, receiver_head.layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names

    def mk_recv_hook(hname: str):
        def fn(tensor, hook):
            if hook.name == hname:
                tensor[:, :, receiver_head.head] = recv_cache[hname][:, :, receiver_head.head]
            return tensor

        return fn

    fwd_hooks = [(lambda n, h=hname: n == h, mk_recv_hook(hname)) for hname in recv_hook_names]
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks=fwd_hooks,
            return_type="logits",
        )
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline

def path_patch_input_to_head_joint(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_channels: Sequence[str],
    orig_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_resid_cache: ActivationCache,
    baseline: float,
) -> float:
    """
    sender='input' → receiver head inputs (q/k/v) joint.
    """
    receiver_layer = receiver_head.layer
    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name
    recv_hook_names = {utils.get_act_name(ch, receiver_layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names
    z_filter = lambda name: name.endswith("z")

    model.reset_hooks()
    model.add_hook(resid0_filter, lambda x, hook: patch_resid_from_cache(x, hook, new_resid_cache), level=0)
    model.add_hook(z_filter, lambda x, hook: freeze_all_head_vectors(x, hook, orig_cache), level=1)
    with t.no_grad():
        _, recv_cache = model.run_with_cache(orig_dataset.toks, names_filter=recv_filter, return_type=None)
    model.reset_hooks()
    def mk_recv_hook(hname: str):
        def fn(tensor, hook):
            if hook.name == hname:
                tensor[:, :, receiver_head.head] = recv_cache[hname][:, :, receiver_head.head]
            return tensor
        return fn
    fwd_hooks = [(lambda n, h=hname: n == h, mk_recv_hook(hname)) for hname in recv_hook_names]
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks,
            fwd_hooks=fwd_hooks,
            return_type="logits",
        )
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline

def path_patch_input_to_logits(
    model: HookedTransformer,
    orig_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_resid_cache: ActivationCache,
    baseline: float,
) -> float:
    """
    sender='input' → logits: patch layer0 resid_pre from corrupted while freezing all z to clean.
    """
    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name
    z_filter = lambda name: name.endswith("z")
    final_resid = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    final_filter = lambda name: name == final_resid

    model.reset_hooks()
    model.add_hook(resid0_filter, lambda x, hook: patch_resid_from_cache(x, hook, new_resid_cache), level=0)
    model.add_hook(z_filter, lambda x, hook: freeze_all_head_vectors(x, hook, orig_cache), level=1)
    with t.no_grad():
        _, cache = model.run_with_cache(orig_dataset.toks, names_filter=final_filter, return_type=None)
    patched_logits = model.unembed(model.ln_final(cache[final_resid]))
    return logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline

# --------------- Residual (to logits) --------------- #

def path_patch_to_residual(
    model: HookedTransformer,
    orig_dataset: IOIDataset,
    new_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    z_filter = lambda name: name.endswith("z")
    final_resid = utils.get_act_name("resid_post", model.cfg.n_layers-1)
    final_filter = lambda name: name == final_resid
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cpu")
    for sender_layer in range(model.cfg.n_layers):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            def z_hook(tensor, hook):  # 原 hk
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
                )
            model.add_hook(z_filter, z_hook)
            with t.no_grad():
                _, cache = model.run_with_cache(
                    orig_dataset.toks,
                    names_filter=final_filter,
                    return_type=None
                )
            patched_logits = model.unembed(model.ln_final(cache[final_resid]))
            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta
    model.reset_hooks()
    return results

# --------------- Orchestration --------------- #

def run_edge_path_patching(
    collapsed_graph: Path,
    output_path: Path,
    prompt_type: str,
    dataset_size: int,
    seed: int,
    channels: Sequence[str],
    joint_patch: bool,
    device: str,
    full_graph: Optional[Path]=None
):
    specs = load_edge_specs(collapsed_graph, channels, full_graph, joint_patch)
    if not specs:
        raise ValueError("没有生成任何 EdgeSpec，检查 collapsed_graph / 参数。")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ioi_ds = IOIDataset(
        prompt_type=prompt_type,
        N=dataset_size,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=device
    )
    abc_ds = ioi_ds.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

    with t.no_grad():
        clean_logits = model(ioi_ds.toks)
        baseline = logits_to_logit_diff(clean_logits, ioi_ds).item()

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        _, orig_cache = model.run_with_cache(ioi_ds.toks, names_filter=z_filter, return_type=None)
        _, new_cache  = model.run_with_cache(abc_ds.toks, names_filter=z_filter, return_type=None)

    # Cache layer0 resid_pre for input sender patching.
    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name
    with t.no_grad():
        _, orig_resid0_cache = model.run_with_cache(ioi_ds.toks, names_filter=resid0_filter, return_type=None)
        _, new_resid0_cache = model.run_with_cache(abc_ds.toks, names_filter=resid0_filter, return_type=None)

    need_logits = any(s.target_kind=="logits" for s in specs)
    residual_matrix = None
    if need_logits:
        residual_matrix = path_patch_to_residual(
            model, ioi_ds, abc_ds, orig_cache, new_cache, baseline
        )

    # 分组：接收头 × 通道
    att_specs = [s for s in specs if s.target_kind=="attention"]
    grouped: Dict[Tuple[HeadRef,str], List[EdgeSpec]] = defaultdict(list)
    for s in att_specs:
        if s.dst is None or s.receiver_input is None:
            continue
        grouped[(s.dst, s.receiver_input)].append(s)

    results = []
    input_recv_cache_single: Dict[Tuple[int, str], ActivationCache] = {}
    input_recv_cache_joint: Dict[Tuple[int, Tuple[str, ...]], ActivationCache] = {}

    # 先处理单通道（排除联合标签 qkv）
    single_keys = [(rh,ch) for (rh,ch) in grouped if ch in ("q","k","v")]
    for (recv_head, recv_ch) in tqdm(single_keys, desc="Single-channel receivers"):
        has_input = False
        head_specs = []
        for spec in grouped[(recv_head, recv_ch)]:
            if spec.src.kind == "input":
                has_input = True
            else:
                head_specs.append(spec)

        # Only compute sender-head deltas for sender heads that actually appear in the collapsed graph for this receiver.
        for spec in head_specs:
            assert spec.src.head is not None
            delta = path_patch_sender_to_head_single_channel(
                model,
                recv_head,
                recv_ch,
                spec.src.head.layer,
                spec.src.head.head,
                ioi_ds,
                orig_cache,
                new_cache,
                baseline,
            )
            results.append({
                "edge": spec.label(),
                "receiver_kind": "attention",
                "src": spec.src.as_name(),
                "dst": spec.dst.as_name(),
                "receiver_input": spec.receiver_input,
                "delta_logit_diff": float(delta),
                "patched_logit_diff": baseline + float(delta),
                "joint": False
            })
        if has_input:
            cache_key = (recv_head.layer, recv_ch)
            recv_cache = input_recv_cache_single.get(cache_key)
            if recv_cache is None:
                recv_cache = _cache_receiver_input_under_input_patch_single(
                    model, recv_head.layer, recv_ch, ioi_ds, orig_cache, new_resid0_cache
                )
                input_recv_cache_single[cache_key] = recv_cache
            delta_in = _delta_from_cached_receiver_input_single(
                model, recv_head, recv_ch, ioi_ds, recv_cache, baseline
            )
            results.append({
                "edge": f"input->{recv_head.as_name()}<{recv_ch}>",
                "receiver_kind": "attention",
                "src": "input",
                "dst": recv_head.as_name(),
                "receiver_input": recv_ch,
                "delta_logit_diff": float(delta_in),
                "patched_logit_diff": baseline + float(delta_in),
                "joint": False
            })

    # 联合 (qkv)
    joint_keys = [(rh,ch) for (rh,ch) in grouped if ch=="qkv"]
    if joint_keys:
        # 收集真实可用通道（可能 full_graph 只给了 q,v 等）
        recv_to_channels: Dict[HeadRef, set] = defaultdict(set)
        for (recv_head, recv_ch) in single_keys:
            recv_to_channels[recv_head].add(recv_ch)
        for (recv_head, _ch) in tqdm(joint_keys, desc="Joint receivers"):
            chs = sorted(recv_to_channels.get(recv_head, set(channels)))
            if len(chs) < 2:
                # 没有足够多的实际单通道记录，跳过
                continue
            # 找到对应联合 spec 列表
            has_input = False
            for spec in grouped[(recv_head, "qkv")]:
                if spec.src.kind == "input":
                    has_input = True
                    continue
                assert spec.src.head is not None
                delta = path_patch_sender_to_head_joint(
                    model,
                    recv_head,
                    chs,
                    spec.src.head.layer,
                    spec.src.head.head,
                    ioi_ds,
                    orig_cache,
                    new_cache,
                    baseline,
                )
                results.append({
                    "edge": spec.label(),
                    "receiver_kind": "attention",
                    "src": spec.src.as_name(),
                    "dst": spec.dst.as_name(),
                    "receiver_input": "qkv",
                    "channels": chs,
                    "delta_logit_diff": delta,
                    "patched_logit_diff": baseline + delta,
                    "joint": True
                })
            if has_input:
                cache_key = (recv_head.layer, tuple(chs))
                recv_cache = input_recv_cache_joint.get(cache_key)
                if recv_cache is None:
                    recv_cache = _cache_receiver_input_under_input_patch_joint(
                        model, recv_head.layer, chs, ioi_ds, orig_cache, new_resid0_cache
                    )
                    input_recv_cache_joint[cache_key] = recv_cache
                delta_in = _delta_from_cached_receiver_input_joint(
                    model, recv_head, chs, ioi_ds, recv_cache, baseline
                )
                results.append({
                    "edge": f"input->{recv_head.as_name()}<qkv>",
                    "receiver_kind": "attention",
                    "src": "input",
                    "dst": recv_head.as_name(),
                    "receiver_input": "qkv",
                    "channels": chs,
                    "delta_logit_diff": float(delta_in),
                    "patched_logit_diff": baseline + float(delta_in),
                    "joint": True
                })

    # logits 边
    for spec in specs:
        if spec.target_kind != "logits":
            continue
        if spec.src.kind == "input":
            delta = float(path_patch_input_to_logits(model, ioi_ds, orig_cache, new_resid0_cache, baseline))
            src_name = "input"
        else:
            assert spec.src.head is not None
            delta = float(residual_matrix[spec.src.head.layer, spec.src.head.head])
            src_name = spec.src.as_name()
        results.append({
            "edge": spec.label(),
            "receiver_kind": "logits",
            "src": src_name,
            "dst": "logits",
            "receiver_input": None,
            "delta_logit_diff": delta,
            "patched_logit_diff": baseline + delta,
            "joint": False
        })

    payload = {
        "meta": {
            "collapsed_graph": str(collapsed_graph),
            "full_graph": str(full_graph) if full_graph else None,
            "prompt_type": prompt_type,
            "dataset_size": dataset_size,
            "seed": seed,
            "baseline_logit_diff": baseline,
            "channels": list(channels),
            "joint_patch": joint_patch
        },
        "edges": results
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[DONE] 写出 {len(results)} 条记录 → {output_path}")

# --------------- CLI --------------- #

def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-channel (q,k,v,qkv) edge path patching")
    p.add_argument("--collapsed-graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-graph", type=Path, help="含 qkv 标注的完整图（可选）")
    p.add_argument("--prompt-type", type=str, default="mixed")
    p.add_argument("--dataset-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--channels", nargs="+", default=["q"], choices=["q","k","v"],
                   help="生成的单通道集合（默认仅 q）")
    p.add_argument("--joint-patch", action="store_true",
                   help="同时生成联合 qkv 边（需 channels 数>1）")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]]=None):
    args = parse_args(argv)
    run_edge_path_patching(
        collapsed_graph=args.collapsed_graph,
        output_path=args.output,
        prompt_type=args.prompt_type,
        dataset_size=args.dataset_size,
        seed=args.seed,
        channels=args.channels,
        joint_patch=args.joint_patch,
        device=args.device,
        full_graph=args.full_graph
    )

if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
