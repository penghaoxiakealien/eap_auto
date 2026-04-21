from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal
from collections import defaultdict

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils, loading_from_pretrained as loading
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from garden_dataset import GardenDataset
from task_dataset_base import TaskDatasetBase


LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


# ---------------- Dataclasses ---------------- #
@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int
    @classmethod
    def from_name(cls, name: str) -> "HeadRef":
        l, h = name.split(".")
        return cls(layer=int(l[1:]), head=int(h[1:]))
    def as_name(self) -> str:
        return f"a{self.layer}.h{self.head}"

@dataclass(frozen=True)
class EdgeSpec:
    src: HeadRef
    target_kind: Literal["attention","logits"]
    dst: Optional[HeadRef] = None
    receiver_input: Optional[str] = None   # q/k/v/qkv/None
    def label(self) -> str:
        if self.target_kind == "attention":
            dst_name = self.dst.as_name() if self.dst else "UNKNOWN"
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
    joint_patch: bool=False,
    joint_only: bool=False,
) -> List[EdgeSpec]:
    if joint_only:
        channels = []
    else:
        valid = {"q","k","v"}
        for c in channels:
            if c not in valid:
                raise ValueError(f"非法通道 {c}")
    edges = load_collapsed_edges(collapsed_graph)

    specs: List[EdgeSpec] = []
    for src, dst in edges:
        if src == "input":
            if dst.startswith("a") and joint_patch:
                specs.append(
                    EdgeSpec(
                        src=HeadRef(layer=-1, head=-1),  # 特殊占位，不用于索引
                        dst=HeadRef.from_name(dst),
                        target_kind="attention",
                        receiver_input="qkv",
                    )
                )
            continue
        if not src.startswith("a"):
            continue
        if dst == "logits":
            specs.append(EdgeSpec(src=HeadRef.from_name(src), target_kind="logits"))
        elif dst.startswith("a"):
            for ch in channels:
                specs.append(EdgeSpec(
                    src=HeadRef.from_name(src),
                    dst=HeadRef.from_name(dst),
                    target_kind="attention",
                    receiver_input=ch,
                ))
            if joint_patch and (joint_only or len(channels) > 1):
                specs.append(EdgeSpec(
                    src=HeadRef.from_name(src),
                    dst=HeadRef.from_name(dst),
                    target_kind="attention",
                    receiver_input="qkv",
                ))
    return specs


# --------------- Metric ---------------- #
def logits_to_logit_diff(logits: t.Tensor, dataset: TaskDatasetBase, per_prompt=False) -> t.Tensor:
    diff = dataset.logit_diff(logits, mean=not per_prompt, loss=False)
    return diff


# --------------- Patch primitives ---------------- #
def patch_or_freeze_head_vectors(
    head_out: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int,int],
):
    head_out[...] = orig_cache[hook.name][...]
    if hook.layer() == head_to_patch[0]:
        head_out[:,:,head_to_patch[1]] = new_cache[hook.name][:,:,head_to_patch[1]]
    return head_out

def patch_receiver_inputs(
    activation: t.Tensor,
    hook: HookPoint,
    recv_cache: ActivationCache,
    receiver_head: HeadRef,
    channels: Sequence[str],
):
    if hook.layer() != receiver_head.layer:
        return activation
    activation[:,:,receiver_head.head] = recv_cache[hook.name][:,:,receiver_head.head]
    return activation


# --------------- Core path patch: single channel --------------- #
def path_patch_to_head_single_channel(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_input: str,
    orig_dataset: TaskDatasetBase,
    new_dataset: TaskDatasetBase,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    receiver_layer = receiver_head.layer
    z_filter = lambda name: name.endswith("z")
    recv_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    recv_filter = lambda name: name == recv_hook_name
    results = t.zeros(receiver_layer, model.cfg.n_heads, device="cpu")
    for sender_layer in range(receiver_layer):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            def z_hook(tensor, hook):
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
                )
            model.add_hook(z_filter, z_hook, level=1)
            with t.no_grad():
                _, recv_cache = model.run_with_cache(
                    orig_dataset.toks,
                    names_filter=recv_filter,
                    return_type=None,
                )
            model.reset_hooks()
            def recv_hook_fn(tensor, hook):
                return patch_receiver_inputs(tensor, hook, recv_cache, receiver_head, [receiver_input])
            with t.no_grad():
                patched_logits = model.run_with_hooks(
                    orig_dataset.toks,
                    fwd_hooks=[(recv_filter, recv_hook_fn)],
                    return_type="logits",
                )
            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta
    model.reset_hooks()
    return results


# --------------- Core path patch: joint channels --------------- #
def path_patch_to_head_joint(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_channels: Sequence[str],
    orig_dataset: TaskDatasetBase,
    new_dataset: TaskDatasetBase,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    receiver_layer = receiver_head.layer
    z_filter = lambda name: name.endswith("z")
    recv_hook_names = {utils.get_act_name(ch, receiver_layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names
    results = t.zeros(receiver_layer, model.cfg.n_heads, device="cpu")
    for sender_layer in range(receiver_layer):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            def z_hook(tensor, hook):
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
                )
            model.add_hook(z_filter, z_hook, level=1)
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
                        tensor[:,:,receiver_head.head] = recv_cache[hname][:,:,receiver_head.head]
                    return tensor
                return fn
            fwd_hooks = [(lambda n, h=hname: n==h, mk_recv_hook(hname)) for hname in recv_hook_names]
            with t.no_grad():
                patched_logits = model.run_with_hooks(
                    orig_dataset.toks,
                    fwd_hooks=fwd_hooks,
                    return_type="logits",
                )
            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta
    model.reset_hooks()
    return results


# --------------- Core path patch: input -> head (joint qkv only) --------------- #
def path_patch_input_to_head_joint(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_channels: Sequence[str],
    orig_dataset: TaskDatasetBase,
    new_dataset: TaskDatasetBase,
    baseline: float,
) -> float:
    receiver_layer = receiver_head.layer
    recv_hook_names = {utils.get_act_name(ch, receiver_layer) for ch in receiver_channels}
    recv_filter = lambda name: name in recv_hook_names

    resid0_name = utils.get_act_name("resid_pre", 0)
    resid0_filter = lambda name: name == resid0_name

    def patch_input(act: t.Tensor, hook: HookPoint):
        act[...] = new_cache[hook.name][...]
        return act

    with t.no_grad():
        _, new_cache = model.run_with_cache(
            new_dataset.toks, names_filter=resid0_filter, return_type=None
        )
        with model.hooks(fwd_hooks=[(resid0_filter, patch_input)]):
            _, recv_cache = model.run_with_cache(
                orig_dataset.toks, names_filter=recv_filter, return_type=None
            )

    def mk_recv_hook(hname: str):
        def fn(tensor, hook):
            if hook.name == hname:
                tensor[:, :, receiver_head.head] = recv_cache[hname][:, :, receiver_head.head]
            return tensor
        return fn

    fwd_hooks = [(lambda n, h=hname: n == h, mk_recv_hook(hname)) for hname in recv_hook_names]
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            orig_dataset.toks, fwd_hooks=fwd_hooks, return_type="logits"
        )
    delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
    return float(delta)


# --------------- Residual (to logits) --------------- #
def path_patch_to_residual(
    model: HookedTransformer,
    orig_dataset: TaskDatasetBase,
    new_dataset: TaskDatasetBase,
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
            def z_hook(tensor, hook):
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (sender_layer, sender_head)
                )
            model.add_hook(z_filter, z_hook)
            with t.no_grad():
                _, cache = model.run_with_cache(
                    orig_dataset.toks,
                    names_filter=final_filter,
                    return_type=None,
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
    dataset: GardenDataset,
    seed: int,
    channels: Sequence[str],
    joint_patch: bool,
    joint_only: bool=False,
    device: str = "cuda",
):
    specs = load_edge_specs(collapsed_graph, channels, joint_patch, joint_only)
    if not specs:
        raise ValueError("没有生成任何 EdgeSpec，检查 collapsed_graph / 参数。")

    model = load_model(device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # dataset 已经构造好，生成 flipped 版本
    flipped_ds = dataset.gen_flipped_prompts()

    with t.no_grad():
        clean_logits = model(dataset.toks)
        baseline = logits_to_logit_diff(clean_logits, dataset).item()

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        _, orig_cache = model.run_with_cache(dataset.toks, names_filter=z_filter, return_type=None)
        _, new_cache  = model.run_with_cache(flipped_ds.toks, names_filter=z_filter, return_type=None)

    need_logits = any(s.target_kind=="logits" for s in specs)
    residual_matrix = None
    if need_logits:
        residual_matrix = path_patch_to_residual(
            model, dataset, flipped_ds, orig_cache, new_cache, baseline
        )

    att_specs = [s for s in specs if s.target_kind=="attention"]
    grouped: Dict[Tuple[HeadRef,str], List[EdgeSpec]] = defaultdict(list)
    for s in att_specs:
        grouped[(s.dst, s.receiver_input)].append(s)

    results = []

    # 单通道（sender=head）
    if not joint_only:
        single_keys = [(rh,ch) for (rh,ch) in grouped if ch in ("q","k","v")]
        for (recv_head, recv_ch) in tqdm(single_keys, desc="Single-channel receivers"):
            mat = path_patch_to_head_single_channel(
                model, recv_head, recv_ch, dataset, flipped_ds, orig_cache, new_cache, baseline
            )
            for spec in grouped[(recv_head, recv_ch)]:
                delta = float(mat[spec.src.layer, spec.src.head])
                results.append({
                    "edge": spec.label(),
                    "receiver_kind": "attention",
                    "src": spec.src.as_name(),
                    "dst": spec.dst.as_name(),
                    "receiver_input": spec.receiver_input,
                    "delta_logit_diff": delta,
                    "patched_logit_diff": baseline + delta,
                    "joint": False,
                })

    # 联合 qkv（sender=head + sender=input）
    joint_keys = [(rh,ch) for (rh,ch) in grouped if ch=="qkv"]
    if joint_keys:
        for (recv_head, _ch) in tqdm(joint_keys, desc="Joint receivers"):
            mat = path_patch_to_head_joint(
                model, recv_head, ["q","k","v"], dataset, flipped_ds, orig_cache, new_cache, baseline
            )
            for spec in grouped[(recv_head, "qkv")]:
                # 约定：src layer/head 为 -1/-1 表示 input
                if spec.src.layer == -1 and spec.src.head == -1:
                    delta = path_patch_input_to_head_joint(
                        model, recv_head, ["q", "k", "v"], dataset, flipped_ds, baseline
                    )
                    results.append(
                        {
                            "edge": f"input->{spec.dst.as_name()}<qkv>",
                            "receiver_kind": "attention",
                            "src": "input",
                            "dst": spec.dst.as_name(),
                            "receiver_input": "qkv",
                            "delta_logit_diff": float(delta),
                            "patched_logit_diff": baseline + float(delta),
                            "joint": True,
                        }
                    )
                else:
                    delta = float(mat[spec.src.layer, spec.src.head])
                    results.append({
                        "edge": spec.label(),
                        "receiver_kind": "attention",
                        "src": spec.src.as_name(),
                        "dst": spec.dst.as_name(),
                        "receiver_input": spec.receiver_input,
                        "delta_logit_diff": delta,
                        "patched_logit_diff": baseline + delta,
                        "joint": True,
                    })

    # logits 边
    if need_logits and residual_matrix is not None:
        logits_specs = [s for s in specs if s.target_kind=="logits"]
        for spec in logits_specs:
            delta = float(residual_matrix[spec.src.layer, spec.src.head])
            results.append({
                "edge": spec.label(),
                "receiver_kind": "logits",
                "src": spec.src.as_name(),
                "dst": "logits",
                "receiver_input": None,
                "delta_logit_diff": delta,
                "patched_logit_diff": baseline + delta,
                "joint": False,
            })

    payload = {
        "meta": {
            "collapsed_graph": str(collapsed_graph),
            "dataset_size": len(dataset),
            "baseline_logit_diff": baseline,
            "channels": (["qkv"] if joint_only else list(channels)),
            "joint_patch": joint_patch,
            "joint_only": joint_only,
        },
        "edges": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[DONE] 写出 {len(results)} 条记录 → {output_path}")


# --------------- CLI --------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Path patching for garden NPZ v-trans (q/k/v + optional qkv)")
    p.add_argument("--collapsed-graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True, help="garden 数据文件")
    p.add_argument("--dataset-size", type=int, default=64, help="采样多少条用于 patching（0=用全部）")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--channels", nargs="+", default=["q","k","v"])
    p.add_argument("--joint-patch", action="store_true")
    p.add_argument("--joint-only", action="store_true", help="只做 joint qkv patching（忽略单通道）")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_local_hooked_transformer(local_model_dir: str, device: str = "cuda") -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)
    cfg = loading.get_pretrained_model_config(
        local_model_dir,
        device=device,
        local_files_only=True,
    )
    model = HookedTransformer(
        cfg,
        tokenizer=tokenizer,
        move_to_device=False,
    )
    state_dict = loading.get_pretrained_state_dict(
        local_model_dir,
        cfg,
        hf_model=hf_model,
        local_files_only=True,
    )
    model.load_and_process_state_dict(state_dict)
    model.move_model_modules_to_device()
    return model


def load_model(device: str = "cuda") -> HookedTransformer:
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return load_local_hooked_transformer(LOCAL_MODEL_DIR, device=device)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True) if os.path.isdir(LOCAL_MODEL_DIR) else AutoTokenizer.from_pretrained("gpt2")
    ds = GardenDataset(
        tokenizer=tokenizer,
        data_path=args.data_path,
        device=args.device,
        seed=args.seed,
    )
    if args.dataset_size and args.dataset_size > 0:
        ds.samples = ds.samples[:args.dataset_size]
        ds = GardenDataset(
            tokenizer=tokenizer,
            samples=ds.samples,
            device=args.device,
            seed=args.seed,
        )

    run_edge_path_patching(
        collapsed_graph=args.collapsed_graph,
        output_path=args.output,
        dataset=ds,
        seed=args.seed,
        channels=args.channels,
        joint_patch=args.joint_patch,
        joint_only=args.joint_only,
        device=args.device,
    )


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
