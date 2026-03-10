#!/usr/bin/env python3
"""
泛化版 edge path patching：依赖 TaskDatasetBase 提供 toks/labels/logit_diff，不再硬编码 IOI。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from task_dataset_base import TaskDatasetBase


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
    target_kind: Literal["attention", "logits"]
    dst: Optional[HeadRef] = None
    receiver_input: Optional[str] = None

    def label(self) -> str:
        if self.target_kind == "attention":
            dst_name = self.dst.as_name() if self.dst is not None else "UNKNOWN"
            return f"{self.src.as_name()}->{dst_name}<{self.receiver_input}>"
        return f"{self.src.as_name()}->logits"


def load_collapsed_edges(path: Path) -> List[Tuple[str, str]]:
    data = json.loads(path.read_text())
    edges = data.get("edge_list")
    if not isinstance(edges, list):
        raise ValueError("collapsed graph 缺少 edge_list 或格式错误")
    return [(a, b) for a, b in edges]


def load_edge_specs(collapsed_graph: Path) -> List[EdgeSpec]:
    edges = load_collapsed_edges(collapsed_graph)
    specs: List[EdgeSpec] = []
    for src, dst in edges:
        if not src.startswith("a"):
            continue
        if dst == "logits":
            specs.append(EdgeSpec(src=HeadRef.from_name(src), target_kind="logits"))
        elif dst.startswith("a"):
            specs.append(
                EdgeSpec(
                    src=HeadRef.from_name(src),
                    dst=HeadRef.from_name(dst),
                    target_kind="attention",
                    receiver_input="q",  # 默认用 q，若有需要可扩展
                )
            )
    return specs


def logits_to_logit_diff(logits: t.Tensor, dataset: TaskDatasetBase, per_prompt=False) -> t.Tensor:
    # 默认使用 dataset.logit_diff
    diff = dataset.logit_diff(logits, mean=not per_prompt, loss=False)
    return diff


def patch_or_freeze_head_vectors(
    head_out: t.Tensor,
    hook: HookPoint,
    new_cache: Dict[str, t.Tensor],
    orig_cache: Dict[str, t.Tensor],
    head_to_patch: Tuple[int, int],
):
    head_out[...] = orig_cache[hook.name][...]
    if hook.layer() == head_to_patch[0]:
        head_out[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return head_out


def logits_delta_matrix(
    model: HookedTransformer,
    dataset: TaskDatasetBase,
    orig_cache: Dict[str, t.Tensor],
    new_cache: Dict[str, t.Tensor],
    baseline: float,
) -> t.Tensor:
    """
    对每个 sender head 计算“将该头替换为 corrupted 版本”对 logits 的 delta。
    返回形状 [n_layers, n_heads] 的矩阵。
    """
    z_filter = lambda name: name.endswith("z")
    L, H = model.cfg.n_layers, model.cfg.n_heads
    results = t.zeros(L, H, device="cpu")
    for layer in range(L):
        for head in range(H):
            model.reset_hooks()
            def z_hook(tensor, hook, layer=layer, head=head):
                return patch_or_freeze_head_vectors(
                    tensor, hook, new_cache, orig_cache, (layer, head)
                )
            model.add_hook(z_filter, z_hook, level=1)
            with t.no_grad():
                patched_logits = model(dataset.toks)
            model.reset_hooks()
            patched_diff = logits_to_logit_diff(patched_logits, dataset, per_prompt=True)
            delta = patched_diff - baseline
            results[layer, head] = delta.mean().cpu()
    return results


def path_patch_to_head_single(
    model: HookedTransformer,
    receiver_head: HeadRef,
    orig_ds: TaskDatasetBase,
    new_ds: TaskDatasetBase,
    orig_cache: Dict[str, t.Tensor],
    new_cache: Dict[str, t.Tensor],
    baseline: float,
) -> t.Tensor:
    receiver_layer = receiver_head.layer
    z_filter = lambda name: name.endswith("z")
    recv_filter = lambda name: name == utils.get_act_name("q", receiver_layer)
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
                    orig_ds.toks,
                    names_filter=recv_filter,
                    return_type=None
                )
            model.reset_hooks()
            patched_logits = model(orig_ds.toks)
            patched_diff = logits_to_logit_diff(patched_logits, orig_ds, per_prompt=True)
            delta = patched_diff - baseline
            results[sender_layer, sender_head] = delta.mean().cpu()
    return results


def run_edge_path_patching(
    collapsed_graph: Path,
    output_path: Path,
    dataset: TaskDatasetBase,
    prompt_type: str,
    seed: int,
    device: str,
):
    specs = load_edge_specs(collapsed_graph)
    if not specs:
        raise ValueError("没有生成任何 EdgeSpec，检查 collapsed_graph。")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # baseline
    with t.no_grad():
        clean_logits = model(dataset.toks)
        baseline = logits_to_logit_diff(clean_logits, dataset, per_prompt=False).item()

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        _, orig_cache = model.run_with_cache(dataset.toks, names_filter=z_filter, return_type=None)
        _, new_cache = model.run_with_cache(dataset.corrupted_toks, names_filter=z_filter, return_type=None)

    need_logits = any(s.target_kind == "logits" for s in specs)
    residual_matrix = None
    if need_logits:
        residual_matrix = logits_delta_matrix(
            model, dataset, orig_cache, new_cache, baseline
        )

    att_specs = [s for s in specs if s.target_kind == "attention"]
    results = []

    for spec in tqdm(att_specs, desc="Receiver heads"):
        recv = spec.dst
        if recv is None:
            continue
        mat = path_patch_to_head_single(
            model, recv, dataset, dataset, orig_cache, new_cache, baseline
        )
        delta = float(mat[spec.src.layer, spec.src.head])
        results.append(
            {
                "edge": spec.label(),
                "receiver_kind": "attention",
                "src": spec.src.as_name(),
                "dst": spec.dst.as_name(),
                "receiver_input": spec.receiver_input,
                "delta_logit_diff": delta,
                "patched_logit_diff": baseline + delta,
                "joint": False,
            }
        )

    for spec in specs:
        if spec.target_kind != "logits":
            continue
        delta = float(residual_matrix[spec.src.layer, spec.src.head])
        results.append(
            {
                "edge": spec.label(),
                "receiver_kind": "logits",
                "src": spec.src.as_name(),
                "dst": "logits",
                "receiver_input": None,
                "delta_logit_diff": delta,
                "patched_logit_diff": baseline + delta,
                "joint": False,
            }
        )

    payload = {
        "meta": {
            "collapsed_graph": str(collapsed_graph),
            "prompt_type": prompt_type,
            "seed": seed,
            "baseline_logit_diff": baseline,
        },
        "edges": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[DONE] 写出 {len(results)} 条记录 → {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic edge path patching (q-channel only)")
    p.add_argument("--collapsed-graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=str, required=True, choices=["ioi", "agr_gender"])
    p.add_argument("--data-path", type=Path, help="任务数据文件，ioi 可忽略")
    p.add_argument("--prompt-type", type=str, default="mixed")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    device = args.device

    if args.task == "agr_gender":
        from transformers import GPT2Tokenizer
        from agr_gender_dataset import AgrGenderDataset
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        dataset = AgrGenderDataset(
            tokenizer=tokenizer,
            data_path=args.data_path,
            prepend_bos=False,
            device=device,
            seed=args.seed,
        )
    else:
        # IOI 回退：使用标准 IOIDataset（需已有标准数据）
        from transformers import GPT2Tokenizer
        from ioi_dataset import IOIDataset
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        dataset = IOIDataset(
            prompt_type=args.prompt_type,
            N=100,
            tokenizer=tokenizer,
            device=device,
        )

    run_edge_path_patching(
        collapsed_graph=args.collapsed_graph,
        output_path=args.output,
        dataset=dataset,
        prompt_type=args.prompt_type,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
