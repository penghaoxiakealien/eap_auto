#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按路径分组执行层级 path patching 并记录 logit diff 变化。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as t

from transformer_lens import utils

from ioi_dataset import IOIDataset

from path_patch_dominant_groups import (
    compute_metric,
    load_model,
    logits_to_avg_logit_diff,
    run_path_patch_analysis,
)


@dataclass
class GroupInfo:
    key: str
    signature: str
    classification: str
    heads: List[str]
    layers: List[int]

    @property
    def max_layer(self) -> int:
        return max(self.layers) if self.layers else -1


def map_head_name(node_name: str) -> Optional[str]:
    if isinstance(node_name, str) and node_name.startswith("a") and ".h" in node_name:
        left, right = node_name.split(".h", 1)
        try:
            layer = int(left[1:])
            head = int(right)
        except ValueError:
            return None
        return f"{layer}.{head}"
    return None


def head_str_to_tuple(head: str) -> Tuple[int, int]:
    layer_str, head_str = head.split(".")
    return int(layer_str), int(head_str)


def build_groups(soft_data: Dict[str, Any]) -> Tuple[Dict[str, GroupInfo], Dict[str, str]]:
    patterns_block = soft_data.get("patterns", {})
    per_head = patterns_block.get("per_head") or soft_data.get("head_patterns")
    if not isinstance(per_head, dict):
        raise ValueError("软分类 JSON 缺少 per_head 信息")

    groups: Dict[str, GroupInfo] = {}
    head_to_group: Dict[str, str] = {}

    for head, info in per_head.items():
        path_patch = info.get("path_patch") or {}
        classification = path_patch.get("classification")
        if classification not in {"positive", "negative"}:
            continue
        signature = info.get("signature") or ""
        group_key = f"{signature}|{classification}"
        layer = int(head.split(".")[0])
        if group_key not in groups:
            groups[group_key] = GroupInfo(
                key=group_key,
                signature=signature,
                classification=classification,
                heads=[],
                layers=[],
            )
        groups[group_key].heads.append(head)
        groups[group_key].layers.append(layer)
        head_to_group[head] = group_key

    return groups, head_to_group


def collect_group_edges(
    graph_data: Dict[str, Any],
    head_to_group: Dict[str, str],
    group_info: Dict[str, GroupInfo],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    edges = {}
    for src_node, dst_node in graph_data.get("edge_list", []):
        src_head = map_head_name(src_node)
        dst_head = map_head_name(dst_node)
        if not src_head or not dst_head:
            continue
        if src_head not in head_to_group or dst_head not in head_to_group:
            continue
        src_group = head_to_group[src_head]
        dst_group = head_to_group[dst_head]
        if src_group == dst_group:
            continue
        src_info = group_info.get(src_group)
        dst_info = group_info.get(dst_group)
        if not src_info or not dst_info:
            continue
        if src_info.classification != dst_info.classification:
            continue
        src_layer = int(src_head.split(".")[0])
        dst_layer = int(dst_head.split(".")[0])
        if src_layer >= dst_layer:
            continue
        key = (src_group, dst_group)
        record = edges.setdefault(key, {"src_heads": set(), "dst_heads": set(), "edges": []})
        record["src_heads"].add(src_head)
        record["dst_heads"].add(dst_head)
        record["edges"].append({"src": src_head, "dst": dst_head})
    # 将集合转换为排序后的列表
    for data in edges.values():
        data["src_heads"] = sorted(data["src_heads"], key=lambda h: tuple(map(int, h.split("."))))
        data["dst_heads"] = sorted(data["dst_heads"], key=lambda h: tuple(map(int, h.split("."))))
    return edges


def make_sender_hook(clean_cache, corrupted_cache, sender_map):
    def hook(act, hook):
        act.copy_(clean_cache[hook.name])
        heads = sender_map.get(hook.layer())
        if heads:
            corrupted_tensor = corrupted_cache[hook.name]
            for head_idx in heads:
                act[:, :, head_idx] = corrupted_tensor[:, :, head_idx]
        return act

    return hook


def make_cache_hook(store: Dict[str, t.Tensor], name: str):
    def cache_fn(act, hook):
        store[name] = act.detach().clone()
        return act

    return cache_fn


def make_patch_hook(tensor: t.Tensor, head_indices: Sequence[int]):
    def patch_fn(act, hook):
        for idx in head_indices:
            act[:, :, idx, :] = tensor[:, :, idx, :]
        return act

    return patch_fn


def run_group_path_patch(
    model,
    dataset: IOIDataset,
    senders: List[Tuple[int, int]],
    receivers: List[Tuple[int, int]],
    receiver_input: str,
    clean_cache,
    corrupted_cache,
    clean_logit_diff: float,
    corrupted_logit_diff: float,
) -> Dict[str, float]:
    if not senders or not receivers:
        raise ValueError("Path patch 需要至少一个发送头和一个接收头")

    sender_map: Dict[int, List[int]] = defaultdict(list)
    for layer, head in senders:
        sender_map[layer].append(head)
    receiver_map: Dict[int, List[int]] = defaultdict(list)
    for layer, head in receivers:
        receiver_map[layer].append(head)

    z_filter = lambda name: name.endswith("z")
    model.reset_hooks()
    model.add_hook(z_filter, make_sender_hook(clean_cache, corrupted_cache, sender_map))

    receiver_cache: Dict[str, t.Tensor] = {}
    hooks = []
    for layer in receiver_map:
        name = utils.get_act_name(receiver_input, layer)
        hooks.append((name, make_cache_hook(receiver_cache, name)))

    model.run_with_hooks(dataset.toks, fwd_hooks=hooks, return_type=None)
    model.reset_hooks()

    patch_hooks = []
    for layer, heads in receiver_map.items():
        name = utils.get_act_name(receiver_input, layer)
        cached = receiver_cache.get(name)
        if cached is None:
            continue
        patch_hooks.append((name, make_patch_hook(cached, heads)))

    patched_logits = model.run_with_hooks(dataset.toks, fwd_hooks=patch_hooks, return_type="logits")
    patched_diff = logits_to_avg_logit_diff(patched_logits, dataset).mean().item()
    delta = patched_diff - clean_logit_diff
    metric = compute_metric(patched_diff, clean_logit_diff, corrupted_logit_diff)
    return {
        "patched_logit_diff": patched_diff,
        "delta_logit_diff": delta,
        "metric": metric,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对分组后的路径执行层级 path patch")
    parser.add_argument("--graph-json", type=Path, required=True, help="筛选后的图 JSON")
    parser.add_argument("--dominant-soft", type=Path, required=True, help="thr75 soft JSON 路径")
    parser.add_argument("--classification-json", type=Path, required=True, help="单头路径插补分类结果输出位置")
    parser.add_argument("--updated-soft", type=Path, required=True, help="写入路径插补信息的 soft JSON")
    parser.add_argument("--group-patch-json", type=Path, required=True, help="分组路径插补结果输出")
    parser.add_argument("--model-name", type=str, default="gpt2", help="HookedTransformer 模型名")
    parser.add_argument("--model-path", type=Path, help="本地模型路径，可选")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-prompts", type=int, default=50, help="IOI 样本数量")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--receiver-input", type=str, default="v", choices=["q", "k", "v"], help="在接收头上修补的输入类型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classification_result = run_path_patch_analysis(
        dominant_soft=args.dominant_soft,
        output_json=args.classification_json,
        updated_soft=args.updated_soft,
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        n_prompts=args.n_prompts,
        seed=args.seed,
    )

    soft_data = json.loads(args.updated_soft.read_text())
    groups, head_to_group = build_groups(soft_data)
    if not groups:
        raise ValueError("未找到带路径插补分类的分组信息（positive/negative）")

    graph_data = json.loads(args.graph_json.read_text())
    group_edges = collect_group_edges(graph_data, head_to_group, groups)
    if not group_edges:
        print("未检测到满足条件的分组路径，不执行路径修补。")
        return

    device = t.device(args.device)
    model = load_model(args.model_name, args.model_path, device)
    t.manual_seed(args.seed)

    dataset = IOIDataset(
        prompt_type="mixed",
        N=args.n_prompts,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=args.seed,
        device=str(device),
    )
    abc_dataset = dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        clean_logits, clean_cache = model.run_with_cache(dataset.toks, names_filter=z_filter)
        corrupted_logits, corrupted_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_filter)

    clean_logit_diff = logits_to_avg_logit_diff(clean_logits, dataset).mean().item()
    corrupted_logit_diff = logits_to_avg_logit_diff(corrupted_logits, dataset).mean().item()

    pair_items = sorted(
        group_edges.items(),
        key=lambda item: (
            groups[item[0][1]].max_layer,
            groups[item[0][0]].max_layer,
        ),
        reverse=True,
    )

    results = []
    for (src_group_key, dst_group_key), data in pair_items:
        src_heads = [head_str_to_tuple(h) for h in data["src_heads"]]
        dst_heads = [head_str_to_tuple(h) for h in data["dst_heads"]]
        stats = run_group_path_patch(
            model,
            dataset,
            src_heads,
            dst_heads,
            args.receiver_input,
            clean_cache,
            corrupted_cache,
            clean_logit_diff,
            corrupted_logit_diff,
        )
        results.append(
            {
                "source_group": {
                    "key": src_group_key,
                    "signature": groups[src_group_key].signature,
                    "classification": groups[src_group_key].classification,
                    "heads": data["src_heads"],
                    "max_layer": groups[src_group_key].max_layer,
                },
                "target_group": {
                    "key": dst_group_key,
                    "signature": groups[dst_group_key].signature,
                    "classification": groups[dst_group_key].classification,
                    "heads": data["dst_heads"],
                    "max_layer": groups[dst_group_key].max_layer,
                },
                "edges": data["edges"],
                "statistics": stats,
            }
        )

    payload = {
        "meta": {
            "graph_json": str(args.graph_json),
            "dominant_soft": str(args.dominant_soft),
            "classification_json": str(args.classification_json),
            "updated_soft": str(args.updated_soft),
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "device": args.device,
            "n_prompts": args.n_prompts,
            "seed": args.seed,
            "receiver_input": args.receiver_input,
            "clean_logit_diff": clean_logit_diff,
            "corrupted_logit_diff": corrupted_logit_diff,
        },
        "classification": classification_result["payload"],
        "group_pairs": results,
    }

    args.group_patch_json.parent.mkdir(parents=True, exist_ok=True)
    args.group_patch_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"写入分组路径插补结果: {args.group_patch_json}")


if __name__ == "__main__":
    main()
