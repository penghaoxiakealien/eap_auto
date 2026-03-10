#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按照路径插补后的 logit diff 变化符号为指定注意力头分类。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch as t
from tqdm import tqdm

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from ioi_dataset import IOIDataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class HeadResult:
    head: str
    patched_logit_diff: float
    delta_logit_diff: float
    metric: float


def parse_heads(values: Iterable[str]) -> List[Tuple[int, int]]:
    heads: List[Tuple[int, int]] = []
    for v in values:
        try:
            layer_str, head_str = v.split(".")
            heads.append((int(layer_str), int(head_str)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"非法 head 格式: {v} (应为 L.H)") from exc
    return heads


def logits_to_ave_logit_diff(logits: t.Tensor, dataset: IOIDataset) -> t.Tensor:
    io_logits = logits[
        range(logits.size(0)), dataset.word_idx["end"], dataset.io_tokenIDs
    ]
    s_logits = logits[
        range(logits.size(0)), dataset.word_idx["end"], dataset.s_tokenIDs
    ]
    return io_logits - s_logits


def compute_metric(
    patched_diff: float,
    clean_logit_diff: float,
    corrupted_logit_diff: float,
) -> float:
    return (patched_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


def patch_or_freeze_head_vectors(
    orig_head_vector: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> t.Tensor:
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def run_path_patching(
    model: HookedTransformer,
    ioi_dataset: IOIDataset,
    head: Tuple[int, int],
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    abc_cache: ActivationCache,
    ioi_cache: ActivationCache,
) -> HeadResult:
    model.reset_hooks()
    resid_post_hook = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_filter = lambda name: name == resid_post_hook
    z_filter = lambda name: name.endswith("z")

    hook_fn = lambda act, hook: patch_or_freeze_head_vectors(
        act, hook, abc_cache, ioi_cache, head
    )
    model.add_hook(z_filter, hook_fn)

    _, patched_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=resid_filter, return_type=None
    )
    patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook]))
    patched_diff = logits_to_ave_logit_diff(patched_logits, ioi_dataset).mean().item()
    metric = compute_metric(patched_diff, clean_logit_diff, corrupted_logit_diff)

    head_name = f"{head[0]}.{head[1]}"
    delta = patched_diff - clean_logit_diff
    return HeadResult(
        head=head_name,
        patched_logit_diff=patched_diff,
        delta_logit_diff=delta,
        metric=metric,
    )


def classify(results: List[HeadResult], threshold: float) -> Dict[str, List[str]]:
    pos, neg, near = [], [], []
    for item in results:
        if item.metric > threshold:
            pos.append(item.head)
        elif item.metric < -threshold:
            neg.append(item.head)
        else:
            near.append(item.head)
    return {"positive": pos, "negative": neg, "near_zero": near}


def save_results(
    path: Path,
    results: List[HeadResult],
    classification: Dict[str, List[str]],
    meta: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "heads": {
            r.head: {
                "patched_logit_diff": r.patched_logit_diff,
                "delta_logit_diff": r.delta_logit_diff,
                "metric": r.metric,
            }
            for r in results
        },
        "classification": classification,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据路径插补后的 logit diff 符号区分注意力头"
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=["9.9", "10.7", "10.2", "11.9"],
        help="待分析的注意力头列表，格式 L.H",
    )
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-prompts", type=int, default=25, help="IOI 样本数量")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="区分正负的 metric 阈值 (默认按符号)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/head_logit_effects.json"),
        help="保存分类与指标的 JSON 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = t.device(args.device)
    heads = parse_heads(args.heads)

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=args.n_prompts,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=args.seed,
        device=str(device),
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

    with t.no_grad():
        clean_logits, _ = model.run_with_cache(ioi_dataset.toks)
        corrupted_logits, _ = model.run_with_cache(abc_dataset.toks)
    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, ioi_dataset).mean().item()
    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, ioi_dataset).mean().item()

    z_filter = lambda name: name.endswith("z")
    _, abc_cache = model.run_with_cache(
        abc_dataset.toks, names_filter=z_filter, return_type=None
    )
    _, ioi_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=z_filter, return_type=None
    )

    results: List[HeadResult] = []
    for head in tqdm(heads, desc="Path patching"):
        res = run_path_patching(
            model,
            ioi_dataset,
            head,
            clean_logit_diff,
            corrupted_logit_diff,
            abc_cache,
            ioi_cache,
        )
        results.append(res)

    classification = classify(results, args.threshold)

    meta = {
        "model": args.model,
        "device": args.device,
        "n_prompts": args.n_prompts,
        "seed": args.seed,
        "threshold": args.threshold,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
    }
    save_results(args.output, results, classification, meta)
    print(f"已保存分类结果: {args.output}")


if __name__ == "__main__":
    main()