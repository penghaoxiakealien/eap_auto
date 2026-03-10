#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对指定的 sender head → receiver head 执行路径插补并保存结果。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch as t
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from ioi_dataset import IOIDataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_head(head: str) -> Tuple[int, int]:
    try:
        layer_str, head_str = head.split(".")
        return int(layer_str), int(head_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"非法 head 格式: {head} (应为 L.H)") from exc


def logits_to_ave_logit_diff(logits: t.Tensor, dataset: IOIDataset) -> t.Tensor:
    io_logits = logits[
        range(logits.size(0)), dataset.word_idx["end"], dataset.io_tokenIDs
    ]
    s_logits = logits[
        range(logits.size(0)), dataset.word_idx["end"], dataset.s_tokenIDs
    ]
    return io_logits - s_logits


def compute_metric(patched_diff: float, clean_diff: float, corrupted_diff: float) -> float:
    return (patched_diff - clean_diff) / (clean_diff - corrupted_diff)


def patch_or_freeze_head_vectors(
    act: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> t.Tensor:
    act[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        act[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return act


def path_patch_head_to_head(
    model: HookedTransformer,
    ioi_dataset: IOIDataset,
    abc_dataset: IOIDataset,
    sender_head: Tuple[int, int],
    receiver_head: Tuple[int, int],
    receiver_input: str,
    clean_diff: float,
    corrupted_diff: float,
) -> Dict[str, float]:
    sender_layer, _ = sender_head
    receiver_layer, _ = receiver_head
    if sender_layer >= receiver_layer:
        raise ValueError("Sender layer 必须小于 receiver layer，才能形成因果路径。")

    z_filter = lambda name: name.endswith("z")
    receiver_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    receiver_filter = lambda name: name == receiver_hook_name

    model.reset_hooks()

    _, new_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_filter, return_type=None)
    _, orig_cache = model.run_with_cache(ioi_dataset.toks, names_filter=z_filter, return_type=None)

    hook_fn = lambda act, hook: patch_or_freeze_head_vectors(act, hook, new_cache, orig_cache, sender_head)
    model.add_hook(z_filter, hook_fn, level=1)

    _, patched_receiver_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=receiver_filter, return_type=None
    )
    model.reset_hooks()

    def patch_receiver(act: t.Tensor, hook: HookPoint) -> t.Tensor:
        act[:, :, receiver_head[1]] = patched_receiver_cache[hook.name][:, :, receiver_head[1]]
        return act

    patched_logits = model.run_with_hooks(
        ioi_dataset.toks,
        fwd_hooks=[(receiver_filter, patch_receiver)],
        return_type="logits",
    )

    patched_diff = logits_to_ave_logit_diff(patched_logits, ioi_dataset).mean().item()
    metric = compute_metric(patched_diff, clean_diff, corrupted_diff)
    delta = patched_diff - clean_diff

    return {
        "patched_logit_diff": patched_diff,
        "delta_logit_diff": delta,
        "metric": metric,
    }


def save_results(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单个 sender→receiver 头路径插补")
    parser.add_argument("--sender-head", required=True, help="发送头，格式 L.H")
    parser.add_argument("--receiver-head", required=True, help="接收头，格式 L.H")
    parser.add_argument(
        "--receiver-input",
        choices=["q", "k", "v"],
        default="q",
        help="接收头的输入类型 (默认 q)",
    )
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/single_head_to_head_patch.json"),
        help="保存 JSON 结果的路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sender_head = parse_head(args.sender_head)
    receiver_head = parse_head(args.receiver_head)
    device = t.device(args.device)

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
    clean_diff = logits_to_ave_logit_diff(clean_logits, ioi_dataset).mean().item()
    corrupted_diff = logits_to_ave_logit_diff(corrupted_logits, ioi_dataset).mean().item()

    result = path_patch_head_to_head(
        model,
        ioi_dataset,
        abc_dataset,
        sender_head,
        receiver_head,
        args.receiver_input,
        clean_diff,
        corrupted_diff,
    )

    payload = {
        "meta": {
            "model": args.model,
            "device": args.device,
            "n_prompts": args.n_prompts,
            "seed": args.seed,
            "sender_head": args.sender_head,
            "receiver_head": args.receiver_head,
            "receiver_input": args.receiver_input,
            "clean_logit_diff": clean_diff,
            "corrupted_logit_diff": corrupted_diff,
        },
        "result": result,
    }

    save_results(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"已保存结果: {args.output}")


if __name__ == "__main__":  # noqa: E305
    main()
