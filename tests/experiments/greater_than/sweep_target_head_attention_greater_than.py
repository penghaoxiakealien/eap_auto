#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep sender heads affecting a greater-than target head's END-position attention
through a chosen receiver input path (q/k/v).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import ActivationCache, HookedTransformer, loading_from_pretrained as loading, utils
from transformer_lens.hook_points import HookPoint

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from greater_than.dataset import GreaterThanDataset


def parse_head(head: str) -> Tuple[int, int]:
    layer_str, head_str = head.split(".", 1)
    return int(layer_str), int(head_str)


def load_local_hooked_transformer(local_model_dir: str, device: str) -> HookedTransformer:
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


def load_model(model_name: str, model_path: Optional[Path], device: str) -> HookedTransformer:
    if model_path:
        model = load_local_hooked_transformer(str(model_path), device)
    else:
        official_name = model_name if model_name != "gpt2-small" else "gpt2"
        model = HookedTransformer.from_pretrained(official_name, device=device)
    model.cfg.use_attn_result = True
    model.eval()
    return model


def patch_or_freeze_head_vectors(
    act: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> t.Tensor:
    act.copy_(orig_cache[hook.name])
    if head_to_patch[0] == hook.layer():
        act[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return act


def patch_target_receiver_input(
    activation: t.Tensor,
    hook: HookPoint,
    patched_receiver_cache: ActivationCache,
    target_head: Tuple[int, int],
) -> t.Tensor:
    patched = activation.clone()
    patched[:, :, target_head[1], :] = patched_receiver_cache[hook.name][:, :, target_head[1], :]
    return patched


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(
        description="Sweep sender heads affecting a greater-than target head END attention row."
    )
    p.add_argument("--target-head", required=True, help="Receiver head layer.head, e.g. 5.5")
    p.add_argument("--receiver-input", default="q", choices=["q", "k", "v"], help="Receiver input path to patch.")
    p.add_argument("--data-path", type=Path, default=repo_root / "greater_than_data.csv")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--model-path", type=Path, help="Local model path.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    p.add_argument("--top-k", type=int, default=5, help="Top increase/decrease tokens to keep.")
    p.add_argument("--top-senders", type=int, default=20, help="How many strongest senders to print.")
    p.add_argument("--include-target", action="store_true", help="Include the target head itself as a sender.")
    p.add_argument("--output-file", type=Path, required=True, help="JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_path = args.data_path.expanduser().resolve()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Greater-than dataset not found: {args.data_path}")

    model = load_model(args.model_name, args.model_path, args.device)
    dataset = GreaterThanDataset(
        tokenizer=model.tokenizer,
        data_path=args.data_path,
        prepend_bos=False,
        device=args.device,
    )
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.N = len(dataset.samples)
        dataset = GreaterThanDataset(
            tokenizer=model.tokenizer,
            samples=dataset.samples,
            prepend_bos=False,
            device=args.device,
        )
    corrupted = dataset.gen_flipped_prompts()

    target_head = parse_head(args.target_head)
    target_layer, target_head_idx = target_head
    all_senders = [
        (layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
        if args.include_target or (layer, head) != target_head
    ]

    z_filter = lambda name: name.endswith("z")
    receiver_hook_name = utils.get_act_name(args.receiver_input, target_layer)
    receiver_filter = lambda name: name == receiver_hook_name
    pattern_hook_name = utils.get_act_name("pattern", target_layer)
    pattern_filter = lambda name: name == pattern_hook_name

    end_position = int(dataset.input_lengths[0].item()) - 1
    str_tokens = model.to_str_tokens(dataset.toks[0])[: end_position + 1]

    with t.no_grad():
        _, clean_cache = model.run_with_cache(
            dataset.toks,
            names_filter=lambda name: z_filter(name) or pattern_filter(name),
            return_type=None,
        )
        _, corrupted_cache = model.run_with_cache(
            corrupted.toks,
            names_filter=z_filter,
            return_type=None,
        )

    clean_pattern = clean_cache[pattern_hook_name][:, target_head_idx, :, :]
    clean_end_vectors = clean_pattern[:, end_position, : end_position + 1].detach().cpu()

    head_results = []
    for idx, sender_head in enumerate(all_senders, start=1):
        print(f"[{idx}/{len(all_senders)}] Sweeping sender {sender_head[0]}.{sender_head[1]} ...")
        model.reset_hooks()
        hook_fn_sender = lambda act, hook, sh=sender_head: patch_or_freeze_head_vectors(
            act, hook, corrupted_cache, clean_cache, sh
        )
        model.add_hook(z_filter, hook_fn_sender)
        with t.no_grad():
            _, patched_receiver_cache = model.run_with_cache(
                dataset.toks,
                names_filter=receiver_filter,
                return_type=None,
            )
        model.reset_hooks()

        patched_pattern_holder: Dict[str, t.Tensor] = {}

        def cache_target_pattern_hook(activation: t.Tensor, hook: HookPoint):
            patched_pattern_holder[hook.name] = activation.detach()
            return activation

        with t.no_grad():
            model.run_with_hooks(
                dataset.toks,
                fwd_hooks=[
                    (receiver_hook_name, lambda act, hook, prc=patched_receiver_cache: patch_target_receiver_input(act, hook, prc, target_head)),
                    (pattern_hook_name, cache_target_pattern_hook),
                ],
                return_type=None,
            )

        patched_pattern = patched_pattern_holder[pattern_hook_name][:, target_head_idx, :, :]
        patched_end_vectors = patched_pattern[:, end_position, : end_position + 1].detach().cpu()
        diff_vectors = patched_end_vectors - clean_end_vectors

        mean_diff = diff_vectors.mean(dim=0)
        mean_abs_diff = diff_vectors.abs().mean(dim=0)
        mean_l1_change = float(diff_vectors.abs().sum(dim=1).mean().item())
        mean_l2_change = float(t.norm(diff_vectors, dim=1).mean().item())

        top_inc_vals, top_inc_idx = t.topk(mean_diff, k=min(args.top_k, mean_diff.numel()))
        top_dec_vals, top_dec_idx = t.topk(mean_diff, k=min(args.top_k, mean_diff.numel()), largest=False)
        top_abs_vals, top_abs_idx = t.topk(mean_abs_diff, k=min(args.top_k, mean_abs_diff.numel()))

        def pack_positions(vals: t.Tensor, idxs: t.Tensor, score_key: str) -> List[dict]:
            rows = []
            for val, pos_idx in zip(vals, idxs):
                pos = int(pos_idx.item())
                rows.append(
                    {
                        "position": pos,
                        "token": str_tokens[pos],
                        score_key: float(val.item()),
                    }
                )
            return rows

        per_sentence = []
        for sample_idx, sample in enumerate(dataset.samples):
            vec = diff_vectors[sample_idx]
            inc_vals, inc_idx = t.topk(vec, k=min(args.top_k, vec.numel()))
            dec_vals, dec_idx = t.topk(vec, k=min(args.top_k, vec.numel()), largest=False)
            per_sentence.append(
                {
                    "sample_id": sample_idx,
                    "sentence_text": sample.clean,
                    "label": sample.label,
                    "top_increases": pack_positions(inc_vals, inc_idx, "delta"),
                    "top_decreases": pack_positions(dec_vals, dec_idx, "delta"),
                }
            )

        head_results.append(
            {
                "sender_head": f"{sender_head[0]}.{sender_head[1]}",
                "scores": {
                    "mean_l1_change": mean_l1_change,
                    "mean_l2_change": mean_l2_change,
                },
                "top_mean_increases": pack_positions(top_inc_vals, top_inc_idx, "mean_delta"),
                "top_mean_decreases": pack_positions(top_dec_vals, top_dec_idx, "mean_delta"),
                "top_mean_abs_changes": pack_positions(top_abs_vals, top_abs_idx, "mean_abs_delta"),
                "per_sentence": per_sentence,
            }
        )

    ranking = sorted(head_results, key=lambda rec: rec["scores"]["mean_l1_change"], reverse=True)
    payload = {
        "meta": {
            "task": "greater_than",
            "target_head": args.target_head,
            "receiver_input": args.receiver_input,
            "data_path": str(args.data_path),
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "device": args.device,
            "max_samples": args.max_samples,
            "num_samples": len(dataset.samples),
            "query_position": "end",
            "end_position": end_position,
            "tokens": str_tokens,
        },
        "ranking_by_mean_l1_change": [
            {
                "sender_head": rec["sender_head"],
                "mean_l1_change": rec["scores"]["mean_l1_change"],
                "mean_l2_change": rec["scores"]["mean_l2_change"],
            }
            for rec in ranking
        ],
        "senders": head_results,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote sender->target sweep: {args.output_file}")

    print(f"\nTop senders by mean L1 change on {args.target_head}.{args.receiver_input}:")
    for rec in ranking[: args.top_senders]:
        print(
            f"  {rec['sender_head']:>5}  "
            f"mean_l1={rec['scores']['mean_l1_change']:.6f}  "
            f"mean_l2={rec['scores']['mean_l2_change']:.6f}"
        )


if __name__ == "__main__":
    main()
