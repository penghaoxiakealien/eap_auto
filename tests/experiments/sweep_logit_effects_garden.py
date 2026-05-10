#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep direct head->logits effects for all attention heads on the Garden task.

This is a bulk version of precompute_logit_effects_garden.py:
  - loads model once
  - runs clean / corrupted caches once
  - patches each head into the clean run
  - reports raw and normalized logit-diff effects
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import torch as t

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from garden_dataset import GardenDataset
from precompute_logit_effects_garden import load_model


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep direct Garden head->logits effects for all heads.")
    p.add_argument("--data-path", type=Path, required=True, help="Garden CSV data path.")
    p.add_argument("--model-name", type=str, default="gpt2", help="Model name.")
    p.add_argument("--model-path", type=Path, help="Local model path.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-file", type=Path, required=True, help="Output JSON file path.")
    p.add_argument("--summary-csv", type=Path, help="Optional CSV ranking output.")
    p.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    p.add_argument("--top-k", type=int, default=20, help="How many heads to print in the terminal summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = t.device(args.device)
    model = load_model(args.model_name, args.model_path, device)

    dataset = GardenDataset(
        tokenizer=model.tokenizer,
        data_path=args.data_path,
        prepend_bos=False,
        device=str(device),
    )
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.N = len(dataset.samples)
        dataset = GardenDataset(
            tokenizer=model.tokenizer,
            samples=dataset.samples,
            prepend_bos=False,
            device=str(device),
        )

    corrupted = dataset.gen_flipped_prompts()
    z_filter = lambda name: name.endswith("z")

    print("Running clean/corrupted baseline passes once...")
    with t.no_grad():
        clean_logits, clean_cache = model.run_with_cache(dataset.toks, names_filter=z_filter)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted.toks, names_filter=z_filter)

    clean_logit_diff = dataset.logit_diff(clean_logits, mean=True).item()
    clean_logit_diff_per = dataset.logit_diff(clean_logits, mean=False).detach().cpu().tolist()
    corrupted_logit_diff = dataset.logit_diff(corrupted_logits, mean=True).item()
    corrupted_logit_diff_per = dataset.logit_diff(corrupted_logits, mean=False).detach().cpu().tolist()

    denom_global = clean_logit_diff - corrupted_logit_diff
    resid_post_hook = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_filter = lambda name: name == resid_post_hook

    head_results = []
    all_heads = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    total_heads = len(all_heads)

    for idx, head_tuple in enumerate(all_heads, start=1):
        layer, head = head_tuple
        print(f"[{idx}/{total_heads}] Sweeping head {layer}.{head} ...")
        model.reset_hooks()
        hook_fn = lambda act, hook, ht=head_tuple: patch_or_freeze_head_vectors(
            act, hook, corrupted_cache, clean_cache, ht
        )
        model.add_hook(z_filter, hook_fn)
        with t.no_grad():
            _, patched_cache = model.run_with_cache(dataset.toks, names_filter=resid_filter, return_type=None)
        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook]))
        patched_logit_diff = dataset.logit_diff(patched_logits, mean=True).item()
        patched_logit_diff_per = dataset.logit_diff(patched_logits, mean=False).detach().cpu().tolist()

        raw_delta = patched_logit_diff - clean_logit_diff
        normalized_delta = raw_delta / denom_global if abs(denom_global) > 1e-12 else None

        per_sentence = []
        increase_count = 0
        decrease_count = 0
        abs_raw_vals = []
        abs_norm_vals = []
        for s_idx, sample in enumerate(dataset.samples):
            clean_val = clean_logit_diff_per[s_idx]
            corr_val = corrupted_logit_diff_per[s_idx]
            patched_val = patched_logit_diff_per[s_idx]
            per_raw = patched_val - clean_val
            per_denom = clean_val - corr_val
            per_norm = per_raw / per_denom if abs(per_denom) > 1e-12 else None
            direction = "increase" if per_raw >= 0 else "decrease"
            if direction == "increase":
                increase_count += 1
            else:
                decrease_count += 1
            abs_raw_vals.append(abs(per_raw))
            if per_norm is not None:
                abs_norm_vals.append(abs(per_norm))
            per_sentence.append(
                {
                    "sentence_id": s_idx,
                    "sentence_text": sample.clean,
                    "clean_logit_diff": clean_val,
                    "corrupted_logit_diff": corr_val,
                    "patched_logit_diff": patched_val,
                    "delta_logit_diff": per_raw,
                    "normalized_delta_logit_diff": per_norm,
                    "direction": direction,
                }
            )

        head_results.append(
            {
                "head": f"{layer}.{head}",
                "layer": layer,
                "head_index": head,
                "scores": {
                    "clean_logit_diff": clean_logit_diff,
                    "corrupted_logit_diff": corrupted_logit_diff,
                    "patched_logit_diff": patched_logit_diff,
                    "delta_logit_diff": raw_delta,
                    "normalized_delta_logit_diff": normalized_delta,
                    "increase_count": increase_count,
                    "decrease_count": decrease_count,
                    "increase_ratio": increase_count / len(dataset.samples),
                    "decrease_ratio": decrease_count / len(dataset.samples),
                    "mean_abs_delta_logit_diff": sum(abs_raw_vals) / len(abs_raw_vals) if abs_raw_vals else None,
                    "mean_abs_normalized_delta_logit_diff": sum(abs_norm_vals) / len(abs_norm_vals) if abs_norm_vals else None,
                },
                "per_sentence": per_sentence,
            }
        )

    model.reset_hooks()

    ranking_by_raw = sorted(
        head_results,
        key=lambda rec: abs(rec["scores"]["delta_logit_diff"]),
        reverse=True,
    )
    ranking_by_normalized = sorted(
        head_results,
        key=lambda rec: abs(rec["scores"]["normalized_delta_logit_diff"]) if rec["scores"]["normalized_delta_logit_diff"] is not None else -1.0,
        reverse=True,
    )

    payload = {
        "meta": {
            "data_path": str(args.data_path),
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "max_samples": args.max_samples,
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "baseline_clean_logit_diff": clean_logit_diff,
            "baseline_corrupted_logit_diff": corrupted_logit_diff,
            "baseline_gap": denom_global,
        },
        "ranking_by_abs_raw_delta": [
            {
                "head": rec["head"],
                "delta_logit_diff": rec["scores"]["delta_logit_diff"],
                "normalized_delta_logit_diff": rec["scores"]["normalized_delta_logit_diff"],
            }
            for rec in ranking_by_raw
        ],
        "ranking_by_abs_normalized_delta": [
            {
                "head": rec["head"],
                "delta_logit_diff": rec["scores"]["delta_logit_diff"],
                "normalized_delta_logit_diff": rec["scores"]["normalized_delta_logit_diff"],
            }
            for rec in ranking_by_normalized
        ],
        "heads": head_results,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote JSON sweep: {args.output_file}")

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_fieldnames = [
            "head",
            "layer",
            "head_index",
            "clean_logit_diff",
            "corrupted_logit_diff",
            "patched_logit_diff",
            "delta_logit_diff",
            "normalized_delta_logit_diff",
            "increase_count",
            "decrease_count",
            "increase_ratio",
            "decrease_ratio",
            "mean_abs_delta_logit_diff",
            "mean_abs_normalized_delta_logit_diff",
        ]
        with args.summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=csv_fieldnames,
            )
            writer.writeheader()
            for rec in ranking_by_normalized:
                row = {
                    "head": rec["head"],
                    "layer": rec["layer"],
                    "head_index": rec["head_index"],
                }
                row.update({k: rec["scores"].get(k) for k in csv_fieldnames if k in rec["scores"]})
                writer.writerow(row)
        print(f"Wrote CSV ranking: {args.summary_csv}")

    print("\nTop heads by |normalized_delta_logit_diff|:")
    for rec in ranking_by_normalized[: args.top_k]:
        norm = rec["scores"]["normalized_delta_logit_diff"]
        raw = rec["scores"]["delta_logit_diff"]
        print(f"  {rec['head']:>5}  normalized={norm:+.6f}  raw={raw:+.6f}")


if __name__ == "__main__":
    main()
