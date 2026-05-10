#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep direct head->logits effects for all attention heads on the greater-than task.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import ActivationCache, HookedTransformer, loading_from_pretrained as loading, utils
from transformer_lens.hook_points import HookPoint

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from greater_than.dataset import GreaterThanDataset


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


def load_local_hooked_transformer(local_model_dir: str, device: t.device) -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(str(local_model_dir), local_files_only=True)
    cfg = loading.get_pretrained_model_config(
        str(local_model_dir),
        device=device,
        local_files_only=True,
    )
    model = HookedTransformer(
        cfg,
        tokenizer=tokenizer,
        move_to_device=False,
    )
    state_dict = loading.get_pretrained_state_dict(
        str(local_model_dir),
        cfg,
        hf_model=hf_model,
        local_files_only=True,
    )
    model.load_and_process_state_dict(state_dict)
    model.move_model_modules_to_device()
    return model


def load_model(model_name: str, model_path: Optional[Path], device: t.device) -> HookedTransformer:
    if model_path:
        model = load_local_hooked_transformer(str(model_path), device)
    else:
        official_name = model_name if model_name != "gpt2-small" else "gpt2"
        model = HookedTransformer.from_pretrained(official_name, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    repo_root = THIS_DIR.parents[2]
    default_data = repo_root / "greater_than_data.csv"

    p = argparse.ArgumentParser(
        description="Sweep direct greater-than head->logits effects for all heads."
    )
    p.add_argument("--data-path", type=Path, default=default_data, help="greater_than CSV data path.")
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
    args.data_path = args.data_path.expanduser().resolve()
    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Greater-than dataset not found: {args.data_path}\n"
            f"Expected default path: {(THIS_DIR.parents[2] / 'greater_than_data.csv').resolve()}\n"
            "You can also pass --data-path explicitly."
        )

    device = t.device(args.device)
    model = load_model(args.model_name, args.model_path, device)

    dataset = GreaterThanDataset(
        tokenizer=model.tokenizer,
        data_path=args.data_path,
        prepend_bos=False,
        device=str(device),
    )
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.N = len(dataset.samples)
        dataset = GreaterThanDataset(
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

    clean_metric = dataset.prob_diff(clean_logits, mean=True).item()
    clean_metric_per = dataset.prob_diff(clean_logits, mean=False).detach().cpu().tolist()
    corrupted_metric = dataset.prob_diff(corrupted_logits, mean=True).item()
    corrupted_metric_per = dataset.prob_diff(corrupted_logits, mean=False).detach().cpu().tolist()

    denom_global = clean_metric - corrupted_metric
    resid_post_hook = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_filter = lambda name: name == resid_post_hook

    head_results = []
    all_heads = [
        (layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
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
            _, patched_cache = model.run_with_cache(
                dataset.toks, names_filter=resid_filter, return_type=None
            )
        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook]))
        patched_metric = dataset.prob_diff(patched_logits, mean=True).item()
        patched_metric_per = dataset.prob_diff(patched_logits, mean=False).detach().cpu().tolist()

        raw_delta = patched_metric - clean_metric
        normalized_delta = raw_delta / denom_global if abs(denom_global) > 1e-12 else None

        per_sentence = []
        increase_count = 0
        decrease_count = 0
        abs_raw_vals = []
        abs_norm_vals = []
        for s_idx, sample in enumerate(dataset.samples):
            clean_val = clean_metric_per[s_idx]
            corr_val = corrupted_metric_per[s_idx]
            patched_val = patched_metric_per[s_idx]
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
                    "label": sample.label,
                    "clean_metric": clean_val,
                    "corrupted_metric": corr_val,
                    "patched_metric": patched_val,
                    "delta_metric": per_raw,
                    "normalized_delta_metric": per_norm,
                    "direction": direction,
                }
            )

        head_results.append(
            {
                "head": f"{layer}.{head}",
                "layer": layer,
                "head_index": head,
                "scores": {
                    "clean_metric": clean_metric,
                    "corrupted_metric": corrupted_metric,
                    "patched_metric": patched_metric,
                    "delta_metric": raw_delta,
                    "normalized_delta_metric": normalized_delta,
                    "increase_count": increase_count,
                    "decrease_count": decrease_count,
                    "increase_ratio": increase_count / len(dataset.samples),
                    "decrease_ratio": decrease_count / len(dataset.samples),
                    "mean_abs_delta_metric": sum(abs_raw_vals) / len(abs_raw_vals) if abs_raw_vals else None,
                    "mean_abs_normalized_delta_metric": sum(abs_norm_vals) / len(abs_norm_vals) if abs_norm_vals else None,
                },
                "per_sentence": per_sentence,
            }
        )

    model.reset_hooks()

    ranking_by_raw = sorted(
        head_results,
        key=lambda rec: abs(rec["scores"]["delta_metric"]),
        reverse=True,
    )
    ranking_by_normalized = sorted(
        head_results,
        key=lambda rec: abs(rec["scores"]["normalized_delta_metric"])
        if rec["scores"]["normalized_delta_metric"] is not None
        else -1.0,
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
            "baseline_clean_metric": clean_metric,
            "baseline_corrupted_metric": corrupted_metric,
            "baseline_gap": denom_global,
        },
        "ranking_by_abs_raw_delta": [
            {
                "head": rec["head"],
                "delta_metric": rec["scores"]["delta_metric"],
                "normalized_delta_metric": rec["scores"]["normalized_delta_metric"],
            }
            for rec in ranking_by_raw
        ],
        "ranking_by_abs_normalized_delta": [
            {
                "head": rec["head"],
                "delta_metric": rec["scores"]["delta_metric"],
                "normalized_delta_metric": rec["scores"]["normalized_delta_metric"],
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
            "clean_metric",
            "corrupted_metric",
            "patched_metric",
            "delta_metric",
            "normalized_delta_metric",
            "increase_count",
            "decrease_count",
            "increase_ratio",
            "decrease_ratio",
            "mean_abs_delta_metric",
            "mean_abs_normalized_delta_metric",
        ]
        with args.summary_csv.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fieldnames)
            writer.writeheader()
            for rec in ranking_by_normalized:
                row = {
                    "head": rec["head"],
                    "layer": rec["layer"],
                    "head_index": rec["head_index"],
                }
                row.update(
                    {k: rec["scores"].get(k) for k in csv_fieldnames if k in rec["scores"]}
                )
                writer.writerow(row)
        print(f"Wrote CSV ranking: {args.summary_csv}")

    print("\nTop heads by |normalized_delta_metric|:")
    for rec in ranking_by_normalized[: args.top_k]:
        norm = rec["scores"]["normalized_delta_metric"]
        raw = rec["scores"]["delta_metric"]
        print(f"  {rec['head']:>5}  normalized={norm:+.6f}  raw={raw:+.6f}")


if __name__ == "__main__":
    main()
