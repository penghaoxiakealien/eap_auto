#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export END-position attention patterns for selected heads on the greater-than task.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, loading_from_pretrained as loading

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from greater_than.dataset import GreaterThanDataset


def parse_head_list(raw: str) -> List[Tuple[int, int]]:
    heads: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        layer_str, head_str = item.split(".")
        heads.append((int(layer_str), int(head_str)))
    if not heads:
        raise ValueError("No valid heads provided.")
    return heads


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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(
        description="Export END-position attention patterns for selected greater-than heads."
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "greater_than_data.csv",
        help="greater_than CSV data path.",
    )
    p.add_argument(
        "--heads",
        type=str,
        required=True,
        help="Comma-separated heads, e.g. 9.1,7.10,8.11",
    )
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--model-path", type=Path, help="Local model path.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    p.add_argument("--top-k", type=int, default=5, help="Number of top tokens to store per sample/head.")
    p.add_argument(
        "--per-head-sample-limit",
        type=int,
        default=0,
        help="Keep only the first N per-sample records for each head in the output (0 = keep all).",
    )
    p.add_argument("--output-file", type=Path, required=True, help="Output JSON path.")
    return p.parse_args()


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    args.data_path = args.data_path.expanduser().resolve()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Greater-than dataset not found: {args.data_path}")

    target_heads = parse_head_list(args.heads)
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

    # All current greater-than prompts share the same tokenized structure.
    str_tokens_template = model.to_str_tokens(dataset.toks[0])
    end_position = int(dataset.input_lengths[0].item()) - 1
    head_to_name = {(layer, head): f"{layer}.{head}" for layer, head in target_heads}

    pattern_cache: Dict[int, t.Tensor] = {}
    needed_layers = sorted({layer for layer, _ in target_heads})
    hook_names = {layer: f"blocks.{layer}.attn.hook_pattern" for layer in needed_layers}

    with t.no_grad():
        _, cache = model.run_with_cache(
            dataset.toks,
            names_filter=lambda name: name in set(hook_names.values()),
            return_type=None,
        )
        for layer in needed_layers:
            pattern_cache[layer] = cache[hook_names[layer]].detach().cpu()

    results_by_head = {}
    for layer, head in target_heads:
        head_name = head_to_name[(layer, head)]
        pattern = pattern_cache[layer][:, head, end_position, : end_position + 1]

        per_sample = []
        position_scores: Dict[int, List[float]] = {pos: [] for pos in range(end_position + 1)}
        for sample_idx, sample in enumerate(dataset.samples):
            scores = pattern[sample_idx].tolist()
            for pos, score in enumerate(scores):
                position_scores[pos].append(float(score))

            token_rows = [
                {
                    "position": pos,
                    "token": str_tokens_template[pos],
                    "score": float(score),
                }
                for pos, score in enumerate(scores)
            ]
            token_rows_sorted = sorted(token_rows, key=lambda x: x["score"], reverse=True)
            per_sample.append(
                {
                    "sample_id": sample_idx,
                    "sentence_text": sample.clean,
                    "label": sample.label,
                    "end_position": end_position,
                    "tokens": str_tokens_template[: end_position + 1],
                    "attention_scores": token_rows,
                    "top_attended_tokens": token_rows_sorted[: args.top_k],
                }
            )

        mean_attention = [
            {
                "position": pos,
                "token": str_tokens_template[pos],
                "mean_score": mean(position_scores[pos]),
            }
            for pos in range(end_position + 1)
        ]
        mean_attention_sorted = sorted(
            mean_attention,
            key=lambda x: -1.0 if x["mean_score"] is None else x["mean_score"],
            reverse=True,
        )

        results_by_head[head_name] = {
            "head": head_name,
            "layer": layer,
            "head_index": head,
            "end_position": end_position,
            "mean_attention": mean_attention,
            "top_mean_attention_tokens": mean_attention_sorted[: args.top_k],
            "per_sample": per_sample[: args.per_head_sample_limit] if args.per_head_sample_limit > 0 else per_sample,
        }

    payload = {
        "meta": {
            "task": "greater_than",
            "data_path": str(args.data_path),
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "device": args.device,
            "max_samples": args.max_samples,
            "per_head_sample_limit": args.per_head_sample_limit,
            "num_samples": len(dataset.samples),
            "query_position": "end",
            "tokenized_length": int(dataset.input_lengths[0].item()),
        },
        "heads": results_by_head,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved END attention export: {args.output_file}")


if __name__ == "__main__":
    main()
