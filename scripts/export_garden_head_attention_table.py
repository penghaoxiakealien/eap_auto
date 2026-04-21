#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, loading_from_pretrained as loading

import sys

sys.path.append("/home/wangziran/eap_auto")
from tests.experiments.precompute_attention_scores_garden import (  # noqa: E402
    _ensure_pad_token,
    _iter_batches,
    compute_attention_for_batch,
    load_standard_garden,
)


DEFAULT_MODEL_PATH = Path("/home/wangziran/gpt2")
DEFAULT_QUERY_POSITIONS = ["subj", "verb", "obj_head", "rel_pron", "rel_verb", "end"]


def load_local_hooked_transformer(local_model_dir: Path, device: str = "cuda") -> HookedTransformer:
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
    model.eval()
    return model


def sample_items(items: List[dict], sample_size: int, seed: int) -> List[dict]:
    if sample_size <= 0 or sample_size >= len(items):
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    selected = sorted(indices[:sample_size])
    return [items[i] for i in selected]


def token_at_position(sentence: str, position: int) -> str:
    words = sentence.split()
    if 0 <= position < len(words):
        return words[position]
    return ""


def flatten_results(results: List[dict], query_label: str) -> List[dict]:
    rows: List[dict] = []
    for item in results:
        top = item.get("top_attended_tokens", [])
        top1 = top[0] if len(top) > 0 else {}
        top2 = top[1] if len(top) > 1 else {}
        qpos = item.get("end_position")
        sentence = item.get("sentence_text", "")
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "sentence_id": item.get("sentence_id"),
                "query_label": query_label.upper(),
                "query_position": qpos,
                "query_token": token_at_position(sentence, qpos) if isinstance(qpos, int) else "",
                "sentence_text": sentence,
                "top1_token": top1.get("token", ""),
                "top1_position": top1.get("position", ""),
                "top1_score": top1.get("score", ""),
                "top2_token": top2.get("token", ""),
                "top2_position": top2.get("position", ""),
                "top2_score": top2.get("score", ""),
            }
        )
    return rows


def write_csv(rows: List[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "sentence_id",
        "query_label",
        "query_position",
        "query_token",
        "sentence_text",
        "top1_token",
        "top1_position",
        "top1_score",
        "top2_token",
        "top2_position",
        "top2_score",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Garden head attention table across query positions.")
    parser.add_argument("--standard-json", type=Path, required=True, help="Path to standard_garden_data.json")
    parser.add_argument("--head", type=str, required=True, help="Head as layer.head, e.g. 7.4")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of sentences to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument(
        "--query-positions",
        type=str,
        default=",".join(DEFAULT_QUERY_POSITIONS),
        help="Comma-separated query positions: subj,verb,obj_head,rel_pron,rel_verb,end",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local model path")
    args = parser.parse_args()

    layer, head_idx = map(int, args.head.split("."))
    target_head = (layer, head_idx)
    query_positions = [x.strip().lower() for x in args.query_positions.split(",") if x.strip()]

    samples = load_standard_garden(args.standard_json)
    samples = sample_items(samples, args.sample_size, args.seed)

    model = load_local_hooked_transformer(args.model_path, device=args.device)
    _ensure_pad_token(model)

    all_rows: List[dict] = []
    for query_label in query_positions:
        results: List[dict] = []
        for batch in _iter_batches(samples, args.batch_size):
            results.extend(compute_attention_for_batch(model, batch, target_head, query_label))
        all_rows.extend(flatten_results(results, query_label))

    write_csv(all_rows, args.output_csv)
    if args.output_json:
        write_json(all_rows, args.output_json)

    print(f"Wrote CSV: {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
