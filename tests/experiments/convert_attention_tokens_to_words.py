#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert BPE token attention scores to strict word-level tokens with positional suffixes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import GPT2TokenizerFast


LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


def load_tokenizer() -> GPT2TokenizerFast:
    print("Loading GPT-2 tokenizer...", flush=True)
    if os.path.isdir(LOCAL_MODEL_DIR):
        return GPT2TokenizerFast.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)


TOKENIZER = None


def tokenize_with_spans(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer()
    enc = TOKENIZER(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = TOKENIZER.convert_ids_to_tokens(enc["input_ids"])
    offsets = enc["offset_mapping"]
    return list(zip(tokens, offsets))


def extract_words_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    words = []
    spans = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans


def build_suffix_map(words: List[str]) -> Dict[int, str]:
    total = defaultdict(int)
    for w in words:
        total[w.lower()] += 1
    running = defaultdict(int)
    suffixed = {}
    for idx, w in enumerate(words):
        key = w.lower()
        running[key] += 1
        if total[key] > 1:
            suffixed[idx] = f"{w}_{running[key]}"
        else:
            suffixed[idx] = w
    return suffixed


def map_bpe_to_word_index(
    bpe_offsets: List[Tuple[str, Tuple[int, int]]],
    word_spans: List[Tuple[int, int]],
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for idx, (_, (start, end)) in enumerate(bpe_offsets):
        if start == end == 0:
            continue
        for w_idx, (w_start, w_end) in enumerate(word_spans):
            if start < w_end and end > w_start:
                mapping[idx] = w_idx
                break
    return mapping


def convert_entry(
    entry: Dict[str, object],
    top_k: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    sentence = str(entry.get("original_sentence", "")).strip()
    if not sentence:
        raise ValueError("missing original_sentence")

    words, spans = extract_words_with_spans(sentence)
    suffixed_map = build_suffix_map(words)
    bpe_tokens = tokenize_with_spans(sentence)
    bpe_to_word = map_bpe_to_word_index(bpe_tokens, spans)

    word_scores: Dict[int, float] = defaultdict(float)
    for item in entry.get("attention_scores", []):
        pos = item.get("position")
        score = float(item.get("score", 0.0))
        if pos is None:
            continue
        if pos in bpe_to_word:
            word_scores[bpe_to_word[pos]] += score

    scored_words = [
        {
            "token": suffixed_map[idx],
            "position": idx,
            "score": score,
        }
        for idx, score in word_scores.items()
    ]
    scored_words.sort(key=lambda x: x["score"], reverse=True)

    top_tokens = [item["token"] for item in scored_words[:top_k]]
    sentence_id = str(entry.get("sentence_id", ""))

    preprocessed_item = {
        "sentence_id": sentence_id,
        "sentence_text": sentence,
        "top_k_tokens": top_tokens,
    }
    ground_truth_item = {
        "key": sentence_id,
        "original_sentence": sentence,
        "attention_scores": scored_words,
    }
    return preprocessed_item, ground_truth_item


def main() -> None:
    p = argparse.ArgumentParser(description="Convert BPE attention tokens to strict word-level tokens.")
    p.add_argument("--input-jsonl", type=Path, required=True, help="preprocessed_for_sampling.jsonl")
    p.add_argument("--output-preprocessed", type=Path, required=True, help="preprocessed_attention_scores.json")
    p.add_argument("--output-ground-truth", type=Path, required=True, help="attention_scores_ground_truth.jsonl (JSON list)")
    p.add_argument("--top-k", type=int, default=2, help="Top-K tokens to keep")
    args = p.parse_args()

    total = 0
    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    print(f"Converting {total} entries from {args.input_jsonl}...", flush=True)

    preprocessed = []
    ground_truth = []
    with args.input_jsonl.open("r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx += 1
            entry = json.loads(line)
            try:
                pre_item, gt_item = convert_entry(entry, args.top_k)
            except Exception:
                continue
            preprocessed.append(pre_item)
            ground_truth.append(gt_item)
            if idx % 100 == 0 or idx == total:
                print(f"processed {idx}/{total}", flush=True)

    args.output_preprocessed.parent.mkdir(parents=True, exist_ok=True)
    args.output_preprocessed.write_text(
        json.dumps(preprocessed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_ground_truth.parent.mkdir(parents=True, exist_ok=True)
    args.output_ground_truth.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ wrote {args.output_preprocessed}")
    print(f"✅ wrote {args.output_ground_truth}")


if __name__ == "__main__":
    main()
