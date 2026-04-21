#!/usr/bin/env python
"""Generate IOI prompts using IOIDataset and export them as structured JSONL.

For each generated prompt we record the sentence text, IO/S tokens, tokenized
sequence, and key indices so the output mirrors `structured_sentences.jsonl`
while guaranteeing identical token lengths (since IOIDataset uses fixed
templates).

Example:
    python generate_ioi_structured_sentences.py \
        --count 50 \
        --output ../../results/ioi/path_patching/structured_sentences_standard.jsonl \
        --tokenizer-path /home/wangziran/gpt2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# ensure project root on path when executed from tests/experiments/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from transformers import AutoTokenizer  # type: ignore

from eap_auto.tests.experiments.ioi_dataset import IOIDataset  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured IOI prompts with IO/S annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=50, help="Number of prompts to generate")
    parser.add_argument(
        "--prompt-type",
        default="mixed",
        choices=["ABBA", "BABA", "mixed", "ABC", "BAC", "ABC mixed"],
        help="Which IOI template family to sample",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for sampling")
    parser.add_argument("--device", default="cuda", help="Device string passed to IOIDataset")
    parser.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Whether to prepend the BOS token when tokenizing prompts.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional local path to a GPT-2 style tokenizer (avoids network fetch).",
    )
    parser.add_argument(
        "--output",
        default="../../results/ioi/path_patching/structured_sentences_standard.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--id-prefix",
        default="ioi_std",
        help="Prefix for sentence_id values, numbers start from 1.",
    )
    return parser.parse_args()


def load_tokenizer(path: Optional[str]) -> AutoTokenizer:
    if path:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)

    dataset = IOIDataset(
        prompt_type=args.prompt_type,
        N=args.count,
        tokenizer=tokenizer,
        seed=args.seed,
        prepend_bos=args.prepend_bos,
        device=args.device,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as handle:
        for idx, prompt in enumerate(dataset.ioi_prompts):
            toks = dataset.toks[idx].tolist()
            str_tokens = dataset.tokenizer.convert_ids_to_tokens(toks)
            entry = {
                "sentence_id": f"{args.id_prefix}_{idx + 1:04d}",
                "sentence_text": dataset.sentences[idx],
                "io_token": f" {prompt['IO']}",
                "s_token": f" {prompt['S']}",
                "io_index": int(dataset.word_idx["IO"][idx]),
                "s1_index": int(dataset.word_idx["S1"][idx]),
                "s2_index": int(dataset.word_idx["S2"][idx]),
                "tokens": str_tokens,
                "token_ids": toks,
            }
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"Wrote {args.count} prompts to {args.output}")


if __name__ == "__main__":
    main()
