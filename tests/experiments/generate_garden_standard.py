#!/usr/bin/env python3
"""
Generate standard garden data JSON for pattern mining and inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from garden_dataset import GardenDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate standard garden data JSON.")
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("datasets/garden/garden_npz_v_trans_mod.csv"),
        help="输入数据文件 (CSV).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/garden/standard_garden_data.json"),
        help="输出 JSON 路径",
    )
    p.add_argument("--prepend-bos", action="store_true", help="在文本前添加 BOS。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    ds = GardenDataset(
        tokenizer=tokenizer,
        data_path=args.data_path,
        prepend_bos=args.prepend_bos,
    )

    records = []
    for i, s in enumerate(ds.samples):
        rec = {
            "clean": s.clean,
            "corrupted": s.corrupted,
            "text": s.clean,
            "corrupted_text": s.corrupted,
            "correct_token": s.correct_token,
            "incorrect_token": s.incorrect_token,
            "template": s.template,
            "tokenized_clean": [tokenizer.decode(tok) for tok in ds.toks[i]],
            "tokenized_corrupted": [tokenizer.decode(tok) for tok in ds.corrupted_toks[i]],
            "end_idx": int(ds.input_lengths[i].item() - 1),
            "word_idx": ds.word_idx[i],
        }
        records.append(rec)

    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"✅ 保存标准数据到: {args.output} (共 {len(records)} 条)")


if __name__ == "__main__":
    main()
