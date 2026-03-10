#!/usr/bin/env python3
"""
生成 agr_gender 的标准数据文件（类似 standard_ioi_data.json），便于查看/复用。
输出结构（每条记录）示例：
{
  "text": "<clean 前缀>",
  "corrupted_text": "<corrupted 前缀>",
  "label": 0/1,          # 0 -> he, 1 -> she
  "S": "<主语名字>",
  "PR": "he/she",
  "verb": "<动词短语>",
  "verb_anchor": "<动词首词>",
  "template": "...",
  "tokenized_clean": [...],
  "tokenized_corrupted": [...],
  "end_idx": <预测位置索引>,
  "word_idx": {"end": <int>, "verb": <int|null>},
  "he_token_id": <int>,
  "she_token_id": <int>
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from agr_gender_dataset import AgrGenderDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate standard agr_gender data JSON.")
    p.add_argument("--data-path", type=Path, help="输入数据文件 (csv/json/jsonl)。若不提供则随机生成。")
    p.add_argument("--generate-n", type=int, default=0, help="随机生成样本数量（当未提供 data-path 时）。")
    p.add_argument("--output", type=Path, default=Path("results/agr_gender/standard_gender_data.json"))
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="可选：同时输出 CSV（列：clean,corrupted,label,S,PR,verb,verb_anchor,template）。",
    )
    p.add_argument("--prepend-bos", action="store_true", help="在文本前添加 BOS。")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 使用 fast tokenizer，保证与 run_agr_gender_gpt2.py 的默认行为一致
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    if args.data_path:
        ds = AgrGenderDataset(
            tokenizer=tokenizer,
            data_path=args.data_path,
            prepend_bos=args.prepend_bos,
            seed=args.seed,
        )
    else:
        if args.generate_n <= 0:
            raise ValueError("未提供 data-path 时需指定 --generate-n > 0")
        ds = AgrGenderDataset(
            tokenizer=tokenizer,
            generate_N=args.generate_n,
            prepend_bos=args.prepend_bos,
            seed=args.seed,
        )

    records = []
    for i, s in enumerate(ds.samples):
        rec = {
            # 兼容旧流水线：clean/corrupted/label
            "clean": s.clean,
            "corrupted": s.corrupted,
            "text": s.clean,
            "corrupted_text": s.corrupted,
            "label": s.label,
            "S": s.S,
            "D": getattr(s, "D", ""),
            "PR": s.PR,
            "verb": s.verb,
            "verb_anchor": s.verb_anchor,
            "template": s.template,
            "tokenized_clean": [tokenizer.decode(tok) for tok in ds.toks[i]],
            "tokenized_corrupted": [tokenizer.decode(tok) for tok in ds.corrupted_toks[i]],
            "end_idx": int(ds.input_lengths[i].item() - 1),
            "word_idx": ds.word_idx[i],
            "he_token_id": int(ds.he_token_id),
            "she_token_id": int(ds.she_token_id),
        }
        records.append(rec)

    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"✅ 保存标准数据到: {args.output} (共 {len(records)} 条)")

    if args.output_csv:
        import csv

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "clean",
                    "corrupted",
                    "label",
                    "S",
                    "PR",
                    "verb",
                    "verb_anchor",
                    "template",
                ],
            )
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"✅ 同时保存 CSV 到: {args.output_csv}")


if __name__ == "__main__":
    main()
