#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def write_csv(rows, path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Garden per-sentence logit direction stats.")
    parser.add_argument("--input-json", type=Path, required=True, help="heads_direct_effect_on_logit_difference.json")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Output summary CSV")
    parser.add_argument("--details-csv", type=Path, help="Optional per-sentence details CSV")
    parser.add_argument(
        "--metric",
        choices=["raw", "normalized"],
        default="normalized",
        help="Which delta metric to summarize. Defaults to normalized.",
    )
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    per_sentence = data.get("per_sentence", [])
    meta = data.get("meta", {})
    head = meta.get("head", "")

    direction_counts = Counter()
    rows = []
    abs_vals = []
    metric_key = "normalized_delta_logit_diff" if args.metric == "normalized" else "delta_logit_diff"
    for rec in per_sentence:
        delta = rec.get(metric_key)
        if delta is None:
            continue
        direction = "increase" if float(delta) >= 0 else "decrease"
        direction_counts[direction] += 1
        abs_vals.append(abs(float(delta)))
        rows.append(
            {
                "head": head,
                "sentence_id": rec.get("sentence_id", ""),
                "sentence_text": rec.get("sentence_text", ""),
                "clean_logit_diff": rec.get("clean_logit_diff", ""),
                "corrupted_logit_diff": rec.get("corrupted_logit_diff", ""),
                "patched_logit_diff": rec.get("patched_logit_diff", ""),
                "delta_logit_diff": rec.get("delta_logit_diff", ""),
                "normalized_delta_logit_diff": rec.get("normalized_delta_logit_diff", ""),
                "selected_metric": args.metric,
                "selected_delta": delta,
                "direction": direction,
            }
        )

    total = sum(direction_counts.values())
    increase = direction_counts["increase"]
    decrease = direction_counts["decrease"]
    summary_rows = [
        {
            "head": head,
            "total_sentences": total,
            "increase_count": increase,
            "decrease_count": decrease,
            "increase_ratio": (increase / total) if total else "",
            "decrease_ratio": (decrease / total) if total else "",
            "metric": args.metric,
            "mean_abs_delta": (sum(abs_vals) / len(abs_vals)) if abs_vals else "",
        }
    ]
    write_csv(
        summary_rows,
        args.summary_csv,
        [
            "head",
            "total_sentences",
            "increase_count",
            "decrease_count",
            "increase_ratio",
            "decrease_ratio",
            "metric",
            "mean_abs_delta",
        ],
    )
    if args.details_csv:
        write_csv(
            rows,
            args.details_csv,
            [
                "head",
                "sentence_id",
                "sentence_text",
                "clean_logit_diff",
                "corrupted_logit_diff",
                "patched_logit_diff",
                "delta_logit_diff",
                "normalized_delta_logit_diff",
                "selected_metric",
                "selected_delta",
                "direction",
            ],
        )

    print(f"Wrote summary: {args.summary_csv}")
    if args.details_csv:
        print(f"Wrote details: {args.details_csv}")
    print(
        f"{head}: metric={args.metric}, total={total}, "
        f"decrease={decrease} ({(decrease/total if total else 0):.3f}), "
        f"increase={increase} ({(increase/total if total else 0):.3f})"
    )


if __name__ == "__main__":
    main()
