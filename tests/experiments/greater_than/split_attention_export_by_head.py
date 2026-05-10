#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a multi-head greater-than attention export JSON into one file per head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a multi-head greater-than attention export into per-head JSON files."
    )
    p.add_argument("--input-file", type=Path, required=True, help="Input multi-head JSON.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for per-head JSON files.")
    p.add_argument(
        "--per-head-sample-limit",
        type=int,
        default=100,
        help="Keep only the first N per-sample records for each head file.",
    )
    return p.parse_args()


def sanitize_head_name(head: str) -> str:
    return head.replace(".", "_")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_file.expanduser().resolve().read_text(encoding="utf-8"))
    meta: Dict[str, Any] = payload.get("meta", {})
    heads: Dict[str, Any] = payload.get("heads", {})
    if not heads:
        raise ValueError("Input file contains no `heads` entries.")

    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    for head_name, record in heads.items():
        per_sample = record.get("per_sample", [])
        if args.per_head_sample_limit > 0:
            per_sample = per_sample[: args.per_head_sample_limit]

        per_head_payload = {
            "meta": {
                **meta,
                "source_file": str(args.input_file.expanduser().resolve()),
                "head": head_name,
                "per_head_sample_limit": args.per_head_sample_limit,
            },
            "head": {
                **record,
                "per_sample": per_sample,
            },
        }
        out_path = args.output_dir / f"head_{sanitize_head_name(head_name)}.json"
        out_path.write_text(
            json.dumps(per_head_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
