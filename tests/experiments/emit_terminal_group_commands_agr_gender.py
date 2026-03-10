#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emit bash commands to run terminal-head explanations for all terminal groups
in an agr_gender grouped graph (groups with edge -> logits).

This is a thin helper so you can do:
  python .../emit_terminal_group_commands_agr_gender.py ... > /tmp/run.sh
  bash /tmp/run.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def infer_attention_position(group_id: str) -> str:
    base = group_id.split("::", 1)[0]
    if ":" in base:
        prefix = base.split(":", 1)[0].strip().lower()
    else:
        prefix = base.strip().lower()
    if prefix in {"end", "verb", "a1", "a2", "b"}:
        return prefix
    return "end"


def split_head(head: str) -> tuple[int, int]:
    head = head.strip()
    if head.startswith("a") and ".h" in head:
        left, right = head.split(".h", 1)
        return int(left[1:]), int(right)
    if "." in head:
        l, h = head.split(".", 1)
        return int(l), int(h)
    raise ValueError(f"Unrecognized head: {head}")


def load_group_members(graph: dict) -> Dict[str, List[str]]:
    raw = graph.get("group_members") or {}
    if not isinstance(raw, dict):
        raise ValueError("group graph missing group_members")
    out: Dict[str, List[str]] = {}
    for gid, hs in raw.items():
        if isinstance(hs, list):
            out[str(gid)] = [str(x) for x in hs if isinstance(x, str)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Emit bash commands for all terminal heads in agr_gender")
    p.add_argument("--group-graph", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--results-root", type=Path, default=Path("results/agr_gender"))
    p.add_argument("--standard-json", type=Path, default=Path("results/agr_gender/standard_gender_data.json"))
    p.add_argument("--script", type=Path, default=Path("tests/experiments/run_gender_terminal_head.sh"))
    args = p.parse_args()

    graph = json.loads(args.group_graph.read_text())
    members = load_group_members(graph)
    edges = graph.get("edges") or []
    downstream: Dict[str, List[str]] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = e.get("src")
        d = e.get("dst")
        if isinstance(s, str) and isinstance(d, str):
            downstream.setdefault(s, []).append(d)

    terminal_groups = [gid for gid in members.keys() if gid not in {"input", "logits"} and "logits" in downstream.get(gid, [])]
    terminal_groups.sort()

    print("#!/bin/bash")
    print("set -euo pipefail")
    for gid in terminal_groups:
        pos = infer_attention_position(gid)
        for head in members.get(gid, []):
            if head in {"input", "logits"}:
                continue
            layer, h = split_head(head)
            cmd = [
                "bash",
                str(args.script),
                "--layer",
                str(layer),
                "--head",
                str(h),
                "--rounds",
                str(args.rounds),
                "--typename",
                "gender_terminal_head",
                "--results-root",
                str(args.results_root),
                "--standard-json",
                str(args.standard_json),
                "--attention-position",
                pos,
            ]
            print(" ".join(cmd))


if __name__ == "__main__":
    main()

