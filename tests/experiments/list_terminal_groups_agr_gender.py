#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List terminal groups (groups with an outgoing edge to 'logits') for agr_gender.

Also prints the heads inside each group, and the inferred attention-position label
used by run_head_graph_agr_gender.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def infer_attention_position(group_id: str) -> str:
    base = group_id.split("::", 1)[0]
    if ":" in base:
        prefix = base.split(":", 1)[0].strip().lower()
    else:
        prefix = base.strip().lower()
    if prefix in {"end", "verb", "a1", "a2", "b"}:
        return prefix
    return "end"


def load_group_members(graph: dict) -> Dict[str, List[str]]:
    raw = graph.get("group_members") or {}
    if isinstance(raw, dict):
        out: Dict[str, List[str]] = {}
        for gid, hs in raw.items():
            if isinstance(hs, list):
                out[str(gid)] = [str(x) for x in hs if isinstance(x, str)]
        return out
    raise ValueError("group graph missing group_members")


def load_terminal_groups(graph: dict) -> List[Tuple[str, List[str]]]:
    members = load_group_members(graph)
    edges = graph.get("edges") or []
    downstream = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = e.get("src")
        d = e.get("dst")
        if isinstance(s, str) and isinstance(d, str):
            downstream.setdefault(s, []).append(d)
    terminal = []
    for gid, hs in members.items():
        if gid in {"input", "logits"}:
            continue
        if "logits" in downstream.get(gid, []):
            terminal.append((gid, hs))
    terminal.sort(key=lambda x: x[0])
    return terminal


def main() -> None:
    p = argparse.ArgumentParser(description="List terminal groups in a grouped agr_gender graph")
    p.add_argument("--group-graph", type=Path, required=True)
    p.add_argument("--print-heads", action="store_true", help="Print full head lists (no truncation)")
    args = p.parse_args()

    graph = json.loads(args.group_graph.read_text())
    terminal = load_terminal_groups(graph)
    print(f"Terminal groups: {len(terminal)}")
    for gid, hs in terminal:
        pos = infer_attention_position(gid)
        print(f"\n[{gid}]  attention_position={pos}  heads={len(hs)}")
        if not hs:
            continue
        if args.print_heads:
            print(", ".join(hs))
        else:
            preview = ", ".join(hs[:12])
            suffix = "" if len(hs) <= 12 else f", ... (+{len(hs)-12})"
            print(preview + suffix)


if __name__ == "__main__":
    main()

