#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keep only selected groups' edges into logits, drop other direct-logits edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune logits edges by allowed group ids.")
    p.add_argument("--group-json", type=Path, required=True, help="Input grouped graph JSON.")
    p.add_argument(
        "--allow-groups",
        required=True,
        help="Comma-separated group ids allowed to connect to logits.",
    )
    p.add_argument(
        "--remove-outgoing",
        action="store_true",
        help="Remove edges from allowed groups to non-logits targets.",
    )
    p.add_argument(
        "--drop-no-outgoing",
        action="store_true",
        help="Drop groups with no outgoing edges (keeps logits/input).",
    )
    p.add_argument(
        "--keep-logits-ancestors",
        action="store_true",
        help="Keep only nodes that can reach logits (always keeps input/logits).",
    )
    p.add_argument("--output-json", type=Path, required=True, help="Output grouped graph JSON.")
    p.add_argument("--output-png", type=Path, help="Optional PNG output path.")
    return p.parse_args()


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object")
    return obj


def keep_edge(edge: dict, allowed: List[str], remove_outgoing: bool) -> bool:
    src = edge.get("source") or edge.get("src")
    dst = edge.get("target") or edge.get("dst")
    if dst == "logits":
        return src in allowed
    if remove_outgoing and src in allowed:
        return False
    return True


def drop_groups_without_outgoing(data: dict) -> None:
    groups = data.get("groups", []) or []
    edges = data.get("edges", []) or []
    # Build outgoing map for group nodes only.
    outgoing = {}
    for g in groups:
        gid = g.get("id")
        if isinstance(gid, str):
            outgoing[gid] = 0
    for e in edges:
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if isinstance(src, str) and src in outgoing and dst is not None:
            outgoing[src] += 1
    keep_groups = {gid for gid, n in outgoing.items() if n > 0}
    # Always keep special nodes if present in groups list.
    keep_groups.update({g.get("id") for g in groups if g.get("id") in ("logits", "input")})
    data["groups"] = [g for g in groups if g.get("id") in keep_groups]
    data["edges"] = [
        e
        for e in edges
        if (e.get("source") or e.get("src")) in keep_groups
        and (e.get("target") or e.get("dst")) in keep_groups
    ]


def keep_logits_ancestors(data: dict) -> None:
    edges = data.get("edges", []) or []
    rev = {}
    for e in edges:
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        rev.setdefault(dst, set()).add(src)
    keep = {"logits", "input"}
    stack = ["logits"]
    while stack:
        node = stack.pop()
        for parent in rev.get(node, set()):
            if parent not in keep:
                keep.add(parent)
                stack.append(parent)
    data["groups"] = [g for g in data.get("groups", []) if g.get("id") in keep]
    data["edges"] = [
        e
        for e in edges
        if (e.get("source") or e.get("src")) in keep
        and (e.get("target") or e.get("dst")) in keep
    ]


def main() -> None:
    args = parse_args()
    allowed = [g.strip() for g in args.allow_groups.split(",") if g.strip()]
    if not allowed:
        raise ValueError("--allow-groups is empty")

    data = load_json(args.group_json)
    edges = data.get("edges", []) or []
    pruned = [e for e in edges if keep_edge(e, allowed, args.remove_outgoing)]
    data["edges"] = pruned
    if args.keep_logits_ancestors:
        keep_logits_ancestors(data)
    elif args.drop_no_outgoing:
        drop_groups_without_outgoing(data)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"✅ wrote pruned grouped graph json: {args.output_json}")

    if args.output_png:
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo_root))
        from tests.experiments.plot_grouped_graph import draw_group_graph

        draw_group_graph(data, args.output_png)
        print(f"✅ wrote pruned grouped graph png: {args.output_png}")


if __name__ == "__main__":
    main()
