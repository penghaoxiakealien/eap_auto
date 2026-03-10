#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a *scheduler-friendly* head graph config from a grouped graph JSON.

Motivation (IOI):
- The grouped graph edges (from plot_grouped_graph.py / pruning) can include edges that do
  not align with an intuitive "upstream -> downstream -> logits" flow (e.g. terminal groups
  pointing to other groups). This makes the automatic explanation scheduler confusing.
- Default behavior: keep group->group edge directions as-is, but remove any
  outgoing edges from terminal groups to other groups (so terminal groups only
  point to logits). This matches the "upstream -> downstream -> logits" intuition
  used by the explanation scheduler.
- Optional behavior: re-orient edges to point toward terminal/logits based on
  undirected distance (legacy mode).

Inputs:
  --input: grouped graph JSON (groups/edges[/sources]/sinks)
Outputs:
  --output: head_graph_config JSON for tests/experiments/run_head_graph.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate scheduler-friendly head graph config from grouped graph JSON.")
    p.add_argument("--input", type=Path, required=True, help="Path to grouped graph JSON.")
    p.add_argument("--output", type=Path, required=True, help="Output path for head graph config JSON.")
    p.add_argument(
        "--mode",
        choices=["prune_terminal_outgoing", "orient_to_terminals"],
        default="prune_terminal_outgoing",
        help="How to transform edges for scheduling.",
    )
    p.add_argument(
        "--drop-equal-level-edges",
        action="store_true",
        help="Drop edges between groups at the same level (default: drop).",
    )
    p.add_argument(
        "--terminal-top-k",
        type=int,
        default=0,
        help=(
            "If >0, only treat the top-K terminal groups (by abs(mean_delta_logit_diff)) as "
            "terminal for scheduler transforms (e.g. pruning outgoing edges). This prevents "
            "accidentally treating many tiny group->logits links as terminal sinks."
        ),
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object")
    return obj


def compute_levels(
    group_ids: Set[str],
    edges: List[Tuple[str, str]],
    terminals: Set[str],
) -> Dict[str, int]:
    """Compute undirected shortest distance to nearest terminal group."""
    undirected: Dict[str, List[str]] = {g: [] for g in group_ids}
    for a, b in edges:
        if a in undirected and b in undirected:
            undirected[a].append(b)
            undirected[b].append(a)

    dist: Dict[str, int] = {g: 10**9 for g in group_ids}
    q = deque()
    for t in terminals:
        if t in dist:
            dist[t] = 0
            q.append(t)
    while q:
        cur = q.popleft()
        for nxt in undirected[cur]:
            if dist[nxt] > dist[cur] + 1:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def main() -> None:
    args = parse_args()
    data = load_json(args.input)

    groups = data.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("input grouped graph missing 'groups' list")

    group_ids: Set[str] = set()
    for g in groups:
        gid = (g or {}).get("id") if isinstance(g, dict) else None
        if isinstance(gid, str):
            group_ids.add(gid)

    # Collect group->group edges and group->logits edges.
    raw_edges = data.get("edges", []) or []
    group_to_group: List[Tuple[str, str]] = []
    group_to_logits: List[Tuple[str, str]] = []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        dst = e.get("target")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if src in group_ids and dst in group_ids:
            group_to_group.append((src, dst))
        elif src in group_ids and dst == "logits":
            group_to_logits.append((src, dst))

    terminals = {src for src, _ in group_to_logits}
    if not terminals:
        raise ValueError("No terminal groups found (no group -> logits edges).")

    # Optionally pick only the most important terminal groups.
    if args.terminal_top_k and args.terminal_top_k > 0:
        group_summary = {
            (g or {}).get("id"): (g or {}).get("summary", {}) or {}
            for g in groups
            if isinstance(g, dict) and isinstance((g or {}).get("id"), str)
        }
        scored: List[Tuple[str, float]] = []
        for gid in terminals:
            delta = (group_summary.get(gid) or {}).get("mean_delta_logit_diff")
            try:
                score = abs(float(delta)) if delta is not None else 0.0
            except Exception:
                score = 0.0
            scored.append((gid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_terminals = {gid for gid, _ in scored[: args.terminal_top_k]}
    else:
        selected_terminals = set(terminals)

    levels = compute_levels(group_ids, group_to_group, selected_terminals)

    downstream_map: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    dropped = 0
    kept = 0

    if args.mode == "orient_to_terminals":
        # Re-orient edges to point toward smaller level (toward terminals/logits).
        oriented: List[Tuple[str, str]] = []
        for a, b in group_to_group:
            la = levels.get(a, 10**9)
            lb = levels.get(b, 10**9)
            if la == 10**9 or lb == 10**9:
                dropped += 1
                continue
            if la == lb and args.drop_equal_level_edges:
                dropped += 1
                continue
            if la > lb:
                oriented.append((a, b))
            elif lb > la:
                oriented.append((b, a))
            else:
                dropped += 1
        oriented = sorted(set(oriented))

        count_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            dst = e.get("target")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src in group_ids and dst in group_ids:
                la = levels.get(src, 10**9)
                lb = levels.get(dst, 10**9)
                if la == 10**9 or lb == 10**9 or (la == lb and args.drop_equal_level_edges):
                    continue
                o = (src, dst) if la > lb else (dst, src)
                if o in oriented:
                    try:
                        count_map[o] += int(e.get("count", 0))
                    except Exception:
                        count_map[o] += 0

        for src, dst in oriented:
            downstream_map[src].append({"target": dst, "count": int(count_map.get((src, dst), 0))})
            kept += 1
    else:
        # Keep original directions, but drop any outgoing edges from *selected* terminal groups to other groups.
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            dst = e.get("target")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src in selected_terminals and dst in group_ids:
                dropped += 1
                continue
            if src in group_ids and dst in group_ids:
                try:
                    cnt = int(e.get("count", 0))
                except Exception:
                    cnt = 0
                downstream_map[src].append({"target": dst, "count": cnt})
                kept += 1

    # Always include group->logits edges.
    for src, dst in group_to_logits:
        downstream_map[src].append({"target": dst, "count": 0})

    nodes: List[Dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        gid = g.get("id")
        if not isinstance(gid, str):
            continue
        node = {
            "id": gid,
            "signature": g.get("signature"),
            "classification": g.get("classification"),
            "heads": g.get("heads", []),
            "layers": g.get("layers", []),
            "summary": g.get("summary", {}),
            "downstreams": downstream_map.get(gid, []),
            "level_to_terminal": int(levels.get(gid, 10**9)),
        }
        nodes.append(node)

    config = {
        "meta": {
            "source_group_json": str(args.input),
            "scheduler_transform": {
                "terminal_groups": sorted(terminals),
                "selected_terminal_groups": sorted(selected_terminals),
                "terminal_top_k": int(args.terminal_top_k),
                "mode": args.mode,
                "kept_group_edges": kept,
                "dropped_group_edges": dropped,
                "drop_equal_level_edges": bool(args.drop_equal_level_edges),
            },
        },
        "nodes": nodes,
        "heads": data.get("heads", {}),
        "sinks": data.get("sinks", [{"id": "logits", "label": "logits"}]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"✅ scheduler head graph config saved to {args.output}")


if __name__ == "__main__":
    main()
