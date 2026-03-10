#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Break cycles in a *group graph* by deleting weakest group->group edges.

This is the IOI-friendly counterpart to break_cycles_by_splitting_heads.py:
- It operates directly on an existing grouped graph JSON (groups/edges/sources/sinks),
  so you don't need dominant-soft or head-level graph inputs.
- Edge "strength" is computed from joint_p.json (delta_logit_diff) aggregated to group edges.

默认会输出 PNG（同名 .png），并写一份 report.json 记录删边过程。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Break cycles in grouped graph using Δlogit weights (from joint_p.json)")
    p.add_argument("--group-json", type=Path, required=True, help="Input grouped graph JSON")
    p.add_argument("--joint-json", type=Path, required=True, help="joint_p.json with per-edge delta_logit_diff")
    p.add_argument("--output-json", type=Path, required=True, help="Output grouped graph JSON (DAG-ish)")
    p.add_argument("--output-png", type=Path, help="Output PNG path (default: output-json with .png)")
    p.add_argument("--report-json", type=Path, help="Optional report JSON path (default: output-json + _report.json)")
    p.add_argument(
        "--agg",
        choices=["mean", "max_abs"],
        default="mean",
        help="How to aggregate head-level deltas into group-edge weight",
    )
    p.add_argument(
        "--min-abs-weight",
        type=float,
        default=0.0,
        help="Ignore head-level edges with |delta| < this when aggregating",
    )
    p.add_argument(
        "--max-deletions",
        type=int,
        default=2000,
        help="Safety cap on number of group edges deleted",
    )
    p.add_argument(
        "--exclude-nodes",
        type=str,
        default="input,logits",
        help="Comma-separated node ids to exclude from cycle handling",
    )
    p.add_argument("--layout", type=str, default="dot", help="Graphviz layout (dot/neato/fdp/sfdp)")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object")
    return obj


def _map_joint_node(name: Any) -> Optional[str]:
    if name in ("input", "logits"):
        return str(name)
    if not isinstance(name, str):
        return None
    if name.startswith("a") and ".h" in name:
        left, right = name.split(".h", 1)
        try:
            return f"{int(left[1:])}.{int(right)}"
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class AggResult:
    weight: float
    n: int


def aggregate_group_edge_weights(
    group_graph: Dict[str, Any],
    joint: Dict[str, Any],
    agg: str,
    min_abs: float,
) -> Dict[Tuple[str, str], AggResult]:
    head_to_group: Dict[str, str] = {}
    for g in group_graph.get("groups", []):
        gid = g.get("id")
        for h in g.get("heads", []) or []:
            if isinstance(gid, str) and isinstance(h, str):
                head_to_group[h] = gid

    present_edges = {
        (e.get("source") or e.get("src"), e.get("target") or e.get("dst"))
        for e in group_graph.get("edges", [])
    }

    bucket: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    for e in joint.get("edges", []):
        src = _map_joint_node(e.get("src"))
        dst = _map_joint_node(e.get("dst"))
        if not src or not dst:
            continue
        if src == "logits" or dst == "input":
            continue
        src_gid = "input" if src == "input" else head_to_group.get(src)
        dst_gid = "logits" if dst == "logits" else head_to_group.get(dst)
        if not src_gid or not dst_gid or src_gid == dst_gid:
            continue
        if (src_gid, dst_gid) not in present_edges:
            continue
        val = e.get("delta_logit_diff")
        if val is None:
            continue
        d = float(val)
        if abs(d) < min_abs:
            continue
        bucket[(src_gid, dst_gid)].append(d)

    out: Dict[Tuple[str, str], AggResult] = {}
    for key, vals in bucket.items():
        if not vals:
            continue
        if agg == "max_abs":
            best = max(vals, key=lambda x: abs(x))
            out[key] = AggResult(weight=float(best), n=len(vals))
        else:
            out[key] = AggResult(weight=float(mean(vals)), n=len(vals))
    return out


def strongly_connected_components(nodes: List[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    """Tarjan SCC."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)

    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def visit(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj[v]:
            if w not in indices:
                visit(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for n in nodes:
        if n not in indices:
            visit(n)
    return sccs


def render_png(group_graph: Dict[str, Any], out_png: Path, layout: str) -> None:
    from plot_grouped_graph import draw_group_graph

    draw_group_graph(group_graph, out_png)


def main() -> None:
    args = parse_args()
    g = load_json(args.group_json)
    joint = load_json(args.joint_json)

    exclude = {x.strip() for x in (args.exclude_nodes or "").split(",") if x.strip()}

    node_ids = [gg.get("id") for gg in g.get("groups", []) if isinstance(gg.get("id"), str)]
    node_set = set(node_ids)

    # group->group edges only (ignore input/logits)
    group_edges: List[Dict[str, Any]] = []
    for e in g.get("edges", []):
        s = e.get("source") or e.get("src")
        t = e.get("target") or e.get("dst")
        if s in node_set and t in node_set:
            group_edges.append(e)

    edge_pairs = [
        (e.get("source") or e.get("src"), e.get("target") or e.get("dst"))
        for e in group_edges
        if (e.get("source") or e.get("src")) and (e.get("target") or e.get("dst"))
    ]

    weights = aggregate_group_edge_weights(g, joint, args.agg, args.min_abs_weight)

    deletions: List[Dict[str, Any]] = []
    kept_pairs: Set[Tuple[str, str]] = set(edge_pairs)

    def current_edges() -> List[Tuple[str, str]]:
        return [p for p in kept_pairs if p[0] in node_set and p[1] in node_set]

    def score_edge(p: Tuple[str, str]) -> Tuple[float, float, int]:
        # weaker = smaller abs(weight); missing weight treated as 0 (weakest)
        w = weights.get(p)
        w_val = 0.0 if w is None else float(w.weight)
        abs_val = abs(w_val)
        # tie-break by count (prefer delete smaller count)
        cnt = 0
        for e in group_edges:
            s = e.get("source") or e.get("src")
            t = e.get("target") or e.get("dst")
            if s == p[0] and t == p[1]:
                try:
                    cnt = int(e.get("count", 0))
                except Exception:
                    cnt = 0
                break
        return (abs_val, abs(w_val), cnt)

    for _ in range(args.max_deletions):
        # SCCs excluding special nodes
        active_nodes = [n for n in node_ids if n not in exclude]
        edges_now = [p for p in current_edges() if p[0] not in exclude and p[1] not in exclude]
        sccs = strongly_connected_components(active_nodes, edges_now)
        cyclic = [c for c in sccs if len(c) > 1]
        if not cyclic:
            break

        # pick one SCC and delete weakest internal edge
        comp = max(cyclic, key=len)
        comp_set = set(comp)
        internal = [p for p in edges_now if p[0] in comp_set and p[1] in comp_set]
        if not internal:
            break
        internal.sort(key=score_edge)  # ascending => weakest first
        victim = internal[0]
        w = weights.get(victim)
        deletions.append(
            {
                "source": victim[0],
                "target": victim[1],
                "weight": None if w is None else w.weight,
                "n": None if w is None else w.n,
            }
        )
        kept_pairs.discard(victim)

    # rebuild edges list
    new_edges: List[Dict[str, Any]] = []
    for e in g.get("edges", []):
        s = e.get("source") or e.get("src")
        t = e.get("target") or e.get("dst")
        if isinstance(s, str) and isinstance(t, str) and (s, t) in kept_pairs:
            out_e = dict(e)
            w = weights.get((s, t))
            if w is not None:
                out_e["weight"] = w.weight
                out_e["n"] = w.n
            new_edges.append(out_e)
        elif not (s in node_set and t in node_set):
            # keep non group->group edges (input/logits edges)
            new_edges.append(e)

    out = dict(g)
    out["edges"] = new_edges
    out.setdefault("meta", {})
    out["meta"] = dict(out.get("meta", {}))
    out["meta"]["cycle_breaking"] = {
        "method": "delete_weakest_group_edges",
        "agg": args.agg,
        "min_abs_weight": args.min_abs_weight,
        "exclude_nodes": sorted(exclude),
        "max_deletions": args.max_deletions,
        "deleted": len(deletions),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ wrote DAG-ish group graph json: {args.output_json}")
    print(f"deleted_group_edges={len(deletions)}")

    out_png = args.output_png or args.output_json.with_suffix(".png")
    render_png(out, out_png, args.layout)
    print(f"✅ wrote DAG-ish group graph png: {out_png}")

    report_path = args.report_json or args.output_json.with_name(args.output_json.stem + "_report.json")
    report = {
        "input_group_json": str(args.group_json),
        "input_joint_json": str(args.joint_json),
        "deleted_edges": deletions,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"✅ wrote report json: {report_path}")


if __name__ == "__main__":
    main()
