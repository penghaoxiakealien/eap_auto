#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prune a grouped graph using Δlogit edge weights aggregated from joint_p.json.

This is intentionally separate from prune_group_graph.py, which uses batch_attention deltas.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune grouped graph edges by Δlogit (from joint_p.json)")
    p.add_argument("--group-json", type=Path, required=True, help="Input grouped graph JSON")
    p.add_argument("--joint-json", type=Path, required=True, help="joint_p.json with per-edge delta_logit_diff")
    p.add_argument("--output-json", type=Path, required=True, help="Output pruned grouped graph JSON")
    p.add_argument(
        "--output-png",
        type=Path,
        help="PNG output path (default: same as --output-json but with .png suffix)",
    )
    p.add_argument(
        "--agg",
        choices=["mean", "max_abs"],
        default="mean",
        help="How to aggregate head-level edge deltas into a group-edge weight",
    )
    p.add_argument(
        "--min-abs-weight",
        type=float,
        default=0.0,
        help="Keep edges with |weight| >= this threshold (applies to selected edge types)",
    )
    p.add_argument(
        "--edge-scope",
        choices=["internal", "all"],
        default="internal",
        help="Prune only group->group edges, or also prune input/logits edges",
    )
    p.add_argument(
        "--drop-isolated",
        action="store_true",
        help="Drop group nodes that become isolated (keeps input/logits if present)",
    )
    p.add_argument(
        "--report-quantiles",
        action="store_true",
        help="Print weight quantiles for the chosen edge-scope before pruning",
    )
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
            layer = int(left[1:])
            head = int(right)
        except ValueError:
            return None
        return f"{layer}.{head}"
    return None


def _quantiles(vals: List[float], ps: Iterable[float]) -> Dict[float, float]:
    if not vals:
        return {}
    s = sorted(vals)
    out: Dict[float, float] = {}
    for p in ps:
        idx = int(round((len(s) - 1) * p))
        out[p] = s[idx]
    return out


@dataclass(frozen=True)
class AggResult:
    weight: float
    n: int


def aggregate_group_edge_weights(
    group_graph: Dict[str, Any],
    joint: Dict[str, Any],
    agg: str,
) -> Dict[Tuple[str, str], AggResult]:
    head_to_group: Dict[str, str] = {}
    for g in group_graph.get("groups", []):
        gid = g.get("id")
        for h in g.get("heads", []):
            if isinstance(gid, str) and isinstance(h, str):
                head_to_group[h] = gid

    present_edges = {(e.get("source"), e.get("target")) for e in group_graph.get("edges", [])}

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
        bucket[(src_gid, dst_gid)].append(float(val))

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


def main() -> None:
    args = parse_args()

    group_graph = load_json(args.group_json)
    joint = load_json(args.joint_json)

    weights = aggregate_group_edge_weights(group_graph, joint, args.agg)

    # Report quantiles (abs weights) before pruning
    if args.report_quantiles:
        chosen: List[float] = []
        for e in group_graph.get("edges", []):
            src = e.get("source")
            dst = e.get("target")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if args.edge_scope == "internal" and (src in ("input", "logits") or dst in ("input", "logits")):
                continue
            w = weights.get((src, dst))
            if w is None:
                continue
            chosen.append(abs(w.weight))
        qs = _quantiles(chosen, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        print(f"[quantiles] n={len(chosen)} (|weight|) " + " ".join(f"p{int(k*100)}={v:.6g}" for k, v in qs.items()))

    kept_edges: List[Dict[str, Any]] = []
    for e in group_graph.get("edges", []):
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue

        is_special = src in ("input", "logits") or dst in ("input", "logits")
        if args.edge_scope == "internal" and is_special:
            # keep special edges unchanged
            out_e = dict(e)
            out_e.setdefault("source", src)
            out_e.setdefault("target", dst)
            kept_edges.append(out_e)
            continue

        w = weights.get((src, dst))
        if w is None:
            continue
        if abs(w.weight) < args.min_abs_weight:
            continue
        out_e = dict(e)
        out_e.setdefault("source", src)
        out_e.setdefault("target", dst)
        out_e["weight"] = w.weight
        out_e["n"] = w.n
        kept_edges.append(out_e)

    out_graph = dict(group_graph)
    out_graph["edges"] = kept_edges
    out_graph.setdefault("meta", {})
    out_graph["meta"] = dict(out_graph.get("meta", {}))
    out_graph["meta"].update(
        {
            "prune": {
                "method": "delta_logit",
                "joint_json": str(args.joint_json),
                "agg": args.agg,
                "edge_scope": args.edge_scope,
                "min_abs_weight": args.min_abs_weight,
            }
        }
    )

    if args.drop_isolated:
        deg: DefaultDict[str, int] = defaultdict(int)
        for e in kept_edges:
            s = e.get("source") or e.get("src")
            t = e.get("target") or e.get("dst")
            if isinstance(s, str) and isinstance(t, str):
                deg[s] += 1
                deg[t] += 1
        keep_group_ids = {g.get("id") for g in out_graph.get("groups", []) if deg.get(g.get("id"), 0) > 0}
        keep_group_ids.discard(None)
        out_graph["groups"] = [g for g in out_graph.get("groups", []) if g.get("id") in keep_group_ids]
        out_graph["heads"] = {
            h: payload for h, payload in out_graph.get("heads", {}).items() if payload.get("group") in keep_group_ids
        }
        out_graph["edges"] = [
            e
            for e in out_graph.get("edges", [])
            if ((e.get("source") or e.get("src")) in keep_group_ids or (e.get("source") or e.get("src")) in ("input", "logits"))
            and ((e.get("target") or e.get("dst")) in keep_group_ids or (e.get("target") or e.get("dst")) in ("input", "logits"))
        ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_graph, ensure_ascii=False, indent=2))
    print(f"✅ wrote pruned grouped graph json: {args.output_json}")
    print(f"kept groups={len(out_graph.get('groups', []))} edges={len(out_graph.get('edges', []))}")

    output_png = args.output_png or args.output_json.with_suffix(".png")
    if output_png:
        from render_group_graph import draw_group_graph  # uses plot_grouped_graph.draw_group_graph

        draw_group_graph(out_graph, output_png)
        print(f"✅ wrote pruned grouped graph png: {output_png}")


if __name__ == "__main__":
    main()
