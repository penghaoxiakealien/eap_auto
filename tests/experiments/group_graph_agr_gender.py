#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 agr_gender 的 head 图按 dominant signature 聚合为“组图”（group graph）。

输入：
  - --graph-json: head 级图（nodes/edge_list），例如 filtered/graph_thr0_001.json
  - --dominant-soft: extract_dominant_patterns_agr_gender.py 输出的 *_soft.json
  - --edge-weights-json: （可选）joint_p_qkv.json，用于给边附上 delta_logit_diff 的聚合权重

输出：
  - --output-json: 聚合后的组图 JSON
  - --output-png: （可选）渲染 PNG

聚合规则：
  - head -> group：按 per_head.signature 分组；缺失则归为 "none"
  - （可选）若提供 --layer-buckets，则在 signature 基础上再按 layer bucket 细分组，
    组 id 变为 "{signature}::L{bucket_label}"，例如 "END:END->A1@+0::L0-4"
  - group 边：只要原图存在 head_src->head_dst，就产生 group_src->group_dst（src!=dst）
    并统计 count（有多少条 head 边落入该 group 边）
  - 若提供 edge-weights-json，则对每条 head 边取 |delta_logit_diff| 最大的那条记录作为该 head 边的权重；
    group 边的 weight_abs 取其下 head 边权重的 max，weight_mean 取 mean（保留符号的 mean）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build agr_gender grouped graph from dominant signatures.")
    p.add_argument("--graph-json", type=Path, required=True)
    p.add_argument("--dominant-soft", type=Path, required=True)
    p.add_argument("--edge-weights-json", type=Path, default=None)
    p.add_argument("--min-edge-abs-score", type=float, default=0.0, help="按 |delta_logit_diff| 裁剪 head 边")
    p.add_argument(
        "--layer-buckets",
        type=str,
        default="",
        help=(
            "可选：按 layer 段进一步细分组，格式如 '0-4,5-8,9-11'。"
            "若为空则不按层切分。"
        ),
    )
    p.add_argument(
        "--max-heads-in-label",
        type=int,
        default=12,
        help="每个组节点标签里最多显示多少个 head（0=全显示）。",
    )
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-png", type=Path, default=None)
    p.add_argument("--layout", type=str, default="dot")
    return p.parse_args()


def load_graph(path: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    data = json.loads(path.read_text())
    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        nodes = [n for n, info in nodes.items() if (info or {}).get("in_graph", False)]
    if not isinstance(nodes, list):
        raise ValueError("graph-json nodes 格式不正确")
    edges = data.get("edge_list") or data.get("edges") or []
    if not isinstance(edges, list):
        raise ValueError("graph-json edge_list 格式不正确")
    return nodes, [(a, b) for a, b in edges]


def head_key(node_name: str) -> str:
    # aL.hH -> "L.H"
    if isinstance(node_name, str) and node_name.startswith("a") and ".h" in node_name:
        left, right = node_name.split(".h", 1)
        return f"{int(left[1:])}.{int(right)}"
    return node_name


def head_layer(node_name: str) -> Optional[int]:
    # aL.hH -> L
    if isinstance(node_name, str) and node_name.startswith("a") and ".h" in node_name:
        left = node_name.split(".h", 1)[0]
        try:
            return int(left[1:])
        except Exception:
            return None
    return None


def parse_layer_buckets(spec: str) -> List[Tuple[int, int, str]]:
    """
    Parse '0-4,5-8,9-11' -> [(0,4,'0-4'), (5,8,'5-8'), (9,11,'9-11')].
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    buckets: List[Tuple[int, int, str]] = []
    for chunk in spec.split(","):
        c = chunk.strip()
        if not c:
            continue
        if "-" not in c:
            raise ValueError(f"layer-buckets 格式错误: {c} (期望 'lo-hi')")
        lo_s, hi_s = [x.strip() for x in c.split("-", 1)]
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            lo, hi = hi, lo
        buckets.append((lo, hi, f"{lo}-{hi}"))
    return buckets


def bucket_for_layer(layer: Optional[int], buckets: List[Tuple[int, int, str]]) -> str:
    if layer is None or not buckets:
        return ""
    for lo, hi, label in buckets:
        if lo <= layer <= hi:
            return label
    return "other"


def load_signatures(dominant_soft: Path) -> Dict[str, str]:
    data = json.loads(dominant_soft.read_text())
    per_head = data.get("per_head", {})
    if not isinstance(per_head, dict):
        raise ValueError("dominant-soft 缺少 per_head dict")
    sigs: Dict[str, str] = {}
    for h, info in per_head.items():
        sig = (info or {}).get("signature") if isinstance(info, dict) else None
        if not isinstance(sig, str) or not sig:
            sig = "none"
        sigs[str(h)] = sig
    return sigs


def load_edge_weights(
    path: Optional[Path],
    min_abs: float,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    返回 head 边 (src,dst) 的聚合权重：
      - abs_max: 该边所有记录里 |delta| 最大值
      - signed_at_abs_max: 对应 abs_max 的 signed delta
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    records = payload.get("edges", [])
    if not isinstance(records, list):
        raise ValueError("edge-weights-json 缺少 edges 列表")

    best: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for e in records:
        src = e.get("src")
        dst = e.get("dst") or "logits"
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        delta = e.get("delta_logit_diff")
        if delta is None:
            continue
        d = float(delta)
        a = abs(d)
        if a < min_abs:
            continue
        key = (src, dst)
        prev = best.get(key)
        if prev is None or a > prev[0]:
            best[key] = (a, d)

    return {k: {"abs_max": v[0], "signed_at_abs_max": v[1]} for k, v in best.items()}


def render_group_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    out_png: Path,
    layout: str,
) -> None:
    import pygraphviz as pgv

    g = pgv.AGraph(directed=True, strict=True, splines="true", overlap="false", layout=layout)
    for n in nodes:
        g.add_node(
            n["id"],
            label=n["label"],
            shape="box",
            style="rounded,filled",
            fillcolor=n.get("fillcolor", "#E8F0FE"),
            fontname="Helvetica",
        )
    for e in edges:
        attrs = {}
        label = e.get("label")
        if label:
            attrs["label"] = label
            attrs["fontsize"] = "10"
        g.add_edge(e["src"], e["dst"], **attrs)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    g.draw(out_png, prog=layout)


def main() -> None:
    args = parse_args()
    nodes, edge_list = load_graph(args.graph_json)
    sigs = load_signatures(args.dominant_soft)
    weights = load_edge_weights(args.edge_weights_json, args.min_edge_abs_score)
    buckets = parse_layer_buckets(args.layer_buckets)

    # head -> group signature
    head_nodes = [n for n in nodes if isinstance(n, str) and n.startswith("a") and ".h" in n]
    head_to_group: Dict[str, str] = {}
    for n in head_nodes:
        k = head_key(n)
        sig = sigs.get(k, "none")
        if buckets:
            b = bucket_for_layer(head_layer(n), buckets)
            head_to_group[n] = f"{sig}::L{b}"
        else:
            head_to_group[n] = sig
    # keep input/logits as-is if present
    for n in nodes:
        if n in {"input", "logits"}:
            head_to_group[n] = n

    # group members
    group_members: Dict[str, List[str]] = defaultdict(list)
    for head, grp in head_to_group.items():
        if head in {"input", "logits"}:
            continue
        group_members[grp].append(head)
    for grp in group_members:
        group_members[grp].sort()

    # group edges aggregation
    group_edge_to_heads: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for s, d in edge_list:
        if s not in head_to_group or d not in head_to_group:
            continue
        gs, gd = head_to_group[s], head_to_group[d]
        if gs == gd:
            continue
        group_edge_to_heads[(gs, gd)].append((s, d))

    group_edges_out: List[Dict[str, Any]] = []
    for (gs, gd), heads_edges in sorted(group_edge_to_heads.items()):
        # weights aggregation
        signed_vals = []
        abs_vals = []
        for s, d in heads_edges:
            w = weights.get((s, d))
            if w is None:
                continue
            abs_vals.append(float(w["abs_max"]))
            signed_vals.append(float(w["signed_at_abs_max"]))
        payload: Dict[str, Any] = {
            "src": gs,
            "dst": gd,
            "count": len(heads_edges),
        }
        if abs_vals:
            payload["weight_abs_max"] = max(abs_vals)
            payload["weight_signed_mean"] = mean(signed_vals) if signed_vals else 0.0
        group_edges_out.append(payload)

    # build group nodes (id=signature)
    group_nodes_out: List[Dict[str, Any]] = []
    for grp, members in sorted(group_members.items(), key=lambda x: x[0]):
        show_n = args.max_heads_in_label
        if show_n == 0 or len(members) <= show_n:
            heads_line = ", ".join(members)
        else:
            heads_line = ", ".join(members[:show_n]) + ", …"
        label = f"{grp}\\nheads: {len(members)}\\n{heads_line}"
        group_nodes_out.append({"id": grp, "label": label})
    # keep input/logits nodes if present
    for n in ["input", "logits"]:
        if n in head_to_group:
            group_nodes_out.append({"id": n, "label": n, "fillcolor": "#EEEEEE"})

    out = {
        "meta": {
            "source_graph": str(args.graph_json),
            "dominant_soft": str(args.dominant_soft),
            "edge_weights": str(args.edge_weights_json) if args.edge_weights_json else None,
            "min_edge_abs_score": args.min_edge_abs_score,
        },
        "groups": group_nodes_out,
        "edges": group_edges_out,
        "group_members": group_members,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ wrote grouped graph json: {args.output_json}")

    if args.output_png:
        # edge labels
        edges_for_draw = []
        for e in group_edges_out:
            lbl = f"{e.get('count', 0)}"
            if "weight_abs_max" in e:
                lbl += f"\n|maxΔ|={float(e['weight_abs_max']):.3g}"
            edges_for_draw.append({"src": e["src"], "dst": e["dst"], "label": lbl})
        render_group_graph(group_nodes_out, edges_for_draw, args.output_png, args.layout)
        print(f"✅ wrote grouped graph png: {args.output_png}")


if __name__ == "__main__":
    main()
