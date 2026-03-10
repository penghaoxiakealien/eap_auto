#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 agr_gender 可视化：
- 输入：筛选/折叠图 JSON（nodes/edge_list），模式 soft JSON（classify_agr_gender 输出）
- 输出：dot/png，节点标签标注 END/VERB 签名
"""
import argparse, json
from pathlib import Path
import pygraphviz as pgv

def parse_args():
    p = argparse.ArgumentParser(description="可视化 agr_gender 图并标注 END/VERB 签名")
    p.add_argument("--graph-json", type=Path, required=True)
    p.add_argument("--soft-json", type=Path, required=True, help="classify_agr_gender 产物")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layout", type=str, default="dot")
    p.add_argument(
        "--show-pattern-stats",
        action="store_true",
        help="在每个 head 节点上额外显示各 query 位置 top-k 模式的 count/total。",
    )
    return p.parse_args()

def load_graph(graph_json: Path):
    data = json.loads(graph_json.read_text())
    nodes = data.get("nodes")
    edges = data.get("edge_list") or data.get("edges") or []
    if isinstance(nodes, dict):
        nodes = [n for n, info in nodes.items() if (info or {}).get("in_graph", False)]
    return nodes, edges

def load_patterns(soft_json: Path):
    data = json.loads(soft_json.read_text())
    per_head = data.get("per_head", {}) or {}
    positions = data.get("positions", []) or []
    return {
        "per_head": per_head,
        "positions": [str(p).upper() for p in positions],
    }

def head_key(node_name: str) -> str:
    # aL.hH -> "L.H"
    if isinstance(node_name, str) and node_name.startswith("a") and ".h" in node_name:
        left, right = node_name.split(".h", 1)
        try:
            layer = int(left[1:])
            head = int(right)
            return f"{layer}.{head}"
        except ValueError:
            return node_name
    return node_name

def format_pos_stats(pos_list):
    # pos_list: [{"pattern":..., "count":..., "total":...}, ...]
    if not isinstance(pos_list, list) or not pos_list:
        return "NA"
    parts = []
    for entry in pos_list:
        if not isinstance(entry, dict):
            continue
        pat = entry.get("pattern") or entry.get("raw_pattern") or ""
        c = entry.get("count")
        t = entry.get("total")
        if isinstance(c, int) and isinstance(t, int) and t > 0:
            parts.append(f"{pat} {c}/{t}")
        else:
            parts.append(str(pat))
    return " | ".join(parts) if parts else "NA"

def build_head_label(node_name: str, info: dict, positions: list, show_stats: bool) -> str:
    sig = ""
    if isinstance(info, dict):
        sig = info.get("signature", "") or ""
    lines = [node_name]
    if sig:
        lines.append(sig)
    if show_stats and isinstance(info, dict):
        pos_map = info.get("positions") or {}
        for pos in positions:
            lst = pos_map.get(pos) or pos_map.get(pos.lower()) or []
            if lst:
                lines.append(f"{pos}: {format_pos_stats(lst)}")
    return "\\n".join(lines)

def main():
    args = parse_args()
    nodes, edges = load_graph(args.graph_json)
    patterns = load_patterns(args.soft_json)
    per_head = patterns["per_head"]
    positions = patterns["positions"]

    g = pgv.AGraph(
        directed=True, strict=True, splines="true", overlap="false", layout=args.layout
    )
    for n in nodes:
        if isinstance(n, str) and n.startswith("a") and ".h" in n:
            hk = head_key(n)
            info = per_head.get(hk, {})
            label = build_head_label(n, info, positions, args.show_pattern_stats)
            g.add_node(n, shape="box", style="rounded,filled", fillcolor="#E8F0FE", label=label, fontname="Helvetica")
        else:
            g.add_node(n, shape="ellipse", style="filled", fillcolor="#EEEEEE", fontname="Helvetica")
    for src, dst in edges:
        g.add_edge(src, dst)
    g.draw(args.output, prog=args.layout)
    print(f"✅ 写出可视化: {args.output}")

if __name__ == "__main__":
    main()
