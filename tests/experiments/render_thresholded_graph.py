#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 joint_p.json 的 delta_logit_diff 阈值筛边并渲染 PNG。
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import List, Tuple, Set
import pygraphviz as pgv

def parse_args():
    p = argparse.ArgumentParser(description="阈值筛边并渲染图")
    p.add_argument("--joint", type=Path, required=True, help="joint_p.json 路径")
    p.add_argument("--threshold", type=float, required=True, help="abs(delta_logit_diff) 阈值")
    p.add_argument("--output-prefix", type=Path, required=True, help="输出前缀（不含扩展名）")
    p.add_argument("--layout", type=str, default="dot", help="布局算法 (dot/neato/sfdp...)")
    return p.parse_args()

def main():
    args = parse_args()
    data = json.loads(args.joint.read_text())
    edges = data.get("edges", [])
    kept = []
    for e in edges:
        if abs(float(e.get("delta_logit_diff", 0))) < args.threshold:
            continue
        # 仅保留 qkv 联合或 logits
        if not (e.get("receiver_input") == "qkv" or e.get("receiver_kind") == "logits"):
            continue
        kept.append(e)

    pairs: Set[Tuple[str, str]] = set()
    nodes: Set[str] = set()
    for e in kept:
        src = e.get("src")
        dst = e.get("dst") or "logits"
        if not src or not dst:
            continue
        pairs.add((src, dst))
        nodes.add(src); nodes.add(dst)

    node_names = sorted(nodes)
    # 写输出 JSON/CSV
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps({
        "meta": {"threshold": args.threshold, "source": str(args.joint)},
        "nodes": node_names,
        "edge_list": sorted(pairs),
    }, indent=2))
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["src","dst"]); w.writerows(sorted(pairs))

    # 渲染 PNG
    png_path = prefix.with_suffix(".png")
    g = pgv.AGraph(directed=True, strict=True, splines="true", overlap="false", layout=args.layout)
    for n in node_names:
        g.add_node(n, shape="box", style="rounded,filled", fillcolor="#E8F0FE", fontname="Helvetica")
    for s,d in pairs:
        g.add_edge(s,d)
    g.draw(png_path, prog=args.layout)
    print(f"kept edges {len(pairs)}, nodes {len(node_names)}")
    print(f"written: {json_path}, {csv_path}, {png_path}")

if __name__ == "__main__":
    main()
