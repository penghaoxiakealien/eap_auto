#!/usr/bin/env python3
"""Render head_graph_config.json into a PNG (or DOT).

This is a convenience viewer for `tests/experiments/run_head_graph.py` configs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render head graph config JSON")
    p.add_argument("config_json", type=Path, help="Path to head_graph_config*.json")
    p.add_argument("--output-png", type=Path, help="Output PNG path")
    p.add_argument("--output-dot", type=Path, help="Optional DOT output path")
    p.add_argument("--layout", type=str, default="dot", help="Graphviz layout prog (dot/neato/fdp/sfdp)")
    p.add_argument("--max-heads", type=int, default=8, help="Max heads to show per node label (0=show none)")
    return p.parse_args()


def _short_heads(heads: List[str], max_heads: int) -> str:
    if max_heads <= 0 or not heads:
        return ""
    if len(heads) <= max_heads:
        return ", ".join(heads)
    return ", ".join(heads[:max_heads]) + ", ..."


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = json.loads(args.config_json.read_text())
    nodes = cfg.get("nodes", [])
    sinks = {s.get("id") for s in cfg.get("sinks", []) if isinstance(s, dict)}
    sinks.discard(None)

    if args.output_dot:
        lines: List[str] = []
        lines.append("digraph G {")
        lines.append('  rankdir="TB";')
        lines.append('  graph [splines=true, overlap=false];')
        lines.append('  node [fontname="Helvetica"];')
        for node in nodes:
            nid = node.get("id")
            if not isinstance(nid, str):
                continue
            signature = node.get("signature") or ""
            classification = node.get("classification") or ""
            heads = node.get("heads", []) or []
            heads_text = _short_heads([h for h in heads if isinstance(h, str)], args.max_heads)
            label_lines = [nid]
            if signature:
                label_lines.append(str(signature))
            if classification:
                label_lines.append(f"class: {classification}")
            if heads_text:
                label_lines.append(f"heads: {heads_text}")
            label = "\\n".join(label_lines).replace('"', '\\"')
            shape = "ellipse" if nid in sinks else "box"
            lines.append(f'  "{nid}" [label="{label}", shape={shape}];')

        for node in nodes:
            src = node.get("id")
            if not isinstance(src, str):
                continue
            for d in node.get("downstreams", []) or []:
                if not isinstance(d, dict):
                    continue
                dst = d.get("target")
                if not isinstance(dst, str):
                    continue
                count = d.get("count", None)
                if count is None:
                    lines.append(f'  "{src}" -> "{dst}";')
                else:
                    lines.append(f'  "{src}" -> "{dst}" [label="{count}"];')
        lines.append("}")
        args.output_dot.parent.mkdir(parents=True, exist_ok=True)
        args.output_dot.write_text("\n".join(lines))
        print(f"✅ wrote DOT: {args.output_dot}")

    if args.output_png:
        try:
            import pygraphviz as pgv
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "未安装 pygraphviz，无法渲染 PNG；请在 eap-ig 环境中安装 pygraphviz，或先输出 DOT 再用 dot 渲染。"
            ) from exc
        g = pgv.AGraph(directed=True, strict=False, splines="spline", overlap="false")
        for node in nodes:
            nid = node.get("id")
            if not isinstance(nid, str):
                continue
            signature = node.get("signature") or ""
            classification = node.get("classification") or ""
            heads = node.get("heads", []) or []
            heads_text = _short_heads([h for h in heads if isinstance(h, str)], args.max_heads)
            label_lines = [nid]
            if signature:
                label_lines.append(str(signature))
            if classification:
                label_lines.append(f"class: {classification}")
            if heads_text:
                label_lines.append(f"heads: {heads_text}")
            shape = "ellipse" if nid in sinks else "box"
            g.add_node(nid, label="\n".join(label_lines), shape=shape, fontname="Helvetica")

        for node in nodes:
            src = node.get("id")
            if not isinstance(src, str):
                continue
            for d in node.get("downstreams", []) or []:
                if not isinstance(d, dict):
                    continue
                dst = d.get("target")
                if not isinstance(dst, str):
                    continue
                count = d.get("count", None)
                if count is None:
                    g.add_edge(src, dst)
                else:
                    g.add_edge(src, dst, label=str(count))
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        g.draw(str(args.output_png), prog=args.layout)
        print(f"✅ wrote PNG: {args.output_png}")


if __name__ == "__main__":
    main()
