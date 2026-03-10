#!/usr/bin/env python3
"""
Render the grouped IOI head graph with custom annotations and colored edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pygraphviz as pgv


ANNOTATION_SPECS = [
    {
        "heads": ["9.6", "9.9", "10.0"],
        "note": "come from logits",
        "target": "logits",
    },
    {
        "heads": ["10.7", "11.10"],
        "note": "come from logits",
        "target": "logits",
    },
    {
        "heads": ["10.6", "10.10"],
        "note": "ineffective explanation",
        "status": "ineffective",
    },
    {
        "heads": ["7.9", "8.6", "8.10"],
        "note": "come from [9.6 9.9 10.0]",
        "target_heads": ["9.6", "9.9", "10.0"],
    },
    {
        "heads": ["5.5"],
        "note": "come from [9.6 9.9 10.0]",
        "target_heads": ["9.6", "9.9", "10.0"],
    },
    {
        "heads": ["6.9"],
        "note": "ineffective explanation",
        "status": "ineffective",
    },
    {
        "heads": ["0.1", "0.10", "3.0"],
        "note": "come from [7.9 8.6 8.10] -> [9.6 9.9 10.0]",
        "target_chain": [
            ["7.9", "8.6", "8.10"],
            ["9.6", "9.9", "10.0"],
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render annotated group graph.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tests/results/group_graph_top60_thr75_logits_pruned_thr0001.json"),
        help="Path to grouped graph JSON (e.g., *_pruned_thr0001.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ioi/hypothesis/Group_Summary/annotated_graph.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def build_annotations(data: Dict) -> Dict[str, Dict]:
    head_to_group = {head: payload["group"] for head, payload in data.get("heads", {}).items()}
    annotations: Dict[str, Dict] = {}
    for spec in ANNOTATION_SPECS:
        group_id = None
        for head in spec["heads"]:
            group_id = head_to_group.get(head)
            if group_id:
                break
        if not group_id:
            continue
        entry: Dict[str, Any] = {
            "note": spec.get("note"),
            "status": spec.get("status"),
        }
        target = spec.get("target")
        chain = spec.get("target_chain")
        if target:
            entry["target"] = target
        elif chain:
            # Resolve first target in chain and create edges later
            resolved_chain: List[str] = []
            for hop in chain:
                hop_target = None
                for tgt_head in hop:
                    tgt_group = head_to_group.get(tgt_head)
                    if tgt_group:
                        hop_target = tgt_group
                        break
                if hop_target:
                    resolved_chain.append(hop_target)
            if resolved_chain:
                entry["target_chain"] = resolved_chain
        else:
            target_heads = spec.get("target_heads", [])
            for tgt_head in target_heads:
                tgt_group = head_to_group.get(tgt_head)
                if tgt_group:
                    entry["target"] = tgt_group
                    break
        annotations[group_id] = entry
    return annotations


def render_graph(data: Dict, annotations: Dict[str, Dict], output: Path) -> None:
    color_map = {
        "positive": "#2E7D32",
        "negative": "#C62828",
        "near_zero": "#546E7A",
        "unknown": "#9E9E9E",
    }
    graph = pgv.AGraph(directed=True, strict=False, splines="spline", overlap="false", layout="dot")

    for group in data.get("nodes", data.get("groups", [])):
        group_id = group.get("id")
        classification = group.get("classification")
        label_lines = [group_id, group.get("signature", ""), f"class: {classification}"]
        label_lines.append(f"heads: {', '.join(group.get('heads', []))}")
        summary = group.get("summary", {})
        delta_val = summary.get("mean_delta_logit_diff")
        if delta_val is not None:
            label_lines.append(f"Δlogit: {delta_val:+.2f}")
        note = None
        fill = color_map.get(classification, "#BDBDBD")
        if group_id in annotations:
            note = annotations[group_id].get("note")
            status = annotations[group_id].get("status")
            if note:
                label_lines.append(f"note: {note}")
            if status == "ineffective":
                fill = "#B0BEC5"
        graph.add_node(
            group_id,
            label="\n".join(label_lines),
            shape="box",
            style="filled,rounded",
            fillcolor=fill,
            fontname="Helvetica",
        )

    graph.add_node(
        "logits",
        label="logits",
        shape="ellipse",
        style="filled",
        fillcolor="#B0BEC5",
    )

    edges = data.get("edges", [])
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], label=str(edge.get("count", 1)))

    for group_id, ann in annotations.items():
        if "target_chain" in ann:
            current = group_id
            for target in ann["target_chain"]:
                if target not in graph.nodes():
                    graph.add_node(target, label=target, shape="ellipse", style="dashed")
                graph.add_edge(
                    current,
                    target,
                    color="#1E88E5",
                    style="dashed",
                    penwidth=2,
                    label="comes from",
                )
                current = target
            continue
        target = ann.get("target")
        if target:
            if target not in graph.nodes():
                graph.add_node(target, label=target, shape="ellipse", style="dashed")
            graph.add_edge(
                group_id,
                target,
                color="#1E88E5",
                style="dashed",
                penwidth=2,
                label="comes from",
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    graph.draw(str(output), prog="dot")
    print(f"✅ Annotated graph saved to {output}")


def main() -> None:
    args = parse_args()
    data = json.loads(args.config.read_text())
    annotations = build_annotations(data)
    render_graph(data, annotations, args.output)


if __name__ == "__main__":
    main()
