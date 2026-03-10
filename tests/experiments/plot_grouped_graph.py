#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据路径插补与模式分组重绘聚合图。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于分组信息重绘聚合拓扑图")
    parser.add_argument("--graph-json", type=Path, required=True, help="原始或筛选后的图 JSON")
    parser.add_argument("--path-patch-json", type=Path, required=True, help="path_patch_thr75.json")
    parser.add_argument("--output-json", type=Path, required=True, help="聚合分组及连边 JSON")
    parser.add_argument("--output-png", type=Path, help="可选，输出 PNG 渲染路径")
    parser.add_argument("--near-threshold", type=float, default=0.1, help="绝对 logit diff 小于该值视为 near_zero")
    parser.add_argument("--layer-gap", type=int, default=4, help="组内层差达到该值时拆分子组")
    parser.add_argument(
        "--layer-buckets",
        type=str,
        default="",
        help="可选：按 layer 段细分组，格式如 '0-4,5-8,9-11'。优先于 --layer-gap",
    )
    parser.add_argument("--edge-weights-json", type=Path, help="可选，提供 edge 权重 JSON 以便按阈值裁剪")
    parser.add_argument(
        "--edge-score-key",
        type=str,
        default="delta_logit_diff",
        help="在 edge 权重 JSON 中用于取值的字段",
    )
    parser.add_argument(
        "--min-edge-abs-score",
        type=float,
        default=0.0,
        help="当提供 edge 权重时，仅保留绝对值不低于该阈值的边",
    )
    parser.add_argument(
        "--drop-head",
        action="append",
        default=[],
        help="可选，指定要从结果中移除的 head（格式如 5.9）；可重复使用",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError(f"{path} 不是期望的 JSON dict")
    return data


def map_head_name(node: str) -> Optional[str]:
    if isinstance(node, str) and node.startswith("a") and ".h" in node:
        left, right = node.split(".h", 1)
        try:
            layer = int(left[1:])
            head = int(right)
        except ValueError:
            return None
        return f"{layer}.{head}"
    if node == "input":
        return "input"
    if node == "logits":
        return "logits"
    return None


def head_layer(head: str) -> int:
    layer, _ = head.split(".")
    return int(layer)


def classify_head(delta: Optional[float], threshold: float) -> str:
    if delta is None:
        return "unknown"
    if abs(delta) < threshold:
        return "near_zero"
    return "positive" if delta > 0 else "negative"


def split_by_layer(heads: List[str], gap: int) -> List[List[str]]:
    if not heads:
        return []
    sorted_heads = sorted(heads, key=head_layer)
    segments: List[List[str]] = [[sorted_heads[0]]]
    for head in sorted_heads[1:]:
        prev_layer = head_layer(segments[-1][-1])
        current_layer = head_layer(head)
        if current_layer - prev_layer >= gap:
            segments.append([head])
        else:
            segments[-1].append(head)
    return segments


def parse_layer_buckets(spec: str) -> List[Tuple[int, int, str]]:
    if not spec:
        return []
    buckets: List[Tuple[int, int, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"layer-buckets 格式错误: {chunk} (期望 'lo-hi')")
        lo_s, hi_s = chunk.split("-", 1)
        lo = int(lo_s.strip())
        hi = int(hi_s.strip())
        label = f"{lo}-{hi}"
        buckets.append((lo, hi, label))
    return buckets


def bucket_for_layer(layer: int, buckets: List[Tuple[int, int, str]]) -> Optional[str]:
    for lo, hi, label in buckets:
        if lo <= layer <= hi:
            return label
    return None

def load_edge_thresholds(
    weights_path: Optional[Path],
    score_key: str,
    min_abs_score: float,
) -> Optional[Set[Tuple[str, str]]]:
    if not weights_path:
        return None
    if not weights_path.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(f"未找到 edge 权重文件: {weights_path}")

    payload = json.loads(weights_path.read_text())
    records = payload.get("edges", [])
    if not isinstance(records, list):
        raise ValueError(f"{weights_path} 缺少 edges 列表")

    max_scores: Dict[Tuple[str, str], float] = {}
    for entry in records:
        score_val = entry.get(score_key)
        if score_val is None:
            continue
        src = map_head_name(entry.get("src"))
        dst_raw = entry.get("dst")
        dst = map_head_name(dst_raw if dst_raw is not None else "")
        if not src or not dst:
            continue
        key = (src, dst)
        current = max_scores.get(key)
        score_abs = abs(float(score_val))
        if current is None or score_abs > current:
            max_scores[key] = score_abs

    if min_abs_score <= 0:
        return set(max_scores.keys())
    allowed = {edge for edge, val in max_scores.items() if val >= min_abs_score}
    return allowed


def main() -> None:
    args = parse_args()

    graph = load_json(args.graph_json)
    path_patch = load_json(args.path_patch_json)

    allowed_edges = load_edge_thresholds(
        args.edge_weights_json, args.edge_score_key, args.min_edge_abs_score
    )

    drop_heads: Set[str] = {head.strip() for head in (args.drop_head or []) if head}

    metrics: Dict[str, Dict[str, Any]] = path_patch.get("metrics", {})
    metrics = {head: stats for head, stats in metrics.items() if head not in drop_heads}
    classification = path_patch.get("classification", {})
    classification = {
        label: [head for head in heads if head not in drop_heads]
        for label, heads in classification.items()
    }
    pattern_groups = path_patch.get("pattern_groups", {})
    pattern_groups = {
        signature: [head for head in heads if head not in drop_heads]
        for signature, heads in pattern_groups.items()
    }

    # 构建签名查找表
    head_to_signature: Dict[str, str] = {}
    for signature, heads in pattern_groups.items():
        for head in heads:
            head_to_signature[head] = signature

    # 原始连边映射到 head 名称
    raw_edges: List[Tuple[str, str]] = []
    for src_node, dst_node in graph.get("edge_list", []):
        src_head = map_head_name(src_node)
        dst_head = map_head_name(dst_node)
        if not src_head or not dst_head:
            continue
        if src_head in drop_heads or dst_head in drop_heads:
            continue
        # input 是特殊源节点：不在 metrics 里，但需要保留 input->head 的边
        if src_head != "input" and src_head not in metrics:
            continue
        if dst_head not in {"logits", "input"} and dst_head not in metrics:
            continue
        edge_key = (src_head, dst_head)
        if allowed_edges is not None and edge_key not in allowed_edges:
            continue
        raw_edges.append((src_head, dst_head))

    outgoing_heads: Set[str] = set()
    candidate_edges: List[Tuple[str, str]] = []
    logits_edges: List[str] = []
    input_edges: List[str] = []
    for src, dst in raw_edges:
        if dst == "logits":
            outgoing_heads.add(src)
            logits_edges.append(src)
        elif src == "input":
            input_edges.append(dst)
        else:
            outgoing_heads.add(src)
            candidate_edges.append((src, dst))

    filtered_metrics = {head: metrics[head] for head in outgoing_heads if head in metrics}

    filtered_edges: List[Tuple[str, str]] = [
        (src, dst)
        for src, dst in candidate_edges
        if src in filtered_metrics and dst in filtered_metrics
    ]

    class_override: Dict[str, str] = {}
    for head, stats in filtered_metrics.items():
        delta = stats.get("delta_logit_diff")
        label = classify_head(delta, args.near_threshold)
        class_override[head] = label

    grouped: Dict[str, Dict[str, Any]] = {}
    head_to_group: Dict[str, str] = {}
    groups_list: List[Dict[str, Any]] = []

    class_map = {head: label for label, heads in classification.items() for head in heads}

    for head, stats in filtered_metrics.items():
        signature = head_to_signature.get(head, "UNASSIGNED")
        base_label = class_override.get(head, class_map.get(head, "unknown"))
        layers = head_layer(head)
        key = (signature, base_label)
        grouped.setdefault(key, {"heads": []})["heads"].append(head)

    group_items: List[Dict[str, Any]] = []
    buckets = parse_layer_buckets(args.layer_buckets)

    for (signature, label), info in grouped.items():
        if buckets:
            bucketed: Dict[str, List[str]] = defaultdict(list)
            for head in info["heads"]:
                layer = head_layer(head)
                bucket = bucket_for_layer(layer, buckets)
                if bucket is None:
                    continue
                bucketed[bucket].append(head)
            segments = []
            for bucket_label in [b[2] for b in buckets]:
                heads = bucketed.get(bucket_label, [])
                if heads:
                    segments.append((bucket_label, heads))
        else:
            segments = [(str(i + 1), seg) for i, seg in enumerate(split_by_layer(info["heads"], args.layer_gap))]

        for seg_label, segment in segments:
            segment_id = f"{signature}|{label}|{seg_label}"
            segment_layers = [head_layer(h) for h in segment]
            metrics_vals = [filtered_metrics[h].get("metric") for h in segment if filtered_metrics[h].get("metric") is not None]
            delta_vals = [filtered_metrics[h].get("delta_logit_diff") for h in segment if filtered_metrics[h].get("delta_logit_diff") is not None]
            payload = {
                "id": segment_id,
                "signature": signature,
                "classification": label,
                "heads": sorted(segment, key=lambda name: tuple(int(x) for x in name.split("."))),
                "layers": segment_layers,
                "summary": {
                    "count": len(segment),
                    "mean_metric": mean(metrics_vals) if metrics_vals else None,
                    "mean_delta_logit_diff": mean(delta_vals) if delta_vals else None,
                    "min_layer": min(segment_layers),
                    "max_layer": max(segment_layers),
                },
            }
            group_items.append(payload)
            for head in segment:
                head_to_group[head] = payload["id"]

    # 聚合连边
    group_edges: Dict[Tuple[str, str], int] = defaultdict(int)
    logits_edge_counts: Dict[str, int] = defaultdict(int)
    input_edge_counts: Dict[str, int] = defaultdict(int)
    for src, dst in filtered_edges:
        src_group = head_to_group.get(src)
        dst_group = head_to_group.get(dst)
        if not src_group or not dst_group or src_group == dst_group:
            continue
        group_edges[(src_group, dst_group)] += 1

    for src in logits_edges:
        if src == "input":
            continue
        src_group = head_to_group.get(src)
        if not src_group:
            continue
        logits_edge_counts[src_group] += 1

    for dst in input_edges:
        dst_group = head_to_group.get(dst)
        if not dst_group:
            continue
        input_edge_counts[dst_group] += 1

    active_ids: Set[str] = {group["id"] for group in group_items}

    final_groups = [g for g in group_items if g["id"] in active_ids]
    final_edges = [
        {"source": src, "target": dst, "count": count}
        for (src, dst), count in group_edges.items()
        if src in active_ids and dst in active_ids
    ]

    for dst_group, count in input_edge_counts.items():
        if dst_group in active_ids:
            final_edges.append({"source": "input", "target": dst_group, "count": count})

    for src, count in logits_edge_counts.items():
        if src in active_ids:
            final_edges.append({"source": src, "target": "logits", "count": count})

    head_payload = {}
    for head, stats in filtered_metrics.items():
        group_id = head_to_group.get(head)
        if group_id not in active_ids:
            continue
        head_payload[head] = {
            "group": group_id,
            "signature": head_to_signature.get(head, "UNASSIGNED"),
            "classification": class_override.get(head, class_map.get(head, "unknown")),
            "layer": head_layer(head),
            "delta_logit_diff": stats.get("delta_logit_diff"),
            "metric": stats.get("metric"),
        }

    sink_nodes = []
    if logits_edge_counts:
        sink_nodes.append({"id": "logits", "label": "logits"})

    source_nodes = []
    if input_edge_counts:
        source_nodes.append({"id": "input", "label": "input"})

    payload = {
        "meta": {
            "graph_json": str(args.graph_json),
            "path_patch_json": str(args.path_patch_json),
            "near_threshold": args.near_threshold,
            "layer_gap": args.layer_gap,
        },
        "groups": final_groups,
        "edges": final_edges,
        "heads": head_payload,
        "sources": source_nodes,
        "sinks": sink_nodes,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"写入聚合分组 JSON: {args.output_json}")

    if args.output_png:
        draw_group_graph(payload, args.output_png)


def draw_group_graph(data: Dict[str, Any], output: Path) -> None:
    try:
        import pygraphviz as pgv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未安装 pygraphviz，无法渲染 PNG；请在包含 pygraphviz 的环境中运行，或仅生成 --output-json。"
        ) from exc

    color_map = {
        "positive": "#2E7D32",
        "negative": "#C62828",
        "near_zero": "#546E7A",
        "unknown": "#9E9E9E",
        "sink": "#B0BEC5",
        "source": "#B0BEC5",
    }
    graph = pgv.AGraph(directed=True, strict=False, splines="spline", overlap="false", layout="dot")

    for source in data.get("sources", []):
        graph.add_node(
            source.get("id"),
            label=source.get("label", source.get("id", "")),
            shape="ellipse",
            style="filled",
            fillcolor=color_map.get("source"),
            fontname="Helvetica",
        )

    group_members = data.get("group_members", {}) or {}
    for group in data.get("groups", []):
        label_lines = [group.get("id", ""), group.get("signature", ""), f"class: {group.get('classification')}" ]
        heads = group.get("heads")
        if heads is None:
            heads = group_members.get(group.get("id", ""), [])
        label_lines.append(
            f"heads: {', '.join(heads)}"
        )
        summary = group.get("summary", {})
        delta_val = summary.get("mean_delta_logit_diff")
        if delta_val is not None:
            label_lines.append(f"Δlogit: {delta_val:+.2f}")
        else:
            label_lines.append("Δlogit: n/a")
        fill = color_map.get(group.get("classification"), "#BDBDBD")
        graph.add_node(
            group.get("id"),
            label="\n".join(label_lines),
            shape="box",
            style="filled,rounded",
            fillcolor=fill,
            fontname="Helvetica",
        )

    for sink in data.get("sinks", []):
        graph.add_node(
            sink.get("id"),
            label=sink.get("label", sink.get("id", "")),
            shape="ellipse",
            style="filled",
            fillcolor=color_map.get("sink"),
            fontname="Helvetica",
        )

    for edge in data.get("edges", []):
        src = edge.get("source") or edge.get("src")
        dst = edge.get("target") or edge.get("dst")
        if not src or not dst:
            continue
        graph.add_edge(src, dst, label=str(edge.get("count", 1)))

    output.parent.mkdir(parents=True, exist_ok=True)
    graph.draw(str(output), prog="dot")
    print(f"写入聚合图: {output}")


if __name__ == "__main__":
    main()
