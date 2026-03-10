from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from graph_collapse import render_collapsed_graph  # 可选
except ImportError:
    render_collapsed_graph = None


def load_relevant_edges(path: Path) -> Dict[str, dict]:
    """
    只加载：
      1. 联合 qkv 边：joint==true 且 receiver_input=='qkv'
      2. logits 边：receiver_kind=='logits'
    其它全部忽略。
    返回 dict[edge_name] = record
    """
    data = json.loads(path.read_text())
    edges = data.get("edges")
    if not isinstance(edges, list):
        raise ValueError(f"{path} 不包含 'edges' 列表。")

    kept: Dict[str, dict] = {}
    for e in edges:
        rk = e.get("receiver_kind")
        ri = e.get("receiver_input")
        joint = e.get("joint", False)
        name = e.get("edge")
        if not name:
            # 尝试构造（对 logits 之外的需要 receiver_input）
            src = e.get("src")
            dst = e.get("dst")
            if rk == "logits":
                if src and dst:
                    name = f"{src}->{dst}"
                else:
                    continue
            else:
                if src and dst and ri:
                    name = f"{src}->{dst}<{ri}>"
                else:
                    continue

        if rk == "logits":
            kept[name] = e
        elif joint and ri == "qkv":
            kept[name] = e
        # else: 忽略单通道 / 其它
    return kept


def rank_and_pick(
    edge_weights: Dict[str, dict],
    percent: float,
    min_edges: int,
    score_key: str,
    use_abs: bool,
) -> List[str]:
    if not edge_weights:
        return []
    scored: List[Tuple[float, str]] = []
    for name, meta in edge_weights.items():
        if score_key not in meta:
            continue
        val = float(meta[score_key])
        scored.append(((abs(val) if use_abs else val), name))
    if not scored:
        return []
    scored.sort(reverse=True)
    keep_n = max(min_edges, math.ceil(len(scored) * percent / 100.0))
    keep_n = min(keep_n, len(scored))
    return [name for _, name in scored[:keep_n]]


def base_pair(edge_name: str) -> Tuple[str, str]:
    # 统一去掉 <qkv> 后缀，logits 无后缀
    if "<" in edge_name:
        left, right = edge_name.split("->", 1)
        dst_part = right.split("<", 1)[0]
        return left, dst_part
    else:
        # logits 或无法匹配的
        left, dst = edge_name.split("->", 1)
        return left, dst


def rebuild_from_selected(
    selected_edge_names: Iterable[str],
) -> Tuple[List[str], List[List[int]], List[Tuple[str, str]]]:
    base_edges_set = set()
    nodes_set = set()
    for name in selected_edge_names:
        src, dst = base_pair(name)
        base_edges_set.add((src, dst))
        nodes_set.add(src)
        nodes_set.add(dst)
    node_names = sorted(nodes_set)
    idx = {n: i for i, n in enumerate(node_names)}
    matrix = [[0] * len(node_names) for _ in node_names]
    for s, d in base_edges_set:
        i, j = idx[s], idx[d]
        matrix[i][j] = 1
    base_edges = sorted(base_edges_set)
    return node_names, matrix, base_edges


def save_outputs(
    prefix: Path,
    node_names: List[str],
    matrix: List[List[int]],
    base_edges: List[Tuple[str, str]],
    edge_weights: Dict[str, dict],
    kept_edge_names: List[str],
    stats: Dict[str, float],
    render: bool,
    layout: str,
):
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # 输出简化图 JSON
    graph_json = {
        "meta": stats,
        "nodes": node_names,
        "adjacency_matrix": matrix,
        "edge_list": base_edges,
    }
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(graph_json, indent=2, ensure_ascii=False))

    # 邻接矩阵 CSV
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node"] + node_names)
        for n, row in zip(node_names, matrix):
            w.writerow([n] + row)

    # base 边 CSV
    edges_csv = prefix.with_name(prefix.name + "_edges.csv")
    with edges_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "dst"])
        w.writerows(base_edges)

    # 筛选后的原始条目
    weights_json = prefix.with_name(prefix.name + "_edge_weights.json")
    weights_payload = {
        "meta": {
            **stats,
            "selected_edge_count": len(kept_edge_names),
        },
        "edges": [edge_weights[n] for n in kept_edge_names],
    }
    weights_json.write_text(json.dumps(weights_payload, indent=2, ensure_ascii=False))

    if render:
        if render_collapsed_graph is None:
            print("提示：未安装 graph_collapse.render_collapsed_graph，跳过绘图。")
        else:
            png_path = prefix.with_suffix(".png")
            # 使用 base_edges 画图
            render_collapsed_graph(node_names, base_edges, png_path, layout=layout)
            print(f"生成图像: {png_path}")

    print(f"[统计] 可用边(联合qkv+logits)总数: {stats['total_original_edges']}  保留: {stats['kept_edges']} ({stats['kept_percent']:.2f}%)")
    print(f"[统计] 节点总数: {stats['total_nodes']}  非孤立节点: {stats['non_isolated_nodes']}")
    print(f"输出: {json_path}")
    print(f"输出: {csv_path}")
    print(f"输出: {edges_csv}")
    print(f"输出: {weights_json}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="仅基于 联合qkv 边 + logits 边 筛选前 X%（按 |delta_logit_diff| 默认）并重建子图。"
    )
    p.add_argument("--edge-weights", type=Path, required=True, help="edge_path_patching 生成的 joint_p.json")
    p.add_argument("--output-prefix", type=Path, required=True, help="输出文件前缀（不含扩展名）")
    p.add_argument("--top-percent", type=float, required=True, help="保留百分比 (0,100]")
    p.add_argument("--min-edges", type=int, default=1, help="至少保留的边数")
    p.add_argument("--score-key", type=str, default="delta_logit_diff", help="排序字段 (默认 delta_logit_diff)")
    p.add_argument("--no-abs-score", action="store_true", help="不取绝对值排序（默认按绝对值）")
    p.add_argument("--no-render", action="store_true", help="不尝试生成 png")
    p.add_argument("--layout", type=str, default="dot", help="pygraphviz 布局 (dot/neato/...)")
    return p


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv).parse_args(argv)

    if not (0 < args.top_percent <= 100):
        raise ValueError("--top-percent 必须在 (0,100]")

    edge_weights = load_relevant_edges(args.edge_weights)
    if not edge_weights:
        raise ValueError("没有找到任何 联合qkv 或 logits 边。请确认 joint_path 输出包含这些条目。")

    kept_names = rank_and_pick(
        edge_weights=edge_weights,
        percent=args.top_percent,
        min_edges=args.min_edges,
        score_key=args.score_key,
        use_abs=not args.no_abs_score,
    )
    node_names, matrix, base_edges = rebuild_from_selected(kept_names)

    incident_nodes = set()
    for s, d in base_edges:
        incident_nodes.add(s); incident_nodes.add(d)

    stats = {
        "top_percent_param": args.top_percent,
        "min_edges_param": args.min_edges,
        "score_key": args.score_key,
        "use_abs_score": (not args.no_abs_score),
        "total_original_edges": len(edge_weights),
        "kept_edges": len(base_edges),
        "kept_percent": (len(base_edges) / len(edge_weights) * 100.0) if edge_weights else 0.0,
        "total_nodes": len(node_names),
        "non_isolated_nodes": len(incident_nodes),
        "filter_mode": "joint_qkv_plus_logits",
    }

    save_outputs(
        prefix=args.output_prefix,
        node_names=node_names,
        matrix=matrix,
        base_edges=base_edges,
        edge_weights=edge_weights,
        kept_edge_names=kept_names,
        stats=stats,
        render=not args.no_render,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()