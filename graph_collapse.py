#!/usr/bin/env python3
"""Collapse an EAP graph to head-only adjacency data."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pygraphviz as pgv


def node_type(name: str) -> str:
    if name == "input":
        return "input"
    if name == "logits":
        return "logits"
    if name.startswith("a"):
        return "attention"
    if name.startswith("m"):
        return "mlp"
    return "other"


def parse_attention(name: str) -> Tuple[int, int]:
    layer_str, head_str = name.split(".")
    return int(layer_str[1:]), int(head_str[1:])


def sort_key(name: str) -> Tuple[int, int, int]:
    kind = node_type(name)
    if kind == "input":
        return (-1, -1, -1)
    if kind == "attention":
        layer, head = parse_attention(name)
        return (layer, head, 0)
    if kind == "logits":
        return (10**6, 0, 0)
    return (10**5, 0, 0)


def load_graph(path: Path) -> Dict:
    with path.open("r") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "graph_data" in data:
        data = data["graph_data"]
    return data


def build_outgoing(edges: Dict[str, Dict], active_nodes: Set[str]) -> Dict[str, Set[str]]:
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges.values():
        if not edge.get("in_graph", False):
            continue
        parent = edge["parent"]
        child = edge["child"]
        if parent not in active_nodes or child not in active_nodes:
            continue
        outgoing[parent].add(child)
    return outgoing


def collapse_connections(
    start: str,
    outgoing: Dict[str, Set[str]],
    active_nodes: Set[str],
    keep_nodes: Set[str],
) -> List[str]:
    targets: Set[str] = set()
    stack: List[str] = list(outgoing.get(start, ()))
    visited: Set[str] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node not in active_nodes or node == start:
            continue
        if node in keep_nodes:
            targets.add(node)
            continue
        if node_type(node) == "mlp":
            stack.extend(outgoing.get(node, ()))
    return sorted(targets, key=sort_key)


def collapse_graph(
    data: Dict,
    include_input: bool,
    include_logits: bool,
) -> Tuple[List[str], Dict[str, List[str]]]:
    if "nodes" not in data or "edges" not in data:
        raise KeyError("graph json 缺少 nodes/edges（请确认输出包含 edges）")
    nodes = data["nodes"]
    edges = data["edges"]
    active_nodes = {name for name, info in nodes.items() if info.get("in_graph", False)}

    keep_nodes: List[str] = []
    for name in active_nodes:
        kind = node_type(name)
        if kind == "attention":
            keep_nodes.append(name)
        elif include_input and kind == "input":
            keep_nodes.append(name)
        elif include_logits and kind == "logits":
            keep_nodes.append(name)

    keep_nodes.sort(key=sort_key)
    keep_set = set(keep_nodes)
    outgoing = build_outgoing(edges, active_nodes)

    adjacency: Dict[str, List[str]] = {}
    for start in keep_nodes:
        adjacency[start] = collapse_connections(start, outgoing, active_nodes, keep_set)
    return keep_nodes, adjacency


def to_matrix(
    nodes: Sequence[str],
    adjacency: Dict[str, Sequence[str]],
) -> Tuple[List[List[int]], List[Tuple[str, str]]]:
    index = {name: idx for idx, name in enumerate(nodes)}
    matrix = [[0 for _ in nodes] for _ in nodes]
    edges: List[Tuple[str, str]] = []
    for src, targets in adjacency.items():
        src_idx = index[src]
        for dst in targets:
            dst_idx = index[dst]
            if matrix[src_idx][dst_idx] == 0:
                matrix[src_idx][dst_idx] = 1
                edges.append((src, dst))
    edges.sort(key=lambda pair: (index[pair[0]], index[pair[1]]))
    return matrix, edges


def write_outputs(
    prefix: Path,
    nodes: List[str],
    matrix: List[List[int]],
    edges: List[Tuple[str, str]],
    include_input: bool,
    include_logits: bool,
) -> Tuple[Path, Path, Path]:
    payload = {
        "nodes": nodes,
        "adjacency_matrix": matrix,
        "edge_list": edges,
        "include_input": include_input,
        "include_logits": include_logits,
    }

    json_path = prefix.with_suffix(".json")
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node"] + list(nodes))
        for idx, src in enumerate(nodes):
            writer.writerow([src] + matrix[idx])

    edges_path = prefix.with_name(prefix.name + "_edges.csv")
    with edges_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "dst"])
        writer.writerows(edges)

    return json_path, csv_path, edges_path


def default_prefix(graph_path: Path) -> Path:
    stem = graph_path.stem if graph_path.suffix else graph_path.name
    return graph_path.with_name(stem + "_collapsed")


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse an EAP graph JSON into a head-only adjacency matrix.",
    )
    parser.add_argument(
        "graph_json",
        type=Path,
        help="Path to the EAP graph JSON (output of Graph.to_json).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for exported files (defaults to <input>_collapsed).",
    )
    parser.add_argument(
        "--drop-input",
        action="store_true",
        help="Exclude the residual stream input node from the collapsed graph.",
    )
    parser.add_argument(
        "--drop-logits",
        action="store_true",
        help="Exclude the logits node from the collapsed graph.",
    )
    return parser.parse_args(argv)

def render_collapsed_graph(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    png_path: Path,
    layout: str = "dot",
) -> None:
    graph = pgv.AGraph(
        directed=True,
        strict=True,
        splines="true",
        overlap="false",
        layout=layout,
    )
    for node in nodes:
        graph.add_node(
            node,
            shape="box",
            style="rounded,filled",
            fillcolor="#EEEEEE",
            fontname="Helvetica",
        )
    for src, dst in edges:
        graph.add_edge(src, dst)
    graph.draw(png_path, prog=layout)
def main() -> None:
    args = parse_args()
    graph_path = args.graph_json.expanduser().resolve()

    graph = load_graph(graph_path)
    include_input = not args.drop_input
    include_logits = not args.drop_logits

    nodes, adjacency = collapse_graph(graph, include_input, include_logits)
    if not nodes:
        raise SystemExit("No nodes selected after collapsing. Check the input graph.")

    matrix, edges = to_matrix(nodes, adjacency)
    prefix = args.output_prefix.expanduser().resolve() if args.output_prefix else default_prefix(graph_path)

    json_path, csv_path, edges_path = write_outputs(prefix, nodes, matrix, edges, include_input, include_logits)
    png_path = prefix.with_suffix(".png")
    render_collapsed_graph(nodes, edges, png_path)
    print(f"Collapsed nodes: {len(nodes)}")
    print(f"Collapsed edges: {len(edges)}")
    print(f"Wrote adjacency JSON to {json_path}")
    print(f"Wrote adjacency CSV to {csv_path}")
    print(f"Wrote edge list CSV to {edges_path}")
    print(f"Wrote collapsed graph PNG to {png_path}")


if __name__ == "__main__":
    main()
