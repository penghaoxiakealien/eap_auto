#!/usr/bin/env python3
"""Prune weak group edges using batch attention summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO_ROOT / "tests/results/group_graph_top60_thr75_logits_filtered.json"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "tests/results/batch_attention"
DEFAULT_OUTPUT = REPO_ROOT / "tests/results/group_graph_top60_thr75_logits_pruned.json"
DEFAULT_SEED_GROUPS = (
    "END-IO|positive|1",
    "END-IO|negative|1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove edges whose attention deltas fall below a threshold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Only report edges to drop")
    parser.add_argument(
        "--seed-group",
        action="append",
        dest="seed_groups",
        help="Group IDs to backtrace from when trimming disconnected nodes",
    )
    return parser.parse_args()


def sanitize_label(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return ascii_text.lower() or "group"


def build_sanitized_map(groups: Iterable[dict]) -> Dict[str, str]:
    return {sanitize_label(group["id"]): group["id"] for group in groups}


def compute_avg_delta(scenario: dict) -> float:
    deltas: List[float] = []
    for sent in scenario.get("per_sentence", []):
        for item in sent.get("top_increases", []):
            deltas.append(abs(float(item.get("delta", 0.0))))
        for item in sent.get("top_decreases", []):
            deltas.append(abs(float(item.get("delta", 0.0))))
    if not deltas:
        return 0.0
    return mean(deltas)


def backtrace_groups(edges: Iterable[dict], seed_groups: Iterable[str]) -> Set[str]:
    incoming: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        src = edge.get("source")
        dst = edge.get("target")
        if isinstance(src, str) and isinstance(dst, str):
            incoming[dst].append(src)

    visited: List[str] = []
    queue: deque[str] = deque([group for group in seed_groups if group and group != "logits"])

    while queue:
        target = queue.popleft()
        if target in visited:
            continue
        visited.append(target)
        for source in incoming.get(target, []):
            if source == "logits":
                continue
            if source not in visited:
                queue.append(source)

    return set(visited)


def main() -> None:
    args = parse_args()
    seed_groups = args.seed_groups or list(DEFAULT_SEED_GROUPS)

    graph_path = args.graph_json if args.graph_json.is_absolute() else REPO_ROOT / args.graph_json
    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else REPO_ROOT / args.summary_dir
    output_path = args.output_json if args.output_json.is_absolute() else REPO_ROOT / args.output_json

    graph = json.loads(graph_path.read_text())
    groups: List[dict] = graph.get("groups", [])
    edges: List[dict] = graph.get("edges", [])

    summary_files = sorted(summary_dir.glob("summary_*.json"))
    print(f"Found {len(summary_files)} summary files in {summary_dir}")

    sanitized_to_group = build_sanitized_map(groups)
    head_to_group: Dict[str, str] = {}
    for group in groups:
        gid = group["id"]
        for head in group.get("heads", []):
            head_to_group[head] = gid

    edges_to_drop: Set[Tuple[str, str]] = set()

    scenario_total = 0
    for summary_path in summary_files:
        data = json.loads(summary_path.read_text())
        for scenario in data.get("scenarios", []):
            avg_delta = compute_avg_delta(scenario)
            if math.isnan(avg_delta):
                continue
            label = scenario.get("label", "")
            if "__to__" not in label:
                continue
            src_sanitized, _, head_token = label.partition("__to__")
            src_group = sanitized_to_group.get(src_sanitized)
            dst_group = head_to_group.get(head_token.replace("_", "."))
            if not src_group or not dst_group:
                continue
            if avg_delta < args.threshold:
                edges_to_drop.add((src_group, dst_group))
            scenario_total += 1

    debug_payload = {
        "threshold": args.threshold,
        "summary_files": [str(path) for path in summary_files],
        "scenarios_processed": scenario_total,
        "edges_to_drop": sorted({f"{src} -> {dst}" for src, dst in edges_to_drop}),
    }

    if not edges_to_drop:
        debug_payload["message"] = "No edges below threshold"
    else:
        debug_payload["message"] = "Edges pruned"

    debug_path = REPO_ROOT / "tests/results/prune_debug.json"
    debug_path.write_text(json.dumps(debug_payload, indent=2))

    if args.dry_run:
        return

    pruned_edges = [edge for edge in edges if (edge.get("source"), edge.get("target")) not in edges_to_drop]

    reachable = backtrace_groups(pruned_edges, seed_groups)
    if not reachable:
        raise ValueError("No groups reachable from seed groups; check graph/topology")

    reachable_edges: List[dict] = []
    for edge in pruned_edges:
        src = edge.get("source")
        dst = edge.get("target")
        if dst == "logits" and isinstance(src, str) and src in reachable:
            reachable_edges.append(edge)
            continue
        if isinstance(src, str) and isinstance(dst, str) and src in reachable and dst in reachable:
            reachable_edges.append(edge)

    graph["edges"] = reachable_edges
    graph["groups"] = [group for group in groups if group.get("id") in reachable]

    heads_section = graph.get("heads") or {}
    filtered_heads = {
        head: info
        for head, info in heads_section.items()
        if isinstance(info, dict) and info.get("group") in reachable
    }
    graph["heads"] = filtered_heads

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2))
    print(f"Pruned graph written to {output_path}")


if __name__ == "__main__":
    main()
