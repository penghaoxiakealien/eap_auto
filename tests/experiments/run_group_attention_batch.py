#!/usr/bin/env python3
"""Batch rerun attention patching for grouped graph edges.

This script walks the grouped attention graph starting from the retained
logits-adjacent groups and automatically issues all required reruns of
``analyze_target_head_attention.py``. Query rows are inferred from the
group signatures so we respect the diagram annotations (END / S2 / S1 / IO).

Typical usage (after activating the ``eap-ig`` conda env)::

    python tests/experiments/run_group_attention_batch.py \
        --device cuda \
        --model-name /data31/private/wangziran/eap-ig/gpt2

All scenario summaries are written under ``tests/results/batch_attention``.
Each target head gets a consolidated summary plus per-edge JSON exports.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from analyze_target_head_attention import run_analysis  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_JSON = REPO_ROOT / "tests/results/group_graph_top60_thr75_logits_filtered.json"
DEFAULT_INPUT_FILE = REPO_ROOT / "results/ioi/path_patching/structured_sentences_standard.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests/results/batch_attention"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch rerun analyze_target_head_attention for grouped edges",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=DEFAULT_GRAPH_JSON,
        help="Grouped graph JSON produced by plot_grouped_graph.py",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Structured IOI sentences JSONL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HookedTransformer model name or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to analyze_target_head_attention (cuda/cpu)",
    )
    parser.add_argument(
        "--receiver-input",
        type=str,
        default="all",
        help="Receiver stream(s) to patch (q,k,v or all)",
    )
    parser.add_argument(
        "--require-same-length",
        action="store_true",
        dest="require_same_length",
        help="Filter prompts to the modal tokenized length (default)",
    )
    parser.add_argument(
        "--no-length-filter",
        action="store_false",
        dest="require_same_length",
        help="Allow prompts with differing tokenized lengths",
    )
    parser.set_defaults(require_same_length=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k tokens to report in mean increases/decreases",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional sentence limit for quick sanity runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store summary JSON files",
    )
    parser.add_argument(
        "--target-groups",
        nargs="+",
        default=["END-IO|positive|1", "END-IO|negative|1"],
        help="Graph group IDs that feed logits and should seed the backtrace",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned runs without executing them",
    )
    return parser.parse_args()


def sanitize_label(text: str) -> str:
    """Produce a lowercase ASCII label safe for file paths."""

    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text)
    ascii_text = ascii_text.strip("_")
    return ascii_text.lower() or "group"


def head_key(head: str) -> Tuple[int, int]:
    layer, h = head.split(".")
    return int(layer), int(h)


def group_query_spec(signature: str) -> str:
    """Infer the diagram query row (end/s2/s1/io) from a group signature."""

    raw = signature.split(":")[-1].strip()
    if not raw:
        return "end"
    base = raw.split()[0]  # drop trailing shift info like "-1"
    prefix = base.split("-", 1)[0].upper()
    mapping = {
        "END": "end",
        "S2": "s2",
        "S1": "s1",
        "IO": "io",
    }
    return mapping.get(prefix, "end")


def is_name_like(token: str) -> bool:
    if not token:
        return False
    stripped = token.strip()
    if not stripped:
        return False
    if stripped == "<|endoftext|>":
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 2:
        return False
    head = stripped[0]
    tail = stripped[1:]
    return head.isupper() and tail.lower() == tail


def shannon_entropy(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log(prob, 2)
    return entropy


def compute_token_stats(entries: List[dict], query_index: int | None) -> Dict[str, object]:
    tokens = [entry.get("token") for entry in entries if entry.get("token")]
    total = len(tokens)
    counter: Counter[str] = Counter(tokens)
    top_token = counter.most_common(1)[0][0] if counter else None
    top_count = counter[top_token] if top_token else 0
    unique = len(counter)
    entropy = shannon_entropy(counter, total)
    norm_entropy = entropy / math.log(unique, 2) if unique > 1 else 0.0
    name_hits = sum(1 for token in tokens if is_name_like(token))
    end_hits = sum(1 for token in tokens if token == "<|endoftext|>")
    query_hits = 0
    for entry in entries:
        pos = entry.get("position")
        if query_index is not None and isinstance(pos, int) and pos == query_index:
            query_hits += 1

    top_preview = [
        {"token": token, "count": count}
        for token, count in counter.most_common(5)
    ]

    return {
        "total": total,
        "unique": unique,
        "top_token": top_token,
        "top_share": (top_count / total) if total else 0.0,
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "name_fraction": (name_hits / total) if total else 0.0,
        "end_fraction": (end_hits / total) if total else 0.0,
        "query_fraction": (query_hits / total) if total else 0.0,
        "top_tokens": top_preview,
    }


def extract_token_entries(scenario: dict, field: str) -> List[dict]:
    entries: List[dict] = []
    for sent in scenario.get("per_sentence", []):
        entries.extend(sent.get(field, []))
    return entries


def annotate_token_concentration(scenario: dict) -> None:
    per_sentence = scenario.get("per_sentence", [])
    query_index = None
    for sent in per_sentence:
        query_pos = sent.get("query_position", {})
        idx = query_pos.get("index")
        if isinstance(idx, int):
            query_index = idx
            break

    increases = extract_token_entries(scenario, "top_increases")
    decreases = extract_token_entries(scenario, "top_decreases")

    scenario["token_concentration"] = {
        "increases": compute_token_stats(increases, query_index),
        "decreases": compute_token_stats(decreases, query_index),
    }


def load_group_graph(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(path.read_text())
    groups = payload.get("groups", [])
    edges = payload.get("edges", [])
    if not groups:
        raise ValueError(f"No groups found in {path}")
    return {"groups": groups, "edges": edges}


def build_backtrace_edges(
    edges: Iterable[Dict[str, object]],
    seed_groups: Sequence[str],
) -> Tuple[Sequence[Tuple[str, str]], Sequence[str]]:
    """Return (source,target) group pairs and visited group IDs via reverse BFS."""

    incoming: Dict[str, List[str]] = defaultdict(list)
    for entry in edges:
        src = entry["source"]
        dst = entry["target"]
        incoming[dst].append(src)

    visited: List[str] = []
    queue: deque[str] = deque(seed_groups)
    edge_pairs: Dict[Tuple[str, str], None] = {}

    while queue:
        target = queue.popleft()
        if target in visited:
            continue
        visited.append(target)
        for source in incoming.get(target, []):
            if source == "logits":
                continue
            edge_pairs[(source, target)] = None
            if source not in visited:
                queue.append(source)

    return list(edge_pairs.keys()), visited


def main() -> None:
    args = parse_args()

    graph_path = resolve_path(Path(args.graph_json))
    input_path = resolve_path(Path(args.input_file))
    output_dir = resolve_path(Path(args.output_dir))

    graph = load_group_graph(graph_path)
    group_map = {group["id"]: group for group in graph["groups"]}
    missing = [gid for gid in args.target_groups if gid not in group_map]
    if missing:
        raise ValueError(f"Target group(s) not found: {missing}")

    edges_to_run, visited_groups = build_backtrace_edges(graph["edges"], args.target_groups)
    if not edges_to_run:
        print("No edges to process. Did you pass the correct target groups?")
        return

    head_to_group: Dict[str, str] = {}
    head_to_query: Dict[str, str] = {}
    for group_id, group in group_map.items():
        signature = group.get("signature", "")
        query_spec = group_query_spec(signature)
        for head in group.get("heads", []):
            head_to_group[head] = group_id
            head_to_query[head] = query_spec

    group_heads: Dict[str, List[str]] = {
        gid: list(group_map[gid]["heads"]) for gid in visited_groups if gid in group_map
    }

    scenarios_by_head: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for source, target in edges_to_run:
        if target not in group_heads or source not in group_map:
            continue
        sender_heads = list(group_map[source]["heads"])
        for head in group_heads[target]:
            scenarios_by_head[head].append((source, sender_heads))

    if not scenarios_by_head:
        print("No scenarios resolved for the selected groups.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Planned target heads:")
    for head in sorted(scenarios_by_head, key=head_key):
        print(
            f"  {head} (query={head_to_query.get(head, 'end')}): "
            f"{len(scenarios_by_head[head])} scenarios"
        )

    if args.dry_run:
        print("Dry run complete. No jobs executed.")
        return

    for head in sorted(scenarios_by_head, key=head_key):
        query_spec = head_to_query.get(head, "end")
        scenario_specs: List[str] = []
        scenario_labels: List[str] = []
        for source_group, sender_heads in sorted(
            scenarios_by_head[head], key=lambda item: sanitize_label(item[0])
        ):
            label = f"{sanitize_label(source_group)}__to__{head.replace('.', '_')}"
            scenario_labels.append(label)
            sender_clause = ",".join(sender_heads)
            scenario_specs.append(f"{label}:{sender_clause}")

        summary_path = output_dir / f"summary_{head.replace('.', '_')}.json"
        head_dir = output_dir / head.replace(".", "_")
        head_dir.mkdir(exist_ok=True)

        if not scenario_specs:
            print(f"Skipping {head}: no scenarios identified")
            continue

        print(f"Running {head}: {len(scenario_specs)} scenarios (query={query_spec})")

        args_namespace = argparse.Namespace(
            target_head=head,
            receiver_input=args.receiver_input,
            model_name=args.model_name,
            device=args.device,
            input_file=str(input_path),
            scenario=scenario_specs,
            limit=args.limit,
            save_json=str(summary_path),
            require_same_length=args.require_same_length,
            query_position=query_spec,
            top_k=args.top_k,
        )

        summary = run_analysis(args_namespace)

        for scenario in summary.get("scenarios", []):
            annotate_token_concentration(scenario)
            label = scenario["label"]
            scenario_path = head_dir / f"{label}.json"
            scenario_payload = {
                "target_head": summary.get("target_head"),
                "query_position_spec": summary.get("query_position_spec"),
                "receiver_inputs": summary.get("receiver_inputs"),
                "scenario": scenario,
            }
            scenario_path.write_text(json.dumps(scenario_payload, indent=2))
            print(f"  wrote {scenario_path}")

        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
