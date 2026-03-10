#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate middle queue scheduling with pending/requeue logic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate middle queue scheduling.")
    p.add_argument("--group-graph-json", type=Path, required=True, help="Group graph JSON (dagish).")
    p.add_argument("--results-root", type=Path, required=True, help="Results root for outputs.")
    p.add_argument("--min-layer", type=int, default=0, help="Minimum sender layer (inclusive).")
    p.add_argument("--max-layer", type=int, default=11, help="Maximum sender layer (inclusive).")
    p.add_argument("--receiver-top-k", type=int, default=0, help="Per child group, top-K receiver heads.")
    p.add_argument("--min-receiver-score", type=float, default=0.2, help="Minimum avg receiver score.")
    p.add_argument("--require-all-receivers", action="store_true", help="Require all receivers ready.")
    p.add_argument("--ignore-receiver-score", action="store_true", help="Only require presence of best_hypothesis.")
    p.add_argument("--max-passes", type=int, default=0, help="Max passes (0 = auto).")
    p.add_argument("--output-json", type=Path, help="Optional path to save simulation output.")
    return p.parse_args()


def load_group_graph(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def parse_head(head: str) -> Tuple[int, int]:
    layer, h = head.split(".", 1)
    return int(layer), int(h)


def compute_score(summary: Dict[str, object]) -> float:
    scores = summary.get("validation_scores", {}) or {}
    att = float(scores.get("direct_attention_f1") or scores.get("attention_f1") or 0.0)
    causal = float(scores.get("causal_f1") or 0.0)
    if att <= 0.0 or causal <= 0.0:
        return 0.0
    return math.sqrt(att * causal)


def load_best_hypothesis_scores(results_root: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for family in ("Terminal", "Middle_Head"):
        root = results_root / "hypothesis" / family
        if not root.exists():
            continue
        for head_dir in root.iterdir():
            if not head_dir.is_dir():
                continue
            best_file = head_dir / "best_hypothesis.json"
            if not best_file.exists():
                continue
            try:
                data = json.loads(best_file.read_text())
            except Exception:
                continue
            head = data.get("head") or head_dir.name.split("_", 1)[0].replace("_", ".")
            score = compute_score(data)
            scores[str(head)] = max(scores.get(str(head), 0.0), score)
    return scores


def load_best_hypothesis_heads(results_root: Path) -> Dict[str, bool]:
    present: Dict[str, bool] = {}
    for family in ("Terminal", "Middle_Head"):
        root = results_root / "hypothesis" / family
        if not root.exists():
            continue
        for head_dir in root.iterdir():
            if not head_dir.is_dir():
                continue
            best_file = head_dir / "best_hypothesis.json"
            if not best_file.exists():
                continue
            try:
                data = json.loads(best_file.read_text())
            except Exception:
                continue
            head = data.get("head") or head_dir.name.split("_", 1)[0].replace("_", ".")
            present[str(head)] = True
    return present


def select_receiver_heads(receivers: List[str], scores: Dict[str, float], top_k: int) -> List[str]:
    if not receivers:
        return []
    if top_k and top_k > 0:
        scored = [(h, scores.get(h, 0.0)) for h in receivers]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored[:top_k]]
    return receivers


def average_score(heads: List[str], scores: Dict[str, float]) -> Optional[float]:
    vals = [scores.get(h, 0.0) for h in heads if scores.get(h, 0.0) > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def receivers_ready(heads: List[str], scores: Dict[str, float]) -> List[str]:
    missing = []
    for h in heads:
        if scores.get(h, 0.0) <= 0:
            missing.append(h)
    return missing


def main() -> None:
    args = parse_args()
    graph = load_group_graph(args.group_graph_json)
    groups = graph.get("groups", []) or []
    edges = graph.get("edges", []) or []

    group_by_id: Dict[str, Dict[str, object]] = {
        g.get("id"): g for g in groups if isinstance(g, dict) and isinstance(g.get("id"), str)
    }

    downstreams: Dict[str, List[str]] = {}
    for e in edges:
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if dst == "logits":
            continue
        downstreams.setdefault(src, []).append(dst)

    scores = load_best_hypothesis_scores(args.results_root)
    present = load_best_hypothesis_heads(args.results_root) if args.ignore_receiver_score else {}

    tasks: List[Tuple[str, str]] = []
    for src_id, child_ids in downstreams.items():
        src_group = group_by_id.get(src_id, {})
        sender_heads = [h for h in src_group.get("heads", []) or [] if isinstance(h, str)]
        if not sender_heads:
            continue
        for sender_head in sender_heads:
            layer, _ = parse_head(sender_head)
            if not (args.min_layer <= layer <= args.max_layer):
                continue
            for child_id in child_ids:
                tasks.append((sender_head, child_id))

    pending = list(tasks)
    max_passes = args.max_passes or max(1, len(pending))
    order: List[str] = []
    pass_logs: List[Dict[str, object]] = []

    for pass_idx in range(1, max_passes + 1):
        if not pending:
            break
        progress = False
        next_pending: List[Tuple[str, str]] = []
        for sender_head, child_id in pending:
            key = f"{sender_head}->{child_id}"
            child_group = group_by_id.get(child_id, {})
            receiver_heads = [h for h in child_group.get("heads", []) or [] if isinstance(h, str)]
            receiver_heads = select_receiver_heads(receiver_heads, scores, args.receiver_top_k)
            if not receiver_heads:
                continue
            if args.ignore_receiver_score:
                if args.require_all_receivers:
                    missing = [h for h in receiver_heads if not present.get(h, False)]
                    if missing:
                        next_pending.append((sender_head, child_id))
                        continue
                elif not any(present.get(h, False) for h in receiver_heads):
                    next_pending.append((sender_head, child_id))
                    continue
                avg = 1.0
            else:
                if args.require_all_receivers:
                    missing = receivers_ready(receiver_heads, scores)
                    if missing:
                        next_pending.append((sender_head, child_id))
                        continue
                avg = average_score(receiver_heads, scores)
                if avg is None:
                    next_pending.append((sender_head, child_id))
                    continue
                if avg < args.min_receiver_score:
                    continue
            # simulate run completion
            scores[sender_head] = max(scores.get(sender_head, 0.0), avg)
            if args.ignore_receiver_score:
                present[sender_head] = True
            order.append(key)
            progress = True
        pass_logs.append(
            {
                "pass": pass_idx,
                "ran": order,
                "pending": [f"{s}->{c}" for s, c in next_pending],
            }
        )
        if not progress:
            pending = next_pending
            break
        pending = next_pending

    summary = {
        "total_tasks": len(tasks),
        "ran": order,
        "pending": [f"{s}->{c}" for s, c in pending],
        "passes": pass_logs,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"✅ wrote {args.output_json}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
