#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run garden middle-head explanations for sender->receiver pairs from a group graph."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def infer_attention_position(signature: Optional[str], group_id: Optional[str]) -> str:
    def pick(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        sig = text.split("|", 1)[0]
        if ":" in sig:
            sig = sig.split(":", 1)[1]
        prefix = sig.split("->", 1)[0].strip().lower()
        if prefix in {"subj", "verb", "obj_head", "rel_pron", "rel_verb", "end"}:
            return prefix
        return None

    pos = pick(signature) or pick(group_id)
    return (pos or "end").upper()


def find_sender_group(groups: List[Dict[str, object]], head: str) -> Optional[Dict[str, object]]:
    for g in groups:
        heads = g.get("heads") or []
        if head in heads:
            return g
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run garden middle-head explanations from group graph with parallelism."
    )
    p.add_argument("--group-graph-json", type=Path, required=True, help="Group graph JSON (dagish).")
    p.add_argument("--results-root", type=Path, required=True, help="Results root for outputs.")
    p.add_argument("--standard-json", type=Path, help="standard_garden_data.json path.")
    p.add_argument("--min-layer", type=int, default=0, help="Minimum sender layer (inclusive).")
    p.add_argument("--max-layer", type=int, default=11, help="Maximum sender layer (inclusive).")
    p.add_argument(
        "--receiver-top-k",
        type=int,
        default=0,
        help="Per child group, top-K receiver heads by score (0 = all).",
    )
    p.add_argument(
        "--min-receiver-score",
        type=float,
        default=0.2,
        help="Minimum average receiver score to run middle (default: 0.2).",
    )
    p.add_argument(
        "--ignore-receiver-score",
        action="store_true",
        help="Ignore receiver scores; only require presence of best_hypothesis.",
    )
    p.add_argument(
        "--require-all-receivers",
        action="store_true",
        help="Require every receiver head to have a score before running.",
    )
    p.add_argument("--rounds", type=int, default=5, help="LLM refinement rounds.")
    p.add_argument(
        "--attention-batch-size",
        type=int,
        default=1,
        help="Batch size for raw attention computation.",
    )
    p.add_argument(
        "--strict-attention-align",
        action="store_true",
        help="Use strict word-level alignment for attention tokens.",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=0,
        help="Maximum scheduling passes (0 = auto based on task count).",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of concurrent runs.",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        help="Optional directory to write per-run stdout/stderr logs.",
    )
    p.add_argument(
        "--use-log-dir",
        action="store_true",
        help="Enable writing per-run logs when --log-dir is provided.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from progress file if available.",
    )
    p.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Progress JSON path (default: results-root/hypothesis/middle_progress.json).",
    )
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


def load_best_hypothesis_scores(results_root: Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    scores: Dict[str, float] = {}
    hypotheses: Dict[str, str] = {}

    terminal_root = results_root / "hypothesis" / "Terminal"
    if terminal_root.exists():
        for head_dir in terminal_root.iterdir():
            if not head_dir.is_dir():
                continue
            best_file = head_dir / "best_hypothesis.json"
            if not best_file.exists():
                continue
            try:
                data = json.loads(best_file.read_text())
            except Exception:
                continue
            head = data.get("head") or head_dir.name.replace("_", ".")
            scores[str(head)] = compute_score(data)
            if data.get("best_hypothesis"):
                hypotheses[str(head)] = str(data.get("best_hypothesis"))

    middle_root = results_root / "hypothesis" / "Middle_Head"
    if middle_root.exists():
        for head_dir in middle_root.iterdir():
            if not head_dir.is_dir():
                continue
            best_file = head_dir / "best_hypothesis.json"
            if not best_file.exists():
                continue
            try:
                data = json.loads(best_file.read_text())
            except Exception:
                continue
            head = data.get("head")
            if not head:
                parts = head_dir.name.split("_", 1)[0].replace("_", ".")
                head = parts
            scores[str(head)] = max(scores.get(str(head), 0.0), compute_score(data))
            if data.get("best_hypothesis"):
                hypotheses.setdefault(str(head), str(data.get("best_hypothesis")))

    return scores, hypotheses


def select_receiver_heads(
    receivers: List[str],
    scores: Dict[str, float],
    top_k: int,
) -> List[str]:
    if not receivers:
        return []
    if top_k and top_k > 0:
        scored = [(h, scores.get(h, 0.0)) for h in receivers]
        scored.sort(key=lambda x: x[1], reverse=True)
        if any(score > 0 for _, score in scored):
            return [h for h, _ in scored[:top_k]]
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


def build_command(
    sender_head: str,
    receiver_heads: List[str],
    results_root: Path,
    data_dir: Path,
    output_dir: Path,
    rounds: int,
    standard_json: Optional[Path],
    receiver_desc_path: Optional[Path],
    attention_batch_size: int,
    strict_attention_align: bool,
) -> List[str]:
    layer_str, head_str = sender_head.split(".")
    cmd = [
        "bash",
        "tests/experiments/run_middle_head_garden.sh",
        "--layer",
        layer_str,
        "--head",
        head_str,
        "--receiver-heads",
        ",".join(receiver_heads),
        "--rounds",
        str(rounds),
        "--results-root",
        str(results_root),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--attention-batch-size",
        str(attention_batch_size),
    ]
    if strict_attention_align:
        cmd.append("--strict-attention-align")
    if standard_json:
        cmd.extend(["--standard-json", str(standard_json)])
    if receiver_desc_path:
        cmd.extend(["--receiver-desc", str(receiver_desc_path)])
    return cmd


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

    scores, hypotheses = load_best_hypothesis_scores(args.results_root)
    log_dir = args.log_dir if args.use_log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    progress_file = args.progress_file or (
        args.results_root / "hypothesis" / "middle_progress.json"
    )
    completed: Dict[str, str] = {}
    if args.resume and progress_file.exists():
        try:
            payload = json.loads(progress_file.read_text())
            completed = payload.get("completed", {}) if isinstance(payload, dict) else {}
        except Exception:
            completed = {}

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

    if not tasks:
        print("未找到可运行的 middle 任务。")
        return

    print(f"Found {len(tasks)} sender->receiver middle tasks.")

    running = []
    max_parallel = max(1, int(args.max_parallel))

    def record_progress(key: str, output_dir: Path) -> None:
        if not output_dir.exists():
            return
        best_file = output_dir / "best_hypothesis.json"
        if not best_file.exists():
            return
        try:
            data = json.loads(best_file.read_text())
        except Exception:
            data = None
        if isinstance(data, dict):
            head = data.get("head")
            score = compute_score(data)
            if head:
                scores[str(head)] = max(scores.get(str(head), 0.0), score)
            if data.get("best_hypothesis") and head:
                hypotheses.setdefault(str(head), str(data.get("best_hypothesis")))
        completed[key] = str(output_dir)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(
            json.dumps(
                {"completed": completed},
                ensure_ascii=False,
                indent=2,
            )
        )

    def enqueue(cmd, key, log_path, log_f, output_dir):
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        running.append((key, proc, log_f, log_path, output_dir))
        if len(running) >= max_parallel:
            key_done, proc_done, log_f_done, log_path_done, out_dir_done = running.pop(0)
            ret = proc_done.wait()
            if log_f_done:
                log_f_done.close()
            if log_path_done:
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[{status}] {key_done} -> {log_path_done}")
            record_progress(key_done, out_dir_done)

    pending = list(tasks)
    max_passes = args.max_passes or max(1, len(pending))
    for pass_idx in range(1, max_passes + 1):
        if not pending:
            break
        print(f"\n=== scheduling pass {pass_idx}/{max_passes} ===")
        progress = False
        next_pending: List[Tuple[str, str]] = []
        for sender_head, child_id in pending:
            key = f"{sender_head}->{child_id}"
            if args.resume and key in completed:
                continue
            child_group = group_by_id.get(child_id, {})
            receiver_heads = [h for h in child_group.get("heads", []) or [] if isinstance(h, str)]
            receiver_heads = select_receiver_heads(receiver_heads, scores, args.receiver_top_k)
            if not receiver_heads:
                continue
            if args.require_all_receivers:
                missing = receivers_ready(receiver_heads, scores)
                if missing:
                    print(f"[PENDING] {key} missing receivers: {', '.join(missing)}")
                    next_pending.append((sender_head, child_id))
                    continue
            avg = average_score(receiver_heads, scores)
            if avg is None:
                if args.ignore_receiver_score and args.require_all_receivers:
                    avg = 0.0
                else:
                    next_pending.append((sender_head, child_id))
                    continue
            if not args.ignore_receiver_score and avg < args.min_receiver_score:
                print(f"[SKIP] {key} avg receiver score {avg:.3f} below threshold")
                continue

            layer, head_idx = parse_head(sender_head)
            data_dir = args.results_root / "path_patching" / "Middle_Head" / f"{layer}_{head_idx}"
            output_dir = (
                args.results_root
                / "hypothesis"
                / "Middle_Head"
                / f"{layer}.{head_idx}_to_{child_id}_{timestamp_suffix()}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                meta = {
                    "run_type": "middle",
                    "sender_head": sender_head,
                    "receiver_group_id": child_id,
                    "receiver_group_signature": child_group.get("signature"),
                    "receiver_group_heads": child_group.get("heads") or [],
                    "selected_receiver_heads": receiver_heads,
                }
                (output_dir / "receiver_group_meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"⚠️ 无法写入 receiver_group_meta.json: {exc}")
            sender_group = find_sender_group(groups, sender_head)
            sender_attn_pos = infer_attention_position(
                sender_group.get("signature") if sender_group else None,
                sender_group.get("id") if sender_group else None,
            )
            receiver_attn_pos = infer_attention_position(
                child_group.get("signature"),
                child_group.get("id"),
            )

            receiver_desc_path = None
            receiver_desc = {
                h: hypotheses.get(h, "")
                for h in receiver_heads
                if hypotheses.get(h)
            }
            if receiver_desc:
                receiver_desc_path = data_dir / "receiver_descriptions.json"
                receiver_desc_path.parent.mkdir(parents=True, exist_ok=True)
                receiver_desc_path.write_text(json.dumps(receiver_desc, ensure_ascii=False, indent=2))

            cmd = build_command(
                sender_head,
                receiver_heads,
                args.results_root,
                data_dir,
                output_dir,
                args.rounds,
                args.standard_json,
                receiver_desc_path,
                args.attention_batch_size,
                args.strict_attention_align,
            )
            cmd.extend(["--attention-position", sender_attn_pos])
            cmd.extend(["--receiver-attention-position", receiver_attn_pos])
            print(" ".join(cmd))
            if args.dry_run:
                continue

            log_path = None
            log_f = None
            if log_dir:
                safe_key = key.replace(".", "_").replace(":", "_")
                log_path = log_dir / f"{safe_key}.log"
                log_f = log_path.open("w", encoding="utf-8")
                log_f.write(f"[CMD] {' '.join(cmd)}\n\n")

            if max_parallel == 1:
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=False)
                if log_f:
                    log_f.close()
                if log_path:
                    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
                    print(f"[{status}] {key} -> {log_path}")
                record_progress(key, output_dir)
                progress = True
                continue

            enqueue(cmd, key, log_path, log_f, output_dir)
            progress = True

        for key_done, proc_done, log_f_done, log_path_done, out_dir_done in running:
            ret = proc_done.wait()
            if log_f_done:
                log_f_done.close()
            if log_path_done:
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[{status}] {key_done} -> {log_path_done}")
            record_progress(key_done, out_dir_done)
        running.clear()

        if not progress:
            print("⚠️ no progress this pass; stopping.")
            pending = next_pending
            break
        pending = next_pending

    if pending:
        print("\n=== pending tasks ===")
        for sender_head, child_id in pending:
            print(f"- {sender_head}->{child_id}")


if __name__ == "__main__":
    main()
