#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-level automated explanation runner for agr_gender, mirroring IOI run_head_graph.py.

Input:
  - a DAG-ish grouped graph JSON (nodes=groups, edges=group->group), e.g.
    results/agr_gender_gpt2_new/group_graph_thr0_001_thr75_layered_nocycle.json

Pipeline:
  - For any group that has an edge to 'logits', treat all its heads as terminal and run:
      bash tests/experiments/run_gender_terminal_head.sh
  - For other heads, run middle-head explanations via each downstream group:
      bash tests/experiments/run_middle_head_agr_gender.sh
    then pick the best candidate by sqrt(causal_f1 * direct_attention_f1).

Downstream best hypotheses are injected via --receiver-desc (head:path pairs).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class CandidateResult:
    head: str
    run_type: str
    output_dir: Path
    best_file: Path
    scores: Dict[str, float]
    description: str
    hypothesis: Optional[str] = None

    @property
    def attention_component(self) -> float:
        return float(self.scores.get("direct_attention_f1") or self.scores.get("attention_f1") or 0.0)

    @property
    def causal_component(self) -> float:
        return float(self.scores.get("causal_f1") or 0.0)

    @property
    def combined_score(self) -> float:
        att = max(self.attention_component, 0.0)
        causal = max(self.causal_component, 0.0)
        if att <= 0.0 or causal <= 0.0:
            return 0.0
        return math.sqrt(att * causal)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traverse agr_gender group graph and run explanations automatically.")
    p.add_argument("--group-graph", type=Path, required=True, help="group_graph_*_nocycle.json")
    p.add_argument("--results-root", type=Path, default=Path("results/agr_gender"), help="Where to write hypothesis/path_patching outputs")
    p.add_argument("--standard-json", type=Path, default=Path("results/agr_gender/standard_gender_data.json"))
    p.add_argument("--terminal-rounds", type=int, default=5)
    p.add_argument("--middle-rounds", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-middle", action="store_true", help="Only run terminal heads.")
    p.add_argument(
        "--terminal-policy",
        choices=["reuse", "run", "skip"],
        default="reuse",
        help="How to handle terminal heads: reuse cached best_hypothesis.json if available (default), always run, or skip running.",
    )
    return p.parse_args()


def infer_attention_position(group_id: str) -> str:
    base = group_id.split("::", 1)[0]
    if ":" in base:
        prefix = base.split(":", 1)[0].strip().lower()
    else:
        prefix = base.strip().lower()
    if prefix == "end":
        return "end"
    if prefix == "verb":
        return "verb"
    if prefix == "a1":
        return "a1"
    if prefix == "a2":
        return "a2"
    if prefix == "b":
        return "b"
    return "end"


def read_best_summary(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"⚠️ 无法解析 {path}: {exc}")
        return None


def split_head(head: str) -> Tuple[int, int]:
    """
    Accept either:
      - "10.7"
      - "a10.h7"
      - "10.7:..." (suffix ignored)
    """
    head = head.split(":", 1)[0].strip()
    if head.startswith("a") and ".h" in head:
        left, right = head.split(".h", 1)
        return int(left[1:]), int(right)
    layer_s, head_s = head.split(".", 1)
    return int(layer_s), int(head_s)


def normalize_head_name(head: str) -> str:
    """Convert 'aL.hH' -> 'L.H'."""
    layer, h = split_head(head)
    return f"{layer}.{h}"


class AgrGenderHeadGraphRunner:
    def __init__(self, graph: Dict[str, any], args: argparse.Namespace):
        self.args = args
        # __file__ = <repo>/tests/experiments/run_head_graph_agr_gender.py
        # parents[0]=experiments, parents[1]=tests, parents[2]=repo root
        self.repo_root = Path(__file__).resolve().parents[2]
        self.graph = graph
        raw_members = graph.get("group_members") or {}
        self.group_members: Dict[str, List[str]] = {}
        for gid, members in raw_members.items():
            gid = str(gid)
            if not isinstance(members, list):
                continue
            normalized = []
            for m in members:
                if not isinstance(m, str):
                    continue
                if m in {"input", "logits"}:
                    continue
                if m.startswith("a") and ".h" in m:
                    normalized.append(normalize_head_name(m))
                else:
                    normalized.append(m)
            self.group_members[gid] = normalized
        # edges: list of {src,dst,...}
        self.edges = graph.get("edges") or []
        self.downstreams: Dict[str, List[str]] = defaultdict(list)
        for e in self.edges:
            s = e.get("src")
            d = e.get("dst")
            if isinstance(s, str) and isinstance(d, str):
                self.downstreams[s].append(d)
        self.groups = [
            g.get("id")
            for g in (graph.get("groups") or [])
            if isinstance(g, dict) and g.get("id") and g.get("id") not in {"input", "logits"}
        ]
        self.processing_order = self._compute_processing_order()
        self.head_records: Dict[str, CandidateResult] = {}
        self.min_receiver_score = 0.2
        self._load_cached_terminal_records()

    def _score_from_scores(self, scores: Dict[str, float]) -> float:
        att = float(scores.get("direct_attention_f1") or scores.get("attention_f1") or 0.0)
        causal = float(scores.get("causal_f1") or 0.0)
        if att <= 0.0 or causal <= 0.0:
            return 0.0
        return math.sqrt(max(att, 0.0) * max(causal, 0.0))

    def _load_cached_terminal_records(self) -> None:
        """
        Pre-load existing terminal best_hypothesis.json under results_root/hypothesis/Terminal.
        If multiple runs exist for the same head, keep the one with highest combined score
        (tie-break by mtime).
        """
        term_dir = self.args.results_root / "hypothesis" / "Terminal"
        if not term_dir.exists():
            return
        best_by_head: Dict[str, Tuple[float, float, CandidateResult]] = {}
        for best_file in term_dir.glob("*/best_hypothesis.json"):
            summary = read_best_summary(best_file)
            if not summary:
                continue
            head = summary.get("head")
            if not isinstance(head, str) or not head.strip():
                # fallback: parse from parent dirname like "9.2_YYYY..."
                head = best_file.parent.name.split("_", 1)[0]
            head = head.strip()
            scores = summary.get("validation_scores", {}) or {}
            if not isinstance(scores, dict):
                scores = {}
            score = self._score_from_scores(scores)
            mtime = best_file.stat().st_mtime
            cand = CandidateResult(
                head=head,
                run_type="terminal_cached",
                output_dir=best_file.parent,
                best_file=best_file,
                scores=scores,
                description=f"{head} direct (logits) [cached]",
                hypothesis=summary.get("best_hypothesis"),
            )
            prev = best_by_head.get(head)
            if prev is None or score > prev[0] or (score == prev[0] and mtime > prev[1]):
                best_by_head[head] = (score, mtime, cand)
        for head, (_s, _t, cand) in best_by_head.items():
            self.head_records[head] = cand

    def _compute_processing_order(self) -> List[str]:
        indegree = {gid: 0 for gid in self.groups}
        for s in self.groups:
            for d in self.downstreams.get(s, []):
                if d in indegree:
                    indegree[d] += 1
        q = deque([gid for gid, deg in indegree.items() if deg == 0])
        topo = []
        while q:
            gid = q.popleft()
            topo.append(gid)
            for d in self.downstreams.get(gid, []):
                if d in indegree:
                    indegree[d] -= 1
                    if indegree[d] == 0:
                        q.append(d)
        return list(reversed(topo))

    def run(self) -> None:
        for gid in self.processing_order:
            if gid in {"input", "logits"}:
                continue
            heads = self.group_members.get(gid, [])
            if not heads:
                continue
            print(f"\n=== 处理 group {gid} (heads={len(heads)}) ===")
            is_terminal = "logits" in self.downstreams.get(gid, [])
            for head in heads:
                print(f"\n--- head {head} ---")
                if is_terminal:
                    if self.args.terminal_policy in {"reuse", "skip"} and head in self.head_records:
                        print(f"↪ 复用已有 terminal 结果: {self.head_records[head].best_file}")
                        continue
                    if self.args.terminal_policy == "skip":
                        print("↪ terminal-policy=skip，跳过运行。")
                        continue
                    cand = self._run_terminal(head, gid)
                    if cand:
                        self.head_records[head] = cand
                    continue
                if self.args.skip_middle:
                    continue
                candidates: List[CandidateResult] = []
                for child in self.downstreams.get(gid, []):
                    if child == "logits":
                        continue
                    cand = self._run_middle(head, gid, child)
                    if cand:
                        candidates.append(cand)
                    cand_abc = self._run_middle_abc(head, gid, child)
                    if cand_abc:
                        candidates.append(cand_abc)
                best = self._pick_best_candidate(candidates)
                if best:
                    print(
                        f"✅ 选出最佳解释: {best.description} | 综合 {best.combined_score:.2f} "
                        f"(causal={best.causal_component:.2f}, attn={best.attention_component:.2f})"
                    )
                    self.head_records[head] = best
                else:
                    print("⚠️ 未找到有效 middle 结果。")

        print("\n===== 最终概览 =====")
        for head, result in sorted(self.head_records.items()):
            print(
                f"{head}: {result.description} | 综合 {result.combined_score:.2f} "
                f"(causal={result.causal_component:.2f}, attn={result.attention_component:.2f}) -> {result.best_file}"
            )

    def _build_receiver_desc(self, heads: List[str]) -> str:
        entries = []
        for h in heads:
            record = self.head_records.get(h)
            if record and record.best_file.exists() and record.combined_score >= self.min_receiver_score:
                entries.append(f"{h}:{record.best_file}")
        return ",".join(entries)

    def _group_is_usable(self, group_id: str) -> bool:
        if self.args.dry_run:
            return True
        heads = self.group_members.get(group_id, [])
        scores = [self.head_records[h].combined_score for h in heads if h in self.head_records]
        if not scores:
            return False
        avg = sum(scores) / len(scores)
        return avg >= self.min_receiver_score

    def _run_terminal(self, head: str, group_id: str) -> Optional[CandidateResult]:
        layer, h = split_head(head)
        attn_pos = infer_attention_position(group_id)
        output_dir = self.args.results_root / "hypothesis" / "Terminal" / f"{layer}.{h}_{timestamp_suffix()}"
        data_dir = self.args.results_root / "path_patching" / "Terminal" / f"{layer}_{h}"
        cmd = [
            "bash",
            str(self.repo_root / "tests/experiments/run_gender_terminal_head.sh"),
            "--layer",
            str(layer),
            "--head",
            str(h),
            "--rounds",
            str(self.args.terminal_rounds),
            "--typename",
            "gender_terminal_head",
            "--results-root",
            str(self.args.results_root),
            "--standard-json",
            str(self.args.standard_json),
            "--attention-position",
            attn_pos,
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(data_dir),
        ]
        self._run_command(cmd)
        best_file = output_dir / "best_hypothesis.json"
        summary = read_best_summary(best_file)
        if not summary:
            return None
        scores = summary.get("validation_scores", {})
        return CandidateResult(
            head=head,
            run_type="terminal",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=f"{head} direct (logits)",
            hypothesis=summary.get("best_hypothesis"),
        )

    def _run_middle(self, head: str, parent_group: str, child_group: str) -> Optional[CandidateResult]:
        receiver_heads = self.group_members.get(child_group, [])
        if not receiver_heads:
            return None
        if not self._group_is_usable(child_group):
            print(f"⚠️ 下游组 {child_group} 得分不足，跳过。")
            return None
        layer, h = split_head(head)
        output_dir = self.args.results_root / "hypothesis" / "Middle_Head" / f"{layer}.{h}_{timestamp_suffix()}"
        data_dir = self.args.results_root / "path_patching" / "Middle_Head" / f"{layer}_{h}"
        attn_pos = infer_attention_position(parent_group)
        desc_files = self._build_receiver_desc(receiver_heads)
        cmd = [
            "bash",
            str(self.repo_root / "tests/experiments/run_middle_head_agr_gender.sh"),
            "--layer",
            str(layer),
            "--head",
            str(h),
            "--receiver-heads",
            ",".join(receiver_heads),
            "--rounds",
            str(self.args.middle_rounds),
            "--results-root",
            str(self.args.results_root),
            "--standard-json",
            str(self.args.standard_json),
            "--attention-position",
            attn_pos,
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(data_dir),
        ]
        if desc_files:
            cmd.extend(["--receiver-desc", desc_files])
        self._run_command(cmd)
        best_file = output_dir / "best_hypothesis.json"
        summary = read_best_summary(best_file)
        if not summary:
            return None
        scores = summary.get("validation_scores", {})
        return CandidateResult(
            head=head,
            run_type="middle",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=f"{head} via {child_group}",
            hypothesis=summary.get("best_hypothesis"),
        )

    def _collect_target_heads(self, child_group: str) -> List[str]:
        targets: List[str] = []
        for g in self.downstreams.get(child_group, []):
            if g == "logits":
                continue
            targets.extend(self.group_members.get(g, []))
        seen = set()
        uniq = []
        for h in targets:
            if h in seen:
                continue
            seen.add(h)
            uniq.append(h)
        return uniq

    def _run_middle_abc(self, head: str, parent_group: str, child_group: str) -> Optional[CandidateResult]:
        receiver_heads = self.group_members.get(child_group, [])
        if not receiver_heads:
            return None
        if not self._group_is_usable(child_group):
            print(f"⚠️ 下游组 {child_group} 得分不足，跳过。")
            return None
        target_heads = self._collect_target_heads(child_group)
        if not target_heads:
            return None

        layer, h = split_head(head)
        output_dir = self.args.results_root / "hypothesis" / "Middle_Head" / f"{layer}.{h}_{timestamp_suffix()}_abc"
        data_dir = self.args.results_root / "path_patching" / "Middle_Head" / f"{layer}_{h}_abc"
        attn_pos = infer_attention_position(parent_group)
        desc_files = self._build_receiver_desc(receiver_heads)
        target_desc_files = self._build_receiver_desc(target_heads)
        cmd = [
            "bash",
            str(self.repo_root / "tests/experiments/run_middle_head_agr_gender.sh"),
            "--layer",
            str(layer),
            "--head",
            str(h),
            "--receiver-heads",
            ",".join(receiver_heads),
            "--target-heads",
            ",".join(target_heads),
            "--rounds",
            str(self.args.middle_rounds),
            "--results-root",
            str(self.args.results_root),
            "--standard-json",
            str(self.args.standard_json),
            "--attention-position",
            attn_pos,
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(data_dir),
            "--use-abc",
        ]
        if desc_files:
            cmd.extend(["--receiver-desc", desc_files])
        if target_desc_files:
            cmd.extend(["--target-desc", target_desc_files])
        self._run_command(cmd)
        best_file = output_dir / "best_hypothesis.json"
        summary = read_best_summary(best_file)
        if not summary:
            return None
        scores = summary.get("validation_scores", {})
        return CandidateResult(
            head=head,
            run_type="middle_abc",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=f"{head} via {child_group} -> targets",
            hypothesis=summary.get("best_hypothesis"),
        )

    def _pick_best_candidate(self, candidates: List[CandidateResult]) -> Optional[CandidateResult]:
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.combined_score)

    def _run_command(self, cmd: List[str]) -> None:
        printable = " ".join(cmd)
        print(f"→ 执行命令: {printable}")
        if self.args.dry_run:
            return
        proc = subprocess.run(cmd, cwd=self.repo_root)
        if proc.returncode != 0:
            raise RuntimeError(f"命令失败: {printable}")


def main() -> None:
    args = parse_args()
    graph = json.loads(args.group_graph.read_text())
    runner = AgrGenderHeadGraphRunner(graph, args)
    runner.run()


if __name__ == "__main__":
    main()
