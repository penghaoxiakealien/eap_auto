#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traverse the garden head graph and run explanation scripts with resume support.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def infer_attention_position(signature: Optional[str], group_id: Optional[str]) -> str:
    """Infer attention query position from group signature/id (fallback to end)."""
    def pick(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        sig = text.split("|", 1)[0]
        if ":" in sig:
            sig = sig.split(":", 1)[1]
        prefix = sig.split("->", 1)[0].strip().lower()
        if prefix in {"subj", "verb", "obj_head", "rel_pron", "rel_verb", "end"}:
            return prefix
        if prefix.startswith("s1"):
            return "s1"
        if prefix.startswith("s2"):
            return "s2"
        if prefix.startswith("io"):
            return "io"
        return None

    pos = pick(signature) or pick(group_id)
    if not pos:
        return "end"
    return pos.upper()


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
        return float(
            self.scores.get("direct_attention_f1")
            or self.scores.get("attention_f1")
            or 0.0
        )

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
    p = argparse.ArgumentParser(description="Traverse the garden head graph and run scripts automatically.")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Graph config JSON (nodes/edges/heads) for garden.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repository root (default: auto-detected from script location).",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="results/garden 根目录（默认: results/garden/garden_npz_v_trans_mod_run）。",
    )
    p.add_argument("--terminal-rounds", type=int, default=5, help="Terminal head rounds.")
    p.add_argument("--middle-rounds", type=int, default=5, help="Middle head rounds.")
    p.add_argument("--middle-plus-rounds", type=int, default=5, help="Middle-plus head rounds.")
    p.add_argument(
        "--middle-target-top-k",
        type=int,
        default=0,
        help="A->B: per receiver group, top-K heads (0 = no limit).",
    )
    p.add_argument(
        "--middle-plus-target-top-k",
        type=int,
        default=1,
        help="A->B->C: per target group, top-K heads (0 = no limit).",
    )
    p.add_argument("--dry-run", action="store_true", help="Only print commands.")
    p.add_argument("--skip-middle-plus", action="store_true", help="Skip A->B->C.")
    p.add_argument("--skip-terminal", action="store_true", help="Reuse cached terminal results.")
    p.add_argument(
        "--min-receiver-score",
        type=float,
        default=0.2,
        help="Minimum receiver combined score to be considered.",
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
        help="Progress JSON path (default: results-root/hypothesis/run_head_graph_progress.json).",
    )
    p.add_argument(
        "--terminal-top-k",
        type=int,
        default=0,
        help="Only run top-K terminal groups by |Δlogit| (0 = no limit).",
    )
    p.add_argument(
        "--two-hop-terminals-only",
        action="store_true",
        help="Restrict to subgraph reachable from selected terminal groups.",
    )
    p.add_argument("--data-path", type=Path, default=None, help="Garden CSV path (optional).")
    p.add_argument("--standard-json", type=Path, default=None, help="standard_garden_data.json path.")
    p.add_argument("--model-path", type=Path, default=None, help="Local model path for terminal runs.")
    return p.parse_args()


def split_head(head: str) -> Tuple[int, int]:
    head = head.split(":", 1)[0].strip()
    if head.startswith("a") and ".h" in head:
        left, right = head.split(".h", 1)
        return int(left[1:]), int(right)
    layer_str, head_str = head.split(".")
    return int(layer_str), int(head_str)


def read_best_summary(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"⚠️ 无法解析 {path}: {exc}")
        return None


class GardenHeadGraphRunner:
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.args = args
        script_dir = Path(__file__).resolve().parent
        default_repo = script_dir.parents[1]
        self.repo_root = args.repo_root or default_repo
        self.results_root = args.results_root or (
            self.repo_root / "results" / "garden" / "garden_npz_v_trans_mod_run"
        )
        self.config = config
        self.node_map = {node["id"]: node for node in config.get("nodes", [])}
        self.head_meta = config.get("heads", {})
        self.head_to_node: Dict[str, str] = {}
        for node_id, node in self.node_map.items():
            for head in node.get("heads", []):
                self.head_to_node[head] = node_id
        self.processing_order: List[str] = []
        self.head_records: Dict[str, CandidateResult] = {}
        self.min_receiver_score = float(args.min_receiver_score)
        self.progress_file = args.progress_file or (
            self.results_root / "hypothesis" / "run_head_graph_progress.json"
        )
        self.terminal_group_ids = self._select_terminal_groups()
        self.reachable_group_ids = self._compute_reachable_groups()
        self.processing_order = self._compute_processing_order()
        self._plan: List[Dict] = []
        if self.args.resume:
            self._load_progress()

    def _has_logits_edge(self, node: Dict) -> bool:
        return any(d.get("target") == "logits" for d in (node.get("downstreams", []) or []))

    def _is_terminal_group_for_run(self, node_id: str, node: Dict) -> bool:
        if self.terminal_group_ids is not None:
            return node_id in self.terminal_group_ids
        return self._has_logits_edge(node)

    def _compute_processing_order(self) -> List[str]:
        node_ids = list(self.node_map.keys())
        if self.reachable_group_ids is not None:
            node_ids = [nid for nid in node_ids if nid in self.reachable_group_ids]

        reverse_adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for src_id in node_ids:
            node = self.node_map[src_id]
            for d in node.get("downstreams", []):
                dst = d.get("target")
                if dst in reverse_adj:
                    reverse_adj[dst].append(src_id)

        dist: Dict[str, int] = {nid: 10**9 for nid in node_ids}
        if self.terminal_group_ids is not None:
            terminals = [nid for nid in node_ids if nid in self.terminal_group_ids]
        else:
            terminals = [nid for nid in node_ids if self._has_logits_edge(self.node_map[nid])]
        q = deque()
        for nid in terminals:
            dist[nid] = 0
            q.append(nid)
        while q:
            cur = q.popleft()
            for parent in reverse_adj.get(cur, []):
                if dist[parent] > dist[cur] + 1:
                    dist[parent] = dist[cur] + 1
                    q.append(parent)

        return sorted(node_ids, key=lambda nid: dist.get(nid, 10**9))

    def _select_terminal_groups(self) -> Optional[set[str]]:
        terminal_nodes: List[tuple[str, float]] = []
        for node_id, node in self.node_map.items():
            if any(d.get("target") == "logits" for d in node.get("downstreams", [])):
                summary = node.get("summary", {}) or {}
                delta = summary.get("mean_delta_logit_diff")
                try:
                    score = abs(float(delta)) if delta is not None else 0.0
                except Exception:
                    score = 0.0
                terminal_nodes.append((node_id, score))

        if self.args.terminal_top_k and self.args.terminal_top_k > 0:
            terminal_nodes.sort(key=lambda x: x[1], reverse=True)
            selected = {nid for nid, _ in terminal_nodes[: self.args.terminal_top_k]}
            print(
                f"📌 terminal-top-k={self.args.terminal_top_k}，选择 terminal 组: "
                + ", ".join(selected)
            )
            return selected
        return None

    def _compute_reachable_groups(self) -> Optional[set[str]]:
        if not self.terminal_group_ids:
            return None
        reverse_adj: Dict[str, List[str]] = {nid: [] for nid in self.node_map}
        for src_id, node in self.node_map.items():
            for d in node.get("downstreams", []):
                dst = d.get("target")
                if dst in reverse_adj:
                    reverse_adj[dst].append(src_id)

        seen: set[str] = set()
        stack: List[str] = list(self.terminal_group_ids)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(reverse_adj.get(cur, []))
        return seen

    def _child_nodes(self, node: Dict) -> List[str]:
        child_ids = []
        for downstream in node.get("downstreams", []):
            target = downstream.get("target")
            if target in self.node_map:
                child_ids.append(target)
        return child_ids

    def _run_command(self, cmd: List[str]) -> None:
        print("→ 执行命令:", " ".join(cmd))
        if self.args.dry_run:
            return
        subprocess.run(cmd, check=False)

    def _score_from_scores(self, scores: Dict[str, float]) -> float:
        att = float(scores.get("direct_attention_f1") or scores.get("attention_f1") or 0.0)
        causal = float(scores.get("causal_f1") or 0.0)
        if att <= 0.0 or causal <= 0.0:
            return 0.0
        return math.sqrt(max(att, 0.0) * max(causal, 0.0))

    def _group_status(self, node: Dict) -> Tuple[bool, str, float]:
        heads = node.get("heads", [])
        if not heads:
            return False, "empty", 0.0
        scores = []
        for head in heads:
            record = self.head_records.get(head)
            if record:
                scores.append(record.combined_score)
        if not scores:
            return False, "incomplete", 0.0
        avg_score = sum(scores) / max(1, len(scores))
        if avg_score < self.min_receiver_score:
            return False, "low_score", avg_score
        return True, "ok", avg_score

    def _group_is_usable(self, node: Dict) -> bool:
        ok, _, _ = self._group_status(node)
        return ok

    def _select_middle_targets(self, child_node: Dict) -> List[str]:
        heads = list(child_node.get("heads", []))
        if not heads:
            return []
        if self.args.middle_target_top_k and self.args.middle_target_top_k > 0:
            scored = []
            for h in heads:
                record = self.head_records.get(h)
                score = record.combined_score if record else 0.0
                scored.append((h, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [h for h, _ in scored[: self.args.middle_target_top_k]]
        return heads

    def _select_middle_plus_targets(self, target_node: Dict) -> List[str]:
        heads = list(target_node.get("heads", []))
        if not heads:
            return []
        if self.args.middle_plus_target_top_k and self.args.middle_plus_target_top_k > 0:
            scored = []
            for h in heads:
                record = self.head_records.get(h)
                score = record.combined_score if record else 0.0
                scored.append((h, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [h for h, _ in scored[: self.args.middle_plus_target_top_k]]
        return heads

    def _build_receiver_desc(self, receiver_heads: List[str]) -> Optional[str]:
        if not receiver_heads:
            return None
        mapping = {}
        for head in receiver_heads:
            record = self.head_records.get(head)
            if record and record.hypothesis:
                mapping[head] = record.hypothesis
        if not mapping:
            return None
        out_path = self.results_root / "hypothesis" / "receiver_desc.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
        return str(out_path)

    def _summarize_middle_output(self, output_dir: Path, rounds: int, head: str) -> Optional[Path]:
        result_file = output_dir / f"final_result_round_{rounds}.json"
        if not result_file.exists():
            return None
        try:
            data = json.loads(result_file.read_text())
        except Exception:
            return None
        results = data.get("all_hypotheses_validation_results", [])
        if not isinstance(results, list) or not results:
            return None
        best = None
        best_score = -1.0
        for entry in results:
            scores = entry.get("validation_scores", {})
            att = float(scores.get("direct_attention_f1") or scores.get("attention_f1") or 0.0)
            causal = float(scores.get("causal_f1") or 0.0)
            score = math.sqrt(att * causal) if att > 0 and causal > 0 else 0.0
            if score > best_score:
                best_score = score
                best = entry
        if not best:
            return None
        summary = {
            "head": head,
            "iteration": None,
            "best_hypothesis": best.get("hypothesis"),
            "validation_scores": best.get("validation_scores", {}),
            "composite_score": best_score,
            "source_file": str(result_file),
        }
        out_path = output_dir / "best_hypothesis.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return out_path

    def _run_terminal_head(self, node: Dict, head: str) -> Optional[CandidateResult]:
        layer, head_idx = split_head(head)
        output_dir = self.results_root / "hypothesis" / "Terminal" / f"{layer}_{head_idx}"
        cmd = [
            "bash",
            str(self.repo_root / "tests/experiments/run_terminal_garden.sh"),
            "--layer",
            str(layer),
            "--head",
            str(head_idx),
            "--rounds",
            str(self.args.terminal_rounds),
            "--results-root",
            str(self.results_root),
        ]
        if self.args.data_path:
            cmd.extend(["--data-path", str(self.args.data_path)])
        if self.args.standard_json:
            cmd.extend(["--standard-json", str(self.args.standard_json)])
        if self.args.model_path:
            cmd.extend(["--model-path", str(self.args.model_path)])
        self._run_command(cmd)
        best_file = output_dir / "best_hypothesis.json"
        summary = read_best_summary(best_file)
        if not summary:
            return None
        scores = summary.get("validation_scores", {}) or {}
        desc = f"{head} terminal"
        return CandidateResult(
            head=head,
            run_type="terminal",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=desc,
            hypothesis=summary.get("best_hypothesis"),
        )

    def _reuse_terminal_head(self, node: Dict, head: str) -> Optional[CandidateResult]:
        layer, head_idx = split_head(head)
        output_dir = self.results_root / "hypothesis" / "Terminal" / f"{layer}_{head_idx}"
        best_file = output_dir / "best_hypothesis.json"
        summary = read_best_summary(best_file)
        if not summary:
            print(f"⚠️ 未找到可复用的 terminal 结果: {head}")
            return None
        scores = summary.get("validation_scores", {}) or {}
        desc = f"{head} reused (Terminal)"
        return CandidateResult(
            head=head,
            run_type="terminal_reuse",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=desc,
            hypothesis=summary.get("best_hypothesis"),
        )

    def _run_middle_head(self, parent_node: Dict, child_node: Dict, head: str) -> Optional[CandidateResult]:
        if self.args.two_hop_terminals_only and not self._is_selected_terminal_node(child_node):
            return None
        receiver_heads = self._select_middle_targets(child_node)
        if not receiver_heads:
            return None
        ok, reason, _ = self._group_status(child_node)
        if not ok:
            if reason == "incomplete":
                print(f"⚠️ 下游节点 {child_node.get('id')} 尚未完成，暂时跳过对 {head} 的 middle 评估。")
            else:
                print(
                    f"⚠️ 下游节点 {child_node.get('id')} 的平均得分低于 {self.min_receiver_score:.2f}，跳过对 {head} 的 middle 评估。"
                )
            return None
        receiver_heads_str = ",".join(receiver_heads)
        layer, head_idx = split_head(head)
        timestamp = timestamp_suffix()
        output_dir = self.results_root / "hypothesis" / "Middle_Head" / f"{layer}.{head_idx}_{timestamp}"
        data_dir = self.results_root / "path_patching" / "Middle_Head" / f"{layer}_{head_idx}"
        desc_file = self._build_receiver_desc(receiver_heads)
        attn_pos = infer_attention_position(parent_node.get("signature"), parent_node.get("id"))
        recv_attn_pos = infer_attention_position(child_node.get("signature"), child_node.get("id"))
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta = {
                "run_type": "middle",
                "sender_head": head,
                "receiver_group_id": child_node.get("id"),
                "receiver_group_signature": child_node.get("signature"),
                "receiver_group_heads": child_node.get("heads") or [],
                "selected_receiver_heads": receiver_heads,
                "attention_position": attn_pos,
            }
            (output_dir / "receiver_group_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"⚠️ 无法写入 receiver_group_meta.json: {exc}")
        cmd = [
            "bash",
            str(self.repo_root / "tests/experiments/run_middle_head_garden.sh"),
            "--layer",
            str(layer),
            "--head",
            str(head_idx),
            "--receiver-heads",
            receiver_heads_str,
            "--rounds",
            str(self.args.middle_rounds),
            "--typename",
            "garden_middle_head",
            "--results-root",
            str(self.results_root),
            "--attention-position",
            attn_pos,
            "--receiver-attention-position",
            recv_attn_pos,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
        if self.args.standard_json:
            cmd.extend(["--standard-json", str(self.args.standard_json)])
        if desc_file:
            cmd.extend(["--receiver-desc", desc_file])
        self._run_command(cmd)
        best_file = self._summarize_middle_output(output_dir, self.args.middle_rounds, head)
        if not best_file:
            return None
        summary = read_best_summary(best_file)
        if not summary:
            return None
        scores = summary.get("validation_scores", {}) or {}
        desc = f"{head} via {child_node.get('id')}"
        return CandidateResult(
            head=head,
            run_type="middle",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=desc,
            hypothesis=summary.get("best_hypothesis"),
        )

    def _run_middle_plus(self, parent_node: Dict, child_node: Dict, head: str) -> List[CandidateResult]:
        results: List[CandidateResult] = []
        if self.args.skip_middle_plus:
            return results
        if not self._group_is_usable(child_node):
            ok, reason, _ = self._group_status(child_node)
            if reason == "incomplete":
                print(f"⚠️ 中间节点 {child_node.get('id')} 尚未完成，跳过 Middle Head Plus。")
            else:
                print(
                    f"⚠️ 中间节点 {child_node.get('id')} 的平均得分低于 {self.min_receiver_score:.2f}，跳过 Middle Head Plus。"
                )
            return results
        intermediate_heads = [h for h in (child_node.get("heads", []) or []) if isinstance(h, str)]
        if not intermediate_heads:
            return results
        for downstream in child_node.get("downstreams", []) or []:
            target_id = downstream.get("target")
            target_node = self.node_map.get(target_id)
            if not target_node:
                continue
            if self.reachable_group_ids is not None and target_id not in self.reachable_group_ids:
                continue
            if not self._group_is_usable(target_node):
                ok, reason, _ = self._group_status(target_node)
                if reason == "incomplete":
                    print(f"⚠️ 目标节点 {target_node.get('id')} 尚未完成，跳过作为 Middle Plus 的 target。")
                else:
                    print(
                        f"⚠️ 目标节点 {target_node.get('id')} 平均得分低于 {self.min_receiver_score:.2f}，跳过作为 Middle Plus 的 target。"
                    )
                continue
            for target_head in self._select_middle_plus_targets(target_node):
                cand = self._run_single_middle_plus(
                    parent_node, child_node, target_node, head, intermediate_heads, target_head
                )
                if cand:
                    results.append(cand)
        return results

    def _run_single_middle_plus(
        self,
        parent_node: Dict,
        child_node: Dict,
        target_node: Dict,
        head: str,
        intermediate_heads: List[str],
        target_head: str,
    ) -> Optional[CandidateResult]:
        layer, head_idx = split_head(head)
        output_dir = (
            self.results_root
            / "hypothesis"
            / "Middle_Head_Plus"
            / f"{layer}.{head_idx}_to_{target_head.replace('.', '_')}_{timestamp_suffix()}"
        )
        data_dir = (
            self.results_root
            / "path_patching"
            / "Middle_Head_Plus"
            / f"{layer}_{head_idx}_to_{target_head.replace('.', '_')}"
        )
        desc_heads = [h for h in intermediate_heads if isinstance(h, str)]
        desc_heads.append(target_head)
        desc_file = self._build_receiver_desc(desc_heads)
        attn_pos = infer_attention_position(parent_node.get("signature"), parent_node.get("id"))
        recv_attn_pos = infer_attention_position(target_node.get("signature"), target_node.get("id"))
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta = {
                "run_type": "middle_plus",
                "sender_head": head,
                "intermediate_heads": intermediate_heads,
                "target_head": target_head,
                "target_group_id": target_node.get("id"),
                "target_group_signature": target_node.get("signature"),
                "target_group_heads": target_node.get("heads") or [],
                "attention_position": attn_pos,
            }
            (output_dir / "receiver_group_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"⚠️ 无法写入 receiver_group_meta.json: {exc}")
        cmd = [
            "bash",
            str(self.repo_root / "tests/experiments/run_middle_head_garden.sh"),
            "--layer",
            str(layer),
            "--head",
            str(head_idx),
            "--intermediate-heads",
            ",".join(intermediate_heads),
            "--target-head",
            target_head,
            "--rounds",
            str(self.args.middle_plus_rounds),
            "--typename",
            "garden_middle_plus_head",
            "--results-root",
            str(self.results_root),
            "--attention-position",
            attn_pos,
            "--receiver-attention-position",
            recv_attn_pos,
            "--output-family",
            "Middle_Head_Plus",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
        if self.args.standard_json:
            cmd.extend(["--standard-json", str(self.args.standard_json)])
        if desc_file:
            cmd.extend(["--receiver-desc", desc_file])
        self._run_command(cmd)
        best_file = self._summarize_middle_output(output_dir, self.args.middle_plus_rounds, head)
        if not best_file:
            return None
        summary = read_best_summary(best_file)
        if not summary:
            return None
        scores = summary.get("validation_scores", {}) or {}
        desc = f"{head} via {child_node.get('id')} -> {target_head}"
        return CandidateResult(
            head=head,
            run_type="middle_plus",
            output_dir=output_dir,
            best_file=best_file,
            scores=scores,
            description=desc,
            hypothesis=summary.get("best_hypothesis"),
        )

    def _load_progress(self) -> None:
        if not self.progress_file.exists():
            return
        try:
            data = json.loads(self.progress_file.read_text())
        except Exception:
            return
        for head, record in data.get("heads", {}).items():
            if not isinstance(record, dict):
                continue
            best_file = Path(record.get("best_file", ""))
            if not best_file.exists():
                continue
            summary = read_best_summary(best_file)
            if not summary:
                continue
            self.head_records[head] = CandidateResult(
                head=head,
                run_type="resume",
                output_dir=Path(record.get("output_dir", "")),
                best_file=best_file,
                scores=summary.get("validation_scores", {}) or {},
                description=record.get("description", "resume"),
                hypothesis=summary.get("best_hypothesis"),
            )

    def _record_progress(self, candidate: CandidateResult) -> None:
        data = {
            "heads": {
                head: {
                    "run_type": res.run_type,
                    "output_dir": str(res.output_dir),
                    "best_file": str(res.best_file),
                    "description": res.description,
                }
                for head, res in self.head_records.items()
            },
            "meta": {
                "config": str(self.args.config),
                "results_root": str(self.results_root),
            },
        }
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _is_selected_terminal_node(self, node: Dict) -> bool:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            return False
        if self.terminal_group_ids is None:
            return self._has_logits_edge(node)
        return node_id in self.terminal_group_ids

    def run(self):
        worklist: List[Tuple[str, str]] = []
        for node_id in self.processing_order:
            node = self.node_map[node_id]
            if self.reachable_group_ids is not None and node_id not in self.reachable_group_ids:
                continue
            for head in node.get("heads", []):
                worklist.append((node_id, head))

        pending = list(worklist)
        processed: Set[str] = set()
        max_passes = max(1, len(worklist))
        for _ in range(max_passes):
            if not pending:
                break
            progress = False
            next_pending: List[Tuple[str, str]] = []
            for node_id, head in pending:
                if head in processed:
                    continue
                if self.args.resume and head in self.head_records:
                    processed.add(head)
                    continue
                node = self.node_map[node_id]
                is_terminal_group = self._is_terminal_group_for_run(node_id, node)
                print(f"\n=== 处理 head {head} ({node_id}) ===")
                if is_terminal_group:
                    candidate = (
                        self._reuse_terminal_head(node, head)
                        if self.args.skip_terminal
                        else self._run_terminal_head(node, head)
                    )
                    if candidate:
                        self.head_records[head] = candidate
                        self._record_progress(candidate)
                        processed.add(head)
                        progress = True
                    else:
                        next_pending.append((node_id, head))
                    continue

                incomplete_children: List[str] = []
                for child_id in self._child_nodes(node):
                    if self.reachable_group_ids is not None and child_id not in self.reachable_group_ids:
                        continue
                    child_node = self.node_map[child_id]
                    ok, reason, _ = self._group_status(child_node)
                    if not ok and reason == "incomplete":
                        incomplete_children.append(child_id)
                if incomplete_children:
                    print(
                        f"⏳ 下游节点 {', '.join(incomplete_children)} 尚未完成，暂缓处理 head {head}。"
                    )
                    next_pending.append((node_id, head))
                    continue

                candidates: List[CandidateResult] = []
                for child_id in self._child_nodes(node):
                    if self.reachable_group_ids is not None and child_id not in self.reachable_group_ids:
                        continue
                    child_node = self.node_map[child_id]
                    if not self._group_is_usable(child_node):
                        print(
                            f"⚠️ 下游节点 {child_node.get('id')} 的平均得分低于 {self.min_receiver_score:.2f}，"
                            f"跳过对 {head} 的普通 middle 评估。"
                        )
                        continue
                    candidate = self._run_middle_head(node, child_node, head)
                    if candidate:
                        candidates.append(candidate)
                    if not self.args.skip_middle_plus:
                        candidates.extend(self._run_middle_plus(node, child_node, head))

                best = self._pick_best_candidate(candidates)
                if best:
                    print(
                        f"✅ 选出最佳解释: {best.description} | 综合得分 {best.combined_score:.2f} "
                        f"(causal={best.causal_component:.2f}, attn={best.attention_component:.2f})"
                    )
                    self.head_records[head] = best
                    self._record_progress(best)
                    processed.add(head)
                    progress = True
                else:
                    next_pending.append((node_id, head))

            if not progress:
                break
            pending = next_pending

        for node_id, head in pending:
            if head in processed:
                continue
            print(f"⚠️ 未能为 head {head} 找到有效结果。")

        print("\n===== 最终概览 =====")
        for head, result in sorted(self.head_records.items()):
            print(
                f"{head}: {result.description} | 综合 {result.combined_score:.2f} "
                f"(causal={result.causal_component:.2f}, attn={result.attention_component:.2f}) "
                f"-> {result.best_file}"
            )

    def _pick_best_candidate(self, candidates: List[CandidateResult]) -> Optional[CandidateResult]:
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.combined_score, c.best_file.stat().st_mtime))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    runner = GardenHeadGraphRunner(config, args)
    runner.run()


if __name__ == "__main__":
    main()
