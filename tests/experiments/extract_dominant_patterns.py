#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 classify.py 生成的注意力模式结果中抽取主导模式。"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PatternStat:
    pattern: str
    count: int
    total: int
    ratio: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PatternStat":
        pattern = str(payload.get("pattern", ""))
        count = int(payload.get("count", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        ratio_val = payload.get("ratio")
        if ratio_val is None:
            ratio = float(count) / total if total else 0.0
        else:
            ratio = float(ratio_val)
        return cls(pattern=pattern, count=count, total=total, ratio=ratio)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "count": self.count,
            "total": self.total,
            "ratio": round(self.ratio, 6),
        }


def friendly_pattern_name(pattern: Optional[str]) -> str:
    if not pattern:
        return ""
    text = str(pattern)
    if "@" in text:
        base, offset = text.split("@", 1)
    else:
        base, offset = text, ""
    base = base.replace("->", "-")
    offset = offset.strip()
    if offset:
        try:
            off_val = int(offset)
        except ValueError:
            if offset.startswith(("+", "-")):
                formatted = offset
            else:
                formatted = f"+{offset}"
        else:
            formatted = f"{off_val:+d}" if off_val else ""
    else:
        formatted = ""
    return f"{base} {formatted}".strip()


def pick_dominant_pattern(
    role_summary: Dict[str, Dict[str, Any]],
    positions: List[str],
    threshold: float,
) -> Dict[str, Any]:
    best_confirmed: Optional[Dict[str, Any]] = None
    best_fallback: Optional[Dict[str, Any]] = None
    for pos in positions:
        info = role_summary.get(pos) or role_summary.get(pos.upper()) or {}
        primary = info.get("primary") if isinstance(info, dict) else None
        if not primary:
            continue
        count = int(primary.get("count", 0) or 0)
        total = int(primary.get("total", 0) or 0)
        ratio_val = primary.get("ratio")
        ratio = float(ratio_val) if ratio_val is not None else (float(count) / total if total else 0.0)
        entry = {
            "position": pos.upper(),
            "raw_pattern": primary.get("pattern"),
            "count": count,
            "total": total,
            "ratio": ratio,
        }
        if ratio >= threshold:
            if best_confirmed is None or ratio > best_confirmed["ratio"]:
                best_confirmed = entry
        if best_fallback is None or ratio > best_fallback["ratio"]:
            best_fallback = entry
    if best_confirmed is not None:
        best_confirmed["status"] = "confirmed"
        return best_confirmed
    if best_fallback is not None:
        best_fallback["status"] = "pending"
        return best_fallback
    return {"status": "none"}


def summarise_position(patterns: List[Dict[str, Any]], threshold: float, min_top_ratio: float) -> Dict[str, Any]:
    stats = [PatternStat.from_dict(p) for p in patterns if p]
    stats.sort(key=lambda s: (s.ratio, s.count), reverse=True)

    if not stats:
        return {
            "selection_mode": "no_data",
            "selected": [],
            "primary": None,
            "threshold": threshold,
            "all_patterns": [],
        }

    selected = [s for s in stats if s.ratio >= threshold]
    if selected:
        selected.sort(key=lambda s: (s.ratio, s.count), reverse=True)
        mode = "threshold"
    else:
        top = stats[0]
        peers = [s for s in stats if s.count >= top.count * min_top_ratio]
        mode = "fallback_top" if len(peers) == 1 else "fallback_top_with_peers"
        selected = peers

    return {
        "selection_mode": mode,
        "selected": [s.to_dict() for s in selected],
        "primary": selected[0].to_dict() if selected else None,
        "threshold": threshold,
        "all_patterns": [s.to_dict() for s in stats],
    }


def build_filtered_soft(
    base_data: Dict[str, Any],
    summary_payload: Dict[str, Any],
    positions: List[str],
    threshold: float,
    min_top_ratio: float,
) -> Dict[str, Any]:
    data = copy.deepcopy(base_data)
    base_patterns = data.get("patterns") or {}
    base_per_head = base_patterns.get("per_head") or data.get("head_patterns") or {}
    assignments = summary_payload.get("dominant_assignments", {}) if isinstance(summary_payload, dict) else {}
    role_summary = summary_payload.get("dominant_patterns", {}) if isinstance(summary_payload, dict) else {}
    all_heads = sorted(set(base_per_head.keys()) | set(role_summary.keys()) | set(assignments.keys()))

    def head_sort_key(name: str):
        parts: List[Any] = []
        for chunk in name.split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                parts.append(chunk)
        return parts

    filtered_per_head: Dict[str, Dict[str, Any]] = {}
    signature_groups: Dict[str, List[str]] = {}

    for head in all_heads:
        per_pos: Dict[str, List[Dict[str, Any]]] = {pos: [] for pos in positions}
        assignment = assignments.get(head, {"status": "none"}) or {"status": "none"}
        status = assignment.get("status", "none")
        pos = assignment.get("position")
        pos = pos.upper() if isinstance(pos, str) else (positions[0] if positions else "")
        raw_pattern = assignment.get("raw_pattern")
        friendly = friendly_pattern_name(raw_pattern)
        count = assignment.get("count")
        total = assignment.get("total")
        ratio = assignment.get("ratio")

        entry: Dict[str, Any] = {}
        if status in {"confirmed", "pending"} and raw_pattern:
            entry = {
                "pattern": friendly or str(raw_pattern),
                "raw_pattern": raw_pattern,
                "count": int(count) if count is not None else None,
                "total": int(total) if total is not None else None,
                "ratio": round(float(ratio), 6) if ratio is not None else None,
                "status": status,
            }
        elif status == "none":
            entry = {
                "pattern": "待定",
                "count": 0,
                "total": 0,
                "ratio": 0.0,
                "status": status,
            }

        if entry:
            per_pos.setdefault(pos, [])
            per_pos[pos] = [drop_none(entry)] if entry else []

        signature = "待定:无主导"
        if status == "confirmed" and friendly:
            signature = friendly
        elif status == "pending" and friendly:
            signature = f"待定:{friendly}"

        filtered_per_head[head] = {
            "positions": per_pos,
            "signature": signature,
            "dominant": {
                **assignment,
                "friendly_pattern": friendly,
                "status": status,
            },
        }
        signature_groups.setdefault(signature, []).append(head)

    for heads in signature_groups.values():
        heads.sort(key=head_sort_key)

    meta = data.setdefault("meta", {})
    meta["dominant_threshold"] = threshold
    meta["dominant_min_top_ratio"] = min_top_ratio

    data["patterns"] = {
        **base_patterns,
        "per_head": filtered_per_head,
        "groups_by_signature": signature_groups,
        "dominant_threshold": threshold,
        "min_top_ratio": min_top_ratio,
        "dominant_assignments": assignments,
    }
    data["head_patterns"] = filtered_per_head
    data["pattern_groups"] = signature_groups
    data["dominant_summary"] = summary_payload

    return data


def drop_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def process_threshold(
    per_head: Dict[str, Any],
    positions: List[str],
    threshold: float,
    min_top_ratio: float,
    meta: Dict[str, Any],
    base_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    positions_upper = [p.upper() for p in positions]
    head_summary: Dict[str, Dict[str, Any]] = {}
    assignments: Dict[str, Dict[str, Any]] = {}

    for head, info in per_head.items():
        pos_map = info.get("positions") if isinstance(info, dict) else {}
        head_summary[head] = {}
        for pos in positions_upper:
            pos_list = []
            if isinstance(pos_map, dict):
                pos_list = pos_map.get(pos) or pos_map.get(pos.upper()) or []
            summary = summarise_position(
                patterns=pos_list,
                threshold=threshold,
                min_top_ratio=min_top_ratio,
            )
            head_summary[head][pos] = summary
        assignments[head] = pick_dominant_pattern(head_summary[head], positions_upper, threshold)

    summary_payload = {
        "meta": {
            **meta,
            "threshold": threshold,
            "min_top_ratio": min_top_ratio,
            "positions": positions_upper,
        },
        "dominant_patterns": head_summary,
        "dominant_assignments": assignments,
    }
    filtered = None
    if base_data is not None:
        filtered = build_filtered_soft(base_data, summary_payload, positions_upper, threshold, min_top_ratio)

    return summary_payload, filtered


def load_per_head(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = json.loads(path.read_text())
    patterns = data.get("patterns") or {}
    per_head = patterns.get("per_head") or data.get("head_patterns")
    if not isinstance(per_head, dict):
        raise ValueError(f"未能在 {path} 中解析 per_head 结构")
    return data, per_head


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="从 classify.py 输出中提取主导模式")
    ap.add_argument("--patterns-json", type=Path, required=True, help="classify.py 生成的 JSON")
    ap.add_argument("--output-prefix", type=Path, required=True, help="主导模式摘要输出前缀")
    ap.add_argument("--threshold", dest="thresholds", type=float, action="append", required=True, help="阈值，可重复指定")
    ap.add_argument("--min-top-ratio", type=float, default=0.7, help="回退时保留与最佳计数比值的下限")
    ap.add_argument("--positions", type=str, default="END,S2,S1,IO", help="关注的查询位置，逗号分隔")
    ap.add_argument("--graph-json", type=Path, help="可选，图结构 JSON")
    ap.add_argument("--visual-prefix", type=Path, help="可选，PNG 输出前缀")
    ap.add_argument("--visualize", action="store_true", help="生成可视化 PNG")
    ap.add_argument("--layout", type=str, default="dot", help="Graphviz 布局算法")
    ap.add_argument("--color-mode", type=str, default="pattern", choices=["pattern", "pos_roles", "soft_label"], help="可视化配色模式")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    data, per_head = load_per_head(args.patterns_json)
    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]

    if args.visualize and (not args.graph_json or not args.visual_prefix):
        raise ValueError("使用 --visualize 时必须同时提供 --graph-json 与 --visual-prefix")

    meta = {
        "source_file": str(args.patterns_json),
        "available_heads": sorted(per_head.keys()),
    }

    for thr in args.thresholds:
        summary, filtered = process_threshold(
            per_head=per_head,
            positions=positions,
            threshold=thr,
            min_top_ratio=args.min_top_ratio,
            meta=meta,
            base_data=data,
        )
        suffix = f"thr{int(round(thr * 100))}"
        out_path = args.output_prefix.parent / f"{args.output_prefix.name}_{suffix}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"写入主导模式: {out_path}")

        if filtered is not None:
            soft_path = args.output_prefix.parent / f"{args.output_prefix.name}_{suffix}_soft.json"
            soft_path.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
            print(f"写入筛选后的 soft JSON: {soft_path}")

            if args.visualize and args.graph_json and args.visual_prefix:
                visual_dir = args.visual_prefix.parent
                base_stem = args.visual_prefix.stem if args.visual_prefix.suffix else args.visual_prefix.name
                png_path = visual_dir / f"{base_stem}_{suffix}.png"
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("visualize.py")),
                    "--graph-json",
                    str(args.graph_json),
                    "--soft-json",
                    str(soft_path),
                    "--output",
                    str(png_path),
                    "--layout",
                    args.layout,
                    "--color-mode",
                    args.color_mode,
                ]
                subprocess.run(cmd, check=True)
                print(f"生成主导模式图: {png_path}")


if __name__ == "__main__":
    main()
