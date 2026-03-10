#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract dominant patterns for garden classify output.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TopPattern:
    pattern: str
    count: int
    total: int
    ratio: float

    @classmethod
    def from_entry(cls, entry: Dict[str, Any]) -> "TopPattern":
        pattern = str(entry.get("pattern", ""))
        count = int(entry.get("count", 0) or 0)
        total = int(entry.get("total", 0) or 0)
        ratio = float(count) / total if total else 0.0
        return cls(pattern=pattern, count=count, total=total, ratio=ratio)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract dominant patterns for garden classify output.")
    p.add_argument("--patterns-json", type=Path, required=True, help="classify_garden 输出 JSON")
    p.add_argument("--output-prefix", type=Path, required=True, help="输出前缀（不含后缀）")
    p.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[0.75, 0.80],
        help="主导阈值，可重复；默认 0.75 与 0.80",
    )
    p.add_argument(
        "--positions",
        type=str,
        default="",
        help="可选：只考虑这些 query 位置（逗号分隔）。默认使用 patterns-json 的 positions。",
    )
    return p.parse_args()


def head_sort_key(h: str) -> Tuple[int, int]:
    try:
        a, b = h.split(".", 1)
        return int(a), int(b)
    except Exception:
        return (10**9, 10**9)


def normalize_signature(pos: str, pattern: str, status: str) -> str:
    base = f"{pos}:{pattern}"
    if status == "confirmed":
        return base
    if status == "pending":
        return f"pending:{base}"
    return "none"


def pick_dominant_for_head(
    head_info: Dict[str, Any],
    positions: List[str],
    threshold: float,
) -> Dict[str, Any]:
    pos_map = (head_info.get("positions") or {})
    best: Optional[Tuple[str, TopPattern]] = None
    best_confirmed: Optional[Tuple[str, TopPattern]] = None

    for pos in positions:
        lst = pos_map.get(pos) or pos_map.get(pos.upper()) or []
        if not isinstance(lst, list) or not lst:
            continue
        top = TopPattern.from_entry(lst[0])
        if best is None or top.ratio > best[1].ratio:
            best = (pos.upper(), top)
        if top.ratio >= threshold:
            if best_confirmed is None or top.ratio > best_confirmed[1].ratio:
                best_confirmed = (pos.upper(), top)

    if best_confirmed is not None:
        pos, top = best_confirmed
        status = "confirmed"
    elif best is not None:
        pos, top = best
        status = "pending"
    else:
        return {"status": "none"}

    return {
        "status": status,
        "position": pos,
        "pattern": top.pattern,
        "count": top.count,
        "total": top.total,
        "ratio": round(top.ratio, 6),
        "signature": normalize_signature(pos, top.pattern, status),
    }


def build_soft(base: Dict[str, Any], assignments: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    per_head = out.get("per_head") or {}
    if not isinstance(per_head, dict):
        per_head = {}
        out["per_head"] = per_head

    groups: Dict[str, List[str]] = {}
    for head, info in per_head.items():
        a = assignments.get(head, {"status": "none"})
        sig = a.get("signature") if isinstance(a, dict) else None
        if not sig:
            sig = info.get("signature", "none")
        info["dominant"] = a
        info["signature"] = sig
        groups.setdefault(sig, []).append(head)

    for sig in list(groups.keys()):
        groups[sig].sort(key=head_sort_key)
    out["groups_by_signature"] = groups
    out["dominant_assignments"] = assignments
    return out


def main() -> None:
    args = parse_args()
    base = json.loads(args.patterns_json.read_text())
    per_head = base.get("per_head")
    if not isinstance(per_head, dict):
        raise ValueError("patterns-json 缺少 per_head dict（请确认输入来自 classify_garden.py）")

    positions = []
    if args.positions.strip():
        positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    else:
        positions = [p.strip().upper() for p in (base.get("positions") or []) if p.strip()]
    if not positions:
        raise ValueError("未找到 positions；请通过 --positions 指定。")

    for thr in args.threshold:
        assignments: Dict[str, Any] = {}
        for head in sorted(per_head.keys(), key=head_sort_key):
            assignments[head] = pick_dominant_for_head(per_head[head] or {}, positions, thr)

        thr_tag = f"thr{int(round(thr * 100)):02d}"
        out_summary = {
            "meta": {
                "threshold": thr,
                "positions": positions,
                "source": str(args.patterns_json),
            },
            "dominant_assignments": assignments,
        }
        summary_path = args.output_prefix.with_name(args.output_prefix.name + f"_{thr_tag}.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2))

        soft = build_soft(base, assignments)
        soft["meta"] = {**(soft.get("meta") or {}), "dominant_threshold": thr, "dominant_positions": positions}
        soft_path = args.output_prefix.with_name(args.output_prefix.name + f"_{thr_tag}_soft.json")
        soft_path.parent.mkdir(parents=True, exist_ok=True)
        soft_path.write_text(json.dumps(soft, ensure_ascii=False, indent=2))

        print(f"✅ 写出 dominant summary: {summary_path}")
        print(f"✅ 写出 dominant soft: {soft_path}")


if __name__ == "__main__":
    main()
