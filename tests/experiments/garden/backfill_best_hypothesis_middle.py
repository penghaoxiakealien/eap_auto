#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill best_hypothesis.json for garden middle-head runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional


def load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def pick_best(result_path: Path) -> Optional[Dict]:
    data = load_json(result_path)
    if not data:
        return None
    results = data.get("all_hypotheses_validation_results", [])
    if not isinstance(results, list) or not results:
        return None
    best = None
    best_score = -1.0
    for entry in results:
        scores = entry.get("validation_scores", {}) if isinstance(entry, dict) else {}
        attn = float(scores.get("direct_attention_f1") or scores.get("attention_f1") or 0.0)
        causal = float(scores.get("causal_f1") or 0.0)
        composite = math.sqrt(attn * causal) if attn > 0 and causal > 0 else 0.0
        if composite > best_score:
            best_score = composite
            best = entry
    if not best:
        return None
    return {
        "head": data.get("head"),
        "typename": data.get("typename", ""),
        "best_hypothesis": best.get("hypothesis") if isinstance(best, dict) else None,
        "validation_scores": best.get("validation_scores", {}) if isinstance(best, dict) else {},
        "composite_score": best_score,
        "source_file": str(result_path),
    }


def backfill_dir(path: Path) -> bool:
    result_files = sorted(path.glob("final_result_round_*.json"))
    if not result_files:
        return False
    result_path = result_files[-1]
    summary = pick_best(result_path)
    if not summary:
        return False
    out_path = path / "best_hypothesis.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill best_hypothesis.json for middle-head runs.")
    p.add_argument("paths", nargs="+", help="Middle_Head run directories")
    args = p.parse_args()

    for raw in args.paths:
        path = Path(raw)
        if not path.is_dir():
            print(f"skip (not dir): {path}")
            continue
        ok = backfill_dir(path)
        if ok:
            print(f"✅ wrote {path / 'best_hypothesis.json'}")
        else:
            print(f"⚠️ failed: {path}")


if __name__ == "__main__":
    main()
