#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api import OpenRouter


DEFAULT_API_KEY = "sk-99F0IFe53pSHOPQ3phWbEAEx86ZDOqkE58Ov9aYCS9AOQ2C7"
DEFAULT_MODEL = "claude-sonnet-4-20250514-thinking"


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _extract_best_hypothesis(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if path.suffix.lower() != ".json":
        txt = path.read_text(encoding="utf-8").strip()
        return txt or None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    hyp = data.get("best_hypothesis")
    if not isinstance(hyp, str) or not hyp.strip():
        hyp = data.get("hypothesis")
    if not isinstance(hyp, str) or not hyp.strip():
        hyp = data.get("final_hypothesis")
    if not isinstance(hyp, str) or not hyp.strip():
        return None
    return hyp.strip()


def _try_parse_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _build_pair_messages(reference: str, hyp_a: str, hyp_b: str, label: str):
    system_prompt = (
        "You are a careful research evaluator. "
        "Your job is to compare candidate mechanistic hypotheses against a reference answer "
        "and judge semantic similarity, not surface form.\n\n"
        "IOI task context (important):\n"
        "- IOI = Indirect Object Identification on truncated transfer/communication sentences.\n"
        "- The model must predict the recipient (IO) at the final prediction position.\n"
        "- Roles/positions used in hypotheses: IO/S1/S2.\n"
        "- Causal claims often refer to IO−S logit difference at the prediction position.\n"
        "Focus on mechanism, causal direction, token/role targeting, and task effect."
    )
    user_prompt = (
        f"Task: Compare two candidate hypotheses (A vs B) to the reference answer for {label}.\n\n"
        "Reference answer (paper standard):\n"
        f"{reference}\n\n"
        "Candidate A:\n"
        f"{hyp_a}\n\n"
        "Candidate B:\n"
        f"{hyp_b}\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "winner": "A" | "B" | "tie",\n'
        '  "confidence": float,\n'
        '  "comments": str,\n'
        '  "a_score": float,\n'
        '  "b_score": float,\n'
        '  "agreement_points": [str],\n'
        '  "mismatch_points": [str]\n'
        "}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


async def _run_once(
    client: OpenRouter,
    reference: str,
    hyp_a: str,
    hyp_b: str,
    label: str,
    output_dir: Path,
    tag: str,
) -> Dict:
    messages = _build_pair_messages(reference, hyp_a, hyp_b, label)
    response = await client.generate(messages=messages, output_dir=str(output_dir))
    parsed = _try_parse_json(response.text) or {}
    winner = str(parsed.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    result = {
        "tag": tag,
        "winner": winner,
        "confidence": _to_float(parsed.get("confidence")),
        "a_score": _to_float(parsed.get("a_score")),
        "b_score": _to_float(parsed.get("b_score")),
        "comments": parsed.get("comments"),
        "raw_text": response.text,
        "parsed": parsed,
    }
    return result


def _mean(xs):
    ys = [x for x in xs if isinstance(x, (int, float))]
    return (sum(ys) / len(ys)) if ys else None


async def main_async(args: argparse.Namespace) -> None:
    reference = _load_text(Path(args.reference))
    cand_a = _extract_best_hypothesis(Path(args.candidate_a))
    cand_b = _extract_best_hypothesis(Path(args.candidate_b))
    if not cand_a or not cand_b:
        raise RuntimeError("候选假设读取失败，请检查 candidate 路径。")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenRouter(model=args.model, api_key=args.api_key)

    runs = []
    for i in range(args.forward_runs):
        runs.append(
            await _run_once(
                client, reference, cand_a, cand_b, args.label, out_dir, f"forward_{i+1}"
            )
        )
    for i in range(args.reverse_runs):
        runs.append(
            await _run_once(
                client, reference, cand_b, cand_a, args.label, out_dir, f"reverse_{i+1}"
            )
        )

    line_path = out_dir / "ab_bias_raw_runs.jsonl"
    with line_path.open("w", encoding="utf-8") as f:
        for row in runs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fwd = [r for r in runs if r["tag"].startswith("forward_")]
    rev = [r for r in runs if r["tag"].startswith("reverse_")]

    # Normalize reverse winner back to original A/B identity.
    normalized = []
    for r in runs:
        w = r["winner"]
        if r["tag"].startswith("reverse_"):
            if w == "A":
                w = "B"
            elif w == "B":
                w = "A"
        normalized.append(w)

    summary = {
        "label": args.label,
        "reference": str(Path(args.reference).resolve()),
        "candidate_a": str(Path(args.candidate_a).resolve()),
        "candidate_b": str(Path(args.candidate_b).resolve()),
        "model": args.model,
        "forward_runs": args.forward_runs,
        "reverse_runs": args.reverse_runs,
        "position_bias_check": {
            "forward_winner_counts": {
                "A": sum(r["winner"] == "A" for r in fwd),
                "B": sum(r["winner"] == "B" for r in fwd),
                "TIE": sum(r["winner"] == "TIE" for r in fwd),
            },
            "reverse_winner_counts_raw": {
                "A": sum(r["winner"] == "A" for r in rev),
                "B": sum(r["winner"] == "B" for r in rev),
                "TIE": sum(r["winner"] == "TIE" for r in rev),
            },
            "normalized_winner_counts": {
                "A": sum(w == "A" for w in normalized),
                "B": sum(w == "B" for w in normalized),
                "TIE": sum(w == "TIE" for w in normalized),
            },
            "forward_mean_scores": {
                "a_score": _mean([r["a_score"] for r in fwd]),
                "b_score": _mean([r["b_score"] for r in fwd]),
            },
            "reverse_mean_scores_raw": {
                "a_score": _mean([r["a_score"] for r in rev]),
                "b_score": _mean([r["b_score"] for r in rev]),
            },
        },
    }

    sum_path = out_dir / "ab_bias_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Raw runs: {line_path}")
    print(f"✅ Summary:  {sum_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check A/B position bias by swapping candidate order.")
    p.add_argument("--reference", required=True, help="Reference answer text file path.")
    p.add_argument("--candidate-a", required=True, help="Candidate A path (json or txt).")
    p.add_argument("--candidate-b", required=True, help="Candidate B path (json or txt).")
    p.add_argument("--label", default="PAIR", help="Label in prompt.")
    p.add_argument("--forward-runs", type=int, default=4, help="A/B normal order runs.")
    p.add_argument("--reverse-runs", type=int, default=4, help="B/A swapped order runs.")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
