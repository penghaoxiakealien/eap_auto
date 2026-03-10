from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from api import OpenRouter


DEFAULT_API_KEY = "sk-99F0IFe53pSHOPQ3phWbEAEx86ZDOqkE58Ov9aYCS9AOQ2C7"
DEFAULT_MODEL = "claude-sonnet-4-20250514-thinking"


@dataclass(frozen=True)
class FamilyConfig:
    family_dir: str
    reference_file: str
    label: str


FAMILY_CONFIGS: Tuple[FamilyConfig, ...] = (
    FamilyConfig("Name_Mover_Head", "NMH.txt", "NMH"),
    FamilyConfig("Negative_Name_Mover_Head", "NNMH.txt", "NNMH"),
    FamilyConfig("SIH", "SIH.txt", "SIH"),
    FamilyConfig("DTH", "DTH.txt", "DTH"),
)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _extract_best_hypothesis(best_path: Path) -> Optional[str]:
    try:
        data = json.loads(best_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    # Support both best_hypothesis.json and test_results.json formats.
    hypothesis = data.get("best_hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        hypothesis = data.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    return hypothesis.strip()

def _load_candidate_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        return _extract_best_hypothesis(path)
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _iter_best_hypotheses(family_root: Path) -> Iterable[Path]:
    if not family_root.exists():
        return []
    return family_root.rglob("best_hypothesis.json")


def _iter_initial_iteration1(family_root: Path) -> Iterable[Path]:
    """Find initial hypotheses from *_all/iteration_results/iteration_1.json."""
    if not family_root.exists():
        return []
    return family_root.rglob("iteration_results/iteration_1.json")


def _extract_initial_hypothesis(iteration1_path: Path) -> Optional[str]:
    try:
        data = json.loads(iteration1_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    hypothesis = data.get("hypothesis") or data.get("hypothesis_before")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    return hypothesis.strip()


def _try_parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to recover JSON from a fenced block or trailing text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _build_messages(reference: str, hypothesis: str, label: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a careful research evaluator. "
        "Your job is to compare a candidate mechanistic hypothesis against a reference answer "
        "and judge semantic similarity, not surface form.\n\n"
        "IOI task context (important):\n"
        "- IOI = Indirect Object Identification on truncated transfer/communication sentences.\n"
        "- Example: \"Anakin and Obi-wan played at Mustafar. Obi-wan gave a lightsaber to ___.\" (the recipient token is withheld).\n"
        "- The model must predict the recipient (IO) at the final prediction position.\n"
        "- Roles/positions used in hypotheses:\n"
        "  - IO: the indirect object (recipient).\n"
        "  - S1: the subject token in the earlier clause.\n"
        "  - S2: the subject token in the main transfer clause.\n"
        "- Causal claims often refer to the IO−S logit difference at the prediction position:\n"
        "  logit_diff = logit(IO) − logit(S).\n"
        "Focus on: core mechanism, causal direction, what tokens/roles are attended to, and task effect."
    )

    user_prompt = (
        f"Task: Compare a candidate hypothesis to the reference answer for {label}.\n\n"
        "Reference answer (paper standard):\n"
        f"{reference}\n\n"
        "Candidate hypothesis:\n"
        f"{hypothesis}\n\n"
        "Evaluation guidance:\n"
        "- Treat semantic alignment as primary; wording differences are fine.\n"
        "- Pay special attention to:\n"
        "  1) the main mechanism, 2) causal direction/sign on IO−S logit difference (i.e., whether it directly or indirectly affects task performance), 3) role/position targeting (IO/S1/S2), 4) stated task effect.\n\n"
        "Return STRICT JSON with this schema:\n"
        "{\n"
        '  "similarity_score": float,  // 0.00 to 1.00\n'
        '  "verdict": "high" | "medium" | "low",\n'
        '  "agreement_points": [str, str, ...],\n'
        '  "mismatch_points": [str, str, ...],\n'
        '  "summary": str\n'
        "}\n\n"
        "Scoring guidance:\n"
        "- 0.9-1.0: Same core mechanism and causal direction.\n"
        "- 0.6-0.8: Mostly aligned but with meaningful mismatches.\n"
        "- 0.3-0.5: Shares some motifs but core mechanism differs.\n"
        "- 0.0-0.2: Fundamentally different.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _build_pair_messages(reference: str, hyp_a: str, hyp_b: str, label: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a careful research evaluator. "
        "Your job is to compare candidate mechanistic hypotheses against a reference answer "
        "and judge semantic similarity, not surface form.\n\n"
        "IOI task context (important):\n"
        "- IOI = Indirect Object Identification on truncated transfer/communication sentences.\n"
        "- Example: \"Anakin and Obi-wan played at Mustafar. Obi-wan gave a lightsaber to ___.\" (the recipient token is withheld).\n"
        "- The model must predict the recipient (IO) at the final prediction position.\n"
        "- Roles/positions used in hypotheses:\n"
        "  - IO: the indirect object (recipient).\n"
        "  - S1: the subject token in the earlier clause.\n"
        "  - S2: the subject token in the main transfer clause.\n"
        "- Causal claims often refer to the IO−S logit difference at the prediction position:\n"
        "  logit_diff = logit(IO) − logit(S).\n"
        "Focus on: core mechanism, causal direction, what tokens/roles are attended to, and task effect."
    )
    user_prompt = (
        f"Task: Compare two candidate hypotheses (A vs B) to the reference answer for {label}.\n\n"
        "Reference answer (paper standard):\n"
        f"{reference}\n\n"
        "Candidate A:\n"
        f"{hyp_a}\n\n"
        "Candidate B:\n"
        f"{hyp_b}\n\n"
        "Evaluation guidance:\n"
        "- Treat semantic alignment as primary; wording differences are fine.\n"
        "- Pay special attention to:\n"
        "  1) the main mechanism, 2) causal direction/sign on IO−S logit difference, 3) role/position targeting (IO/S1/S2), 4) stated task effect.\n\n"
        "Return STRICT JSON with this schema:\n"
        "{\n"
        '  "winner": "A" | "B" | "tie",\n'
        '  "confidence": float,  // 0.00 to 1.00\n'
        '  "comments": str,\n'
        '  "a_score": float,  // 0.00 to 1.00\n'
        '  "b_score": float,  // 0.00 to 1.00\n'
        '  "agreement_points": [str, str, ...],\n'
        '  "mismatch_points": [str, str, ...]\n'
        "}\n\n"
        "Scoring guidance:\n"
        "- 0.9-1.0: Same core mechanism and causal direction.\n"
        "- 0.6-0.8: Mostly aligned but with meaningful mismatches.\n"
        "- 0.3-0.5: Shares some motifs but core mechanism differs.\n"
        "- 0.0-0.2: Fundamentally different.\n"
        "Set a_score/b_score using this scale, and pick winner accordingly."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _score_one(
    client: OpenRouter,
    reference: str,
    best_path: Path,
    label: str,
) -> Optional[Dict]:
    hypothesis = _extract_best_hypothesis(best_path)
    if not hypothesis:
        return None

    run_dir = best_path.parent
    messages = _build_messages(reference, hypothesis, label)
    response = await client.generate(messages=messages, output_dir=str(run_dir))
    parsed = _try_parse_json(response.text)
    if parsed is None:
        parsed = {
            "similarity_score": None,
            "verdict": "low",
            "agreement_points": [],
            "mismatch_points": [],
            "summary": "Failed to parse LLM response as JSON.",
            "raw_response": response.text,
        }

    result = {
        "reference_label": label,
        "reference_path": str(best_path.parents[2] / "answer"),
        "best_hypothesis_path": str(best_path),
        "best_hypothesis": hypothesis,
        "llm_evaluation": parsed,
    }
    out_path = run_dir / "reference_similarity.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def _score_initial_one(
    client: OpenRouter,
    reference: str,
    iteration1_path: Path,
    label: str,
) -> Optional[Dict]:
    hypothesis = _extract_initial_hypothesis(iteration1_path)
    if not hypothesis:
        return None

    run_dir = iteration1_path.parents[1]  # *_all directory
    messages = _build_messages(reference, hypothesis, label)
    response = await client.generate(messages=messages, output_dir=str(run_dir))
    parsed = _try_parse_json(response.text)
    if parsed is None:
        parsed = {
            "similarity_score": None,
            "verdict": "low",
            "agreement_points": [],
            "mismatch_points": [],
            "summary": "Failed to parse LLM response as JSON.",
            "raw_response": response.text,
        }

    result = {
        "reference_label": label,
        "source_type": "initial_iteration_1",
        "iteration_1_path": str(iteration1_path),
        "initial_hypothesis": hypothesis,
        "llm_evaluation": parsed,
    }
    out_path = run_dir / "reference_similarity_initial.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

async def _score_pair(
    client: OpenRouter,
    reference: str,
    cand_a: Path,
    cand_b: Path,
    label: str,
    output_path: Optional[Path],
) -> Optional[Dict]:
    hyp_a = _load_candidate_text(cand_a)
    hyp_b = _load_candidate_text(cand_b)
    if not hyp_a or not hyp_b:
        return None
    messages = _build_pair_messages(reference, hyp_a, hyp_b, label)
    response = await client.generate(messages=messages, output_dir=str(output_path.parent if output_path else cand_a.parent))
    parsed = _try_parse_json(response.text)
    if parsed is None:
        parsed = {
            "winner": "tie",
            "confidence": None,
            "comments": "Failed to parse LLM response as JSON.",
            "a_score": None,
            "b_score": None,
            "agreement_points": [],
            "mismatch_points": [],
            "raw_response": response.text,
        }
    result = {
        "reference_label": label,
        "reference_path": str(cand_a.parent / "answer"),
        "candidate_a_path": str(cand_a),
        "candidate_b_path": str(cand_b),
        "candidate_a": hyp_a,
        "candidate_b": hyp_b,
        "llm_evaluation": parsed,
    }
    if output_path:
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def main_async(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    answer_root = results_root / "answer"
    hypothesis_root = results_root / "hypothesis"
    only_run_dir = Path(args.only_run_dir).resolve() if args.only_run_dir else None

    client = OpenRouter(model=args.model, api_key=args.api_key)

    if args.pair_reference_path and args.candidate_a and args.candidate_b:
        reference = _load_text(Path(args.pair_reference_path))
        output_path = Path(args.output) if args.output else None
        result = await _score_pair(
            client,
            reference,
            Path(args.candidate_a),
            Path(args.candidate_b),
            args.pair_label or "PAIR",
            output_path,
        )
        if result and output_path:
            print(f"[ok] Wrote {output_path}")
        elif not result:
            print("[warn] Failed to score pair (missing candidate text?)")
        return

    total_best = 0
    written_best = 0
    total_initial = 0
    written_initial = 0

    for cfg in FAMILY_CONFIGS:
        reference_path = answer_root / cfg.reference_file
        family_root = hypothesis_root / cfg.family_dir
        if not reference_path.exists():
            print(f"[skip] Missing reference: {reference_path}")
            continue
        reference = _load_text(reference_path)

        best_paths = list(_iter_best_hypotheses(family_root))
        if only_run_dir:
            best_paths = [p for p in best_paths if p.parent.resolve() == only_run_dir]
        if not best_paths:
            print(f"[skip] No best_hypothesis.json under: {family_root}")
        elif args.mode in {"both", "best"}:
            print(f"[info] {cfg.label}: reference={reference_path.name}, best_runs={len(best_paths)}")
            for best_path in best_paths:
                total_best += 1
                try:
                    result = await _score_one(client, reference, best_path, cfg.label)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] Failed for {best_path}: {exc}")
                    continue
                if result:
                    written_best += 1
                    print(f"[ok] Wrote {best_path.parent / 'reference_similarity.json'}")

        initial_paths = [
            p for p in _iter_initial_iteration1(family_root) if "_all" in str(p.parents[1].name)
        ]
        if only_run_dir:
            initial_paths = [p for p in initial_paths if p.parents[1].resolve() == only_run_dir]
        if not initial_paths:
            print(f"[skip] No *_all/iteration_results/iteration_1.json under: {family_root}")
            continue

        if args.mode in {"both", "initial"}:
            print(f"[info] {cfg.label}: initial_runs={len(initial_paths)}")
            for iteration1_path in initial_paths:
                total_initial += 1
                try:
                    result = await _score_initial_one(client, reference, iteration1_path, cfg.label)
                except Exception as exc:  
                    print(f"[warn] Failed for {iteration1_path}: {exc}")
                    continue
                if result:
                    written_initial += 1
                    print(f"[ok] Wrote {iteration1_path.parents[1] / 'reference_similarity_initial.json'}")

    print(
        "\nDone. "
        f"Best hypotheses: {written_best}/{total_best}. "
        f"Initial iteration_1: {written_initial}/{total_initial}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based similarity scoring against reference answers.")
    parser.add_argument(
        "--results-root",
        default="/data31/private/wangziran/eap_auto/results/ioi_0126",
        help="Root directory containing answer/ and hypothesis/ subfolders.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="OpenRouter API key.")
    parser.add_argument(
        "--only-run-dir",
        default="",
        help="Optional: evaluate only this run directory (path to a run folder).",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "best", "initial"],
        default="both",
        help="Which sources to evaluate: best_hypothesis, initial iteration_1, or both.",
    )
    parser.add_argument("--pair-reference-path", default="", help="Reference answer path for pairwise compare.")
    parser.add_argument("--pair-label", default="", help="Label for pairwise compare output.")
    parser.add_argument("--candidate-a", default="", help="Path to candidate A (json or txt).")
    parser.add_argument("--candidate-b", default="", help="Path to candidate B (json or txt).")
    parser.add_argument("--output", default="", help="Output JSON path for pairwise compare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
