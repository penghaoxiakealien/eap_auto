#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BOUNDARY_TOKENS = {"then", "when", "after"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_token(token: str) -> str:
    return token.strip()


def build_occurrence_labels(tokens: list[str]) -> dict[int, str]:
    counts: Counter[str] = Counter()
    labels: dict[int, str] = {}
    for pos, tok in enumerate(tokens):
        base = normalize_token(tok)
        counts[base] += 1
        if not base:
            labels[pos] = f"EMPTY_{counts[base]}"
        else:
            labels[pos] = f"{base}_{counts[base]}"
    return labels


def structural_label(
    pos: int,
    token: str,
    sample: dict[str, Any],
    occurrence_labels: dict[int, str],
) -> str:
    tok = normalize_token(token)
    tok_lower = tok.lower()
    positions = sample.get("positions", {})
    if pos == positions.get("io"):
        return "IO_1"
    if pos == positions.get("s1"):
        return "S1_1"
    if pos == positions.get("s2"):
        return "S2_1"
    if pos == 0 and tok_lower in BOUNDARY_TOKENS:
        return "BOUNDARY"
    if tok_lower == "and":
        return "AND"
    if tok_lower == "to":
        return "TO"
    if tok_lower == ",":
        return "COMMA"
    if tok_lower == normalize_token(sample["clean"].get("io_token", "")).lower():
        return "IO_other"
    if tok_lower == normalize_token(sample["clean"].get("s_token", "")).lower():
        return "S_other"
    return occurrence_labels.get(pos, tok or f"POS_{pos}")


def top_direction_tokens(
    diff_vector: list[float],
    tokens: list[str],
    sample: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    occurrence_labels = build_occurrence_labels(tokens)
    indexed = list(enumerate(diff_vector))
    inc = sorted(indexed, key=lambda item: item[1], reverse=True)[:top_k]
    dec = sorted(indexed, key=lambda item: item[1])[:top_k]

    def convert(items: list[tuple[int, float]]) -> list[dict[str, Any]]:
        out = []
        for pos, score in items:
            out.append(
                {
                    "position": pos,
                    "token": normalize_token(tokens[pos]),
                    "score": score,
                    "occurrence_label": occurrence_labels[pos],
                    "structural_label": structural_label(pos, tokens[pos], sample, occurrence_labels),
                }
            )
        return out

    return convert(inc), convert(dec)


def summarize_causal_dataset(
    head: str,
    causal_dataset_path: Path,
    standard_data: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    causal_dataset = load_json(causal_dataset_path)
    sample_map = {str(item["sample_id"]): item for item in standard_data["samples"]}

    top1_inc_counter: Counter[str] = Counter()
    top1_dec_counter: Counter[str] = Counter()
    topk_inc_counter: Counter[str] = Counter()
    topk_dec_counter: Counter[str] = Counter()
    raw_top1_inc_counter: Counter[str] = Counter()
    raw_top1_dec_counter: Counter[str] = Counter()
    examples = []

    diff_key = f"{head}->ALL"
    for sid, item in sorted(causal_dataset.items(), key=lambda pair: int(pair[0])):
        sample = sample_map[str(item["sentence_id"])]
        tokens = item["tokens"]
        diff_vector = item["diff_vectors"][diff_key]
        inc, dec = top_direction_tokens(diff_vector, tokens, sample, top_k=top_k)
        top1_inc_counter[inc[0]["structural_label"]] += 1
        top1_dec_counter[dec[0]["structural_label"]] += 1
        raw_top1_inc_counter[inc[0]["occurrence_label"]] += 1
        raw_top1_dec_counter[dec[0]["occurrence_label"]] += 1
        for ent in inc:
            topk_inc_counter[ent["structural_label"]] += 1
        for ent in dec:
            topk_dec_counter[ent["structural_label"]] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "sentence_id": item["sentence_id"],
                    "sentence_text": item["sentence_text"],
                    "increase_top1": inc[0],
                    "decrease_top1": dec[0],
                }
            )

    return {
        "head": head,
        "dataset_size": len(causal_dataset),
        "top1_increase_structural_counts": dict(top1_inc_counter.most_common()),
        "top1_decrease_structural_counts": dict(top1_dec_counter.most_common()),
        "topk_increase_structural_counts": dict(topk_inc_counter.most_common()),
        "topk_decrease_structural_counts": dict(topk_dec_counter.most_common()),
        "top1_increase_raw_counts": dict(raw_top1_inc_counter.most_common(20)),
        "top1_decrease_raw_counts": dict(raw_top1_dec_counter.most_common(20)),
        "examples": examples,
    }


def direction_choice(scores: dict[str, Any]) -> dict[str, Any]:
    inc = scores.get("causal_increase_f1")
    dec = scores.get("causal_decrease_f1")
    chosen = None
    if inc is not None and dec is not None:
        chosen = "increase" if inc >= dec else "decrease"
    return {
        "causal_f1": scores.get("causal_f1"),
        "causal_increase_f1": inc,
        "causal_decrease_f1": dec,
        "chosen_direction": chosen,
        "margin": None if inc is None or dec is None else abs(float(inc) - float(dec)),
        "direct_attention_f1": scores.get("direct_attention_f1"),
        "composite_score": scores.get("composite_score", scores.get("composite_f1")),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    validation_results = load_json(run_dir / "validation_results.json")
    test_results = load_json(run_dir / "test_results.json")
    initial_test_results = load_json(run_dir / "initial_test_results.json")
    direction_timeline = []
    for item in validation_results:
        direction_timeline.append(
            {
                "label": item.get("label"),
                "decision": item.get("decision"),
                **direction_choice(item.get("validation_scores", {})),
            }
        )
    return {
        "run_dir": str(run_dir),
        "validation_timeline": direction_timeline,
        "test_final": direction_choice(test_results.get("validation_scores", {})),
        "test_initial": direction_choice(initial_test_results.get("validation_scores", {})),
        "best_hypothesis": load_json(run_dir / "best_hypothesis.json"),
        "final_hypothesis": load_json(run_dir / "final_hypothesis.json"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize IOI Middle_Head increase/decrease behavior.")
    p.add_argument(
        "--results-root",
        default="/home/wangziran/eap_auto/results/ioi_0312",
        help="Base results directory containing path_patching and hypothesis subdirs.",
    )
    p.add_argument(
        "--heads",
        nargs="+",
        default=["7.9", "8.6", "8.10"],
        help="Head ids to summarize, e.g. 7.9 8.6 8.10",
    )
    p.add_argument(
        "--run-dirs",
        nargs="*",
        default=[
            "/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Middle_Head/7.9_rerun_20260408_0615",
            "/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Middle_Head/8.6_rerun_20260408_0615",
            "/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Middle_Head/8.10_rerun_20260408_0615",
        ],
        help="Optional rerun directories matching --heads order.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many top increase/decrease tokens to count from each diff vector.",
    )
    p.add_argument(
        "--output",
        default="/home/wangziran/eap_auto/results/ioi_0312/summary/middle_head_direction_analysis.json",
        help="Output JSON path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    standard_data = load_json(results_root / "path_patching" / "standard_ioi_data.json")

    output: dict[str, Any] = {
        "results_root": str(results_root),
        "heads": {},
    }

    run_dirs = [Path(p) for p in args.run_dirs]
    run_map = {}
    for run_dir in run_dirs:
        head = run_dir.name.split("_rerun_", 1)[0]
        run_map[head] = run_dir

    for head in args.heads:
        layer, h = head.split(".")
        causal_dataset_path = (
            results_root / "path_patching" / "Middle_Head" / f"{layer}_{h}" / f"causal_dataset_{layer}_{h}.json"
        )
        head_summary = {
            "causal_dataset_summary": summarize_causal_dataset(
                head=head,
                causal_dataset_path=causal_dataset_path,
                standard_data=standard_data,
                top_k=args.top_k,
            )
        }
        if head in run_map:
            head_summary["rerun_summary"] = summarize_run(run_map[head])
        output["heads"][head] = head_summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
