#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BOUNDARY_TOKENS = {"when", "while", "after", "as"}


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
        labels[pos] = f"{base}_{counts[base]}" if base else f"EMPTY_{counts[base]}"
    return labels


def structural_label(
    pos: int,
    token: str,
    sample: dict[str, Any],
    occurrence_labels: dict[int, str],
) -> str:
    tok = normalize_token(token)
    tok_lower = tok.lower()
    word_idx = sample.get("word_idx", {})

    if pos == 0 and tok_lower in BOUNDARY_TOKENS:
        return "SUBORD"
    if pos == word_idx.get("subj"):
        return "SUBJ"
    if pos == word_idx.get("verb"):
        return "VERB"
    if pos == word_idx.get("obj_head"):
        return "OBJ_HEAD"
    if pos == word_idx.get("rel_pron"):
        return "REL_PRON"
    if pos == word_idx.get("rel_verb"):
        return "REL_VERB"
    if tok_lower == "the":
        return occurrence_labels.get(pos, "the")
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
    sender_head: str,
    causal_dataset_path: Path,
    standard_data: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    causal_dataset = load_json(causal_dataset_path)
    sample_map = {str(idx): item for idx, item in enumerate(standard_data)}

    top1_inc_counter: Counter[str] = Counter()
    top1_dec_counter: Counter[str] = Counter()
    topk_inc_counter: Counter[str] = Counter()
    topk_dec_counter: Counter[str] = Counter()
    raw_top1_inc_counter: Counter[str] = Counter()
    raw_top1_dec_counter: Counter[str] = Counter()
    examples = []

    diff_key = f"{sender_head}->ALL"
    for sid, item in sorted(causal_dataset.items(), key=lambda pair: int(pair[0])):
        sample = sample_map.get(str(item.get("sentence_id", sid)))
        if not sample:
            continue
        tokens = item["tokens"]
        diff_vector = item["diff_vectors"][diff_key]
        inc, dec = top_direction_tokens(diff_vector, tokens, sample, top_k=top_k)
        if not inc or not dec:
            continue
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
        "sender_head": sender_head,
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
    margin = None

    if inc is not None and dec is not None:
        chosen = "increase" if inc >= dec else "decrease"
        margin = abs(float(inc) - float(dec))
    elif scores.get("causal_direction"):
        chosen = scores.get("causal_direction")
        margin = 0.0

    return {
        "causal_f1": scores.get("causal_f1"),
        "causal_increase_f1": inc,
        "causal_decrease_f1": dec,
        "causal_direction": scores.get("causal_direction"),
        "causal_direction_f1": scores.get("causal_direction_f1"),
        "chosen_direction": chosen,
        "margin": margin,
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


def infer_sender_head(run_dir: Path) -> str:
    meta_path = run_dir / "receiver_group_meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        sender = str(meta.get("sender_head", "")).strip()
        if sender:
            return sender

    name = run_dir.name
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    raise ValueError(f"Cannot infer sender head from run dir: {run_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Garden Middle_Head increase/decrease behavior.")
    p.add_argument(
        "--standard-json",
        default="/home/wangziran/eap_auto/results/garden_0131/standard_garden_data.json",
        help="Path to standard_garden_data.json.",
    )
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more Garden middle run directories to summarize.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many top increase/decrease tokens to count from each diff vector.",
    )
    p.add_argument(
        "--output",
        default="/home/wangziran/eap_auto/results/garden_0412/summary/middle_head_direction_analysis.json",
        help="Output JSON path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    standard_data = load_json(Path(args.standard_json))

    output: dict[str, Any] = {
        "standard_json": str(args.standard_json),
        "runs": {},
    }

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        sender_head = infer_sender_head(run_dir)
        run_summary = {
            "sender_head": sender_head,
            "causal_dataset_summary": summarize_causal_dataset(
                sender_head=sender_head,
                causal_dataset_path=run_dir / "causal_dataset.json",
                standard_data=standard_data,
                top_k=args.top_k,
            ),
        }
        if (run_dir / "validation_results.json").exists() and (run_dir / "test_results.json").exists():
            run_summary["rerun_summary"] = summarize_run(run_dir)
        output["runs"][run_dir.name] = run_summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
