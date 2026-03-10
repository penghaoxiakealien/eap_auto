#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def pick_best(entry_list):
    if not entry_list:
        return None
    def composite(item):
        scores = item.get("validation_scores", {})
        explicit = scores.get("composite_score")
        if explicit is None:
            explicit = scores.get("composite_f1")
        if explicit is not None:
            return float(explicit or 0.0)
        attn = float(scores.get("direct_attention_f1", 0.0) or 0.0)
        causal = float(scores.get("causal_f1", 0.0) or 0.0)
        return (attn * causal) ** 0.5

    return max(entry_list, key=composite)


def main():
    parser = argparse.ArgumentParser(description="Save the best hypothesis summary from an auto_s_att run.")
    parser.add_argument("--result_file", required=True, help="Path to final_result_round_X.json")
    parser.add_argument("--output_file", required=True, help="Path to write the summary JSON")
    args = parser.parse_args()

    result_path = Path(args.result_file)
    if not result_path.exists():
        raise FileNotFoundError(f"{result_path} not found")

    data = json.loads(result_path.read_text())
    candidates = data.get("all_hypotheses_validation_results", [])
    best_entry = pick_best(candidates)
    summary = {
        "head": data.get("head"),
        "typename": data.get("typename"),
        "result_file": str(result_path),
        "best_hypothesis": best_entry.get("hypothesis") if best_entry else None,
        "validation_scores": best_entry.get("validation_scores") if best_entry else {},
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ Saved best hypothesis summary to {out_path}")


if __name__ == "__main__":
    main()
