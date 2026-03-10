#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def pick_best(entries):
    def key_fn(item):
        scores = item.get("scores", {})
        attn = scores.get("attention_f1", 0.0) or 0.0
        causal = scores.get("causal_f1", 0.0) or 0.0
        return (attn * causal) ** 0.5

    return max(entries, key=key_fn)


def main():
    parser = argparse.ArgumentParser(description="Summarize legacy iteration logs (e.g., 10.7_1.json).")
    parser.add_argument("--input_file", required=True, help="Path to iteration list JSON (e.g., 10.7_1.json)")
    parser.add_argument("--output_file", required=True, help="Where to write best summary JSON")
    parser.add_argument("--head", default="", help="Optional head identifier (e.g., 10.7)")
    parser.add_argument("--typename", default="", help="Optional typename label")
    args = parser.parse_args()

    path = Path(args.input_file)
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} does not contain iteration list")

    best = pick_best(entries)
    scores = best.get("scores", {})
    summary = {
        "head": args.head or path.stem,
        "typename": args.typename or "",
        "iteration": best.get("iteration"),
        "best_hypothesis": best.get("hypothesis"),
        "validation_scores": scores,
        "source_file": str(path),
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ Saved legacy best hypothesis to {out_path}")


if __name__ == "__main__":
    main()
