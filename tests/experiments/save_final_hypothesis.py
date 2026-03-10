#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save final hypothesis summary from test_results.json")
    parser.add_argument("--test_file", required=True, help="Path to test_results.json")
    parser.add_argument("--output_file", required=True, help="Path to write final_hypothesis.json")
    parser.add_argument("--head", default="", help="Optional head id")
    parser.add_argument("--typename", default="", help="Optional type name")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found")

    data = json.loads(test_path.read_text(encoding="utf-8"))
    hypothesis = data.get("hypothesis")
    if not isinstance(hypothesis, str):
        hypothesis = ""

    scores = data.get("validation_scores", {})
    summary = {
        "head": args.head,
        "typename": args.typename,
        "source_test_file": str(test_path),
        "final_hypothesis": hypothesis.strip(),
        "test_scores": scores if isinstance(scores, dict) else {},
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved final hypothesis to {out_path}")


if __name__ == "__main__":
    main()

