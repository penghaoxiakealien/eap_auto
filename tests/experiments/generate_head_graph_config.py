#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict


def generate_config(input_path: Path, output_path: Path):
    data = json.loads(input_path.read_text())

    groups = data.get("groups", [])
    edges = data.get("edges", [])
    head_details = data.get("heads", {})

    downstream_map = defaultdict(list)
    for edge in edges:
        src = edge["source"]
        dst = edge["target"]
        downstream_map[src].append({"target": dst, "count": edge.get("count", 0)})

    nodes = []
    for group in groups:
        node = {
            "id": group["id"],
            "signature": group.get("signature"),
            "classification": group.get("classification"),
            "heads": group.get("heads", []),
            "layers": group.get("layers", []),
            "summary": group.get("summary", {}),
            "downstreams": downstream_map.get(group["id"], []),
        }
        nodes.append(node)

    config = {
        "meta": data.get("meta", {}),
        "nodes": nodes,
        "heads": head_details,
        "sinks": data.get("sinks", []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"✅ Head graph config saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate simplified head graph config.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/results/group_graph_top60_thr75_logits_pruned_thr0001.json"),
        help="Path to the original group graph JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/results/head_graph_config.json"),
        help="Output path for the simplified config.",
    )
    args = parser.parse_args()
    generate_config(args.input, args.output)


if __name__ == "__main__":
    main()
