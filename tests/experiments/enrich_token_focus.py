#!/usr/bin/env python3
"""Augment existing batch attention results with token concentration statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from run_group_attention_batch import annotate_token_concentration, sanitize_label


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "tests/results/batch_attention"
DEFAULT_OUTPUT = REPO_ROOT / "tests/results/token_focus/batch_attention"
DEFAULT_REPORT = REPO_ROOT / "tests/results/token_focus/token_focus_overview.json"
DEFAULT_GRAPH = REPO_ROOT / "tests/results/group_graph_top60_thr75_logits_filtered.json"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate token concentration metrics")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH, help="Grouped graph JSON for label->group mapping")
    return parser.parse_args()


def build_group_maps(graph_path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    payload = json.loads(graph_path.read_text())
    groups = payload.get("groups", [])
    sanitized_to_group = {
        sanitize_label(group["id"]): group["id"]
        for group in groups
        if isinstance(group, dict) and group.get("id")
    }
    head_to_group: Dict[str, str] = {}
    group_heads: Dict[str, List[str]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = group.get("id")
        heads = [str(head) for head in group.get("heads", [])]
        if gid:
            group_heads[gid] = sorted(
                heads,
                key=lambda name: tuple(int(x) for x in name.split(".")) if "." in name else (name,),
            )
        for head in heads:
            head_to_group[str(head)] = gid
    return sanitized_to_group, head_to_group, group_heads


def update_summary(
    summary_path: Path,
    output_summary: Path,
    head_dir_in: Path,
    head_dir_out: Path,
    records: List[Dict[str, object]],
    sanitized_to_group: Dict[str, str],
    head_to_group: Dict[str, str],
    group_heads: Dict[str, List[str]],
) -> None:
    data = json.loads(summary_path.read_text())
    target_head = data.get("target_head", "")

    for scenario in data.get("scenarios", []):
        annotate_token_concentration(scenario)
        label = scenario.get("label")
        if not label:
            continue

        src_token, dst_token = label, ""
        if "__to__" in label:
            src_token, _, dst_token = label.partition("__to__")

        source_group = sanitized_to_group.get(src_token, src_token)
        target_group = head_to_group.get(dst_token.replace("_", "."), head_to_group.get(str(target_head), target_head))

        source_heads = group_heads.get(source_group)
        target_heads = group_heads.get(target_group)

        def format_heads(heads: Optional[List[str]]) -> Optional[str]:
            if not heads:
                return None
            return "[" + ", ".join(heads) + "]"

        scenario_input = head_dir_in / f"{label}.json"
        if scenario_input.exists():
            payload = json.loads(scenario_input.read_text())
            payload["scenario"] = scenario
            head_dir_out.mkdir(parents=True, exist_ok=True)
            scenario_output = head_dir_out / f"{label}.json"
            scenario_output.write_text(json.dumps(payload, indent=2))

        records.append(
            {
                "target_head": target_head,
                "target_group": target_group,
                "target_heads": target_heads,
                "source_group": source_group,
                "source_heads": source_heads,
                "edge": {
                    "from": source_group,
                    "to": target_group,
                    "from_heads": source_heads,
                    "to_heads": target_heads,
                    "description": f"{format_heads(source_heads) or source_group} -> {format_heads(target_heads) or target_group}",
                },
                "label": label,
                "token_concentration": scenario.get("token_concentration", {}),
                "mean_top_increases": scenario.get("mean_top_increases", []),
                "mean_top_decreases": scenario.get("mean_top_decreases", []),
            }
        )

    output_summary.write_text(json.dumps(data, indent=2))


def main() -> None:
    args = parse_args()

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    report_path = resolve(args.report)
    graph_path = resolve(args.graph_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    sanitized_to_group, head_to_group, group_heads = build_group_maps(graph_path)

    for summary_path in sorted(input_dir.glob("summary_*.json")):
        head_token = summary_path.stem.replace("summary_", "")
        head_dir_in = input_dir / head_token
        head_dir_out = output_dir / head_token
        head_dir_out.mkdir(parents=True, exist_ok=True)
        update_summary(
            summary_path,
            output_dir / summary_path.name,
            head_dir_in,
            head_dir_out,
            records,
            sanitized_to_group,
            head_to_group,
            group_heads,
        )

    report_payload = {
        "records": records,
        "total_records": len(records),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2))
    print(f"Wrote augmented summaries to {output_dir}")
    print(f"Token focus report: {report_path}")


if __name__ == "__main__":
    main()
