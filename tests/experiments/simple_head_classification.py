#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将注意力模式签名与路径插补分类汇总为简明 JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def load_path_patch(path: Path) -> Dict[str, Any]:
	data = json.loads(path.read_text())
	if "metrics" not in data or "classification" not in data:
		raise ValueError("path patch JSON 缺少 metrics 或 classification 字段")
	return data


def invert_pattern_groups(pattern_groups: Dict[str, List[str]]) -> Dict[str, str]:
	head_to_signature: Dict[str, str] = {}
	for signature, heads in pattern_groups.items():
		for head in heads:
			head_to_signature[head] = signature
	return head_to_signature


def combine_groups(
	metrics: Dict[str, Dict[str, float]],
	classification: Dict[str, List[str]],
	head_to_signature: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
	grouped: Dict[str, Dict[str, Any]] = {}
	heads_payload: Dict[str, Dict[str, Any]] = {}

	class_map = {head: label for label, items in classification.items() for head in items}

	for head, stat in metrics.items():
		signature = head_to_signature.get(head, "UNASSIGNED")
		label = class_map.get(head, "unknown")
		key = f"{signature}|{label}"
		bucket = grouped.setdefault(
			key,
			{
				"signature": signature,
				"classification": label,
				"heads": [],
				"metrics": {},
			},
		)
		bucket["heads"].append(head)
		bucket["metrics"][head] = {
			"metric": stat.get("metric"),
			"delta_logit_diff": stat.get("delta_logit_diff"),
			"patched_logit_diff": stat.get("patched_logit_diff"),
		}
		heads_payload[head] = {
			"signature": signature,
			"classification": label,
			"metric": stat.get("metric"),
			"delta_logit_diff": stat.get("delta_logit_diff"),
			"patched_logit_diff": stat.get("patched_logit_diff"),
		}

	for bucket in grouped.values():
		metrics_vals = [m.get("metric") for m in bucket["metrics"].values() if m.get("metric") is not None]
		delta_vals = [m.get("delta_logit_diff") for m in bucket["metrics"].values() if m.get("delta_logit_diff") is not None]
		bucket["summary"] = {
			"count": len(bucket["heads"]),
			"mean_metric": mean(metrics_vals) if metrics_vals else None,
			"mean_delta_logit_diff": mean(delta_vals) if delta_vals else None,
		}
		bucket["heads"].sort(key=lambda name: tuple(int(x) for x in name.split(".")))

	return grouped, heads_payload


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="输出按模式与路径插补分类的注意力头分组")
	parser.add_argument("--path-patch-json", type=Path, required=True, help="path_patch_dominant_groups.py 生成的 JSON")
	parser.add_argument("--output", type=Path, required=True, help="输出 JSON 文件路径")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data = load_path_patch(args.path_patch_json)

	metrics = data.get("metrics", {})
	classification = data.get("classification", {})
	pattern_groups = data.get("pattern_groups", {})
	head_to_signature = invert_pattern_groups(pattern_groups)

	grouped, heads_payload = combine_groups(metrics, classification, head_to_signature)

	output_payload = {
		"meta": data.get("meta", {}),
		"groups": sorted(grouped.values(), key=lambda item: (item["classification"], item["signature"])),
		"heads": heads_payload,
	}

	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))
	print(f"写入简化分类: {args.output}")

