#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run garden terminal explanations for heads that connect to logits in a group graph."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run garden auto_terminal for heads connected to logits in group graph."
    )
    p.add_argument("--group-graph-json", type=Path, required=True, help="Group graph JSON (dagish).")
    p.add_argument("--results-root", type=Path, required=True, help="Results root for outputs.")
    p.add_argument(
        "--data-source-base",
        type=Path,
        help="Base dir containing per-head data_source_dir (default: results-root/path_patching/Terminal).",
    )
    p.add_argument(
        "--output-base",
        type=Path,
        help="Base dir for hypothesis outputs (default: results-root/hypothesis/Terminal).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        help="Optional directory to write per-head stdout/stderr logs.",
    )
    p.add_argument(
        "--use-log-dir",
        action="store_true",
        help="Enable writing per-head logs when --log-dir is provided.",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of concurrent runs.",
    )
    p.add_argument("--rounds", type=int, default=5, help="LLM refinement rounds.")
    p.add_argument("--min-layer", type=int, default=0, help="Minimum layer (inclusive).")
    p.add_argument("--max-layer", type=int, default=11, help="Maximum layer (inclusive).")
    p.add_argument(
        "--auto-generate",
        action="store_true",
        help="Auto-generate data_source_dir by running run_terminal_garden.sh when missing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    return p.parse_args()


def load_group_graph(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def collect_logits_heads(group_graph: Dict[str, object], min_layer: int, max_layer: int) -> List[str]:
    groups = group_graph.get("groups", [])
    group_by_id = {
        g.get("id"): g
        for g in groups
        if isinstance(g, dict) and isinstance(g.get("id"), str)
    }
    logits_sources: Set[str] = set()
    for e in group_graph.get("edges", []) or []:
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if dst == "logits" and isinstance(src, str):
            logits_sources.add(src)

    heads: Set[str] = set()
    for gid in sorted(logits_sources):
        grp = group_by_id.get(gid, {})
        for h in grp.get("heads", []) or []:
            if isinstance(h, str):
                try:
                    layer = int(h.split(".", 1)[0])
                except ValueError:
                    continue
                if min_layer <= layer <= max_layer:
                    heads.add(h)
    return sorted(heads, key=lambda x: tuple(int(p) for p in x.split(".")))


def build_command(
    head: str,
    data_source_dir: Path,
    output_dir: Path,
    rounds: int,
) -> List[str]:
    layer_str, head_str = head.split(".")
    return [
        "python",
        "tests/experiments/garden/auto_terminal.py",
        "--layer",
        layer_str,
        "--head",
        head_str,
        "--data_source_dir",
        str(data_source_dir),
        "--output_dir",
        str(output_dir),
        "--rounds",
        str(rounds),
    ]


def build_generate_command(
    head: str,
    results_root: Path,
) -> List[str]:
    layer_str, head_str = head.split(".")
    return [
        "bash",
        "tests/experiments/run_terminal_garden.sh",
        "--layer",
        layer_str,
        "--head",
        head_str,
        "--results-root",
        str(results_root),
    ]


def has_required_files(data_source_dir: Path) -> bool:
    required = [
        "preprocessed_for_sampling.jsonl",
        "preprocessed_attention_scores.json",
        "heads_direct_effect_on_logit_difference.json",
    ]
    return all((data_source_dir / name).exists() for name in required)


def main() -> None:
    args = parse_args()
    data = load_group_graph(args.group_graph_json)
    heads = collect_logits_heads(data, args.min_layer, args.max_layer)

    if not heads:
        print("未找到与 logits 相连的 heads。")
        return

    results_root = args.results_root
    data_source_base = args.data_source_base or (results_root / "path_patching" / "Terminal")
    output_base = args.output_base or (results_root / "hypothesis" / "Terminal")

    print(f"Found {len(heads)} heads connected to logits.")
    log_dir = args.log_dir if args.use_log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    max_parallel = max(1, int(args.max_parallel))
    running = []

    def enqueue(cmd, head, log_path, log_f):
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        running.append((head, proc, log_f, log_path))
        if len(running) >= max_parallel:
            head_done, proc_done, log_f_done, log_path_done = running.pop(0)
            ret = proc_done.wait()
            if log_f_done:
                log_f_done.close()
            if log_path_done:
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[{status}] {head_done} -> {log_path_done}")
    for head in heads:
        data_source_dir = data_source_base / head.replace(".", "_")
        output_dir = output_base / head.replace(".", "_")
        cmd = build_command(head, data_source_dir, output_dir, args.rounds)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        if not data_source_dir.exists() or not has_required_files(data_source_dir):
            if not args.auto_generate:
                missing = "不存在" if not data_source_dir.exists() else "关键文件缺失"
                print(f"⚠️ data_source_dir {missing}，跳过: {data_source_dir}")
                continue
            gen_cmd = build_generate_command(head, results_root)
            print(" ".join(gen_cmd))
            gen_log = None
            gen_f = None
            if log_dir:
                gen_log = log_dir / f"{head.replace('.', '_')}_generate.log"
                gen_f = gen_log.open("w", encoding="utf-8")
                gen_f.write(f"[CMD] {' '.join(gen_cmd)}\n\n")
            if max_parallel == 1:
                ret = subprocess.run(gen_cmd, stdout=gen_f, stderr=subprocess.STDOUT, check=False).returncode
                if gen_f:
                    gen_f.close()
                if gen_log:
                    status = "OK" if ret == 0 else f"FAIL({ret})"
                    print(f"[{status}] {head} -> {gen_log}")
            else:
                enqueue(gen_cmd, head, gen_log, gen_f)
            # run_terminal_garden.sh already runs auto_terminal; skip duplicate run
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = None
        log_f = None
        if log_dir:
            log_path = log_dir / f"{head.replace('.', '_')}.log"
            log_f = log_path.open("w", encoding="utf-8")
            log_f.write(f"[CMD] {' '.join(cmd)}\n\n")

        if max_parallel == 1:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=False)
            if log_f:
                log_f.close()
            if log_path:
                status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
                print(f"[{status}] {head} -> {log_path}")
            continue

        enqueue(cmd, head, log_path, log_f)

    for head_done, proc_done, log_f_done, log_path_done in running:
        ret = proc_done.wait()
        if log_f_done:
            log_f_done.close()
        if log_path_done:
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"[{status}] {head_done} -> {log_path_done}")


if __name__ == "__main__":
    main()
