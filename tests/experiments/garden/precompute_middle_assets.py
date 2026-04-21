#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch precompute attention files and sender->receiver diff datasets for garden middle heads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path("/home/wangziran/eap_auto")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute attention and diff datasets for garden.")
    p.add_argument("--group-graph-json", type=Path, required=True, help="group_graph_thr75_dagish.json")
    p.add_argument("--sim-queue-json", type=Path, help="sim_middle_queue.json (optional)")
    p.add_argument("--results-root", type=Path, required=True, help="Results root for outputs.")
    p.add_argument("--standard-json", type=Path, required=True, help="standard_garden_data.json")
    p.add_argument("--mode", choices=("attention", "diff", "both"), default="both")
    p.add_argument("--attention-batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--attention-position", type=str, default="end")
    p.add_argument("--strict-attention-align", action="store_true")
    p.add_argument("--include-abc", action="store_true", help="Also precompute A->B->C diffs.")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--log-dir", type=Path, help="Optional log dir for per-task logs.")
    p.add_argument("--cuda-visible-devices", type=str, default="7")
    return p.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def parse_head(head: str) -> Tuple[int, int]:
    layer, h = head.split(".", 1)
    return int(layer), int(h)


def head_to_str(head: Tuple[int, int]) -> str:
    return f"{head[0]}.{head[1]}"


def head_dir_name(head: Tuple[int, int]) -> str:
    return f"{head[0]}_{head[1]}"


def load_group_graph(path: Path) -> Dict[str, object]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def load_sim_queue(path: Path | None) -> Tuple[List[str], List[str]]:
    if not path:
        return [], []
    data = load_json(path)
    if not isinstance(data, dict):
        return [], []
    ran = [str(x) for x in data.get("ran", []) if isinstance(x, str)]
    pending = [str(x) for x in data.get("pending", []) if isinstance(x, str)]
    return ran, pending


def build_tasks(group_graph: Dict[str, object], sim_queue: Path | None) -> Tuple[List[Tuple[str, str]], List[str]]:
    groups = group_graph.get("groups", []) or []
    edges = group_graph.get("edges", []) or []
    group_by_id = {g.get("id"): g for g in groups if isinstance(g, dict) and isinstance(g.get("id"), str)}

    ran, pending = load_sim_queue(sim_queue)
    if ran or pending:
        all_tasks = ran + pending
        tasks: List[Tuple[str, str]] = []
        for item in all_tasks:
            if "->" not in item:
                continue
            sender, child = item.split("->", 1)
            tasks.append((sender.strip(), child.strip()))
    else:
        tasks = []
        downstreams: Dict[str, List[str]] = {}
        for e in edges:
            src = e.get("source") or e.get("src")
            dst = e.get("target") or e.get("dst")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if dst == "logits":
                continue
            downstreams.setdefault(src, []).append(dst)
        for src_id, child_ids in downstreams.items():
            src_group = group_by_id.get(src_id, {})
            sender_heads = [h for h in src_group.get("heads", []) or [] if isinstance(h, str)]
            for sender in sender_heads:
                for child in child_ids:
                    tasks.append((sender, child))

    sender_heads = sorted({sender for sender, _ in tasks})
    return tasks, sender_heads


def resolve_receivers(group_graph: Dict[str, object], child_id: str) -> List[str]:
    for g in group_graph.get("groups", []) or []:
        if g.get("id") == child_id:
            return [h for h in g.get("heads", []) or [] if isinstance(h, str)]
    return []


def iter_downstreams(group_graph: Dict[str, object], node_id: str) -> List[str]:
    for g in group_graph.get("groups", []) or []:
        if g.get("id") != node_id:
            continue
        downstreams = [d.get("target") for d in g.get("downstreams", []) or [] if isinstance(d, dict)]
        if downstreams:
            return downstreams
        break
    targets = []
    for e in group_graph.get("edges", []) or []:
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if src == node_id and isinstance(dst, str):
            targets.append(dst)
    return targets


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_preprocessed_sampling(raw_path: Path, out_path: Path) -> None:
    data = load_json(raw_path)
    if not isinstance(data, list):
        raise ValueError(f"{raw_path} is not a list")
    with out_path.open("w", encoding="utf-8") as out_f:
        for item in data:
            out_f.write(
                json.dumps(
                    {
                        "sentence_id": str(item.get("sample_id", "")),
                        "original_sentence": item["sentence_text"],
                        "indirect_object": item.get("io_token", ""),
                        "number_of_important_tokens": len(item.get("top_attended_tokens", [])),
                        "attention_scores": [
                            {
                                "token": tok.get("token", "").strip(),
                                "position": tok.get("position", -1),
                                "score": tok.get("score", 0.0),
                            }
                            for tok in item.get("top_attended_tokens", [])
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_attention_ground_truth(preprocessed_jsonl: Path, out_path: Path) -> None:
    records = []
    with preprocessed_jsonl.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = item.get("sentence_id") or str(idx)
            records.append(
                {
                    "key": key,
                    "original_sentence": item["original_sentence"],
                    "attention_scores": item["attention_scores"],
                }
            )
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_cmd(cmd: Sequence[str], env: Dict[str, str], log_path: Path | None = None) -> int:
    if log_path:
        ensure_dir(log_path.parent)
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=log_f, stderr=log_f)
            return await proc.wait()
    proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    return await proc.wait()


async def run_with_semaphore(
    sem: asyncio.Semaphore,
    cmd: Sequence[str],
    env: Dict[str, str],
    label: str,
    log_path: Path | None,
) -> None:
    async with sem:
        print(f"[RUN] {label}")
        code = await run_cmd(cmd, env, log_path)
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"[{status}] {label}")


def attention_outputs(att_dir: Path) -> Tuple[Path, Path, Path, Path]:
    raw = att_dir / "raw_attention.json"
    preprocessed_jsonl = att_dir / "preprocessed_for_sampling.jsonl"
    preprocessed_attention = att_dir / "preprocessed_attention_scores.json"
    attention_gt = att_dir / "attention_scores_ground_truth.jsonl"
    return raw, preprocessed_jsonl, preprocessed_attention, attention_gt


async def precompute_attention_for_head(
    head: str,
    args: argparse.Namespace,
    env: Dict[str, str],
    sem: asyncio.Semaphore,
) -> None:
    layer, h = parse_head(head)
    att_dir = args.results_root / "precompute" / "attention" / head_dir_name((layer, h))
    ensure_dir(att_dir)
    raw, preprocessed_jsonl, preprocessed_attention, attention_gt = attention_outputs(att_dir)

    if args.skip_existing and preprocessed_attention.exists() and attention_gt.exists():
        return

    cmd = [
        "python",
        str(REPO_ROOT / "tests/experiments/precompute_attention_scores_garden.py"),
        "--standard-json",
        str(args.standard_json),
        "--output-file",
        str(raw),
        "--head",
        head,
        "--batch-size",
        str(args.attention_batch_size),
    ]
    log_path = args.log_dir / f"attention_{head}.log" if args.log_dir else None
    await run_with_semaphore(sem, cmd, env, f"attention {head}", log_path)

    write_preprocessed_sampling(raw, preprocessed_jsonl)

    if args.strict_attention_align:
        cmd = [
            "python",
            str(REPO_ROOT / "tests/experiments/convert_attention_tokens_to_words.py"),
            "--input-jsonl",
            str(preprocessed_jsonl),
            "--output-preprocessed",
            str(preprocessed_attention),
            "--output-ground-truth",
            str(attention_gt),
            "--top-k",
            str(args.top_k),
        ]
        log_path = args.log_dir / f"attention_align_{head}.log" if args.log_dir else None
        await run_with_semaphore(sem, cmd, env, f"align {head}", log_path)
    else:
        cmd = [
            "python",
            str(REPO_ROOT / "tests/experiments/preprocess_attention_scores.py"),
            "--input",
            str(preprocessed_jsonl),
            "--output",
            str(preprocessed_attention),
            "--top_k",
            str(args.top_k),
        ]
        log_path = args.log_dir / f"attention_preprocess_{head}.log" if args.log_dir else None
        await run_with_semaphore(sem, cmd, env, f"preprocess {head}", log_path)
        write_attention_ground_truth(preprocessed_jsonl, attention_gt)


async def precompute_diffs_for_task(
    sender: str,
    child_id: str,
    receivers: List[str],
    args: argparse.Namespace,
    env: Dict[str, str],
    sem: asyncio.Semaphore,
) -> None:
    if not receivers:
        return
    layer, h = parse_head(sender)
    out_dir = args.results_root / "precompute" / "diffs" / head_dir_name((layer, h)) / child_id
    ensure_dir(out_dir)
    out_file = out_dir / "causal_dataset.json"
    if args.skip_existing and out_file.exists():
        return
    cmd = [
        "python",
        str(REPO_ROOT / "tests/experiments/precompute_middle_head_garden.py"),
        "--sender_head",
        sender,
        "--receiver_heads",
        ",".join(receivers),
        "--standard-json",
        str(args.standard_json),
        "--output_file",
        str(out_file),
        "--attention_position",
        str(args.attention_position),
    ]
    safe_child = child_id.replace("/", "_")
    log_path = args.log_dir / f"diff_{sender}_to_{safe_child}.log" if args.log_dir else None
    await run_with_semaphore(sem, cmd, env, f"diff {sender}->{child_id}", log_path)


async def precompute_abc_for_task(
    sender: str,
    child_id: str,
    intermediate_heads: List[str],
    target_head: str,
    args: argparse.Namespace,
    env: Dict[str, str],
    sem: asyncio.Semaphore,
) -> None:
    if not intermediate_heads:
        return
    layer, h = parse_head(sender)
    safe_child = child_id.replace("/", "_")
    safe_target = target_head.replace(".", "_")
    out_dir = args.results_root / "precompute" / "diffs_abc" / head_dir_name((layer, h)) / f"{safe_child}_to_{safe_target}"
    ensure_dir(out_dir)
    out_file = out_dir / "causal_dataset.json"
    if args.skip_existing and out_file.exists():
        return
    cmd = [
        "python",
        str(REPO_ROOT / "tests/experiments/precompute_sender_target_via_intermediates.py"),
        "--sender_head",
        sender,
        "--intermediate_heads",
        ",".join(intermediate_heads),
        "--target_head",
        target_head,
        "--standard-json",
        str(args.standard_json),
        "--output_file",
        str(out_file),
        "--attention_position",
        str(args.attention_position),
    ]
    log_path = args.log_dir / f"diff_abc_{sender}_via_{safe_child}_to_{safe_target}.log" if args.log_dir else None
    await run_with_semaphore(sem, cmd, env, f"diff A->B->C {sender}->{child_id}->{target_head}", log_path)


async def main_async() -> None:
    args = parse_args()
    group_graph = load_group_graph(args.group_graph_json)
    tasks, sender_heads = build_tasks(group_graph, args.sim_queue_json)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    sem = asyncio.Semaphore(max(1, args.max_parallel))

    if args.mode in ("attention", "both"):
        print(f"Precomputing attention for {len(sender_heads)} heads...")
        await asyncio.gather(
            *[
                precompute_attention_for_head(head, args, env, sem)
                for head in sender_heads
            ]
        )

    if args.mode in ("diff", "both"):
        print(f"Precomputing diffs for {len(tasks)} sender->child tasks...")
        diff_tasks = []
        for sender, child_id in tasks:
            receivers = resolve_receivers(group_graph, child_id)
            diff_tasks.append(precompute_diffs_for_task(sender, child_id, receivers, args, env, sem))
        await asyncio.gather(*diff_tasks)

        if args.include_abc:
            abc_tasks = []
            for sender, child_id in tasks:
                intermediate_heads = resolve_receivers(group_graph, child_id)
                if not intermediate_heads:
                    continue
                for target_id in iter_downstreams(group_graph, child_id):
                    if not isinstance(target_id, str):
                        continue
                    for target_head in resolve_receivers(group_graph, target_id):
                        abc_tasks.append(
                            precompute_abc_for_task(
                                sender,
                                child_id,
                                intermediate_heads,
                                target_head,
                                args,
                                env,
                                sem,
                            )
                        )
            if abc_tasks:
                print(f"Precomputing A->B->C diffs for {len(abc_tasks)} tasks...")
                await asyncio.gather(*abc_tasks)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
