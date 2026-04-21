#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_NMH import (  # type: ignore
    initialize_openrouter as init_nmh_router,
    predict_causal_effects_for_sentences,
    format_examples_and_tokens,
    underscore_important_tokens,
    convert_predict_attention_to_json,
    compare_attention_f1,
    chunk_list,
    load_preprocessed_attention_dataset,
    load_preprocessed_causal_dataset,
    build_sentences_from_ids,
    build_causal_examples_from_ids,
    get_real_attention_pattern_for_sampling,
)
from auto_s_att import (  # type: ignore
    initialize_openrouter as init_middle_router,
    predict_attention_changes_for_sentences,
    predict_top_k_attenders_batch,
)


FAMILY_DIRS = [
    "Name_Mover_Head",
    "Negative_Name_Mover_Head",
    "SIH",
    "DTH",
]
MODES = ["all", "causal", "att"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate initial hypothesis on the same test set as final test_results.json.")
    p.add_argument(
        "--results-root",
        default="/home/wangziran/eap_auto/results/ioi_0304",
        help="Root containing hypothesis/...",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing initial_test_results.json.",
    )
    p.add_argument(
        "--families",
        default="",
        help="Comma-separated family dirs to run: Name_Mover_Head,Negative_Name_Mover_Head,SIH,DTH. Empty means all.",
    )
    return p.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _f1_pair(pred: List[str], truth: List[str]) -> float:
    ps, ts = set(pred or []), set(truth or [])
    if not ps and not ts:
        return 1.0
    if not ps or not ts:
        return 0.0
    inter = len(ps & ts)
    return 2 * inter / (len(ps) + len(ts)) if (len(ps) + len(ts)) > 0 else 0.0


def _composite(causal_f1: float, att_f1: float) -> float:
    return math.sqrt(causal_f1 * att_f1) if causal_f1 > 0 and att_f1 > 0 else 0.0


def _parse_head_from_run_dir(run_dir: Path) -> Tuple[int, int]:
    head_prefix = run_dir.name.split("_", 1)[0]
    layer, head = head_prefix.split(".")
    return int(layer), int(head)


def _build_sentence_dict_from_test_details(details: List[dict]) -> Dict[str, dict]:
    out = {}
    for d in details:
        sid = str(d.get("sentence_id", ""))
        txt = d.get("sentence_text", "")
        if sid and txt:
            out[sid] = {"sentence_text": txt}
    return out


async def eval_initial_for_nmh_family(run_dir: Path, family: str, hypothesis: str, test_data: dict) -> dict:
    layer, head = _parse_head_from_run_dir(run_dir)
    sender_head = (layer, head)
    details = test_data.get("validation_details", []) or []
    test_ids = [str(d.get("sentence_id", "")).strip() for d in details if str(d.get("sentence_id", "")).strip()]
    if not test_ids:
        return {}

    open_router = init_nmh_router()
    task_prefix = "NMH" if family == "Name_Mover_Head" else "NNMH"
    data_source_dir = run_dir.parents[2] / "path_patching" / f"{task_prefix}_{layer}_{head}"
    preprocessed_attention_path = data_source_dir / "preprocessed_attention_scores.json"
    preprocessed_causal_path = data_source_dir / f"causal_effects_{layer}_{head}.json"
    preprocessed_attention_data = load_preprocessed_attention_dataset(str(preprocessed_attention_path))
    preprocessed_causal_dataset = load_preprocessed_causal_dataset(str(preprocessed_causal_path))
    full_causal_dataset = preprocessed_causal_dataset
    sentences = build_sentences_from_ids(test_ids, preprocessed_attention_data)
    if not sentences:
        return {}
    causal_examples = build_causal_examples_from_ids(test_ids, full_causal_dataset, sentences)
    if not causal_examples:
        return {}

    # Causal (direction labels)
    causal_preds = await predict_causal_effects_for_sentences(
        open_router, hypothesis, sentences, sender_head, str(run_dir)
    )
    gt_map = {str(item.get("key", "")): str(item.get("direction", "")).lower() for item in causal_examples}
    correct = 0
    count = 0
    for sid, pred in causal_preds.items():
        gt = gt_map.get(str(sid), "")
        pd = str((pred or {}).get("direction", "")).lower()
        if gt:
            count += 1
            if pd == gt:
                correct += 1
    causal_f1 = (correct / count) if count > 0 else 0.0

    # Attention (top-2 token F1, same NMH scorer)
    real_attention_json = get_real_attention_pattern_for_sampling(sentences, preprocessed_attention_data)
    if not real_attention_json:
        return {}

    pred_attention_json = []
    for chunk in chunk_list(real_attention_json, 5):
        formatted = format_examples_and_tokens(chunk)
        highlighted = await underscore_important_tokens(
            open_router, layer, head, hypothesis, formatted, str(run_dir)
        )
        pred_attention_json.extend(convert_predict_attention_to_json(highlighted, chunk))
    att_f1, _ = compare_attention_f1(pred_attention_json, real_attention_json)

    return {
        "hypothesis": hypothesis,
        "validation_scores": {
            "causal_f1": float(causal_f1),
            "attention_f1": float(att_f1),
            "composite_score": float(_composite(causal_f1, att_f1)),
        },
        "source_test_file": str(run_dir / "test_results.json"),
    }


async def eval_initial_for_middle_family(run_dir: Path, hypothesis: str, test_data: dict) -> dict:
    layer, head = _parse_head_from_run_dir(run_dir)
    sender_head = (layer, head)
    details = test_data.get("validation_details", []) or []
    sentences = _build_sentence_dict_from_test_details(details)
    if not sentences:
        return {}

    open_router = init_middle_router()

    # causal token-level predictions
    causal_preds = {}
    for i in range(0, len(sentences), 5):
        batch_items = list(sentences.items())[i:i + 5]
        batch = {k: v for k, v in batch_items}
        batch_preds, _, _ = await predict_attention_changes_for_sentences(
            open_router, hypothesis, batch, sender_head, 1, str(run_dir)
        )
        causal_preds.update(batch_preds)

    # attention top-2 predictions
    att_preds = {}
    for i in range(0, len(sentences), 5):
        batch_items = list(sentences.items())[i:i + 5]
        batch = {k: v for k, v in batch_items}
        batch_pred = await predict_top_k_attenders_batch(
            open_router, hypothesis, batch, sender_head, 2, str(run_dir)
        )
        att_preds.update(batch_pred)

    # compute causal F1 (sentence-level inc/dec mean)
    c_scores = []
    for d in details:
        sid = str(d.get("sentence_id", ""))
        gt = d.get("causal_truth")
        pred = causal_preds.get(sid, {})
        if not sid or not isinstance(gt, dict):
            continue
        gt_inc = gt.get("increase", []) or []
        gt_dec = gt.get("decrease", []) or []
        pd_inc = pred.get("increase", []) or []
        pd_dec = pred.get("decrease", []) or []
        c_scores.append((_f1_pair(pd_inc, gt_inc) + _f1_pair(pd_dec, gt_dec)) / 2.0)
    causal_f1 = sum(c_scores) / len(c_scores) if c_scores else 0.0

    # compute attention F1 (global token-level)
    total_correct = 0
    total_pred = 0
    total_truth = 0
    for d in details:
        sid = str(d.get("sentence_id", ""))
        truth = set((d.get("attn_truth") or [])[:2])
        pred = set((att_preds.get(sid, {}) or {}).get("predicted_tokens", [])[:2])
        total_correct += len(pred & truth)
        total_pred += len(pred)
        total_truth += len(truth)
    precision = (total_correct / total_pred) if total_pred > 0 else 0.0
    recall = (total_correct / total_truth) if total_truth > 0 else 0.0
    att_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "hypothesis": hypothesis,
        "validation_scores": {
            "causal_f1": float(causal_f1),
            "attention_f1": float(att_f1),
            "composite_score": float(_composite(causal_f1, att_f1)),
        },
        "source_test_file": str(run_dir / "test_results.json"),
    }


async def main_async(args: argparse.Namespace) -> None:
    root = Path(args.results_root)
    hyp_root = root / "hypothesis"
    done = 0
    skipped = 0

    selected_families = set()
    if args.families.strip():
        selected_families = {x.strip() for x in args.families.split(",") if x.strip()}

    for family in FAMILY_DIRS:
        if selected_families and family not in selected_families:
            continue
        fam_dir = hyp_root / family
        if not fam_dir.exists():
            continue
        for run_dir in sorted([p for p in fam_dir.iterdir() if p.is_dir()]):
            if not any(run_dir.name.endswith(f"_{m}") for m in MODES):
                continue
            test_path = run_dir / "test_results.json"
            iter1_path = run_dir / "iteration_results" / "iteration_1.json"
            out_path = run_dir / "initial_test_results.json"
            if not test_path.exists() or not iter1_path.exists():
                skipped += 1
                continue
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            test_data = _load_json(test_path)
            iter1 = _load_json(iter1_path)
            initial_hyp = (
                iter1.get("hypothesis_before")
                or iter1.get("hypothesis")
                or ""
            )
            if not initial_hyp:
                skipped += 1
                continue

            try:
                if family in {"Name_Mover_Head", "Negative_Name_Mover_Head"}:
                    result = await eval_initial_for_nmh_family(run_dir, family, initial_hyp, test_data)
                else:
                    result = await eval_initial_for_middle_family(run_dir, initial_hyp, test_data)
                if not result:
                    skipped += 1
                    continue
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[ok] {out_path}")
                done += 1
            except Exception as e:  # noqa: BLE001
                print(f"[warn] failed {run_dir}: {e}")
                skipped += 1

    print(f"Done. written={done}, skipped={skipped}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
