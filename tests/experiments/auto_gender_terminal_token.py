#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal head auto-explanation for agr_gender, IOI-style causal evaluation.

This script mirrors the IOI "token-level causal" idea:
  - causal_f1: predict which sentence token's logit increases most (<< >>) and decreases most ([[ ]])
              after corrupting the sender head (patch sender z from corrupted prompt into clean prompt).
  - direct_attention_f1: predict which tokens the head attends to at a chosen query position
                         (top-k by attention weights), by marking tokens in the sentence.

IMPORTANT: We do NOT provide answer candidates to the LLM. It must pick from the sentence itself.
We enforce a strict output protocol (IOI-style): a [REASONING] block and a single-line [PREDICTION].
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

sys.path.append("/home/wangziran/eap_auto/")
from api import OpenRouter


def initialize_openrouter(model: str = "claude-sonnet-4-20250514-thinking") -> OpenRouter:
    api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO"
    return OpenRouter(model=model, api_key=api_key)


def normalize_token(token: str) -> str:
    s = (token or "").strip().lower()
    return re.sub(r"_\d+$", "", s)


def combined_score(scores: Dict[str, float]) -> float:
    att = float(scores.get("direct_attention_f1") or scores.get("attention_f1") or 0.0)
    causal = float(scores.get("causal_f1") or 0.0)
    if att <= 0.0 or causal <= 0.0:
        return 0.0
    return math.sqrt(att * causal)


def parse_prediction_blocks(text: str) -> str:
    """
    Return the single-line sentence from [PREDICTION] if present, else fall back to last non-empty line.
    """
    t = (text or "").strip()
    m = re.search(r"\[PREDICTION\]\s*(.*)", t, re.DOTALL | re.IGNORECASE)
    if m:
        line = m.group(1).strip().strip("`")
        # prediction should be a single line; pick last non-empty line if multi-line
        lines = [ln.strip().strip("`") for ln in line.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    lines = [ln.strip().strip("`") for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _get_words(sentence: str) -> List[str]:
    # Word-level tokenization used for mapping duplicate occurrences (e.g., Paul_1 / Paul_2).
    # This must NOT be over-escaped; we want regex tokens like \w and \s to work.
    return re.findall(r"\w+|[^\w\s]", sentence)


def _get_suffixed_word_map(sentence: str) -> Tuple[List[str], Dict[int, str]]:
    words = _get_words(sentence)
    counts: Dict[str, int] = {}
    for w in words:
        counts[normalize_token(w)] = counts.get(normalize_token(w), 0) + 1
    running: Dict[str, int] = {}
    out: Dict[int, str] = {}
    for i, w in enumerate(words):
        n = normalize_token(w)
        if counts.get(n, 0) > 1:
            running[n] = running.get(n, 0) + 1
            out[i] = f"{w.strip()}_{running[n]}"
        else:
            out[i] = w.strip()
    return words, out


def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str) -> Tuple[List[str], List[str]]:
    """
    Parse << >> as increase, [[ ]] as decrease. Map marked tokens back to suffixed form.
    Supports punctuation tokens.
    """
    words, suffixed_map = _get_suffixed_word_map(original_sentence_text)

    def pick_suffix(raw_token: str) -> str:
        n = normalize_token(raw_token)
        for i, w in enumerate(words):
            if normalize_token(w) == n:
                return suffixed_map[i]
        return raw_token.strip()

    inc, dec = [], []
    for chunk in re.findall(r"<<(.*?)>>", marked_sentence):
        m = re.search(r"\w+|[^\w\s]", chunk)
        if m:
            inc.append(pick_suffix(m.group(0)))
    for chunk in re.findall(r"\[\[(.*?)\]\]", marked_sentence):
        m = re.search(r"\w+|[^\w\s]", chunk)
        if m:
            dec.append(pick_suffix(m.group(0)))
    return inc, dec


def f1_set(pred_set: set[str], gold_set: set[str]) -> float:
    if not pred_set and not gold_set:
        return 0.0
    inter = len(pred_set & gold_set)
    denom = len(pred_set) + len(gold_set)
    return (2 * inter) / denom if denom else 0.0


def load_attention_ground_truth(path: Path, top_k: int) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("attention_scores_ground_truth.jsonl must be a JSON list")
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        sid = str(item.get("key"))
        sent = str(item.get("original_sentence", ""))
        scores = item.get("attention_scores") or []
        if not sid or not sent or not isinstance(scores, list):
            continue
        scores_sorted = sorted(scores, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        gold_tokens = [str(s.get("token", "")).strip() for s in scores_sorted[:top_k] if str(s.get("token", "")).strip()]
        out[sid] = {"sentence": sent, "gold_top": gold_tokens}
    return out


def load_causal_ground_truth(path: Path, top_k: int) -> Dict[str, Dict[str, Any]]:
    """
    Reads preprocess_causal_effects.py output (list of items with ground_truth increase/decrease).
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("causal_effects must be a JSON list")
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        sid = str(item.get("sentence_id"))
        sent = str(item.get("sentence_text", ""))
        gt = item.get("ground_truth") or {}
        inc = [str(x).strip() for x in (gt.get("increase") or [])][:top_k]
        dec = [str(x).strip() for x in (gt.get("decrease") or [])][:top_k]
        if not sid or not sent or not inc or not dec:
            continue
        out[sid] = {"sentence": sent, "increase": inc, "decrease": dec}
    return out


async def llm_mark_tokens_strict(
    client: OpenRouter,
    sender: str,
    hypothesis: str,
    sid: str,
    sentence: str,
    top_k: int,
    output_dir: str,
    task_label: str,
) -> str:
    """
    Ask LLM to mark tokens directly in the sentence. No candidates given.
    Enforce strict output with retries.
    """
    system = (
        "You are a meticulous AI researcher applying a hypothesis to predict experimental outcomes.\n\n"
        f"Task: {task_label}\n"
        "You MUST follow the output protocol.\n"
    )
    base_user = (
        f"Sender head: {sender}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Rules:\n"
        "- Mark tokens whose value INCREASES with <<token>> (exactly the required count).\n"
        "- Mark tokens whose value DECREASES with [[token]] (exactly the required count).\n"
        "- Do not add or remove words. Do not rewrite the sentence.\n"
        "- If a name appears multiple times, mark the specific occurrence (the evaluator will suffix it as _1/_2).\n\n"
        "Output format (STRICT):\n"
        "[REASONING]\\n...\\n[PREDICTION]\\n`<the full original sentence with markings>`\n\n"
        f"Exact Count: {top_k} increase and {top_k} decrease.\n\n"
        f"Sentence id: {sid}\n"
        f"Sentence: `{sentence}`\n"
    )

    def ok(pred_line: str) -> bool:
        inc = re.findall(r"<<(.*?)>>", pred_line)
        dec = re.findall(r"\[\[(.*?)\]\]", pred_line)
        return len(inc) == top_k and len(dec) == top_k

    last_err = ""
    for attempt in range(1, 6):
        user = base_user
        if last_err:
            user += f"\nYour previous output was invalid because: {last_err}\nFix it.\n"
        print(f"[LLM] {task_label} | sid={sid} | attempt {attempt}/5")
        resp = await client.generate(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            output_dir=output_dir,
        )
        pred_line = parse_prediction_blocks(resp.text)
        if ok(pred_line):
            return pred_line
        last_err = "wrong marker counts or missing [PREDICTION] sentence"
        print(f"[LLM] sid={sid} invalid format -> retry ({last_err})")
    return pred_line


def eval_causal_batch(
    preds: Dict[str, str],
    gt: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Tuple[float, str]:
    scores = []
    lines = []
    for sid, g in gt.items():
        if sid not in preds:
            continue
        sent = g["sentence"]
        inc_pred, dec_pred = _parse_and_suffix_tokens(sent, preds[sid])
        # compare ignoring suffix (IOI-style robustness for duplicates)
        pred_inc = {normalize_token(t) for t in inc_pred[:top_k]}
        pred_dec = {normalize_token(t) for t in dec_pred[:top_k]}
        gold_inc = {normalize_token(t) for t in g["increase"][:top_k]}
        gold_dec = {normalize_token(t) for t in g["decrease"][:top_k]}
        f1_inc = f1_set(pred_inc, gold_inc)
        f1_dec = f1_set(pred_dec, gold_dec)
        scores.append((f1_inc + f1_dec) / 2)
        lines.append(f"- {sid}: pred_inc={list(pred_inc)} gold_inc={list(gold_inc)} pred_dec={list(pred_dec)} gold_dec={list(gold_dec)}")
    return (float(sum(scores) / len(scores)) if scores else 0.0), "\n".join(lines[:10])


def eval_attention_batch(
    preds: Dict[str, str],
    gt: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Tuple[float, str]:
    scores = []
    lines = []
    for sid, g in gt.items():
        if sid not in preds:
            continue
        sent = g["sentence"]
        inc_pred, dec_pred = _parse_and_suffix_tokens(sent, preds[sid])
        # For attention, treat << >> as top1 and [[ ]] as top2 (orderless for F1@k)
        pred = {normalize_token(t) for t in (inc_pred + dec_pred)[:top_k]}
        gold = {normalize_token(t) for t in g["gold_top"][:top_k]}
        scores.append(f1_set(pred, gold))
        lines.append(f"- {sid}: pred={list(pred)} gold={list(gold)}")
    return (float(sum(scores) / len(scores)) if scores else 0.0), "\n".join(lines[:10])


def _write_debug_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    p = argparse.ArgumentParser(description="IOI-style terminal auto explanation for agr_gender")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--typename", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_source_dir", type=str, required=True)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--topk-attn", type=int, default=2)
    args = p.parse_args()

    sender = f"{args.layer}.{args.head}"
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attn_gt_path = output_dir / "attention_scores_ground_truth.jsonl"
    causal_gt_path = data_dir / f"causal_effects_{args.layer}_{args.head}.json"

    attention_gt = load_attention_ground_truth(attn_gt_path, top_k=args.topk_attn)
    causal_gt = load_causal_ground_truth(causal_gt_path, top_k=args.top_k)
    usable_ids = [sid for sid in causal_gt.keys() if sid in attention_gt]
    if not usable_ids:
        raise RuntimeError(
            "No usable ids for terminal token task: causal ground truth ids do not overlap attention ground truth ids. "
            "Check that causal_effects_<L>_<H>.json has non-empty sentence_id values and matches attention_scores_ground_truth.jsonl keys."
        )
    rng = random.Random(123)
    rng.shuffle(usable_ids)
    split = int(len(usable_ids) * 0.8)
    train_ids = usable_ids[:split]
    val_ids = usable_ids[split:]

    client = initialize_openrouter()

    # initial hypothesis
    init_system = "You are a mechanistic interpretability assistant. Provide [HYPOTHESIS]: one paragraph."
    init_user = (
        "Task: gender pronoun prediction (he/she).\n"
        f"Sender head: {sender}\n"
        "We corrupt this head by patching its head output (z) from a corrupted prompt into the clean prompt.\n"
        "Write a mechanistic hypothesis for what this head does and what it attends to."
    )
    init_resp = await client.generate(
        messages=[{"role": "system", "content": init_system}, {"role": "user", "content": init_user}],
        output_dir=str(output_dir),
    )
    hypothesis = re.sub(r"^\\[HYPOTHESIS\\]:\\s*", "", init_resp.text.strip(), flags=re.IGNORECASE).strip()

    all_val_results: List[Dict[str, Any]] = []

    for it in range(1, args.max_iters + 1):
        print(f"\\n--- Iteration {it}/{args.max_iters} ---")
        batch_train = rng.sample(train_ids, k=min(args.batch_size, len(train_ids)))

        # causal: mark inc/dec tokens (logit change)
        causal_preds: Dict[str, str] = {}
        for sid in batch_train:
            causal_preds[sid] = await llm_mark_tokens_strict(
                client,
                sender,
                hypothesis,
                sid,
                causal_gt[sid]["sentence"],
                args.top_k,
                str(output_dir),
                task_label="Predict token-level logit change at the prediction position.",
            )
        causal_f1, causal_fb = eval_causal_batch(causal_preds, {sid: causal_gt[sid] for sid in batch_train}, args.top_k)
        print(f"Causal F1 (train batch): {causal_f1:.2f}")
        _write_debug_json(
            output_dir / f"debug_causal_iter_{it}.json",
            {
                "iteration": it,
                "sender": sender,
                "top_k": args.top_k,
                "causal_f1": float(causal_f1),
                "predictions": [{"key": sid, "marked_sentence": causal_preds.get(sid, "")} for sid in batch_train],
                "ground_truth": [
                    {
                        "key": sid,
                        "sentence": causal_gt[sid]["sentence"],
                        "increase_tokens": causal_gt[sid]["increase"],
                        "decrease_tokens": causal_gt[sid]["decrease"],
                    }
                    for sid in batch_train
                ],
                "feedback": causal_fb,
            },
        )

        # attention: mark top tokens attended by the head at the chosen query position
        attn_preds: Dict[str, str] = {}
        for sid in batch_train:
            attn_preds[sid] = await llm_mark_tokens_strict(
                client,
                sender,
                hypothesis,
                sid,
                attention_gt[sid]["sentence"],
                args.topk_attn,
                str(output_dir),
                task_label="Predict which tokens the head attends to (direct attention).",
            )
        attn_f1, attn_fb = eval_attention_batch(attn_preds, {sid: attention_gt[sid] for sid in batch_train}, args.topk_attn)
        print(f"Direct Attention F1 (train batch): {attn_f1:.2f}")
        _write_debug_json(
            output_dir / f"debug_attention_iter_{it}.json",
            {
                "iteration": it,
                "sender": sender,
                "top_k": args.topk_attn,
                "direct_attention_f1": float(attn_f1),
                "predictions": [{"key": sid, "marked_sentence": attn_preds.get(sid, "")} for sid in batch_train],
                "ground_truth": [
                    {"key": sid, "sentence": attention_gt[sid]["sentence"], "gold_top": attention_gt[sid].get("gold_top", [])}
                    for sid in batch_train
                ],
                "feedback": attn_fb,
            },
        )

        # validation
        batch_val = rng.sample(val_ids, k=min(args.batch_size, len(val_ids))) if val_ids else []
        val_causal_preds: Dict[str, str] = {}
        val_attn_preds: Dict[str, str] = {}
        for sid in batch_val:
            val_causal_preds[sid] = await llm_mark_tokens_strict(
                client, sender, hypothesis, sid, causal_gt[sid]["sentence"], args.top_k, str(output_dir),
                task_label="Predict token-level logit change at the prediction position."
            )
            val_attn_preds[sid] = await llm_mark_tokens_strict(
                client, sender, hypothesis, sid, attention_gt[sid]["sentence"], args.topk_attn, str(output_dir),
                task_label="Predict which tokens the head attends to (direct attention)."
            )
        val_causal_f1, _ = eval_causal_batch(val_causal_preds, {sid: causal_gt[sid] for sid in batch_val}, args.top_k) if batch_val else (0.0, "")
        val_attn_f1, _ = eval_attention_batch(val_attn_preds, {sid: attention_gt[sid] for sid in batch_val}, args.topk_attn) if batch_val else (0.0, "")

        all_val_results.append(
            {
                "hypothesis": hypothesis,
                "validation_scores": {"causal_f1": float(val_causal_f1), "direct_attention_f1": float(val_attn_f1)},
                "train_debug": {"causal_f1": float(causal_f1), "direct_attention_f1": float(attn_f1)},
                "meta": {
                    "sender": sender,
                    "top_k_causal": args.top_k,
                    "top_k_attention": args.topk_attn,
                },
            }
        )

        if it >= args.max_iters:
            break

        refine_system = (
            "You are refining a mechanistic hypothesis about a single transformer attention head on a gender pronoun "
            "prediction task.\n\n"
            "Feedback meanings:\n"
            "- In the causal feedback, gold is which sentence token's logit increases/decreases most under corruption.\n"
            "- In the attention feedback, gold is which tokens are top-attended at the query position.\n\n"
            "Output format:\n"
            "- Return exactly ONE paragraph starting with [HYPOTHESIS]:\n"
            "- Do NOT output JSON, lists, or analysis text."
        )
        refine_user = (
            f"Old hypothesis:\n{hypothesis}\n\n"
            "Causal feedback (pred vs gold):\n"
            f"{causal_fb}\n\n"
            "Attention feedback (pred vs gold):\n"
            f"{attn_fb}\n\n"
            "Write an improved hypothesis."
        )
        resp = await client.generate(
            messages=[{"role": "system", "content": refine_system}, {"role": "user", "content": refine_user}],
            output_dir=str(output_dir),
        )
        hypothesis = re.sub(r"^\\[HYPOTHESIS\\]:\\s*", "", resp.text.strip(), flags=re.IGNORECASE).strip()

    final = {"head": sender, "typename": args.typename, "all_hypotheses_validation_results": all_val_results}
    out_path = output_dir / f"final_result_round_{args.rounds}.json"
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved final result to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
