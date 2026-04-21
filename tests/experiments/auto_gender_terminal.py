#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated hypothesis generation + refinement for agr_gender terminal heads.

Two evaluation axes (mirrors IOI logic):
  - causal_f1 (IOI-style): predict which pronoun token's logit increases most vs decreases most under corruption
  - direct_attention_f1: predict which tokens the head attends to (choose top-2 from candidates)

Inputs are produced by run_gender_terminal_head.sh:
  - output_dir/attention_scores_ground_truth.jsonl (JSON list)
  - data_source_dir/terminal_effects_{L}_{H}.json (JSON list)

Outputs:
  - output_dir/final_result_round_{rounds}.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys_path_added = False
try:
    import sys

    sys.path.append("/home/wangziran/eap_auto/")
    sys_path_added = True
except Exception:
    pass

from api import OpenRouter


def initialize_openrouter(model: str = "claude-sonnet-4-20250514-thinking") -> OpenRouter:
    api_key = "sk-Z3pwy4dD8WY2XZlbzch66NP5hQIoFKeU7KvI2XD8bQSyFVGO"
    return OpenRouter(model=model, api_key=api_key)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(text: str) -> Any:
    """
    Best-effort JSON extraction for LLM replies.
    Handles common patterns like ```json ...```, and stray prose before/after.
    """
    txt = (text or "").strip()
    if not txt:
        return None

    m = _FENCE_RE.search(txt)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            txt = candidate

    try:
        return json.loads(txt)
    except Exception:
        pass

    # substring parse: first {...} or [...]
    brace_positions = [p for p in (txt.find("{"), txt.find("[")) if p != -1]
    if not brace_positions:
        return None
    start = min(brace_positions)
    end = max(txt.rfind("}"), txt.rfind("]"))
    if end == -1 or end <= start:
        return None
    snippet = txt[start : end + 1].strip()
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _clean_key(k: str) -> str:
    s = (k or "").strip().strip(",")
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    return s.strip()


def _clean_value(v: str) -> str:
    s = (v or "").strip().strip(",")
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    return s.strip().lower()


def split_ids(ids: List[str], validation_split: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    split = int(len(ids) * (1 - validation_split))
    return ids[:split], ids[split:]


def f1_binary(pred: List[int], gold: List[int]) -> float:
    # positive class = 1
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    if tp == 0 and (fp + fn) == 0:
        return 0.0
    return (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def f1_set(pred_set: set[str], gold_set: set[str]) -> float:
    if not pred_set and not gold_set:
        return 0.0
    inter = len(pred_set & gold_set)
    denom = len(pred_set) + len(gold_set)
    return (2 * inter) / denom if denom else 0.0


def parse_hypothesis(text: str) -> str:
    m = re.search(r"\[HYPOTHESIS\]:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def load_attention_ground_truth(path: Path, top_k: int = 2) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    gt: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, list):
        raise ValueError("attention_scores_ground_truth.jsonl must be a JSON list")
    for item in data:
        key = str(item.get("key"))
        scores = item.get("attention_scores") or []
        if not key or not isinstance(scores, list) or not scores:
            continue
        # sort by score desc, keep tokens
        scores_sorted = sorted(scores, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        tokens = [str(s.get("token", "")).strip() for s in scores_sorted[: max(top_k, 10)]]
        gt[key] = {
            "sentence": str(item.get("original_sentence", "")),
            "candidates": tokens,  # candidates to choose from
            "gold_top": [t for t in tokens[:top_k] if t],
        }
    return gt


def load_terminal_effects(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("terminal_effects must be a JSON list")
    out = {}
    for item in data:
        sid = str(item.get("sentence_id"))
        if not sid:
            continue
        # prefer IOI-style token labels if present
        inc_tok = str(item.get("increase_token") or "").strip().lower()
        dec_tok = str(item.get("decrease_token") or "").strip().lower()
        eff = str(item.get("effect") or "")
        if inc_tok in {"he", "she"} and dec_tok in {"he", "she"} and inc_tok != dec_tok:
            out[sid] = item
            continue
        # backward-compat: legacy help/hurt/neutral only
        if eff in {"hurt", "help", "neutral"}:
            out[sid] = item
    return out


async def llm_predict_causal(
    client: OpenRouter,
    hypothesis: str,
    sender: str,
    examples: List[Tuple[str, str]],
    output_dir: str,
) -> Dict[str, Dict[str, str]]:
    """
    Return mapping sid -> {"increase": "he"|"she", "decrease": "he"|"she"}.
    """
    system = (
        "You are analyzing a transformer attention head for a gender pronoun prediction task.\n"
        "We perform an intervention: we corrupt the sender head by patching its activation from a corrupted prompt into the clean prompt.\n"
        "For each sentence, predict which pronoun token's logit increases most vs decreases most under this corruption.\n"
        "Return JSON mapping sentence_id -> {\"increase\": \"he\"|\"she\", \"decrease\": \"he\"|\"she\"}.\n"
        "Return JSON only.\n"
    )
    user_lines = [f"Sender head: {sender}", f"Hypothesis: {hypothesis}", ""]
    for sid, sent in examples:
        user_lines.append(f"- {sid}: {sent}")
    user_lines.append("\nReturn JSON only.")
    resp = await client.generate(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_lines)}],
        output_dir=output_dir,
    )
    txt = resp.text.strip()
    obj = _try_parse_json(txt)
    out: Dict[str, Dict[str, str]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            sid = _clean_key(str(k))
            if isinstance(v, dict):
                inc = _clean_value(str(v.get("increase", "")))
                dec = _clean_value(str(v.get("decrease", "")))
                if inc in {"he", "she"} and dec in {"he", "she"} and inc != dec:
                    out[sid] = {"increase": inc, "decrease": dec}
    return out


async def llm_predict_attention(
    client: OpenRouter,
    hypothesis: str,
    sender: str,
    examples: List[Tuple[str, str, List[str]]],
    top_k: int,
    output_dir: str,
) -> Dict[str, List[str]]:
    """
    For each sentence, choose exactly top_k tokens from the provided candidate list.
    Return mapping sid -> list[str].
    """
    system = (
        "You are analyzing what an attention head LOOKS AT.\n"
        "For each sentence, you are given a candidate list of tokens.\n"
        f"Choose exactly {top_k} tokens (from the candidate list) that the head most strongly attends to.\n"
        "Return a JSON object mapping sentence_id to a JSON array of the chosen tokens.\n"
    )
    user_lines = [f"Sender head: {sender}", f"Hypothesis: {hypothesis}", ""]
    for sid, sent, cands in examples:
        c = ", ".join([repr(t) for t in cands])
        user_lines.append(f"- {sid}: {sent}\n  candidates=[{c}]")
    user_lines.append("\nReturn JSON only.")
    resp = await client.generate(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_lines)}],
        output_dir=output_dir,
    )
    txt = resp.text.strip()
    obj = _try_parse_json(txt)
    if isinstance(obj, dict):
        out: Dict[str, List[str]] = {}
        for k, v in obj.items():
            if isinstance(v, list):
                out[_clean_key(str(k))] = [str(x).strip() for x in v][:top_k]
        return out
    return {}


def eval_causal(
    pred_map: Dict[str, Dict[str, str]],
    gold_map: Dict[str, Dict[str, str]],
) -> Tuple[float, str]:
    ids = [sid for sid in gold_map.keys() if sid in pred_map]
    if not ids:
        return 0.0, "no overlapping ids"
    scores = []
    lines = []
    for sid in ids[:10]:
        p = pred_map[sid]
        g = gold_map[sid]
        pred_inc = set([_clean_value(p.get("increase", ""))])
        pred_dec = set([_clean_value(p.get("decrease", ""))])
        gold_inc = set([_clean_value(g.get("increase", ""))])
        gold_dec = set([_clean_value(g.get("decrease", ""))])
        f1_inc = f1_set(pred_inc, gold_inc)
        f1_dec = f1_set(pred_dec, gold_dec)
        scores.append((f1_inc + f1_dec) / 2)
        lines.append(f"- {sid}: pred_inc={list(pred_inc)} gold_inc={list(gold_inc)} pred_dec={list(pred_dec)} gold_dec={list(gold_dec)}")
    score = float(sum(scores) / len(scores)) if scores else 0.0
    return score, "\n".join(lines)


def eval_attention(
    pred_map: Dict[str, List[str]],
    gt: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Tuple[float, str]:
    ids = [sid for sid in gt.keys() if sid in pred_map]
    if not ids:
        return 0.0, "no overlapping ids"
    scores = []
    lines = []
    for sid in ids:
        gold = set(gt[sid]["gold_top"][:top_k])
        pred = set([t for t in pred_map.get(sid, [])[:top_k] if t])
        scores.append(f1_set(pred, gold))
        lines.append(f"- {sid}: pred={list(pred)} gold={list(gold)}")
    return float(sum(scores) / len(scores)), "\n".join(lines[:10])


async def main() -> None:
    p = argparse.ArgumentParser(description="auto terminal hypothesis for agr_gender")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--typename", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_source_dir", type=str, required=True)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--topk-attn", type=int, default=2)
    args = p.parse_args()

    sender = f"{args.layer}.{args.head}"
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attn_gt_path = output_dir / "attention_scores_ground_truth.jsonl"
    effects_path = data_dir / f"terminal_effects_{args.layer}_{args.head}.json"

    attention_gt = load_attention_ground_truth(attn_gt_path, top_k=args.topk_attn)
    effects = load_terminal_effects(effects_path)

    # align ids: prefer IOI-style token labels; drop rows missing token labels
    usable_ids = []
    for sid, e in effects.items():
        inc = str(e.get("increase_token") or "").strip().lower()
        dec = str(e.get("decrease_token") or "").strip().lower()
        if inc in {"he", "she"} and dec in {"he", "she"} and inc != dec and sid in attention_gt:
            usable_ids.append(sid)
    train_ids, val_ids = split_ids(usable_ids, validation_split=0.2, seed=42)

    client = initialize_openrouter()

    # initial hypothesis
    init_system = "You are a mechanistic interpretability assistant. Provide [HYPOTHESIS]: one paragraph."
    init_user = (
        "Task: gender pronoun prediction (he/she).\n"
        f"Sender head: {sender}\n"
        "We corrupt this head (patch from corrupted prompt into clean prompt).\n"
        "Write a mechanistic hypothesis for what this head does and what it attends to."
    )
    init_resp = await client.generate(
        messages=[{"role": "system", "content": init_system}, {"role": "user", "content": init_user}],
        output_dir=str(output_dir),
    )
    hypothesis = parse_hypothesis(init_resp.text)

    all_val_results: List[Dict[str, Any]] = []
    rng = random.Random(123)

    for it in range(1, args.max_iters + 1):
        print(f"\n--- Iteration {it}/{args.max_iters} ---")
        batch_train = rng.sample(train_ids, k=min(args.batch_size, len(train_ids)))
        # causal examples
        causal_examples = [(sid, attention_gt[sid]["sentence"]) for sid in batch_train]
        gold_causal = {sid: {"increase": str(effects[sid]["increase_token"]).lower(), "decrease": str(effects[sid]["decrease_token"]).lower()} for sid in batch_train}
        pred_causal = await llm_predict_causal(client, hypothesis, sender, causal_examples, str(output_dir))
        causal_f1, causal_feedback = eval_causal(pred_causal, gold_causal)
        print(f"Causal F1 (train batch): {causal_f1:.2f}")

        # attention examples (provide candidates)
        attn_examples = [(sid, attention_gt[sid]["sentence"], attention_gt[sid]["candidates"]) for sid in batch_train]
        pred_attn = await llm_predict_attention(client, hypothesis, sender, attn_examples, args.topk_attn, str(output_dir))
        attn_f1, attn_feedback = eval_attention(pred_attn, attention_gt, args.topk_attn)
        print(f"Direct Attention F1 (train batch): {attn_f1:.2f}")

        # validation scoring on a fixed small batch
        batch_val = rng.sample(val_ids, k=min(args.batch_size, len(val_ids))) if val_ids else []
        gold_val = {sid: {"increase": str(effects[sid]["increase_token"]).lower(), "decrease": str(effects[sid]["decrease_token"]).lower()} for sid in batch_val}
        pred_val = await llm_predict_causal(client, hypothesis, sender, [(sid, attention_gt[sid]["sentence"]) for sid in batch_val], str(output_dir))
        val_causal_f1, _ = eval_causal(pred_val, gold_val) if batch_val else (0.0, "")
        pred_val_attn = await llm_predict_attention(
            client,
            hypothesis,
            sender,
            [(sid, attention_gt[sid]["sentence"], attention_gt[sid]["candidates"]) for sid in batch_val],
            args.topk_attn,
            str(output_dir),
        ) if batch_val else {}
        val_attn_f1, _ = eval_attention(pred_val_attn, attention_gt, args.topk_attn) if batch_val else (0.0, "")

        all_val_results.append(
            {
                "hypothesis": hypothesis,
                "validation_scores": {
                    "causal_f1": float(val_causal_f1),
                    "direct_attention_f1": float(val_attn_f1),
                },
                "train_debug": {
                    "causal_f1": float(causal_f1),
                    "direct_attention_f1": float(attn_f1),
                },
            }
        )

        if it >= args.max_iters:
            break

        # refine (make the meaning of pred/gold explicit, mirroring the more contextual IOI prompts)
        refine_system = (
            "You are refining a mechanistic hypothesis about a single transformer attention head on a gender pronoun "
            "prediction task (he/she).\n\n"
            "Intervention setup:\n"
            "- We CORRUPT this sender head by patching its head output (z) from a corrupted prompt into the clean prompt.\n"
            "- We then measure how the next-token logits for the pronouns 'he' and 'she' change.\n"
            "- Ground truth is token-level (IOI-style): which pronoun logit increases most vs decreases most.\n\n"
            "Feedback meanings:\n"
            "- In the causal feedback, 'pred_inc/dec' are your predicted pronoun tokens; "
            "'gold_inc/dec' are the ground truth tokens computed from the intervention.\n"
            "- In the attention feedback, 'pred' is your previous prediction of which tokens are most attended; "
            "'gold' is the ground truth top tokens from the model's attention at the specified query position.\n\n"
            "Goal:\n"
            "- Update the hypothesis to better predict BOTH (1) hurt/help and (2) top-attended tokens.\n"
            "- Be specific and mechanistic: state what the head attends to (which role/position/token), and how that "
            "information changes the computation relevant to predicting he vs she.\n\n"
            "Output format:\n"
            "- Return exactly ONE paragraph starting with [HYPOTHESIS]:\n"
            "- Do NOT output JSON, lists, or analysis text."
        )
        refine_user = (
            f"Old hypothesis:\n{hypothesis}\n\n"
            "Causal feedback (examples of pred vs gold):\n"
            f"{causal_feedback}\n\n"
            "Attention feedback (examples of pred vs gold):\n"
            f"{attn_feedback}\n\n"
            "Write an improved hypothesis that explains BOTH what the head does (effect on he/she prediction) and what it attends to."
        )
        resp = await client.generate(
            messages=[{"role": "system", "content": refine_system}, {"role": "user", "content": refine_user}],
            output_dir=str(output_dir),
        )
        hypothesis = parse_hypothesis(resp.text)

    final = {
        "head": sender,
        "typename": args.typename,
        "all_hypotheses_validation_results": all_val_results,
    }
    out_path = output_dir / f"final_result_round_{args.rounds}.json"
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved final result to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
