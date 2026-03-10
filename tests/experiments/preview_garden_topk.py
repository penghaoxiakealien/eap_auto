#!/usr/bin/env python3
"""
Preview GPT-2 next-token top-k for garden_npz_v-trans_mod sentences.

Example:
  python tests/experiments/preview_garden_topk.py \
    --data datasets/garden/garden_npz_v_trans_mod.csv \
    --field clean --topk 10 --limit 5
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview next-token top-k for garden dataset.")
    p.add_argument("--data", type=Path, required=True, help="CSV with clean/corrupted columns.")
    p.add_argument(
        "--field",
        choices=["clean", "corrupted", "both"],
        default="clean",
        help="Which column to evaluate.",
    )
    p.add_argument("--topk", type=int, default=10, help="Top-k tokens to show.")
    p.add_argument("--limit", type=int, default=10, help="How many rows to print.")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass.")
    p.add_argument("--model-name", type=str, default="gpt2", help="HF model name.")
    p.add_argument("--model-path", type=str, default=None, help="Local model path (optional).")
    p.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto.")
    p.add_argument("--pos-token", type=str, default=" was", help="Positive token (with leading space).")
    p.add_argument("--neg-token", type=str, default=" for", help="Negative token (with leading space).")
    return p.parse_args()


def load_texts(path: Path) -> List[Tuple[str, str, str | None, str | None]]:
    rows: List[Tuple[str, str, str | None, str | None]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = (row.get("clean") or "").strip()
            corrupted = (row.get("corrupted") or "").strip()
            pos_tok = (row.get("correct_token") or "").strip() or None
            neg_tok = (row.get("incorrect_token") or "").strip() or None
            if clean and corrupted:
                rows.append((clean, corrupted, pos_tok, neg_tok))
    return rows


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def topk_for_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    topk: int,
    batch_size: int,
    device: str,
) -> List[List[Tuple[str, float]]]:
    results: List[List[Tuple[str, float]]] = []
    model.eval()
    with torch.no_grad():
        for batch in iter_batches(texts, batch_size):
            toks = tokenizer(batch, return_tensors="pt", padding=True)
            input_ids = toks["input_ids"].to(device)
            attention_mask = toks["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            lengths = attention_mask.sum(dim=1) - 1
            idx = torch.arange(input_ids.size(0), device=device)
            last_logits = logits[idx, lengths]
            probs = torch.softmax(last_logits, dim=-1)
            top_vals, top_ids = torch.topk(probs, k=topk, dim=-1)
            for row_vals, row_ids in zip(top_vals, top_ids):
                row = []
                for score, tok_id in zip(row_vals.tolist(), row_ids.tolist()):
                    tok = tokenizer.decode([tok_id])
                    row.append((tok, float(score)))
                results.append(row)
    return results


def normalize_token_str(token: str) -> str:
    token = token.strip()
    if token and not token[0].isspace():
        return " " + token
    return token


def single_token_id(tokenizer: AutoTokenizer, token: str) -> int:
    token = normalize_token_str(token)
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} is not a single token: {ids}")
    return ids[0]


def logit_diff_for_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    pos_tokens: List[str],
    neg_tokens: List[str],
    batch_size: int,
    device: str,
) -> List[Tuple[float, float, float]]:
    if len(pos_tokens) != len(texts) or len(neg_tokens) != len(texts):
        raise ValueError("pos_tokens/neg_tokens must align with texts length.")
    pos_ids = [single_token_id(tokenizer, t) for t in pos_tokens]
    neg_ids = [single_token_id(tokenizer, t) for t in neg_tokens]
    results: List[Tuple[float, float, float]] = []
    model.eval()
    with torch.no_grad():
        offset = 0
        for batch in iter_batches(texts, batch_size):
            toks = tokenizer(batch, return_tensors="pt", padding=True)
            input_ids = toks["input_ids"].to(device)
            attention_mask = toks["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            lengths = attention_mask.sum(dim=1) - 1
            idx = torch.arange(input_ids.size(0), device=device)
            last_logits = logits[idx, lengths]
            batch_pos = torch.tensor(pos_ids[offset : offset + input_ids.size(0)], device=device)
            batch_neg = torch.tensor(neg_ids[offset : offset + input_ids.size(0)], device=device)
            pos_logits = last_logits.gather(1, batch_pos.unsqueeze(1)).squeeze(1)
            neg_logits = last_logits.gather(1, batch_neg.unsqueeze(1)).squeeze(1)
            diffs = (pos_logits - neg_logits).tolist()
            for p, n, d in zip(pos_logits.tolist(), neg_logits.tolist(), diffs):
                results.append((float(p), float(n), float(d)))
            offset += input_ids.size(0)
    return results


def preview(
    label: str,
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    topk: int,
    limit: int,
    batch_size: int,
    device: str,
    pos_token: str,
    neg_token: str,
    per_row_tokens: List[Tuple[str | None, str | None]] | None = None,
) -> None:
    print(f"\n== {label} ==")
    topk_rows = topk_for_texts(texts, tokenizer, model, topk, batch_size, device)
    if per_row_tokens is None:
        pos_tokens = [pos_token for _ in texts]
        neg_tokens = [neg_token for _ in texts]
    else:
        pos_tokens = [p if p is not None else pos_token for p, _ in per_row_tokens]
        neg_tokens = [n if n is not None else neg_token for _, n in per_row_tokens]
    diffs = logit_diff_for_texts(texts, tokenizer, model, pos_tokens, neg_tokens, batch_size, device)
    counts = Counter()
    for i, row in enumerate(topk_rows[:limit]):
        pretty = ", ".join([f"{tok}:{score:.3f}" for tok, score in row])
        pos_logit, neg_logit, diff = diffs[i]
        print(f"[{i}] {texts[i]}")
        print(f"    {pretty}")
        row_pos = pos_tokens[i]
        row_neg = neg_tokens[i]
        print(f"    logit({row_pos!r})={pos_logit:.3f} logit({row_neg!r})={neg_logit:.3f} diff={diff:.3f}")
    for row in topk_rows:
        counts.update(tok for tok, _ in row)
    print("\nTop-k token frequency:")
    for tok, cnt in counts.most_common(20):
        print(f"  {tok!r}: {cnt}")


def main() -> None:
    args = parse_args()
    rows = load_texts(args.data)
    if not rows:
        raise SystemExit(f"No rows found in {args.data}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    clean_texts = [c for c, _, _, _ in rows]
    corrupted_texts = [x for _, x, _, _ in rows]
    per_row_tokens = [(p, n) for _, _, p, n in rows]

    if args.field in ("clean", "both"):
        preview(
            "clean",
            clean_texts,
            tokenizer,
            model,
            args.topk,
            args.limit,
            args.batch_size,
            device,
            args.pos_token,
            args.neg_token,
            per_row_tokens=per_row_tokens,
        )
    if args.field in ("corrupted", "both"):
        preview(
            "corrupted",
            corrupted_texts,
            tokenizer,
            model,
            args.topk,
            args.limit,
            args.batch_size,
            device,
            args.pos_token,
            args.neg_token,
            per_row_tokens=per_row_tokens,
        )


if __name__ == "__main__":
    main()
