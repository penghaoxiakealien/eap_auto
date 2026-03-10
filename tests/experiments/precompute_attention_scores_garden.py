#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute raw attention scores for a specific head on Garden data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer


LOCAL_MODEL_DIR = "/data31/private/wangziran/eap-ig/gpt2"


def load_model(device: str = "cuda") -> HookedTransformer:
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return HookedTransformer.from_pretrained("gpt2", device=device, cache_dir=LOCAL_MODEL_DIR)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


def load_standard_garden(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是标准 garden JSON 列表")
    for idx, item in enumerate(data):
        if isinstance(item, dict) and "sample_id" not in item:
            item["sample_id"] = idx
    return data


def _ensure_pad_token(model: HookedTransformer) -> None:
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _iter_batches(items: List[Dict[str, object]], batch_size: int) -> Iterable[List[Dict[str, object]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _query_index(sample: Dict[str, object], sentence_len: int, attention_position: str) -> int:
    pos = (attention_position or "end").lower()
    wd = sample.get("word_idx") or {}
    if pos in {"subj", "verb", "obj_head", "rel_pron", "rel_verb"}:
        idx = wd.get(pos)
        if isinstance(idx, int) and 0 <= idx < sentence_len:
            return idx
    if pos == "end":
        end_idx = sample.get("end_idx")
        if isinstance(end_idx, int):
            return min(end_idx, sentence_len - 1)
    return max(0, sentence_len - 1)


def compute_attention_for_sample(
    model: HookedTransformer,
    sample: Dict[str, object],
    target_head: tuple[int, int],
    attention_position: str,
) -> Dict[str, object]:
    sentence = str(sample.get("text", "")).strip()
    if not sentence:
        raise ValueError("missing sentence text")

    correct = str(sample.get("correct_token", ""))
    incorrect = str(sample.get("incorrect_token", ""))
    tokens = model.to_tokens(sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
    str_tokens = model.to_str_tokens(tokens[0])
    actual_end_pos = _query_index(sample, tokens.shape[1], attention_position)

    layer, head_idx = target_head
    attn_hook_name = f"blocks.{layer}.attn.hook_pattern"

    with t.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == attn_hook_name)

    attention_scores = cache[attn_hook_name][0, head_idx, actual_end_pos, :].cpu()
    token_analysis = []
    for pos, score in enumerate(attention_scores):
        if pos <= actual_end_pos:
            token_analysis.append(
                {
                    "token": str_tokens[pos],
                    "position": pos,
                    "score": score.item(),
                }
            )
    token_analysis.sort(key=lambda x: x["score"], reverse=True)

    sample_id = sample.get("sample_id", sample.get("id", None))
    return {
        "sample_id": sample_id,
        "sentence_id": sample_id,
        "sentence_text": sentence,
        "io_token": correct,
        "s_token": incorrect,
        "target_head": f"{layer}.{head_idx}",
        "end_position": actual_end_pos,
        "top_attended_tokens": token_analysis,
    }


def compute_attention_for_batch(
    model: HookedTransformer,
    batch: List[Dict[str, object]],
    target_head: tuple[int, int],
    attention_position: str,
) -> List[Dict[str, object]]:
    sentences = []
    for sample in batch:
        sentence = str(sample.get("text", "")).strip()
        if not sentence:
            sentences.append(None)
        else:
            sentences.append(sentence)

    valid_items = [(idx, s) for idx, s in enumerate(sentences) if s]
    if not valid_items:
        return []

    indices, texts = zip(*valid_items)
    tokenizer = model.tokenizer
    enc = tokenizer(list(texts), return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.cfg.device)
    attention_mask = enc["attention_mask"].to(model.cfg.device)

    layer, head_idx = target_head
    attn_hook_name = f"blocks.{layer}.attn.hook_pattern"

    with t.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=lambda name: name == attn_hook_name)

    results = []
    for local_idx, batch_idx in enumerate(indices):
        sample = batch[batch_idx]
        correct = str(sample.get("correct_token", ""))
        incorrect = str(sample.get("incorrect_token", ""))
        length = int(attention_mask[local_idx].sum().item())
        actual_end_pos = _query_index(sample, length, attention_position)

        str_tokens = tokenizer.convert_ids_to_tokens(input_ids[local_idx][:length].tolist())
        scores = cache[attn_hook_name][local_idx, head_idx, actual_end_pos, :actual_end_pos + 1].cpu()

        token_analysis = []
        for pos, score in enumerate(scores):
            token_analysis.append(
                {
                    "token": str_tokens[pos],
                    "position": pos,
                    "score": score.item(),
                }
            )
        token_analysis.sort(key=lambda x: x["score"], reverse=True)

        sample_id = sample.get("sample_id", sample.get("id", None))
        results.append(
            {
                "sample_id": sample_id,
                "sentence_id": sample_id,
                "sentence_text": str(sample.get("text", "")).strip(),
                "io_token": correct,
                "s_token": incorrect,
                "target_head": f"{layer}.{head_idx}",
                "end_position": actual_end_pos,
                "top_attended_tokens": token_analysis,
            }
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Compute raw attention scores for Garden data.")
    p.add_argument("--standard-json", type=Path, required=True, help="standard_garden_data.json")
    p.add_argument("--output-file", type=Path, required=True, help="输出 raw attention JSON")
    p.add_argument("--head", type=str, required=True, help="目标头 (layer.head)")
    p.add_argument(
        "--attention-position",
        type=str,
        default="end",
        help="Query position (end/subj/verb/obj_head/rel_pron/rel_verb).",
    )
    p.add_argument("--max-samples", type=int, default=None, help="处理最大样本数")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for attention computation")
    args = p.parse_args()

    layer, head_idx = map(int, args.head.split("."))
    target_head = (layer, head_idx)

    print(f"--- Garden raw attention for head {args.head} ---")
    model = load_model()
    _ensure_pad_token(model)
    samples = load_standard_garden(args.standard_json)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"将处理 {len(samples)} 个样本...")

    results = []
    if args.batch_size <= 1:
        for sample in tqdm(samples, desc="计算注意力"):
            try:
                results.append(compute_attention_for_sample(model, sample, target_head, args.attention_position))
            except Exception:
                continue
    else:
        for batch in tqdm(list(_iter_batches(samples, args.batch_size)), desc="计算注意力"):
            try:
                results.extend(compute_attention_for_batch(model, batch, target_head, args.attention_position))
            except Exception:
                continue

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"✅ 保存 raw attention: {args.output_file}")


if __name__ == "__main__":
    main()
