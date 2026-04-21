#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute raw attention scores for a single head on the agr_gender dataset.

This mirrors precompute_attention_scores.py (IOI) but uses the agr_gender standard JSON
which includes word_idx positions (end/verb/a1/b/a2).

Output JSON schema (list):
  [
    {
      "sample_id": "...",
      "sentence_text": "...",
      "attention_position": "end|verb|a1|b|a2",
      "query_pos": <int>,
      "tokens": ["..."],
      "top_attended_tokens": [{"token": "...", "position": <int>, "score": <float>}]
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils


LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


def load_model(device: str = "cuda") -> HookedTransformer:
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return HookedTransformer.from_pretrained("gpt2", device=device, cache_dir=LOCAL_MODEL_DIR)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


def parse_head(head_str: str) -> tuple[int, int]:
    head_str = head_str.split(":", 1)[0].strip()
    layer_s, head_s = head_str.split(".", 1)
    return int(layer_s), int(head_s)


def load_standard_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {path}")


def resolve_query_pos(record: Dict[str, Any], attention_position: str) -> int:
    pos = (attention_position or "end").lower().strip()
    word_idx = record.get("word_idx") or {}
    if not isinstance(word_idx, dict):
        word_idx = {}
    if pos in {"end", "verb", "a1", "b", "a2"}:
        v = word_idx.get(pos)
        if v is None:
            # Backwards compatibility: some generators stored uppercase keys
            v = word_idx.get(pos.upper())
        if isinstance(v, int):
            return v
    # fallback: try int
    try:
        return int(pos)
    except Exception:
        pass
    # last token fallback
    toks = record.get("tokenized_clean") or record.get("tokens") or []
    return max(0, len(toks) - 1) if isinstance(toks, list) else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute raw attention scores on agr_gender dataset.")
    p.add_argument("--head", required=True, help="Head as L.H (e.g. 10.7)")
    p.add_argument("--input_json", required=True, help="Path to standard_gender_data.json")
    p.add_argument("--output_file", required=True, help="Where to write raw attention JSON")
    p.add_argument("--attention-position", default="end", help="Query position key: end/verb/a1/b/a2")
    p.add_argument("--topk", type=int, default=10, help="How many top attended tokens to keep")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    layer, head_idx = parse_head(args.head)
    records = load_standard_json(Path(args.input_json))
    model = load_model(device=args.device)
    model.cfg.use_attn_result = True

    pattern_hook = utils.get_act_name("pattern", layer)
    results: List[Dict[str, Any]] = []

    for i, rec in tqdm(list(enumerate(records)), desc="计算注意力"):
        text = rec.get("clean") or rec.get("text") or rec.get("sentence_text")
        if not isinstance(text, str) or not text.strip():
            continue
        sample_id = rec.get("sentence_id") or rec.get("id") or str(i)
        query_pos = resolve_query_pos(rec, args.attention_position)
        toks = model.to_tokens(text, prepend_bos=False)
        str_toks = model.to_str_tokens(toks[0])
        query_pos = min(max(int(query_pos), 0), len(str_toks) - 1)

        _, cache = model.run_with_cache(toks, names_filter=lambda name: name == pattern_hook, return_type=None)
        pattern = cache[pattern_hook][0, head_idx]  # [pos_q, pos_k]
        row = pattern[query_pos].detach().float().cpu()
        k = min(args.topk, row.numel())
        top_vals, top_idx = torch.topk(row, k=k)
        top_tokens = []
        for score, pos_k in zip(top_vals.tolist(), top_idx.tolist()):
            top_tokens.append(
                {"token": str_toks[pos_k].strip(), "position": int(pos_k), "score": float(score)}
            )
        results.append(
            {
                "sample_id": str(sample_id),
                "sentence_text": text,
                "attention_position": str(args.attention_position),
                "query_pos": int(query_pos),
                "tokens": str_toks,
                "top_attended_tokens": top_tokens,
            }
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 原始注意力分数计算完成！结果已保存到: {out_path}")


if __name__ == "__main__":
    main()

