#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-sentence terminal causal effects for agr_gender for a single sender head.

We follow the IOI path-patching style:
  - clean prompt = record["clean"]
  - corrupted prompt = record["corrupted"]
  - cache all head outputs ("z") on clean and corrupted
  - run on clean, freezing all heads to clean-cache, except patch sender head z from corrupted-cache
  - measure the change in gender logit diff at the prediction position (word_idx["end"])

We output a JSON list suitable for an "auto" script to score terminal causal predictions:
  [
    {
      "sentence_id": "...",
      "sentence_text": "...",
      "label": 0|1,
      "end_idx": <int>,
      "delta_logit_diff": <float>,
      "effect": "hurt"|"help"|"neutral",
      "delta_logits": {"he": <float>, "she": <float>},
      "increase_token": "he"|"she",
      "decrease_token": "he"|"she"
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint


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


def patch_or_freeze_head_vectors(
    orig_head_vector: torch.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
):
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def gender_logit_diff(
    logits: torch.Tensor,
    end_idx: int,
    label: int,
    he_token_id: int,
    she_token_id: int,
) -> float:
    # logits: [1, seq, vocab]
    he = logits[0, end_idx, he_token_id]
    she = logits[0, end_idx, she_token_id]
    if int(label) == 0:
        return float((he - she).detach().cpu())
    return float((she - he).detach().cpu())


def gender_logits(
    logits: torch.Tensor,
    end_idx: int,
    he_token_id: int,
    she_token_id: int,
) -> tuple[float, float]:
    he = float(logits[0, end_idx, he_token_id].detach().cpu())
    she = float(logits[0, end_idx, she_token_id].detach().cpu())
    return he, she


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute terminal (logits) causal effects for agr_gender.")
    p.add_argument("--head", required=True, help="Sender head L.H")
    p.add_argument("--data_json", required=True, help="standard_gender_data.json")
    p.add_argument("--output_file", required=True)
    p.add_argument("--dataset-size", type=int, default=200, help="How many samples to process (0=all)")
    p.add_argument("--neutral-eps", type=float, default=1e-4, help="|delta| < eps => neutral")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sender = parse_head(args.head)
    records = load_standard_json(Path(args.data_json))
    if args.dataset_size and args.dataset_size > 0:
        records = records[: args.dataset_size]
    print(f"Processing {len(records)} samples for sender {sender[0]}.{sender[1]}")

    model = load_model(device=args.device)
    model.cfg.use_attn_result = True

    z_name_filter = lambda name: name.endswith("z")
    out: List[Dict[str, Any]] = []

    for i, rec in tqdm(list(enumerate(records)), desc="terminal effects"):
        clean_text = rec.get("clean") or rec.get("text")
        corrupted_text = rec.get("corrupted") or rec.get("corrupted_text")
        if not isinstance(clean_text, str) or not clean_text.strip():
            continue
        if not isinstance(corrupted_text, str) or not corrupted_text.strip():
            continue
        label = int(rec.get("label", 0))
        he_id = int(rec.get("he_token_id", 339))
        she_id = int(rec.get("she_token_id", 673))
        word_idx = rec.get("word_idx") or {}
        end_idx = word_idx.get("end")
        if end_idx is None:
            end_idx = rec.get("end_idx")
        if end_idx is None:
            # fallback: last token
            end_idx = len((rec.get("tokenized_clean") or [])) - 1
        end_idx = int(end_idx)

        clean_toks = model.to_tokens(clean_text, prepend_bos=False)
        corrupted_toks = model.to_tokens(corrupted_text, prepend_bos=False)

        # baseline
        clean_logits = model(clean_toks)
        base_he, base_she = gender_logits(clean_logits, end_idx, he_id, she_id)
        base = gender_logit_diff(clean_logits, end_idx, label, he_id, she_id)

        # caches for path patching
        _, clean_cache = model.run_with_cache(clean_toks, names_filter=z_name_filter, return_type=None)
        _, corrupted_cache = model.run_with_cache(corrupted_toks, names_filter=z_name_filter, return_type=None)

        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=corrupted_cache,
            orig_cache=clean_cache,
            head_to_patch=sender,
        )
        model.add_hook(z_name_filter, hook_fn)
        patched_logits = model(clean_toks)
        model.reset_hooks()

        patched_he, patched_she = gender_logits(patched_logits, end_idx, he_id, she_id)
        patched = gender_logit_diff(patched_logits, end_idx, label, he_id, she_id)
        delta = float(patched - base)
        delta_he = float(patched_he - base_he)
        delta_she = float(patched_she - base_she)

        # IOI-style token labels: which pronoun token increases/decreases most under corruption
        if delta_he >= delta_she:
            inc_tok, dec_tok = "he", "she"
        else:
            inc_tok, dec_tok = "she", "he"

        if abs(delta) < float(args.neutral_eps):
            eff = "neutral"
        elif delta < 0:
            eff = "hurt"  # corruption makes performance worse => head is contributory
        else:
            eff = "help"  # corruption improves => head is inhibitory

        sid = rec.get("sentence_id") or rec.get("id") or str(i)
        out.append(
            {
                "sentence_id": str(sid),
                "sentence_text": clean_text,
                "label": label,
                "end_idx": end_idx,
                "delta_logit_diff": delta,
                "effect": eff,
                "delta_logits": {"he": delta_he, "she": delta_she},
                "increase_token": inc_tok,
                "decrease_token": dec_tok,
            }
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved terminal effects to {out_path}")


if __name__ == "__main__":
    main()
