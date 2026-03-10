#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IOI-style terminal causal ground truth for agr_gender:

For each sample, run head->logits path patching (patch sender head z from corrupted prompt into clean prompt),
then compute per-token logit changes at the prediction position. We restrict tokens to those appearing in the
clean prompt (unique token IDs), mirroring IOI's precompute_logit_effects.py.

Outputs a JSON file with the same high-level schema expected by preprocess_causal_effects.py:
{
  "experiment_info": {...},
  "results": [
    {
      "sample_id": ...,
      "sentence_text": ...,
      "tokens": [...],
      "head_patched": "L.H",
      "logit_analysis_at_end_pos": {"sentence_token_logits": [{"token":..., "clean_logit":..., "patched_logit":..., "change":...}, ...]}
    }, ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

LOCAL_MODEL_DIR = "/data31/private/wangziran/eap-ig/gpt2"


def load_model(device: str = "cuda") -> HookedTransformer:
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return HookedTransformer.from_pretrained("gpt2", device=device, cache_dir=LOCAL_MODEL_DIR)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


def parse_head(head_str: str) -> Tuple[int, int]:
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
    orig_head_vector: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
):
    """
    Freeze all heads to orig_cache, except head_to_patch uses new_cache.
    """
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def analyze_single_sample(
    model: HookedTransformer,
    sample: Dict[str, Any],
    head_to_patch: Tuple[int, int],
    device: str,
    fallback_id: str,
) -> Dict[str, Any] | None:
    clean_text = sample.get("clean") or sample.get("text")
    corrupted_text = sample.get("corrupted") or sample.get("corrupted_text")
    if not isinstance(clean_text, str) or not clean_text.strip():
        return None
    if not isinstance(corrupted_text, str) or not corrupted_text.strip():
        return None

    word_idx = sample.get("word_idx") or {}
    end_idx = word_idx.get("end")
    if end_idx is None:
        end_idx = sample.get("end_idx")
    if end_idx is None:
        # fallback: last token
        end_idx = len((sample.get("tokenized_clean") or [])) - 1
    end_idx = int(end_idx)

    clean_tokens = model.to_tokens(clean_text, prepend_bos=False).to(device)
    corrupted_tokens = model.to_tokens(corrupted_text, prepend_bos=False).to(device)

    # ensure end_idx in bounds
    actual_end_idx = min(end_idx, clean_tokens.shape[1] - 1)

    model.reset_hooks()
    z_name_filter = lambda name: name.endswith("z")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=z_name_filter)

    model.reset_hooks()
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=z_name_filter)

    hook_fn = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=head_to_patch,
    )
    model.add_hook(z_name_filter, hook_fn)
    patched_logits = model(clean_tokens)
    model.reset_hooks()

    clean_end_logits = clean_logits[0, actual_end_idx]
    patched_end_logits = patched_logits[0, actual_end_idx]

    # Only consider token IDs that appear in the clean prompt (IOI-style)
    sentence_token_ids = t.unique(clean_tokens[0]).detach().cpu()
    sentence_token_logits = []
    for token_id in sentence_token_ids:
        tid = int(token_id)
        sentence_token_logits.append(
            {
                "token": model.to_string(tid),
                "clean_logit": float(clean_end_logits[tid].detach().cpu()),
                "patched_logit": float(patched_end_logits[tid].detach().cpu()),
                "change": float((patched_end_logits[tid] - clean_end_logits[tid]).detach().cpu()),
            }
        )
    sentence_token_logits.sort(key=lambda x: float(x.get("change", 0.0)), reverse=True)

    sid = sample.get("sentence_id") or sample.get("id") or sample.get("sample_id") or fallback_id
    return {
        "sample_id": str(sid) if sid is not None else "",
        "sentence_text": clean_text,
        "tokens": model.to_str_tokens(clean_tokens[0].detach().cpu()),
        "head_patched": f"{head_to_patch[0]}.{head_to_patch[1]}",
        "logit_analysis_at_end_pos": {"sentence_token_logits": sentence_token_logits},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute IOI-style token logit effects for agr_gender terminal heads.")
    p.add_argument("--input_file", required=True, help="standard_gender_data.json")
    p.add_argument("--output_file", required=True, help="Path to final_logit_effects_head_L_H.json")
    p.add_argument("--head_to_patch", required=True, help="Sender head (L.H)")
    p.add_argument("--max_samples", type=int, default=200, help="How many samples to process (0=all)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    head_to_patch = parse_head(args.head_to_patch)
    device = args.device

    print(f"--- 开始分析 (agr_gender), Patching Head: {args.head_to_patch} ---")
    model = load_model(device=device)
    model.cfg.use_attn_result = True

    records = load_standard_json(Path(args.input_file))
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"将处理 {len(records)} 个样本")

    results: List[Dict[str, Any]] = []
    for i, sample in tqdm(list(enumerate(records)), desc="分析句子"):
        try:
            r = analyze_single_sample(model, sample, head_to_patch, device=device, fallback_id=str(i))
            if r:
                results.append(r)
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

    out = {
        "experiment_info": {
            "head_patched": f"{head_to_patch[0]}.{head_to_patch[1]}",
            "total_samples": len(results),
            "model_name": "gpt2",
            "analysis_type": "sentence_token_logits_before_after_patch",
            "task": "agr_gender",
        },
        "results": results,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 分析完成，结果已保存到: {out_path}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
