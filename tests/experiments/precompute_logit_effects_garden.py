#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute direct logit-diff effect for a single head on the Garden task.
Outputs heads_direct_effect_on_logit_difference.json with one entry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import ActivationCache, HookedTransformer, utils, loading_from_pretrained as loading
from transformer_lens.hook_points import HookPoint

from garden_dataset import GardenDataset


def patch_or_freeze_head_vectors(
    act: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> t.Tensor:
    act.copy_(orig_cache[hook.name])
    if head_to_patch[0] == hook.layer():
        act[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return act


def load_local_hooked_transformer(local_model_dir: str, device: t.device) -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(str(local_model_dir), local_files_only=True)
    cfg = loading.get_pretrained_model_config(
        str(local_model_dir),
        device=device,
        local_files_only=True,
    )
    model = HookedTransformer(
        cfg,
        tokenizer=tokenizer,
        move_to_device=False,
    )
    state_dict = loading.get_pretrained_state_dict(
        str(local_model_dir),
        cfg,
        hf_model=hf_model,
        local_files_only=True,
    )
    model.load_and_process_state_dict(state_dict)
    model.move_model_modules_to_device()
    return model


def load_model(model_name: str, model_path: Optional[Path], device: t.device) -> HookedTransformer:
    if model_path:
        model = load_local_hooked_transformer(model_path, device)
    else:
        official_name = model_name if model_name != "gpt2-small" else "gpt2"
        model = HookedTransformer.from_pretrained(official_name, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute garden head direct logit-diff effect.")
    p.add_argument("--data-path", type=Path, required=True, help="Garden CSV data path.")
    p.add_argument("--head", type=str, required=True, help="Head to patch (layer.head).")
    p.add_argument("--model-name", type=str, default="gpt2", help="Model name.")
    p.add_argument("--model-path", type=Path, help="Local model path.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-file", type=Path, required=True, help="Output JSON file path.")
    p.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layer, head = map(int, args.head.split("."))
    head_tuple = (layer, head)

    device = t.device(args.device)
    model = load_model(args.model_name, args.model_path, device)

    dataset = GardenDataset(
        tokenizer=model.tokenizer,
        data_path=args.data_path,
        prepend_bos=False,
        device=str(device),
    )
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.N = len(dataset.samples)
        dataset = GardenDataset(
            tokenizer=model.tokenizer,
            samples=dataset.samples,
            prepend_bos=False,
            device=str(device),
        )

    corrupted = dataset.gen_flipped_prompts()

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        clean_logits, clean_cache = model.run_with_cache(dataset.toks, names_filter=z_filter)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted.toks, names_filter=z_filter)

    clean_logit_diff = dataset.logit_diff(clean_logits, mean=True).item()
    clean_logit_diff_per = dataset.logit_diff(clean_logits, mean=False).detach().cpu().tolist()
    corrupted_logit_diff = dataset.logit_diff(corrupted_logits, mean=True).item()
    corrupted_logit_diff_per = dataset.logit_diff(corrupted_logits, mean=False).detach().cpu().tolist()

    model.reset_hooks()
    hook_fn = lambda act, hook: patch_or_freeze_head_vectors(
        act, hook, corrupted_cache, clean_cache, head_tuple
    )
    model.add_hook(z_filter, hook_fn)

    resid_post_hook = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_filter = lambda name: name == resid_post_hook
    with t.no_grad():
        _, patched_cache = model.run_with_cache(dataset.toks, names_filter=resid_filter, return_type=None)
    patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook]))
    patched_logit_diff = dataset.logit_diff(patched_logits, mean=True).item()
    patched_logit_diff_per = dataset.logit_diff(patched_logits, mean=False).detach().cpu().tolist()

    raw_delta = patched_logit_diff - clean_logit_diff
    denom = clean_logit_diff - corrupted_logit_diff
    normalized_delta = raw_delta / denom if abs(denom) > 1e-12 else None
    per_sentence = []
    for idx, sample in enumerate(dataset.samples):
        sentence_id = idx
        sentence_text = sample.clean
        clean_val = clean_logit_diff_per[idx]
        corrupted_val = corrupted_logit_diff_per[idx]
        patched_val = patched_logit_diff_per[idx]
        per_denom = clean_val - corrupted_val
        per_raw_delta = patched_val - clean_val
        per_normalized_delta = per_raw_delta / per_denom if abs(per_denom) > 1e-12 else None
        per_sentence.append(
            {
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "clean_logit_diff": clean_val,
                "corrupted_logit_diff": corrupted_val,
                "patched_logit_diff": patched_val,
                "delta_logit_diff": per_raw_delta,
                "normalized_delta_logit_diff": per_normalized_delta,
            }
        )

    payload = {
        f"{layer}.{head}": raw_delta,
        "per_sentence": per_sentence,
        "scores": {
            "clean_logit_diff": clean_logit_diff,
            "corrupted_logit_diff": corrupted_logit_diff,
            "patched_logit_diff": patched_logit_diff,
            "delta_logit_diff": raw_delta,
            "normalized_delta_logit_diff": normalized_delta,
        },
        "meta": {
            "head": f"{layer}.{head}",
            "data_path": str(args.data_path),
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "max_samples": args.max_samples,
        },
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    normalized_display = "nan" if normalized_delta is None else f"{normalized_delta:.6f}"
    print(
        f"✅ 写入 logit 贡献: {args.output_file} "
        f"({layer}.{head}: raw_delta={raw_delta:.6f}, normalized_delta={normalized_display})"
    )


if __name__ == "__main__":
    main()
