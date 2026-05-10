#!/usr/bin/env python3
"""
Verify copying behavior for IOI Name Mover Heads on real prompts.

Checks two related notions:
1. OV copying score, following the IOI tutorial logic.
2. On actual IOI prompts at END, whether ablating a head reduces the logit of
   the token it attends to most strongly.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, loading_from_pretrained as loading

from ioi_dataset import IOIDataset, NAMES


NAME_MOVER_HEADS: List[Tuple[int, int]] = [(9, 6), (9, 9), (10, 0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify copying behavior for IOI name mover heads.")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--model-path", type=Path, default=Path("/home/wangziran/gpt2"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--top-k-copy", type=int, default=5, help="Top-k used for OV copying score.")
    p.add_argument("--output-file", type=Path, help="Optional JSON output path.")
    return p.parse_args()


def load_local_hooked_transformer(local_model_dir: str, device: str, dtype: t.dtype) -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    )
    cfg = loading.get_pretrained_model_config(
        local_model_dir,
        device=device,
        local_files_only=True,
    )
    model = HookedTransformer(
        cfg,
        tokenizer=tokenizer,
        move_to_device=False,
    )
    state_dict = loading.get_pretrained_state_dict(
        local_model_dir,
        cfg,
        hf_model=hf_model,
        local_files_only=True,
    )
    model.load_and_process_state_dict(state_dict)
    model.move_model_modules_to_device()
    return model


def load_model(model_name: str, model_path: Optional[Path], device: str) -> HookedTransformer:
    dtype = t.float16 if "cuda" in device else t.float32
    if model_path:
        model = load_local_hooked_transformer(str(model_path), device=device, dtype=dtype)
    else:
        official_name = "gpt2" if model_name == "gpt2-small" else model_name
        model = HookedTransformer.from_pretrained(official_name, device=device, dtype=dtype)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def get_copying_scores(model: HookedTransformer, heads: List[Tuple[int, int]], k: int) -> Dict[str, Dict[str, float]]:
    embed = model.embed
    mlp0 = model.blocks[0].mlp
    ln0 = model.blocks[0].ln2
    unembed = model.unembed
    ln_final = model.ln_final

    name_tokens = model.to_tokens(NAMES, prepend_bos=False)
    name_embeddings = embed(name_tokens)
    resid_after_mlp1 = name_embeddings + mlp0(ln0(name_embeddings))

    results: Dict[str, Dict[str, float]] = {}
    for layer, head in heads:
        w_ov = model.W_V[layer, head] @ model.W_O[layer, head]
        resid_after_ov_pos = resid_after_mlp1 @ w_ov
        resid_after_ov_neg = resid_after_mlp1 @ (-w_ov)

        logits_pos = unembed(ln_final(resid_after_ov_pos)).squeeze(1)
        logits_neg = unembed(ln_final(resid_after_ov_neg)).squeeze(1)

        topk_logits = t.topk(logits_pos, dim=-1, k=k).indices
        in_topk = (topk_logits == name_tokens).any(-1)
        bottomk_logits = t.topk(logits_neg, dim=-1, k=k).indices
        in_bottomk = (bottomk_logits == name_tokens).any(-1)

        results[f"{layer}.{head}"] = {
            "positive_copying_score_topk": float(in_topk.float().mean().item()),
            "negative_copying_score_topk": float(in_bottomk.float().mean().item()),
        }
    return results


def ablate_single_head_result(act: t.Tensor, hook, target_head: int) -> t.Tensor:
    patched = act.clone()
    patched[:, :, target_head, :] = 0.0
    return patched


def analyze_real_prompt_copying(
    model: HookedTransformer,
    dataset: IOIDataset,
    heads: List[Tuple[int, int]],
) -> Dict[str, Dict]:
    pattern_names = {f"blocks.{layer}.attn.hook_pattern" for layer, _ in heads}
    result_names = {f"blocks.{layer}.attn.hook_result" for layer, _ in heads}
    names_to_cache = pattern_names | result_names

    with t.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            dataset.toks,
            names_filter=lambda n: n in names_to_cache,
        )

    end_positions = dataset.word_idx["end"].tolist()
    str_tokens_per_prompt = [model.to_str_tokens(dataset.toks[i]) for i in range(dataset.N)]

    analyses: Dict[str, Dict] = {}

    def _to_int_token_id(value) -> int:
        if isinstance(value, int):
            return value
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    for layer, head in heads:
        head_name = f"{layer}.{head}"
        hook_pattern_name = f"blocks.{layer}.attn.hook_pattern"
        hook_result_name = f"blocks.{layer}.attn.hook_result"
        pattern = clean_cache[hook_pattern_name][:, head]  # [batch, q_pos, k_pos]

        ablate_hook = (
            hook_result_name,
            lambda act, hook, target_head=head: ablate_single_head_result(act, hook, target_head),
        )
        with t.no_grad():
            ablated_logits = model.run_with_hooks(dataset.toks, fwd_hooks=[ablate_hook], return_type="logits")

        per_prompt = []
        delta_values: List[float] = []
        positive_count = 0
        attended_token_counter: Counter[str] = Counter()
        attended_position_counter: Counter[int] = Counter()
        io_match_count = 0
        s_match_count = 0

        for i in range(dataset.N):
            end_pos = int(end_positions[i])
            attn_row = pattern[i, end_pos, : end_pos + 1]
            top_src = int(attn_row.argmax().item())
            top_weight = float(attn_row[top_src].item())
            attended_token_id = int(dataset.toks[i, top_src].item())
            attended_token_str = str_tokens_per_prompt[i][top_src]

            clean_target_logit = float(clean_logits[i, end_pos, attended_token_id].item())
            ablated_target_logit = float(ablated_logits[i, end_pos, attended_token_id].item())
            delta = clean_target_logit - ablated_target_logit

            io_token_id = _to_int_token_id(dataset.io_tokenIDs[i])
            s_token_id = _to_int_token_id(dataset.s_tokenIDs[i])
            if attended_token_id == io_token_id:
                io_match_count += 1
            if attended_token_id == s_token_id:
                s_match_count += 1

            delta_values.append(delta)
            if delta > 0:
                positive_count += 1
            attended_token_counter[attended_token_str] += 1
            attended_position_counter[top_src] += 1

            per_prompt.append(
                {
                    "prompt_index": i,
                    "sentence": dataset.sentences[i],
                    "end_position": end_pos,
                    "attended_position": top_src,
                    "attended_token": attended_token_str,
                    "attention_weight": top_weight,
                    "clean_logit_of_attended_token": clean_target_logit,
                    "ablated_logit_of_attended_token": ablated_target_logit,
                    "delta_logit_of_attended_token": delta,
                    "attended_token_is_io": attended_token_id == io_token_id,
                    "attended_token_is_s": attended_token_id == s_token_id,
                }
            )

        per_prompt.sort(key=lambda x: x["delta_logit_of_attended_token"], reverse=True)
        analyses[head_name] = {
            "summary": {
                "mean_delta_logit_of_attended_token": float(sum(delta_values) / len(delta_values)),
                "positive_delta_fraction": float(positive_count / len(delta_values)),
                "attended_token_is_io_fraction": float(io_match_count / len(delta_values)),
                "attended_token_is_s_fraction": float(s_match_count / len(delta_values)),
                "most_common_attended_tokens": attended_token_counter.most_common(10),
                "most_common_attended_positions": attended_position_counter.most_common(10),
            },
            "per_prompt": per_prompt,
        }
    return analyses


def main() -> None:
    args = parse_args()
    model = load_model(args.model_name, args.model_path, args.device)
    dataset = IOIDataset(
        prompt_type="mixed",
        N=args.n_prompts,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=args.seed,
        device=args.device,
    )

    ov_scores = get_copying_scores(model, NAME_MOVER_HEADS, args.top_k_copy)
    real_prompt_scores = analyze_real_prompt_copying(model, dataset, NAME_MOVER_HEADS)

    payload = {
        "meta": {
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "device": args.device,
            "n_prompts": args.n_prompts,
            "seed": args.seed,
            "top_k_copy": args.top_k_copy,
        },
        "heads": {
            head: {
                "ov_copying_score": ov_scores[head],
                "real_prompt_copying": real_prompt_scores[head],
            }
            for head in ov_scores
        },
    }

    for head_name in payload["heads"]:
        ov = payload["heads"][head_name]["ov_copying_score"]
        real = payload["heads"][head_name]["real_prompt_copying"]["summary"]
        print(f"\nHead {head_name}")
        print(
            f"  OV positive copying score@{args.top_k_copy}: "
            f"{ov['positive_copying_score_topk']:.3f}"
        )
        print(
            f"  Mean delta logit of attended token after ablation: "
            f"{real['mean_delta_logit_of_attended_token']:.4f}"
        )
        print(
            f"  Positive delta fraction: {real['positive_delta_fraction']:.3f}"
        )
        print(
            f"  Attended token is IO fraction: {real['attended_token_is_io_fraction']:.3f}"
        )
        print(
            f"  Most common attended tokens: {real['most_common_attended_tokens'][:5]}"
        )

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
