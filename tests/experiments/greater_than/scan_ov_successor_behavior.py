#!/usr/bin/env python3
"""
Scan OV-style successor-token support on the greater-than task.

For each attention head, on each prompt:
  - take the head output vector at END
  - identify the "successor token" after the earlier repeated prefix
  - compute x_out · W_U[successor_token]

This is a direct implementation of the copying-style logit support check,
without path patching.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, loading_from_pretrained as loading

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from greater_than.dataset import GreaterThanDataset


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="Scan OV successor-token support on greater-than prompts.")
    p.add_argument("--data-path", type=Path, default=repo_root / "greater_than_data.csv")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--model-path", type=Path, default=Path("/home/wangziran/gpt2"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--top-k", type=int, default=20, help="How many heads to print.")
    p.add_argument("--output-file", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    args.data_path = args.data_path.expanduser().resolve()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Greater-than dataset not found: {args.data_path}")

    model = load_model(args.model_name, args.model_path, args.device)
    dataset = GreaterThanDataset(
        tokenizer=model.tokenizer,
        data_path=args.data_path,
        prepend_bos=False,
        device=args.device,
    )
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
        dataset.N = len(dataset.samples)
        dataset = GreaterThanDataset(
            tokenizer=model.tokenizer,
            samples=dataset.samples,
            prepend_bos=False,
            device=args.device,
        )

    seq_len = int(dataset.input_lengths[0].item())
    end_pos = seq_len - 1
    earlier_prefix_pos = end_pos - 5
    successor_pos = end_pos - 4
    str_tokens = model.to_str_tokens(dataset.toks[0])

    # Cache all head outputs and patterns once.
    result_names = {f"blocks.{layer}.attn.hook_result" for layer in range(model.cfg.n_layers)}
    pattern_names = {f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)}
    with t.no_grad():
        _, cache = model.run_with_cache(
            dataset.toks,
            names_filter=lambda n: n in result_names or n in pattern_names,
            return_type=None,
        )

    successor_token_ids = dataset.toks[:, successor_pos].detach().cpu()
    earlier_prefix_token_ids = dataset.toks[:, earlier_prefix_pos].detach().cpu()

    head_results: List[Dict] = []
    for layer in range(model.cfg.n_layers):
        result = cache[f"blocks.{layer}.attn.hook_result"][:, end_pos, :, :].detach().cpu()  # [batch, head, d_model]
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, end_pos, : end_pos + 1].detach().cpu()  # [batch, head, pos]
        for head in range(model.cfg.n_heads):
            head_out = result[:, head, :]  # [batch, d_model]
            support_scores: List[float] = []
            support_by_prompt: List[Dict] = []
            prefix_attn: List[float] = []
            successor_attn: List[float] = []

            for i in range(dataset.N):
                succ_tok_id = int(successor_token_ids[i].item())
                unembed_vec = model.W_U[:, succ_tok_id].detach().cpu()
                score = float(t.dot(head_out[i], unembed_vec).item())
                support_scores.append(score)

                prefix_attn_val = float(pattern[i, head, earlier_prefix_pos].item())
                successor_attn_val = float(pattern[i, head, successor_pos].item())
                prefix_attn.append(prefix_attn_val)
                successor_attn.append(successor_attn_val)

                support_by_prompt.append(
                    {
                        "sample_id": i,
                        "sentence_text": dataset.samples[i].clean,
                        "successor_token": str_tokens[successor_pos],
                        "successor_token_id": succ_tok_id,
                        "earlier_prefix_token": str_tokens[earlier_prefix_pos],
                        "earlier_prefix_token_id": int(earlier_prefix_token_ids[i].item()),
                        "ov_successor_support": score,
                        "attention_to_earlier_prefix": prefix_attn_val,
                        "attention_to_successor_token": successor_attn_val,
                    }
                )

            mean_score = float(sum(support_scores) / len(support_scores))
            positive_fraction = float(sum(1 for x in support_scores if x > 0) / len(support_scores))
            mean_prefix_attn = float(sum(prefix_attn) / len(prefix_attn))
            mean_successor_attn = float(sum(successor_attn) / len(successor_attn))
            support_by_prompt.sort(key=lambda x: x["ov_successor_support"], reverse=True)

            head_results.append(
                {
                    "head": f"{layer}.{head}",
                    "scores": {
                        "mean_ov_successor_support": mean_score,
                        "positive_support_fraction": positive_fraction,
                        "mean_attention_to_earlier_prefix": mean_prefix_attn,
                        "mean_attention_to_successor_token": mean_successor_attn,
                    },
                    "top_prompts": support_by_prompt[:10],
                }
            )

    ranking = sorted(head_results, key=lambda rec: rec["scores"]["mean_ov_successor_support"], reverse=True)
    payload = {
        "meta": {
            "task": "greater_than",
            "data_path": str(args.data_path),
            "model_name": args.model_name,
            "model_path": str(args.model_path) if args.model_path else None,
            "device": args.device,
            "num_samples": dataset.N,
            "sequence_length": seq_len,
            "end_position": end_pos,
            "earlier_prefix_position": earlier_prefix_pos,
            "successor_position": successor_pos,
            "tokens_template": str_tokens,
        },
        "ranking_by_mean_ov_successor_support": [
            {
                "head": rec["head"],
                **rec["scores"],
            }
            for rec in ranking
        ],
        "heads": head_results,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTop heads by mean OV successor support:")
    for rec in ranking[: args.top_k]:
        s = rec["scores"]
        print(
            f"  {rec['head']:>5}  "
            f"mean_support={s['mean_ov_successor_support']:+.4f}  "
            f"pos_frac={s['positive_support_fraction']:.3f}  "
            f"attn_prefix={s['mean_attention_to_earlier_prefix']:.3f}  "
            f"attn_succ={s['mean_attention_to_successor_token']:.3f}"
        )
    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
