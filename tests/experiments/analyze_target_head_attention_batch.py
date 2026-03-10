#!/usr/bin/env python
"""Analyze target-head attention changes using batched IOIDataset caching.

This version mirrors `heads_contribution_to_heads.py` by loading all prompts into
an `IOIDataset`, caching clean & corrupted runs once, and reusing them for each
path-patching scenario. It supports multi-head-to-one patching and reports which
tokens gain or lose the most END-position attention after patching.

Example:
    python analyze_target_head_attention_batch.py \
        --target-head 10.7 \
        --scenario node10_2:10.2 \
        --scenario node9_group:9.6,9.9,10.0 \
        --scenario node10_group:10.6,10.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

# Ensure project root on import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from transformer_lens import HookedTransformer, utils, ActivationCache  # type: ignore
from transformer_lens.hook_points import HookPoint  # type: ignore

from eap_auto.tests.experiments.ioi_dataset import IOIDataset  # type: ignore

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@dataclass
class Scenario:
    label: str
    sender_heads: List[Tuple[int, int]]


def str_to_head(head: str) -> Tuple[int, int]:
    layer_str, head_str = head.split(".")
    return int(layer_str), int(head_str)


def parse_scenarios(raw: Optional[Iterable[str]]) -> List[Scenario]:
    if not raw:
        return [
            Scenario("node10_2", [(10, 2)]),
            Scenario("node9_group", [(9, 6), (9, 9), (10, 0)]),
            Scenario("node10_group", [(10, 6), (10, 10)]),
        ]
    scenarios: List[Scenario] = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"Scenario '{entry}' must be in label:head,... format")
        label, heads = entry.split(":", 1)
        heads = heads.strip()
        if not heads:
            raise ValueError(f"Scenario '{entry}' must list at least one head")
        sender_heads = [str_to_head(h.strip()) for h in heads.split(",") if h.strip()]
        scenarios.append(Scenario(label.strip(), sender_heads))
    return scenarios


def load_dataset_from_jsonl(
    path: str,
    tokenizer,
    limit: Optional[int] = None,
    prepend_bos: bool = False,
    seed: int = 1,
    device: str = "cuda",
) -> IOIDataset:
    """Load a JSONL of prompts (structured format) into an IOIDataset-like object."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entry = {
                "text": row["sentence_text"],
                "IO": row["io_token"].strip(),
                "S": row["s_token"].strip(),
                "TEMPLATE_IDX": -1,
            }
            prompts.append(entry)
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise FileNotFoundError(f"No valid prompts found in {path}")

    dataset = IOIDataset(
        prompt_type="BABA",  # dummy, not used when prompts are provided
        N=len(prompts),
        tokenizer=tokenizer,
        prompts=prompts,
        prepend_bos=prepend_bos,
        seed=seed,
        device=device,
    )
    return dataset


def group_heads_by_layer(heads: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for layer, head in heads:
        grouped.setdefault(layer, []).append(head)
    return grouped


def patch_selected_heads(
    orig_head_vector: torch.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    heads_by_layer: Dict[int, List[int]],
):
    layer = hook.layer()
    orig_head_vector[...] = orig_cache[hook.name][...]
    heads = heads_by_layer.get(layer)
    if heads:
        orig_head_vector[:, :, heads, :] = new_cache[hook.name][:, :, heads, :]
    return orig_head_vector


def main(argv: Optional[List[str]] = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Analyze target-head attention with batched path patching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-head", default="10.7", help="Target head layer.head")
    parser.add_argument("--receiver-input", choices=["q", "k", "v"], default="v")
    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input-file",
        default="../../results/ioi/path_patching/structured_sentences_standard.jsonl",
    )
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--tokenizer-path")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    if os.path.isdir(args.model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading HF model/tokenizer from local path: {args.model_name}")
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model, tokenizer=tokenizer, device=device)
    else:
        model = HookedTransformer.from_pretrained(args.model_name, device=device)
        tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.cfg.use_attn_result = True

    dataset = load_dataset_from_jsonl(
        args.input_file, tokenizer, limit=args.limit, prepend_bos=args.prepend_bos, seed=args.seed, device=str(device)
    )

    scenarios = parse_scenarios(args.scenario)
    target_layer, target_head = str_to_head(args.target_head)
    receiver_input_hook_name = utils.get_act_name(args.receiver_input, target_layer)
    pattern_hook_name = utils.get_act_name("pattern", target_layer)

    z_filter = lambda name: name.endswith("z")
    _, clean_cache_z = model.run_with_cache(dataset.toks, names_filter=z_filter, return_type=None)
    corrupted_prompts = [
        prompt["text"].replace(prompt["IO"], prompt["S"]) for prompt in dataset.ioi_prompts
    ]
    corrupted_tokens = model.to_tokens(corrupted_prompts)
    _, corrupted_cache_z = model.run_with_cache(
        corrupted_tokens, names_filter=z_filter, return_type=None
    )

    _, clean_pattern_cache = model.run_with_cache(
        dataset.toks, names_filter=lambda name: name == pattern_hook_name, return_type=None
    )
    clean_pattern = clean_pattern_cache[pattern_hook_name][:, target_head]
    clean_end_vector = clean_pattern[:, -1].detach()

    summary = {
        "target_head": args.target_head,
        "receiver_input": args.receiver_input,
        "scenarios": [],
    }

    for scenario in scenarios:
        heads_by_layer = group_heads_by_layer(scenario.sender_heads)

        hook_fn = lambda act, hook, *, heads_by_layer=heads_by_layer: patch_selected_heads(
            act, hook, corrupted_cache_z, clean_cache_z, heads_by_layer
        )

        model.add_hook(z_filter, hook_fn)
        _, patched_receiver_cache = model.run_with_cache(
            dataset.toks,
            names_filter=lambda name: name == receiver_input_hook_name,
            return_type=None,
        )
        model.reset_hooks()

        patched_receiver_tensor = patched_receiver_cache[receiver_input_hook_name]
        patched_pattern_holder: dict[str, torch.Tensor] = {}

        def cache_target_pattern_hook(activation: torch.Tensor, hook: HookPoint):
            patched_pattern_holder[hook.name] = activation.detach()

        def patch_target_input_hook(activation: torch.Tensor, hook: HookPoint):
            patched_activation = activation.clone()
            patched_activation[:, :, target_head, :] = patched_receiver_tensor[:, :, target_head, :]
            return patched_activation

        model.run_with_hooks(
            dataset.toks,
            fwd_hooks=[
                (receiver_input_hook_name, patch_target_input_hook),
                (pattern_hook_name, cache_target_pattern_hook),
            ],
            return_type=None,
        )

        patched_pattern = patched_pattern_holder[pattern_hook_name][:, target_head]
        patched_end_vector = patched_pattern[:, -1].detach()
        diff_vector = patched_end_vector - clean_end_vector

        summary["scenarios"].append(
            {
                "label": scenario.label,
                "sender_heads": [f"{layer}.{head}" for layer, head in scenario.sender_heads],
                "diff_per_prompt": diff_vector.tolist(),
            }
        )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary to {args.save_json}")

    return summary


if __name__ == "__main__":
    main()
