#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute sender->receiver attention diff vectors for Garden middle heads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import einops
from tqdm import tqdm

from transformer_lens import HookedTransformer, utils, loading_from_pretrained as loading
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from transformers import AutoModelForCausalLM, AutoTokenizer


LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


def load_local_hooked_transformer(local_model_dir: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(local_model_dir, local_files_only=True)
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


def load_model(device: str = "cuda"):
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return load_local_hooked_transformer(LOCAL_MODEL_DIR, device=device)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


def patch_or_freeze_head_vectors(orig_head_vector: torch.Tensor, hook: HookPoint, new_cache: ActivationCache,
                                 orig_cache: ActivationCache, head_to_patch: tuple[int, int]):
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def pad_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[1] == b.shape[1]:
        return a, b
    max_len = max(a.shape[1], b.shape[1])
    if a.shape[1] < max_len:
        a = torch.cat([a, a[:, -1:].repeat(1, max_len - a.shape[1])], dim=1)
    if b.shape[1] < max_len:
        b = torch.cat([b, b[:, -1:].repeat(1, max_len - b.shape[1])], dim=1)
    return a, b


def normalize_token(token: str) -> str:
    """Normalize token text for suffix counting."""
    return token.strip().lower()


def get_suffixed_word_map(original_text: str):
    """Map each word index to a suffixed token for disambiguating duplicates."""
    words = re.findall(r"\w+|[^\w\s]", original_text)
    global_counts = defaultdict(int)
    for w in words:
        global_counts[normalize_token(w)] += 1

    running_counts = defaultdict(int)
    suffixed_map = {}
    for i, word in enumerate(words):
        norm_word = normalize_token(word)
        running_counts[norm_word] += 1
        if global_counts[norm_word] > 1:
            suffixed_map[i] = f"{word.strip()}_{running_counts[norm_word]}"
        else:
            suffixed_map[i] = word.strip()
    return words, suffixed_map


def calculate_attention_difference(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    sender_head: tuple[int, int],
    receiver_head: tuple[int, int],
    target_pos_idx: int,
    receiver_inputs: list[str],
):
    sender_layer, sender_h_idx = sender_head
    receiver_layer, receiver_h_idx = receiver_head

    z_name_filter = lambda name: name.endswith("z")
    pattern_name_filter = lambda name: name == utils.get_act_name("pattern", receiver_layer)

    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name))
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=z_name_filter)
    clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    receiver_hook_names = {
        utils.get_act_name(ch, receiver_layer) for ch in receiver_inputs
    }
    receiver_input_filter = lambda name: name in receiver_hook_names
    _, patched_receiver_input_cache = model.run_with_cache(
        clean_tokens, names_filter=receiver_input_filter
    )
    model.reset_hooks()

    patched_pattern_cache: Dict[str, torch.Tensor] = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_hook(activation, hook):
        if hook.name in patched_receiver_input_cache:
            activation[:, :, receiver_h_idx] = patched_receiver_input_cache[hook.name][:, :, receiver_h_idx]
        return activation

    model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[
            (receiver_input_filter, patch_receiver_input_hook),
            (pattern_name_filter, cache_patched_pattern_hook),
        ],
        return_type=None,
    )
    patched_receiver_pattern = patched_pattern_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]
    avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    diff = avg_patched_pattern - avg_clean_pattern
    return diff[target_pos_idx]

def calculate_attention_difference_multi(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    sender_head: tuple[int, int],
    receiver_heads: list[tuple[int, int]],
    target_pos_idx: int,
    receiver_inputs: list[str],
):
    sender_layer, sender_h_idx = sender_head
    receiver_layers = sorted({layer for layer, _ in receiver_heads})
    receiver_layer_to_heads: Dict[int, List[int]] = {}
    for layer, head in receiver_heads:
        receiver_layer_to_heads.setdefault(layer, []).append(head)

    z_name_filter = lambda name: name.endswith("z")
    pattern_names = {utils.get_act_name("pattern", layer) for layer in receiver_layers}
    pattern_name_filter = lambda name: name in pattern_names

    _, clean_cache = model.run_with_cache(
        clean_tokens, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name)
    )
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=z_name_filter)

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    receiver_hook_names = {
        utils.get_act_name(ch, layer)
        for layer in receiver_layers
        for ch in receiver_inputs
    }
    receiver_input_filter = lambda name: name in receiver_hook_names
    _, patched_receiver_input_cache = model.run_with_cache(
        clean_tokens, names_filter=receiver_input_filter
    )
    model.reset_hooks()

    patched_pattern_cache: Dict[str, torch.Tensor] = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_hook(activation, hook):
        if hook.name in patched_receiver_input_cache:
            try:
                layer = int(hook.name.split(".")[1])
            except (IndexError, ValueError):
                return activation
            heads = receiver_layer_to_heads.get(layer, [])
            for head_idx in heads:
                activation[:, :, head_idx] = patched_receiver_input_cache[hook.name][:, :, head_idx]
        return activation

    model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[
            (receiver_input_filter, patch_receiver_input_hook),
            (pattern_name_filter, cache_patched_pattern_hook),
        ],
        return_type=None,
    )

    diffs = []
    for receiver_layer, receiver_h_idx in receiver_heads:
        clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]
        patched_receiver_pattern = patched_pattern_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]
        avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
        avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
        diff = avg_patched_pattern - avg_clean_pattern
        diffs.append(diff[target_pos_idx])

    if not diffs:
        raise ValueError("No receiver diffs computed.")
    combined = torch.stack(diffs, dim=0).mean(dim=0)
    return combined


def parse_head(head_str: str) -> tuple[int, int]:
    head_str = head_str.split(":", 1)[0]
    layer, head = head_str.split(".", 1)
    return int(layer), int(head)


def load_standard_garden(path: Path):
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是标准 garden JSON 列表")
    for idx, item in enumerate(data):
        if isinstance(item, dict) and "sample_id" not in item:
            item["sample_id"] = idx
    return data


def _select_query_index(sample: dict, sentence_len: int, attention_position: str) -> int:
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
    try:
        idx = int(pos)
        if 0 <= idx < sentence_len:
            return idx
    except ValueError:
        pass
    return max(0, sentence_len - 1)


def main():
    parser = argparse.ArgumentParser(description="Precompute sender->receiver diff vectors for garden middle heads.")
    parser.add_argument("--sender_head", required=True, help="Sender head formatted as L.H (e.g., 10.6).")
    parser.add_argument("--receiver_heads", required=True, help="Comma separated receiver heads, e.g., 10.7,11.10.")
    parser.add_argument("--standard-json", required=True, help="standard_garden_data.json")
    parser.add_argument(
        "--receiver-attention-position",
        type=str,
        default="end",
        help="Row of receiver attention pattern to evaluate (default: end).",
    )
    parser.add_argument(
        "--attention_position",
        type=str,
        default=None,
        help="(Deprecated) Alias of --receiver-attention-position.",
    )
    parser.add_argument(
        "--receiver-inputs",
        type=str,
        default="q,k,v",
        help="Comma separated receiver inputs to patch (default: q,k,v).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples from standard_garden_data.json (0 means all).",
    )
    parser.add_argument("--output_file", required=True, help="Path to write the diff dataset JSON.")
    args = parser.parse_args()

    sender_head = parse_head(args.sender_head)
    receiver_heads = [parse_head(h.strip()) for h in args.receiver_heads.split(",") if h.strip()]
    if not receiver_heads:
        raise ValueError("receiver_heads 不能为空")

    print(f"Sender head: {sender_head}, receivers: {receiver_heads}")

    records = load_standard_garden(Path(args.standard_json))
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} samples from {args.standard_json}")

    model = load_model()
    model.cfg.use_attn_result = True

    receiver_inputs = [c.strip().lower() for c in args.receiver_inputs.split(",") if c.strip()]
    if not receiver_inputs:
        raise ValueError("receiver_inputs 不能为空")
    for ch in receiver_inputs:
        if ch not in {"q", "k", "v"}:
            raise ValueError(f"非法 receiver_input: {ch}")
    attention_position = args.receiver_attention_position or args.attention_position or "end"

    dataset = {}
    for item in tqdm(records, desc="Computing diffs"):
        sentence_id = item.get("sample_id")
        clean = item.get("text") or item.get("clean")
        corrupted = item.get("corrupted_text") or item.get("corrupted")
        correct = item.get("correct_token", "")
        incorrect = item.get("incorrect_token", "")
        if clean is None or corrupted is None:
            continue

        clean_tokens = model.to_tokens(clean, prepend_bos=False)
        corrupted_tokens = model.to_tokens(corrupted, prepend_bos=False)
        clean_tokens, corrupted_tokens = pad_to_match(clean_tokens, corrupted_tokens)

        str_tokens = model.to_str_tokens(clean_tokens[0])
        target_pos_idx = _select_query_index(item, len(str_tokens), attention_position)

        record = {
            "sentence_id": sentence_id,
            "sentence_text": clean,
            "io_token": correct,
            "s_token": incorrect,
            "tokens": str_tokens,
            "diff_vectors": {},
        }

        try:
            diff = calculate_attention_difference_multi(
                model,
                clean_tokens,
                corrupted_tokens,
                sender_head,
                receiver_heads,
                target_pos_idx,
                receiver_inputs=receiver_inputs,
            )
            record["diff_vectors"][f"{sender_head[0]}.{sender_head[1]}->ALL"] = diff.cpu().tolist()
        except Exception as e:
            print(f"Warning: failed on sentence {sentence_id} for receiver group: {e}")
            continue

        # Precompute causal ground truth tokens for easy inspection.
        diff_values = [v for v in record["diff_vectors"].values() if v]
        if diff_values:
            min_len = min(len(v) for v in diff_values)
            avg_diff_vector = torch.tensor([sum(v[i] for v in diff_values) / len(diff_values) for i in range(min_len)])
            _, suffixed_map = get_suffixed_word_map(clean)
            token_scores = []
            for i in range(min_len):
                token = suffixed_map.get(i)
                if token:
                    token_scores.append({"token": token, "score": float(avg_diff_vector[i])})
            token_scores.sort(key=lambda x: x["score"], reverse=True)
            record["causal_token_scores"] = token_scores
            record["causal_ground_truth"] = {
                "top_k": 1,
                "increase": [token_scores[0]["token"]] if token_scores else [],
                "decrease": [token_scores[-1]["token"]] if token_scores else [],
            }

        dataset[str(sentence_id)] = record

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved middle-head diff dataset to {args.output_file}")


if __name__ == "__main__":
    main()
