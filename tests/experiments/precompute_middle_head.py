#!/usr/bin/env python3
import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import List

import torch
import einops
from tqdm import tqdm

from transformer_lens import HookedTransformer, utils, loading_from_pretrained as loading
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from ioi_dataset import IOIDataset

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


def calculate_attention_difference(model, ioi_dataset, abc_dataset, sender_head, receiver_head, target_pos_idx):
    sender_layer, sender_h_idx = sender_head
    receiver_layer, receiver_h_idx = receiver_head

    z_name_filter = lambda name: name.endswith("z")
    pattern_name_filter = lambda name: name == utils.get_act_name("pattern", receiver_layer)

    _, clean_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name))
    _, corrupted_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter)
    clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    receiver_inputs = ["q", "k", "v"]
    receiver_input_cache = {}
    for receiver_input in receiver_inputs:
        receiver_input_hook_name = utils.get_act_name(receiver_input, receiver_layer)
        receiver_input_filter = lambda name, target=receiver_input_hook_name: name == target
        _, patched_receiver_input_cache = model.run_with_cache(
            ioi_dataset.toks, names_filter=receiver_input_filter
        )
        receiver_input_cache[receiver_input_hook_name] = patched_receiver_input_cache[receiver_input_hook_name]
    model.reset_hooks()

    patched_pattern_cache = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_hook(activation, hook):
        activation[:, :, receiver_h_idx] = receiver_input_cache[hook.name][:, :, receiver_h_idx]
        return activation

    receiver_filters = [
        (lambda name, target=utils.get_act_name(inp, receiver_layer): name == target)
        for inp in receiver_inputs
    ]
    fwd_hooks = [(filt, patch_receiver_input_hook) for filt in receiver_filters]
    fwd_hooks.append((pattern_name_filter, cache_patched_pattern_hook))

    model.run_with_hooks(ioi_dataset.toks, fwd_hooks=fwd_hooks, return_type=None)
    patched_receiver_pattern = patched_pattern_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]
    avg_clean_pattern = einops.reduce(clean_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_receiver_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    diff = avg_patched_pattern - avg_clean_pattern
    return diff[target_pos_idx]


def calculate_attention_difference_multi(model, ioi_dataset, abc_dataset, sender_head, receiver_heads, target_pos_idx):
    sender_layer, sender_h_idx = sender_head
    receiver_layers = sorted({layer for layer, _ in receiver_heads})
    receiver_layer_to_heads = {}
    for layer, head in receiver_heads:
        receiver_layer_to_heads.setdefault(layer, []).append(head)

    z_name_filter = lambda name: name.endswith("z")
    pattern_names = {utils.get_act_name("pattern", layer) for layer in receiver_layers}
    pattern_name_filter = lambda name: name in pattern_names

    _, clean_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name)
    )
    _, corrupted_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter)

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    receiver_inputs = ["q", "k", "v"]
    receiver_hook_names = {
        utils.get_act_name(inp, layer) for layer in receiver_layers for inp in receiver_inputs
    }
    receiver_input_filter = lambda name: name in receiver_hook_names
    _, patched_receiver_input_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=receiver_input_filter
    )
    model.reset_hooks()

    patched_pattern_cache = {}

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
        ioi_dataset.toks,
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


def load_sentences(path: Path):
    """同时兼容 standard JSON（含 samples 字段）与 JSONL，并归一化字段。"""
    text = path.read_text().strip()
    if not text:
        return []
    def _normalize(item, idx):
        if not isinstance(item, dict):
            return None
        if "clean" in item and isinstance(item.get("clean"), dict):
            clean = item["clean"]
            return {
                "sentence_id": str(item.get("sample_id", idx)),
                "sentence_text": clean.get("sentence", ""),
                "io_token": clean.get("io_token", ""),
                "s_token": clean.get("s_token", ""),
                "positions": item.get("positions", {}),
            }
        return {
            "sentence_id": str(item.get("sentence_id", item.get("sample_id", idx))),
            "sentence_text": item.get("sentence_text", item.get("text", "")),
            "io_token": item.get("io_token", ""),
            "s_token": item.get("s_token", ""),
            "positions": item.get("positions", {}),
        }
    # 优先尝试整体 JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "samples" in data:
            out = []
            for idx, item in enumerate(data["samples"]):
                normalized = _normalize(item, idx)
                if normalized:
                    out.append(normalized)
            return out
        if isinstance(data, list):
            out = []
            for idx, item in enumerate(data):
                normalized = _normalize(item, idx)
                if normalized:
                    out.append(normalized)
            return out
    except json.JSONDecodeError:
        pass

    # 回退为 JSONL
    sentences = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                normalized = _normalize(item, len(sentences))
                if normalized:
                    sentences.append(normalized)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse line in {path}: {line[:50]}...") from e
    return sentences


def find_token_occurrence(tokens: List[str], target: str, occurrence: int = 1):
    count = 0
    for idx, tok in enumerate(tokens):
        if tok.strip() == target:
            count += 1
            if count == occurrence:
                return idx
    return None


def select_position_index(tokens: List[str], attention_position: str, s_token: str, io_token: str):
    attention_position = (attention_position or "end").lower()
    if attention_position == "end":
        return len(tokens) - 1
    if attention_position == "s1":
        return find_token_occurrence(tokens, s_token, 1) or len(tokens) - 1
    if attention_position == "s2":
        return find_token_occurrence(tokens, s_token, 2) or find_token_occurrence(tokens, s_token, 1) or len(tokens) - 1
    if attention_position in {"io", "io1"}:
        return find_token_occurrence(tokens, io_token, 1) or len(tokens) - 1
    if attention_position == "io2":
        return find_token_occurrence(tokens, io_token, 2) or find_token_occurrence(tokens, io_token, 1) or len(tokens) - 1
    try:
        idx = int(attention_position)
        if 0 <= idx < len(tokens):
            return idx
    except ValueError:
        pass
    return len(tokens) - 1


def main():
    parser = argparse.ArgumentParser(description="Precompute sender->receiver diff vectors for middle heads.")
    parser.add_argument("--sender_head", required=True, help="Sender head formatted as L.H (e.g., 10.6).")
    parser.add_argument("--receiver_heads", required=True, help="Comma separated receiver heads, e.g., 10.7,11.10.")
    parser.add_argument("--sentences_file", type=str, default="/home/wangziran/eap_auto/results/ioi/path_patching/structured_sentences.jsonl", help="Structured sentences JSONL.")
    parser.add_argument("--attention_position", type=str, default="end", help="Row of attention pattern to evaluate (e.g., end, s2).")
    parser.add_argument("--output_file", required=True, help="Path to write the diff dataset JSON.")
    args = parser.parse_args()

    sender_head = parse_head(args.sender_head)
    receiver_heads = [parse_head(h.strip()) for h in args.receiver_heads.split(",") if h.strip()]
    if not receiver_heads:
        raise ValueError("receiver_heads 不能为空")

    print(f"Sender head: {sender_head}, receivers: {receiver_heads}")

    sentences = load_sentences(Path(args.sentences_file))
    print(f"Loaded {len(sentences)} sentences from {args.sentences_file}")

    model = load_model()
    model.cfg.use_attn_result = True

    dataset = {}

    for item in tqdm(sentences, desc="Computing diffs"):
        sentence_id = item.get("sentence_id")
        sentence_text = item.get("sentence_text")
        io_token_raw = item.get("io_token", "")
        s_token_raw = item.get("s_token", "")
        io_token = io_token_raw.strip()
        s_token = s_token_raw.strip()
        if not sentence_id or not sentence_text:
            continue

        prompt_entry = {
            "text": sentence_text,
            "IO": io_token,
            "S": s_token,
            "TEMPLATE_IDX": 0,
        }
        ioi_dataset = IOIDataset(
            prompt_type="mixed",
            N=1,
            tokenizer=model.tokenizer,
            prepend_bos=False,
            device="cuda",
            prompts=[prompt_entry],
        )
        ioi_dataset.toks = model.to_tokens(sentence_text, prepend_bos=False)
        abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

        str_tokens = model.to_str_tokens(ioi_dataset.toks[0])
        target_pos_idx = select_position_index(str_tokens, args.attention_position, s_token, io_token)
        record = {
            "sentence_id": sentence_id,
            "sentence_text": sentence_text,
            "io_token": io_token,
            "s_token": s_token,
            "tokens": str_tokens,
            "diff_vectors": {},
        }

        try:
            diff = calculate_attention_difference_multi(
                model, ioi_dataset, abc_dataset, sender_head, receiver_heads, target_pos_idx
            )
            record["diff_vectors"][f"{sender_head[0]}.{sender_head[1]}->ALL"] = diff.cpu().tolist()
        except Exception as e:
            print(f"Warning: failed on sentence {sentence_id} for receiver group: {e}")
            continue

        dataset[sentence_id] = record

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved middle-head diff dataset to {args.output_file}")


if __name__ == "__main__":
    main()
