#!/usr/bin/env python3
import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache

from ioi_dataset import IOIDataset


LOCAL_MODEL_DIR = "/data31/private/wangziran/eap-ig/gpt2"


def load_model(device: str = "cuda"):
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return HookedTransformer.from_pretrained("gpt2", device=device, cache_dir=LOCAL_MODEL_DIR)
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)


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


def compute_target_diff(
    model,
    tokens,
    corrupted_tokens,
    sender_head,
    intermediate_head,
    target_head,
    target_pos_idx,
):
    sender_layer, sender_idx = sender_head
    inter_layer, inter_idx = intermediate_head
    target_layer, target_idx = target_head

    z_filter = lambda name: name.endswith("z")
    _, clean_cache = model.run_with_cache(tokens, names_filter=z_filter)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=z_filter)

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_idx),
    )
    model.add_hook(z_filter, hook_fn_sender)

    receiver_inputs = ["q", "k", "v"]
    inter_input_cache = {}
    for receiver_input in receiver_inputs:
        receiver_input_hook_name = utils.get_act_name(receiver_input, inter_layer)
        _, patched_inter_cache = model.run_with_cache(
            tokens, names_filter=lambda name, target=receiver_input_hook_name: name == target
        )
        inter_input_cache[receiver_input_hook_name] = patched_inter_cache[receiver_input_hook_name]
    model.reset_hooks()

    patched_target_cache = {}

    def cache_target_pattern_hook(act, hook):
        patched_target_cache[hook.name] = act.detach()
        return act

    def patch_intermediate_input_hook(act, hook):
        act[:, :, inter_idx, :] = inter_input_cache[hook.name][:, :, inter_idx, :]
        return act

    pattern_hook_name = utils.get_act_name("pattern", target_layer)
    receiver_filters = [
        (lambda name, target=utils.get_act_name(inp, inter_layer): name == target)
        for inp in receiver_inputs
    ]
    fwd_hooks = [(filt, patch_intermediate_input_hook) for filt in receiver_filters]
    fwd_hooks.append((pattern_hook_name, cache_target_pattern_hook))

    model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)

    patched_pattern = patched_target_cache[pattern_hook_name][0, target_idx]
    _, clean_target_cache = model.run_with_cache(
        tokens, names_filter=lambda name: name == pattern_hook_name
    )
    clean_pattern = clean_target_cache[pattern_hook_name][0, target_idx]
    diff = patched_pattern - clean_pattern
    return diff[target_pos_idx].cpu().tolist()


def compute_target_diff_multi(
    model,
    tokens,
    corrupted_tokens,
    sender_head,
    intermediate_heads,
    target_heads,
    target_pos_idx,
):
    sender_layer, sender_idx = sender_head
    inter_layers = sorted({layer for layer, _ in intermediate_heads})
    inter_layer_to_heads = {}
    for layer, head in intermediate_heads:
        inter_layer_to_heads.setdefault(layer, []).append(head)

    target_layers = sorted({layer for layer, _ in target_heads})
    target_layer_to_heads = {}
    for layer, head in target_heads:
        target_layer_to_heads.setdefault(layer, []).append(head)

    z_filter = lambda name: name.endswith("z")
    pattern_names = {utils.get_act_name("pattern", layer) for layer in target_layers}
    pattern_name_filter = lambda name: name in pattern_names

    _, clean_cache = model.run_with_cache(tokens, names_filter=lambda name: z_filter(name) or pattern_name_filter(name))
    _, corrupted_cache = model.run_with_cache(corrupted_tokens, names_filter=z_filter)

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_idx),
    )
    model.add_hook(z_filter, hook_fn_sender)

    receiver_inputs = ["q", "k", "v"]
    inter_hook_names = {
        utils.get_act_name(inp, layer) for layer in inter_layers for inp in receiver_inputs
    }
    inter_input_filter = lambda name: name in inter_hook_names
    _, patched_inter_cache = model.run_with_cache(tokens, names_filter=inter_input_filter)
    model.reset_hooks()

    patched_target_cache = {}

    def cache_target_pattern_hook(act, hook):
        patched_target_cache[hook.name] = act.detach()
        return act

    def patch_intermediate_input_hook(act, hook):
        if hook.name in patched_inter_cache:
            try:
                layer = int(hook.name.split(".")[1])
            except (IndexError, ValueError):
                return act
            heads = inter_layer_to_heads.get(layer, [])
            for head_idx in heads:
                act[:, :, head_idx, :] = patched_inter_cache[hook.name][:, :, head_idx, :]
        return act

    fwd_hooks = [(inter_input_filter, patch_intermediate_input_hook)]
    fwd_hooks.append((pattern_name_filter, cache_target_pattern_hook))

    model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)

    diffs = []
    for target_layer, target_idx in target_heads:
        pattern_hook_name = utils.get_act_name("pattern", target_layer)
        patched_pattern = patched_target_cache[pattern_hook_name][0, target_idx]
        clean_pattern = clean_cache[pattern_hook_name][0, target_idx]
        diff = patched_pattern - clean_pattern
        diffs.append(diff[target_pos_idx])

    if not diffs:
        raise ValueError("No target diffs computed.")
    combined = torch.stack(diffs, dim=0).mean(dim=0)
    return combined.cpu().tolist()


def load_structured_sentences(path: Path):
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

    sentences = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            normalized = _normalize(item, len(sentences))
            if normalized:
                sentences.append(normalized)
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


def pad_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[1] == b.shape[1]:
        return a, b
    max_len = max(a.shape[1], b.shape[1])
    if a.shape[1] < max_len:
        a = torch.cat([a, a[:, -1:].repeat(1, max_len - a.shape[1])], dim=1)
    if b.shape[1] < max_len:
        b = torch.cat([b, b[:, -1:].repeat(1, max_len - b.shape[1])], dim=1)
    return a, b


def load_garden_standard(path: Path):
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    for idx, item in enumerate(data):
        if isinstance(item, dict) and "sample_id" not in item:
            item["sample_id"] = idx
    return data


def select_position_index_garden(item: dict, tokens: List[str], attention_position: str) -> int:
    attention_position = (attention_position or "end").lower()
    wd = item.get("word_idx") or {}
    if attention_position in {"subj", "verb", "obj_head", "rel_pron", "rel_verb"}:
        idx = wd.get(attention_position)
        if isinstance(idx, int) and 0 <= idx < len(tokens):
            return idx
    if attention_position == "end":
        end_idx = item.get("end_idx")
        if isinstance(end_idx, int):
            return min(end_idx, len(tokens) - 1)
    try:
        idx = int(attention_position)
        if 0 <= idx < len(tokens):
            return idx
    except ValueError:
        pass
    return len(tokens) - 1


def main():
    parser = argparse.ArgumentParser(description="Precompute sender→target effects via intermediate heads.")
    parser.add_argument("--sender_head", required=True, help="Sender head (e.g., 0.1)")
    parser.add_argument("--intermediate_heads", required=True, help="Comma separated intermediate heads (e.g., 7.9,8.6,8.10)")
    parser.add_argument("--target_head", help="Target head to observe (e.g., 9.6)")
    parser.add_argument("--target_heads", help="Comma separated target heads (e.g., 9.6,9.9,10.0)")
    parser.add_argument("--sentences_file", help="Structured sentences JSONL (IOI)")
    parser.add_argument("--standard-json", help="standard_garden_data.json (Garden)")
    parser.add_argument("--output_file", required=True, help="Output dataset path")
    parser.add_argument("--receiver_input", type=str, default="q", help="(Deprecated) Input hook for intermediates.")
    parser.add_argument("--attention_position", type=str, default="end", help="Row of attention pattern to evaluate (e.g., end, s2).")
    args = parser.parse_args()

    sender_head = tuple(map(int, args.sender_head.split(".")))
    intermediate_heads = [tuple(map(int, h.split("."))) for h in args.intermediate_heads.split(",") if h.strip()]
    if args.target_heads:
        target_heads = [tuple(map(int, h.split("."))) for h in args.target_heads.split(",") if h.strip()]
    elif args.target_head:
        target_heads = [tuple(map(int, args.target_head.split(".")))]
    else:
        raise ValueError("Either --target_head or --target_heads must be provided.")

    print(f"Sender head: {sender_head}, target heads: {target_heads}, intermediates: {intermediate_heads}")

    use_garden = bool(args.standard_json)
    if use_garden:
        sentences = load_garden_standard(Path(args.standard_json))
        print(f"Loaded {len(sentences)} garden samples from {args.standard_json}")
    else:
        if not args.sentences_file:
            raise ValueError("Either --sentences_file or --standard-json must be provided.")
        sentences = load_structured_sentences(Path(args.sentences_file))
        print(f"Loaded {len(sentences)} sentences from {args.sentences_file}")

    model = load_model()
    model.cfg.use_attn_result = True

    dataset = {}
    sender_prefix = f"{sender_head[0]}.{sender_head[1]}->"

    for entry in tqdm(sentences, desc="Computing sender→target diffs"):
        if use_garden:
            sentence_id = entry.get("sample_id")
            text = entry.get("text") or entry.get("clean")
            corrupted = entry.get("corrupted_text") or entry.get("corrupted")
            io_token = entry.get("correct_token", "").strip()
            s_token = entry.get("incorrect_token", "").strip()
            if sentence_id is None or not text or not corrupted:
                continue
            tokens = model.to_tokens(text, prepend_bos=False)
            corrupted_tokens = model.to_tokens(corrupted, prepend_bos=False)
            tokens, corrupted_tokens = pad_to_match(tokens, corrupted_tokens)
            str_tokens = model.to_str_tokens(tokens[0])
            target_pos_idx = select_position_index_garden(entry, str_tokens, args.attention_position)
        else:
            sentence_id = entry.get("sentence_id")
            text = entry.get("sentence_text")
            io_token = entry.get("io_token", "").strip()
            s_token = entry.get("s_token", "").strip()
            if not sentence_id or not text:
                continue

            prompt = {
                "text": text,
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
                prompts=[prompt],
            )
            ioi_dataset.toks = model.to_tokens(text, prepend_bos=False)
            abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

            tokens = ioi_dataset.toks
            corrupted_tokens = abc_dataset.toks
            str_tokens = model.to_str_tokens(tokens[0])
            target_pos_idx = select_position_index(str_tokens, args.attention_position, s_token, io_token)

        diff_vectors = {}
        try:
            diff_sum = compute_target_diff_multi(
                model,
                tokens,
                corrupted_tokens,
                sender_head,
                intermediate_heads,
                target_heads,
                target_pos_idx,
            )
        except Exception as e:
            print(f"Warning: failed for sentence {sentence_id} with intermediates {intermediate_heads} targets {target_heads}: {e}")
            diff_sum = [0.0 for _ in str_tokens]

        diff_vectors[f"{sender_prefix}ALL"] = diff_sum

        dataset[str(sentence_id)] = {
            "sentence_id": sentence_id,
            "sentence_text": text,
            "io_token": io_token,
            "s_token": s_token,
            "tokens": str_tokens,
            "diff_vectors": diff_vectors,
        }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved sender→target dataset to {out_path}")


if __name__ == "__main__":
    main()
