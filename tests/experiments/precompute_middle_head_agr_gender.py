#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute sender->receiver attention-pattern diff vectors for agr_gender.

Modes:
  - A->B (group avg): patch sender head A, compute per-receiver diffs for B heads,
    then store ONLY the average diff (as a single vector).
  - A->B->C (optional): patch sender A into B-group inputs (jointly), then measure
    target head C's attention pattern change; store under diff_vectors_abc.

Output JSON dict keyed by sentence_id:
  {
    "<id>": {
      "sentence_id": "...",
      "sentence_text": "...",
      "tokens": [...],
      "diff_vectors": {"A->B_GROUP": [...]},
      "diff_vectors_abc": {"A->B_GROUP->C": [...], ...}  # optional
    },
    ...
  }
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import einops
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint


LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


def load_model(device: str = "cuda") -> HookedTransformer:
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


def _receiver_input_hook_name(receiver_input: str, layer: int) -> str:
    return utils.get_act_name(receiver_input, layer)


def _pattern_hook_name(layer: int) -> str:
    return utils.get_act_name("pattern", layer)


def calculate_attention_difference(
    model: HookedTransformer,
    clean_toks: torch.Tensor,
    corrupted_toks: torch.Tensor,
    sender_head: tuple[int, int],
    receiver_head: tuple[int, int],
    query_pos_idx: int,
    receiver_input: str = "q",
) -> torch.Tensor:
    sender_layer, sender_h_idx = sender_head
    receiver_layer, receiver_h_idx = receiver_head

    z_name_filter = lambda name: name.endswith("z")
    pattern_name_filter = lambda name: name == utils.get_act_name("pattern", receiver_layer)

    _, clean_cache = model.run_with_cache(
        clean_toks,
        names_filter=lambda name: z_name_filter(name) or pattern_name_filter(name),
        return_type=None,
    )
    _, corrupted_cache = model.run_with_cache(
        corrupted_toks,
        names_filter=z_name_filter,
        return_type=None,
    )

    clean_receiver_pattern = clean_cache[utils.get_act_name("pattern", receiver_layer)][:, receiver_h_idx, :, :]

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)

    receiver_input_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    receiver_input_filter = lambda name: name == receiver_input_hook_name
    _, patched_receiver_input_cache = model.run_with_cache(
        clean_toks, names_filter=receiver_input_filter, return_type=None
    )
    model.reset_hooks()

    patched_pattern_cache: Dict[str, torch.Tensor] = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_hook(activation, hook):
        activation[:, :, receiver_h_idx] = patched_receiver_input_cache[hook.name][:, :, receiver_h_idx]
        return activation

    model.run_with_hooks(
        clean_toks,
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
    return diff[query_pos_idx]


def calculate_attention_difference_group(
    model: HookedTransformer,
    clean_toks: torch.Tensor,
    corrupted_toks: torch.Tensor,
    sender_head: tuple[int, int],
    receiver_heads: Sequence[tuple[int, int]],
    query_pos_idx: int,
    receiver_input: str = "q",
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    A->B group patching in one pass:
      - Patch sender A
      - Capture all receiver inputs for B heads (same receiver_input, all layers)
      - Inject all B receiver inputs jointly
      - Cache patterns for all B heads, compute diff at query_pos
    Returns per-receiver diff vectors.
    """
    sender_layer, sender_h_idx = sender_head
    recv_layers = sorted({lh for (lh, _h) in receiver_heads})
    recv_head_set = {(lh, h) for (lh, h) in receiver_heads}

    z_name_filter = lambda name: name.endswith("z")
    recv_pattern_names = {utils.get_act_name("pattern", l) for l in recv_layers}
    recv_pattern_filter = lambda name: name in recv_pattern_names
    recv_input_names = {_receiver_input_hook_name(receiver_input, l) for l in recv_layers}
    recv_input_filter = lambda name: name in recv_input_names

    # clean patterns for all receiver layers
    _, clean_cache = model.run_with_cache(
        clean_toks,
        names_filter=lambda name: z_name_filter(name) or recv_pattern_filter(name),
        return_type=None,
    )
    # corrupted cache for z
    _, corrupted_cache = model.run_with_cache(
        corrupted_toks,
        names_filter=z_name_filter,
        return_type=None,
    )

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)
    _, patched_recv_input_cache = model.run_with_cache(
        clean_toks, names_filter=recv_input_filter, return_type=None
    )
    model.reset_hooks()

    patched_pattern_cache: Dict[str, torch.Tensor] = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_group(activation, hook):
        if hook.name not in patched_recv_input_cache:
            return activation
        for (layer, h) in recv_head_set:
            if layer == hook.layer():
                activation[:, :, h] = patched_recv_input_cache[hook.name][:, :, h]
        return activation

    model.run_with_hooks(
        clean_toks,
        fwd_hooks=[
            (recv_input_filter, patch_receiver_input_group),
            (recv_pattern_filter, cache_patched_pattern_hook),
        ],
        return_type=None,
    )

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for (layer, h) in receiver_heads:
        pat_name = utils.get_act_name("pattern", layer)
        clean_pattern = clean_cache[pat_name][:, h, :, :]
        patched_pattern = patched_pattern_cache[pat_name][:, h, :, :]
        avg_clean = einops.reduce(clean_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
        avg_patched = einops.reduce(patched_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
        diff = avg_patched - avg_clean
        out[(layer, h)] = diff[query_pos_idx]
    return out


def calculate_attention_difference_group_on_target(
    model: HookedTransformer,
    clean_toks: torch.Tensor,
    corrupted_toks: torch.Tensor,
    sender_head: tuple[int, int],
    receiver_heads: Sequence[tuple[int, int]],
    target_head: tuple[int, int],
    query_pos_idx: int,
    receiver_input: str = "q",
) -> torch.Tensor:
    """
    A->B->C: patch sender A, capture B-group receiver inputs, inject them jointly,
    and measure target head C's attention pattern change.
    """
    sender_layer, sender_h_idx = sender_head
    target_layer, target_h_idx = target_head

    z_name_filter = lambda name: name.endswith("z")
    recv_layers = sorted({lh for (lh, _h) in receiver_heads})
    recv_hook_names = {_receiver_input_hook_name(receiver_input, l) for l in recv_layers}
    recv_filter = lambda name: name in recv_hook_names
    target_pattern_name = _pattern_hook_name(target_layer)
    target_pattern_filter = lambda name: name == target_pattern_name

    # clean & corrupted caches for z
    _, clean_cache = model.run_with_cache(clean_toks, names_filter=z_name_filter, return_type=None)
    _, corrupted_cache = model.run_with_cache(corrupted_toks, names_filter=z_name_filter, return_type=None)

    # patch sender z, then cache receiver inputs for all B heads
    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)
    _, patched_recv_cache = model.run_with_cache(clean_toks, names_filter=recv_filter, return_type=None)
    model.reset_hooks()

    # cache clean target pattern
    _, clean_target_cache = model.run_with_cache(clean_toks, names_filter=target_pattern_filter, return_type=None)
    clean_pattern = clean_target_cache[target_pattern_name][:, target_h_idx, :, :]

    patched_pattern_cache: Dict[str, torch.Tensor] = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    # patch receiver inputs for ALL B heads jointly
    def patch_receiver_input_group(activation, hook):
        # hook.name corresponds to one layer's receiver_input
        if hook.name not in patched_recv_cache:
            return activation
        # patch all heads in this layer that are in receiver_heads
        for (layer, h) in receiver_heads:
            if layer == hook.layer():
                activation[:, :, h] = patched_recv_cache[hook.name][:, :, h]
        return activation

    model.run_with_hooks(
        clean_toks,
        fwd_hooks=[
            (recv_filter, patch_receiver_input_group),
            (target_pattern_filter, cache_patched_pattern_hook),
        ],
        return_type=None,
    )
    patched_pattern = patched_pattern_cache[target_pattern_name][:, target_h_idx, :, :]

    avg_clean_pattern = einops.reduce(clean_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    avg_patched_pattern = einops.reduce(patched_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
    diff = avg_patched_pattern - avg_clean_pattern
    return diff[query_pos_idx]


def calculate_attention_difference_group_on_targets(
    model: HookedTransformer,
    clean_toks: torch.Tensor,
    corrupted_toks: torch.Tensor,
    sender_head: tuple[int, int],
    receiver_heads: Sequence[tuple[int, int]],
    target_heads: Sequence[tuple[int, int]],
    query_pos_idx: int,
    receiver_input: str = "q",
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    A->B->C group patching in one pass:
      - Patch sender A
      - Capture receiver inputs for B group
      - Inject all B inputs jointly
      - Cache patterns for all target heads C
    Returns per-target diff vectors.
    """
    sender_layer, sender_h_idx = sender_head
    recv_layers = sorted({lh for (lh, _h) in receiver_heads})
    recv_head_set = {(lh, h) for (lh, h) in receiver_heads}
    target_layers = sorted({lh for (lh, _h) in target_heads})

    z_name_filter = lambda name: name.endswith("z")
    recv_input_names = {_receiver_input_hook_name(receiver_input, l) for l in recv_layers}
    recv_input_filter = lambda name: name in recv_input_names
    target_pattern_names = {utils.get_act_name("pattern", l) for l in target_layers}
    target_pattern_filter = lambda name: name in target_pattern_names

    # clean cache for target patterns
    _, clean_target_cache = model.run_with_cache(
        clean_toks,
        names_filter=lambda name: z_name_filter(name) or target_pattern_filter(name),
        return_type=None,
    )
    # corrupted cache for z
    _, corrupted_cache = model.run_with_cache(
        corrupted_toks,
        names_filter=z_name_filter,
        return_type=None,
    )

    hook_fn_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=corrupted_cache,
        orig_cache=clean_target_cache,
        head_to_patch=(sender_layer, sender_h_idx),
    )
    model.add_hook(z_name_filter, hook_fn_sender)
    _, patched_recv_input_cache = model.run_with_cache(
        clean_toks, names_filter=recv_input_filter, return_type=None
    )
    model.reset_hooks()

    patched_pattern_cache: Dict[str, torch.Tensor] = {}

    def cache_patched_pattern_hook(activation, hook):
        patched_pattern_cache[hook.name] = activation
        return activation

    def patch_receiver_input_group(activation, hook):
        if hook.name not in patched_recv_input_cache:
            return activation
        for (layer, h) in recv_head_set:
            if layer == hook.layer():
                activation[:, :, h] = patched_recv_input_cache[hook.name][:, :, h]
        return activation

    model.run_with_hooks(
        clean_toks,
        fwd_hooks=[
            (recv_input_filter, patch_receiver_input_group),
            (target_pattern_filter, cache_patched_pattern_hook),
        ],
        return_type=None,
    )

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for (layer, h) in target_heads:
        pat_name = utils.get_act_name("pattern", layer)
        clean_pattern = clean_target_cache[pat_name][:, h, :, :]
        patched_pattern = patched_pattern_cache[pat_name][:, h, :, :]
        avg_clean = einops.reduce(clean_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
        avg_patched = einops.reduce(patched_pattern, "batch pos_q pos_k -> pos_q pos_k", "mean")
        diff = avg_patched - avg_clean
        out[(layer, h)] = diff[query_pos_idx]
    return out


def parse_head(head_str: str) -> tuple[int, int]:
    head_str = head_str.split(":", 1)[0].strip()
    layer, head = head_str.split(".", 1)
    return int(layer), int(head)


def load_standard_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {path}")


def resolve_query_pos(record: Dict[str, Any], attention_position: str, n_tokens: int) -> int:
    pos = (attention_position or "end").lower().strip()
    word_idx = record.get("word_idx") or {}
    if isinstance(word_idx, dict) and pos in {"end", "verb", "a1", "b", "a2"}:
        v = word_idx.get(pos)
        if v is None:
            v = word_idx.get(pos.upper())
        if isinstance(v, int):
            return min(max(v, 0), n_tokens - 1)
    try:
        v = int(pos)
        return min(max(v, 0), n_tokens - 1)
    except Exception:
        return n_tokens - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute sender->receiver diff vectors for agr_gender.")
    parser.add_argument("--sender_head", required=True, help="Sender head formatted as L.H (e.g., 10.6).")
    parser.add_argument("--receiver_heads", required=True, help="Comma separated receiver heads, e.g., 10.7,11.10.")
    parser.add_argument(
        "--target_heads",
        type=str,
        default="",
        help="Optional comma-separated target heads for A->B->C (e.g., 10.7,11.10).",
    )
    parser.add_argument("--data_json", required=True, help="standard_gender_data.json")
    parser.add_argument("--attention_position", type=str, default="end", help="end/verb/a1/b/a2")
    parser.add_argument("--receiver_input", type=str, default="q", help="receiver input hook (q/k/v)")
    parser.add_argument("--output_file", required=True, help="Path to write the diff dataset JSON.")
    parser.add_argument(
        "--group_avg_only",
        action="store_true",
        help="If set, store only the average A->B group diff (single vector).",
    )
    parser.add_argument(
        "--group_patch",
        action="store_true",
        help="Use group patching (one pass for all receivers/targets).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sender_head = parse_head(args.sender_head)
    receiver_heads = [parse_head(h.strip()) for h in args.receiver_heads.split(",") if h.strip()]
    if not receiver_heads:
        raise ValueError("receiver_heads 不能为空")
    target_heads = [parse_head(h.strip()) for h in args.target_heads.split(",") if h.strip()]

    records = load_standard_json(Path(args.data_json))
    print(f"Sender head: {sender_head}, receivers: {receiver_heads}")
    print(f"Loaded {len(records)} samples from {args.data_json}")

    model = load_model(device=args.device)
    model.cfg.use_attn_result = True

    dataset: Dict[str, Dict[str, Any]] = {}
    for i, rec in tqdm(list(enumerate(records)), desc="Computing diffs"):
        clean_text = rec.get("clean") or rec.get("text") or rec.get("sentence_text")
        corrupted_text = rec.get("corrupted") or rec.get("corrupted_text")
        if not isinstance(clean_text, str) or not clean_text.strip():
            continue
        if not isinstance(corrupted_text, str) or not corrupted_text.strip():
            continue

        sentence_id = rec.get("sentence_id") or rec.get("id") or str(i)
        clean_toks = model.to_tokens(clean_text, prepend_bos=False)
        corrupted_toks = model.to_tokens(corrupted_text, prepend_bos=False)
        str_tokens = model.to_str_tokens(clean_toks[0])
        query_pos = resolve_query_pos(rec, args.attention_position, len(str_tokens))

        record: Dict[str, Any] = {
            "sentence_id": str(sentence_id),
            "sentence_text": clean_text,
            "tokens": str_tokens,
            "diff_vectors": {},
        }

        # A->B (per-receiver) diffs
        per_receiver_vecs: List[torch.Tensor] = []
        if args.group_patch:
            try:
                diff_map = calculate_attention_difference_group(
                    model,
                    clean_toks,
                    corrupted_toks,
                    sender_head,
                    receiver_heads,
                    query_pos,
                    receiver_input=args.receiver_input,
                )
                for receiver in receiver_heads:
                    diff = diff_map.get(receiver)
                    if diff is None:
                        continue
                    per_receiver_vecs.append(diff.detach().cpu())
                    if not args.group_avg_only:
                        record["diff_vectors"][
                            f"{sender_head[0]}.{sender_head[1]}->{receiver[0]}.{receiver[1]}"
                        ] = diff.detach().cpu().tolist()
            except Exception as e:
                print(f"Warning: failed on {sentence_id} receiver group: {e}")
        else:
            for receiver in receiver_heads:
                try:
                    diff = calculate_attention_difference(
                        model,
                        clean_toks,
                        corrupted_toks,
                        sender_head,
                        receiver,
                        query_pos,
                        receiver_input=args.receiver_input,
                    )
                    per_receiver_vecs.append(diff.detach().cpu())
                    if not args.group_avg_only:
                        record["diff_vectors"][
                            f"{sender_head[0]}.{sender_head[1]}->{receiver[0]}.{receiver[1]}"
                        ] = diff.detach().cpu().tolist()
                except Exception as e:
                    print(f"Warning: failed on {sentence_id} receiver {receiver}: {e}")
                    continue

        if per_receiver_vecs and args.group_avg_only:
            min_len = min(v.numel() for v in per_receiver_vecs)
            vecs = [v[:min_len] for v in per_receiver_vecs]
            avg_vec = torch.mean(torch.stack(vecs, dim=0), dim=0)
            record["diff_vectors"][f"{sender_head[0]}.{sender_head[1]}->B_GROUP_AVG"] = avg_vec.tolist()

        # A->B->C (optional)
        if target_heads:
            record["diff_vectors_abc"] = {}
            if args.group_patch:
                try:
                    diff_map = calculate_attention_difference_group_on_targets(
                        model,
                        clean_toks,
                        corrupted_toks,
                        sender_head,
                        receiver_heads,
                        target_heads,
                        query_pos,
                        receiver_input=args.receiver_input,
                    )
                    for target in target_heads:
                        diff_c = diff_map.get(target)
                        if diff_c is None:
                            continue
                        key = f"{sender_head[0]}.{sender_head[1]}->B_GROUP->{target[0]}.{target[1]}"
                        record["diff_vectors_abc"][key] = diff_c.detach().cpu().tolist()
                except Exception as e:
                    print(f"Warning: failed on {sentence_id} target group: {e}")
            else:
                for target in target_heads:
                    try:
                        diff_c = calculate_attention_difference_group_on_target(
                            model,
                            clean_toks,
                            corrupted_toks,
                            sender_head,
                            receiver_heads,
                            target,
                            query_pos,
                            receiver_input=args.receiver_input,
                        )
                        key = f"{sender_head[0]}.{sender_head[1]}->B_GROUP->{target[0]}.{target[1]}"
                        record["diff_vectors_abc"][key] = diff_c.detach().cpu().tolist()
                    except Exception as e:
                        print(f"Warning: failed on {sentence_id} target {target}: {e}")
                        continue

        dataset[str(sentence_id)] = record

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved middle-head diff dataset to {out_path}")


if __name__ == "__main__":
    main()
