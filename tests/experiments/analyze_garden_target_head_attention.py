#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze how each sender head affects a Garden target head's attention pattern
through a chosen receiver input path (q/k/v).

This is the Garden analogue of IOI analyze_target_head_attention.py, but instead
of a few named scenarios it sweeps (by default) all sender heads individually.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import einops
import torch
from tqdm import tqdm

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from precompute_middle_head_garden import (
    load_model,
    pad_to_match,
    patch_or_freeze_head_vectors,
)


SUBORD_WORDS = {"When", "While", "After", "As"}


def parse_head(head_str: str) -> Tuple[int, int]:
    layer_str, head_str = head_str.split(".", 1)
    return int(layer_str), int(head_str)


def load_standard_garden(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list")
    for idx, item in enumerate(data):
        if isinstance(item, dict) and "sample_id" not in item:
            item["sample_id"] = idx
    return data


def normalize_token(token: str) -> str:
    return token.strip().lower()


def get_suffixed_word_map(original_text: str):
    import re

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


def select_query_index(sample: dict, sentence_len: int, attention_position: str) -> int:
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


def label_for_position(sample: dict, token: str, position: int) -> str:
    token = token.strip()
    toks = [t.strip() for t in sample.get("tokenized_clean", [])]
    wd = sample.get("word_idx") or {}
    if toks and position == 0 and toks[0] in SUBORD_WORDS:
        return "SUBORD"
    mapping = {
        "SUBJ": wd.get("subj"),
        "VERB": wd.get("verb"),
        "OBJ_HEAD": wd.get("obj_head"),
        "REL_PRON": wd.get("rel_pron"),
        "REL_VERB": wd.get("rel_verb"),
    }
    for label, idx in mapping.items():
        if idx == position:
            return label
    return "OTHER"


def compress_counts(counter: Counter) -> Dict[str, int]:
    order = ["SUBORD", "SUBJ", "VERB", "OBJ_HEAD", "REL_PRON", "REL_VERB", "OTHER"]
    return {k: int(counter[k]) for k in order if counter.get(k, 0) > 0}


def patch_target_receiver_input(
    activation: torch.Tensor,
    hook: HookPoint,
    patched_receiver_cache: ActivationCache,
    target_head: Tuple[int, int],
) -> torch.Tensor:
    patched = activation.clone()
    patched[:, :, target_head[1], :] = patched_receiver_cache[hook.name][:, :, target_head[1], :]
    return patched


def make_sender_list(
    n_layers: int,
    n_heads: int,
    target_head: Tuple[int, int],
    include_target: bool,
) -> List[Tuple[int, int]]:
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    if not include_target:
        heads = [x for x in heads if x != target_head]
    return heads


def top_changes(
    diff_vector: torch.Tensor,
    sample: dict,
    top_k: int,
) -> Tuple[List[dict], List[dict]]:
    _, suffixed_map = get_suffixed_word_map(sample.get("text") or sample.get("clean") or "")
    k = min(top_k, diff_vector.numel())
    top_vals, top_idx = torch.topk(diff_vector, k)
    bot_vals, bot_idx = torch.topk(diff_vector, k, largest=False)

    def pack(val, idx):
        pos = int(idx.item())
        token = suffixed_map.get(pos, str(pos))
        return {
            "position": pos,
            "token": token,
            "label": label_for_position(sample, token, pos),
            "delta": float(val.item()),
        }

    return [pack(v, i) for v, i in zip(top_vals, top_idx)], [pack(v, i) for v, i in zip(bot_vals, bot_idx)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep sender heads affecting a Garden target head attention row.")
    parser.add_argument("--target-head", required=True, help="Receiver head layer.head, e.g. 7.4")
    parser.add_argument("--receiver-input", default="q", choices=["q", "k", "v"], help="Receiver input path to patch.")
    parser.add_argument("--standard-json", required=True, help="standard_garden_data.json")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--model-path", type=Path, default=Path("/home/wangziran/gpt2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-position", default="end", help="Query row to inspect on the target head.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples.")
    parser.add_argument("--top-k", type=int, default=3, help="Top increase/decrease tokens to keep.")
    parser.add_argument("--top-senders", type=int, default=20, help="How many strongest senders to print.")
    parser.add_argument("--include-target", action="store_true", help="Include the target head itself as a sender.")
    parser.add_argument(
        "--require-same-length",
        action="store_true",
        help="Filter to the modal tokenized length before averaging attention rows.",
    )
    parser.add_argument("--output-file", type=Path, required=True, help="JSON output path.")
    args = parser.parse_args()

    model = load_model(device=args.device)
    model.cfg.use_attn_result = True
    records = load_standard_garden(Path(args.standard_json))
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    if args.require_same_length:
        token_lens = []
        for sample in records:
            clean_text = sample.get("text") or sample.get("clean")
            if not clean_text:
                token_lens.append(None)
                continue
            toks = model.to_tokens(clean_text, prepend_bos=False)
            token_lens.append(int(toks.size(1)))
        valid_lens = [x for x in token_lens if x is not None]
        if not valid_lens:
            raise ValueError("No valid token lengths found in standard garden records.")
        modal_len, _ = Counter(valid_lens).most_common(1)[0]
        print(f"Filtering to modal tokenized length = {modal_len}")
        records = [sample for sample, tok_len in zip(records, token_lens) if tok_len == modal_len]
        if not records:
            raise ValueError("No records remain after filtering to modal tokenized length.")
        print(f"Remaining samples after length filter: {len(records)}")

    target_head = parse_head(args.target_head)
    target_layer, target_h_idx = target_head
    sender_heads = make_sender_list(model.cfg.n_layers, model.cfg.n_heads, target_head, args.include_target)

    z_filter = lambda name: name.endswith("z")
    receiver_hook_name = utils.get_act_name(args.receiver_input, target_layer)
    receiver_filter = lambda name: name == receiver_hook_name
    pattern_hook_name = utils.get_act_name("pattern", target_layer)
    pattern_filter = lambda name: name == pattern_hook_name

    grouped_records: Dict[int, List[dict]] = defaultdict(list)
    for sample in records:
        clean_text = sample.get("text") or sample.get("clean")
        corrupted_text = sample.get("corrupted_text") or sample.get("corrupted")
        if not clean_text or not corrupted_text:
            continue
        clean_tokens = model.to_tokens(clean_text, prepend_bos=False)
        seq_len = int(clean_tokens.size(1))
        grouped_records[seq_len].append(sample)

    if not grouped_records:
        raise ValueError("No valid Garden samples found after preprocessing.")

    prepared_groups = []
    query_position_example = None
    for seq_len, samples in sorted(grouped_records.items()):
        clean_texts = [(s.get("text") or s.get("clean")) for s in samples]
        corrupted_texts = [(s.get("corrupted_text") or s.get("corrupted")) for s in samples]
        clean_tokens = model.to_tokens(clean_texts, prepend_bos=False)
        corrupted_tokens = model.to_tokens(corrupted_texts, prepend_bos=False)
        clean_tokens, corrupted_tokens = pad_to_match(clean_tokens, corrupted_tokens)
        str_tokens = model.to_str_tokens(clean_tokens[0])
        query_positions = [select_query_index(s, len(str_tokens), args.attention_position) for s in samples]
        if query_position_example is None and samples:
            query_position_example = {
                "sample_id": samples[0].get("sample_id"),
                "query_index": query_positions[0],
                "query_token": str_tokens[query_positions[0]],
            }
        print(f"Preparing group: seq_len={seq_len}, samples={len(samples)}")
        _, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: z_filter(name) or pattern_filter(name),
            return_type=None,
        )
        _, corrupted_cache = model.run_with_cache(
            corrupted_tokens,
            names_filter=z_filter,
            return_type=None,
        )
        clean_pattern = clean_cache[pattern_hook_name][:, target_h_idx, :, :]
        clean_query_vectors = torch.stack(
            [clean_pattern[i, qpos].detach() for i, qpos in enumerate(query_positions)],
            dim=0,
        )
        prepared_groups.append(
            {
                "seq_len": seq_len,
                "samples": samples,
                "clean_tokens": clean_tokens,
                "clean_cache": clean_cache,
                "corrupted_cache": corrupted_cache,
                "str_tokens": str_tokens,
                "query_positions": query_positions,
                "clean_query_vectors": clean_query_vectors,
            }
        )

    state = {}
    for sender in sender_heads:
        state[f"{sender[0]}.{sender[1]}"] = {
            "sender_head": f"{sender[0]}.{sender[1]}",
            "count": 0,
            "by_seq_len": {},
            "per_sentence": [],
        }

    for sender_head in tqdm(sender_heads, desc="Sender heads"):
        sender_key = f"{sender_head[0]}.{sender_head[1]}"
        sender_state = state[sender_key]

        for group in prepared_groups:
            clean_tokens = group["clean_tokens"]
            clean_cache = group["clean_cache"]
            corrupted_cache = group["corrupted_cache"]
            query_positions = group["query_positions"]
            clean_query_vectors = group["clean_query_vectors"]
            samples = group["samples"]
            str_tokens = group["str_tokens"]

            hook_fn_sender = lambda act, hook, sh=sender_head, cc=corrupted_cache, oc=clean_cache: patch_or_freeze_head_vectors(
                act, hook, cc, oc, sh
            )
            model.add_hook(z_filter, hook_fn_sender)
            _, patched_receiver_cache = model.run_with_cache(
                clean_tokens,
                names_filter=receiver_filter,
                return_type=None,
            )
            model.reset_hooks()

            patched_pattern_holder: Dict[str, torch.Tensor] = {}

            def cache_target_pattern_hook(activation: torch.Tensor, hook: HookPoint):
                patched_pattern_holder[hook.name] = activation.detach()
                return activation

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (receiver_hook_name, lambda act, hook, prc=patched_receiver_cache: patch_target_receiver_input(act, hook, prc, target_head)),
                    (pattern_hook_name, cache_target_pattern_hook),
                ],
                return_type=None,
            )

            patched_patterns = patched_pattern_holder[pattern_hook_name][:, target_h_idx, :, :]
            patched_query_vectors = torch.stack(
                [patched_patterns[i, qpos].detach() for i, qpos in enumerate(query_positions)],
                dim=0,
            )
            diff_vectors = patched_query_vectors - clean_query_vectors

            for i, sample in enumerate(samples):
                diff_vector = diff_vectors[i]
                clean_query_vector = clean_query_vectors[i]
                patched_query_vector = patched_query_vectors[i]
                qpos = query_positions[i]

                increases, decreases = top_changes(diff_vector, sample, args.top_k)
                l1_norm = float(diff_vector.abs().sum().item())
                max_abs = float(diff_vector.abs().max().item())

                seq_state = sender_state["by_seq_len"].setdefault(
                    len(clean_query_vector),
                    {
                        "sum_clean": None,
                        "sum_patched": None,
                        "sum_diff": None,
                        "count": 0,
                        "reference_sample": sample,
                    },
                )
                seq_state["sum_clean"] = clean_query_vector.clone() if seq_state["sum_clean"] is None else seq_state["sum_clean"] + clean_query_vector
                seq_state["sum_patched"] = patched_query_vector.clone() if seq_state["sum_patched"] is None else seq_state["sum_patched"] + patched_query_vector
                seq_state["sum_diff"] = diff_vector.clone() if seq_state["sum_diff"] is None else seq_state["sum_diff"] + diff_vector
                seq_state["count"] += 1
                sender_state["count"] += 1
                sender_state["per_sentence"].append(
                    {
                        "sentence_id": sample.get("sample_id"),
                        "sentence_text": sample.get("text") or sample.get("clean"),
                        "query_position": {"index": qpos, "token": str_tokens[qpos]},
                        "l1_norm": l1_norm,
                        "max_abs_delta": max_abs,
                        "top_increases": increases,
                        "top_decreases": decreases,
                    }
                )

    results = []
    for sender_key, sender_state in state.items():
        if sender_state["count"] == 0:
            continue
        representative_seq_len, representative_state = max(
            sender_state["by_seq_len"].items(),
            key=lambda item: item[1]["count"],
        )
        mean_diff = representative_state["sum_diff"] / representative_state["count"]
        mean_clean = representative_state["sum_clean"] / representative_state["count"]
        mean_patched = representative_state["sum_patched"] / representative_state["count"]

        reference_sample = representative_state["reference_sample"]
        mean_increases, mean_decreases = top_changes(mean_diff, reference_sample, args.top_k)

        top_inc_labels = Counter()
        top_dec_labels = Counter()
        for sent in sender_state["per_sentence"]:
            top_inc_labels.update(x["label"] for x in sent["top_increases"])
            top_dec_labels.update(x["label"] for x in sent["top_decreases"])

        by_seq_len_summary = []
        for seq_len, seq_state in sorted(sender_state["by_seq_len"].items()):
            group_mean_diff = seq_state["sum_diff"] / seq_state["count"]
            group_increases, group_decreases = top_changes(group_mean_diff, seq_state["reference_sample"], args.top_k)
            by_seq_len_summary.append(
                {
                    "seq_len": seq_len,
                    "count": seq_state["count"],
                    "mean_l1_diff": float(group_mean_diff.abs().sum().item()),
                    "mean_max_abs_diff": float(group_mean_diff.abs().max().item()),
                    "mean_top_increases": group_increases,
                    "mean_top_decreases": group_decreases,
                }
            )

        result = {
            "sender_head": sender_key,
            "receiver_head": args.target_head,
            "receiver_input": args.receiver_input,
            "query_position": args.attention_position,
            "representative_seq_len": representative_seq_len,
            "mean_l1_diff": float(mean_diff.abs().sum().item()),
            "mean_max_abs_diff": float(mean_diff.abs().max().item()),
            "mean_top_increases": mean_increases,
            "mean_top_decreases": mean_decreases,
            "mean_top_increase_label_counts": compress_counts(top_inc_labels),
            "mean_top_decrease_label_counts": compress_counts(top_dec_labels),
            "by_seq_len": by_seq_len_summary,
            "per_sentence": sender_state["per_sentence"],
        }
        # enrich clean/patched values on top positions
        for entry in result["mean_top_increases"]:
            pos = entry["position"]
            entry["clean"] = float(mean_clean[pos].item())
            entry["patched"] = float(mean_patched[pos].item())
        for entry in result["mean_top_decreases"]:
            pos = entry["position"]
            entry["clean"] = float(mean_clean[pos].item())
            entry["patched"] = float(mean_patched[pos].item())
        results.append(result)

    results.sort(key=lambda x: x["mean_l1_diff"], reverse=True)

    payload = {
        "target_head": args.target_head,
        "receiver_input": args.receiver_input,
        "attention_position": args.attention_position,
        "query_position_example": query_position_example,
        "num_samples": len(records),
        "ranked_senders": results,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {args.output_file}")
    print(f"\nTop senders by mean L1 change on {args.target_head}.{args.receiver_input}:")
    for row in results[: args.top_senders]:
        print(
            f"  {row['sender_head']:>5}  mean_l1={row['mean_l1_diff']:.6f}  "
            f"↑ {[x['label']+':'+x['token'] for x in row['mean_top_increases']]}  "
            f"↓ {[x['label']+':'+x['token'] for x in row['mean_top_decreases']]}"
        )


if __name__ == "__main__":
    main()
