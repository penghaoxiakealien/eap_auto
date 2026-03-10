#!/usr/bin/env python
"""Analyze how upstream heads affect a target head's end-token attention pattern.

This script focuses on a single target head (default: 10.7) and measures how
patching selected sender heads changes the attention distribution from a chosen
query position (defaults to the end token) to every other token. For each patch
scenario, it reports the top-k tokens whose attention increases the most and
the top-k tokens whose attention decreases the most.

Example:
    python analyze_target_head_attention.py \
        --target-head 10.7 \
        --scenario node10_2:10.2 \
        --scenario node9_group:9.6,9.9,10.0 \
        --scenario node10_group:10.6,10.10

The script assumes the structured IOI sentences JSONL file used elsewhere in
this project. It saves no files by default but can optionally emit a JSON
summary via --save-json.
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

# Allow imports from project root when run from tests/experiments/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from transformer_lens import HookedTransformer, utils, ActivationCache  # type: ignore
from transformer_lens.hook_points import HookPoint  # type: ignore

from eap_auto.tests.experiments.ioi_dataset import IOIDataset  # type: ignore

# Use the HF mirror when available (important in mainland China)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass
class Scenario:
    label: str
    sender_heads: List[Tuple[int, int]]


def str_to_head(head: str) -> Tuple[int, int]:
    """Convert "layer.head" into (layer, head) ints."""
    layer_str, head_str = head.split(".")
    return int(layer_str), int(head_str)


def _normalize_token(token: str) -> str:
    """Map GPT encoder tokens (Ġ prefix) to strings returned by to_str_tokens."""
    if token.startswith("Ġ"):
        return " " + token[1:]
    return token


def _token_occurrence(tokens: List[str], index: int) -> Tuple[str, int]:
    """Return the target token and its occurrence count up to the given index."""
    if index < 0 or index >= len(tokens):
        raise ValueError(f"Token index {index} out of range for token list of length {len(tokens)}")
    normalized_tokens = [_normalize_token(tok) for tok in tokens]
    target = normalized_tokens[index]
    occurrence = sum(1 for tok in normalized_tokens[: index + 1] if tok == target)
    return target, occurrence


def _find_nth_occurrence(tokens: List[str], target: str, occurrence: int) -> int:
    """Find the index of the nth occurrence of target within tokens."""
    count = 0
    for idx, tok in enumerate(tokens):
        if tok == target:
            count += 1
            if count == occurrence:
                return idx
    raise ValueError(f"Could not find {occurrence}-th occurrence of token '{target}' in sequence")


def resolve_query_position(spec: str, sentence: dict, str_tokens: List[str]) -> int:
    """Resolve which query row of the attention pattern to analyze."""
    spec_lower = spec.lower()
    if spec_lower in {"end", "last"}:
        return len(str_tokens) - 1
    if spec_lower in {"bos", "start"}:
        return 0

    index_fields = {
        "io": "io_index",
        "s1": "s1_index",
        "s2": "s2_index",
    }

    if spec_lower in index_fields:
        field = index_fields[spec_lower]
        raw_index = sentence.get(field)
        if raw_index is None:
            raise ValueError(f"Sentence missing field '{field}' required for query position '{spec}'.")
        sentence_tokens = sentence.get("tokens")
        if not sentence_tokens:
            raise ValueError("Structured sentence entry lacks 'tokens' needed for named query positions.")
        target_token, occurrence = _token_occurrence(sentence_tokens, int(raw_index))
        return _find_nth_occurrence(str_tokens, target_token, occurrence)

    try:
        numeric_index = int(spec)
    except ValueError as exc:
        raise ValueError(
            "Unsupported query position '{}'. Use an integer index (can be negative) or one of "
            "'end', 'bos', 'io', 's1', 's2'.".format(spec)
        ) from exc

    if numeric_index < 0:
        numeric_index = len(str_tokens) + numeric_index
    if not 0 <= numeric_index < len(str_tokens):
        raise ValueError(
            f"Query position {numeric_index} out of range for sequence length {len(str_tokens)}"
        )
    return numeric_index


def parse_scenarios(raw: Optional[Iterable[str]]) -> List[Scenario]:
    """Parse --scenario entries of the form label:head1,head2,..."""
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
        if not heads:
            raise ValueError(f"Scenario '{entry}' must list at least one head")
        sender_heads = [str_to_head(h.strip()) for h in heads.split(",") if h.strip()]
        scenarios.append(Scenario(label.strip(), sender_heads))
    return scenarios


def load_sentences(path: str, limit: Optional[int] = None) -> List[dict]:
    """Load structured sentences JSONL, filtering malformed rows."""
    sentences: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sentence_text" not in row or "io_token" not in row or "s_token" not in row:
                continue
            sentences.append(row)
            if limit is not None and len(sentences) >= limit:
                break
    if not sentences:
        raise FileNotFoundError(f"No valid sentences found in {path}")
    return sentences


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
    """Freeze all heads except those listed in heads_by_layer, which get patched."""
    layer = hook.layer()
    orig_head_vector[...] = orig_cache[hook.name][...]
    heads = heads_by_layer.get(layer)
    if heads:
        orig_head_vector[:, :, heads, :] = new_cache[hook.name][:, :, heads, :]
    return orig_head_vector


def run_analysis(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    # If a local model path is provided, pre-load the HF model/tokenizer and
    # pass them into HookedTransformer to avoid transformers trying to download
    # resources. This uses the `hf_model` and `tokenizer` kwargs which
    # HookedTransformer.from_pretrained accepts.
    if os.path.isdir(args.model_name):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading HF model/tokenizer from local path: {args.model_name}")
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model_name, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
            # Use an official model name ('gpt2') for the loader but provide
            # the preloaded hf_model/tokenizer to avoid network calls.
            model = HookedTransformer.from_pretrained(
                "gpt2", hf_model=hf_model, tokenizer=tokenizer, device=device, hf_config=hf_model.config
            )
        except Exception as e:  # fall back to default behavior and let errors surface
            print(f"Failed to load local HF model/tokenizer: {e}. Falling back to normal load.")
            model = HookedTransformer.from_pretrained(args.model_name, device=device)
    else:
        model = HookedTransformer.from_pretrained(args.model_name, device=device)
    model.cfg.use_attn_result = True

    sentences = load_sentences(args.input_file, args.limit)

    # Optionally filter sentences to a single tokenized length (modal length) to
    # make attention vectors align across prompts. This avoids mismatches when
    # averaging attention distributions from the chosen query token.
    if getattr(args, "require_same_length", False):
        token_lens = []
        for s in sentences:
            toks = model.to_tokens(s["sentence_text"])
            token_lens.append(toks.size(1))
        from collections import Counter

        modal_len, _ = Counter(token_lens).most_common(1)[0]
        print(f"Filtering to modal tokenized length = {modal_len} (of {len(sentences)} sentences)")
        sentences = [s for i, s in enumerate(sentences) if token_lens[i] == modal_len]
        if not sentences:
            raise ValueError("No sentences remain after filtering to modal token length")

    target_layer, target_h_idx = str_to_head(args.target_head)

    if args.receiver_input.lower() == "all":
        receiver_inputs = ["q", "k", "v"]
    else:
        receiver_inputs = [item.strip() for item in args.receiver_input.split(",") if item.strip()]
        for item in receiver_inputs:
            if item not in {"q", "k", "v"}:
                raise ValueError(f"Unsupported receiver input '{item}'. Use q,k,v or all.")
    receiver_hook_names = [utils.get_act_name(inp, target_layer) for inp in receiver_inputs]

    pattern_hook_name = utils.get_act_name("pattern", target_layer)

    scenarios = parse_scenarios(args.scenario)
    scenario_metadata = {
        s.label: {
            "heads": s.sender_heads,
            "heads_by_layer": group_heads_by_layer(s.sender_heads),
            "sum_clean": None,
            "sum_patched": None,
            "sum_diff": None,
            "count": 0,
            "reference_tokens": None,
            "per_sentence": [],
        }
        for s in scenarios
    }

    top_k = max(1, getattr(args, "top_k", 1))
    query_position_spec = getattr(args, "query_position", "end")
    print(f"Analyzing attention row for query position spec '{query_position_spec}'")
    query_position_example: Optional[dict] = None

    for sentence in tqdm(sentences, desc="Sentences"):
        clean_text = sentence["sentence_text"]
        io_token = sentence["io_token"]
        s_token = sentence["s_token"]
        corrupted_text = clean_text.replace(io_token, s_token)

        clean_tokens = model.to_tokens(clean_text)
        corrupted_tokens = model.to_tokens(corrupted_text)
        str_tokens = model.to_str_tokens(clean_tokens)

        query_pos = resolve_query_position(query_position_spec, sentence, str_tokens)
        if query_position_example is None:
            query_position_example = {
                "sentence_id": sentence.get("sentence_id"),
                "index": query_pos,
                "token": str_tokens[query_pos],
            }

        # Cache z activations for clean & corrupted runs
        z_name_filter = lambda name: name.endswith("z")
        _, clean_cache_z = model.run_with_cache(
            clean_tokens, names_filter=z_name_filter, return_type=None
        )
        _, corrupted_cache_z = model.run_with_cache(
            corrupted_tokens, names_filter=z_name_filter, return_type=None
        )

        # Baseline target head attention pattern (clean)
        _, clean_pattern_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: name == pattern_hook_name,
            return_type=None,
        )
        clean_pattern = clean_pattern_cache[pattern_hook_name][0, target_h_idx]
        clean_query_vector = clean_pattern[query_pos].detach()

        for scenario in scenarios:
            state = scenario_metadata[scenario.label]
            heads_by_layer = state["heads_by_layer"]

            hook_fn = lambda act, hook, *, heads_by_layer=heads_by_layer: patch_selected_heads(
                act, hook, corrupted_cache_z, clean_cache_z, heads_by_layer
            )

            model.add_hook(z_name_filter, hook_fn)
            _, patched_receiver_cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda name: name in receiver_hook_names,
                return_type=None,
            )
            model.reset_hooks()

            patched_pattern_holder: dict[str, torch.Tensor] = {}

            def cache_target_pattern_hook(activation: torch.Tensor, hook: HookPoint):
                patched_pattern_holder[hook.name] = activation.detach()

            def make_patch_target_input_hook(hook_name: str):
                def _hook(activation: torch.Tensor, hook: HookPoint):
                    patched_activation = activation.clone()
                    patched_activation[:, :, target_h_idx, :] = patched_receiver_cache[hook_name][
                        :, :, target_h_idx, :
                    ]
                    return patched_activation

                return _hook

            fwd_hooks = [(hook_name, make_patch_target_input_hook(hook_name)) for hook_name in receiver_hook_names]
            fwd_hooks.append((pattern_hook_name, cache_target_pattern_hook))

            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=fwd_hooks,
                return_type=None,
            )

            patched_pattern = patched_pattern_holder[pattern_hook_name][0, target_h_idx]
            patched_query_vector = patched_pattern[query_pos].detach()
            diff_vector = patched_query_vector - clean_query_vector

            k_local = min(top_k, diff_vector.numel())
            top_values, top_indices = torch.topk(diff_vector, k_local)
            bottom_values, bottom_indices = torch.topk(diff_vector, k_local, largest=False)

            # Ensure token length consistency across prompts for this scenario
            if state["reference_tokens"] is None:
                state["reference_tokens"] = list(str_tokens)
            elif len(state["reference_tokens"]) != len(str_tokens):
                raise ValueError("Token sequence length mismatch across prompts")

            # Accumulate sums for later averaging
            state["sum_clean"] = (
                clean_query_vector.clone()
                if state["sum_clean"] is None
                else state["sum_clean"] + clean_query_vector
            )
            state["sum_patched"] = (
                patched_query_vector.clone()
                if state["sum_patched"] is None
                else state["sum_patched"] + patched_query_vector
            )
            state["sum_diff"] = (
                diff_vector.clone()
                if state["sum_diff"] is None
                else state["sum_diff"] + diff_vector
            )
            state["count"] += 1

            increases = [
                {
                    "position": int(idx.item()),
                    "token": str_tokens[int(idx.item())],
                    "delta": float(val.item()),
                }
                for val, idx in zip(top_values, top_indices)
            ]

            decreases = [
                {
                    "position": int(idx.item()),
                    "token": str_tokens[int(idx.item())],
                    "delta": float(val.item()),
                }
                for val, idx in zip(bottom_values, bottom_indices)
            ]

            state["per_sentence"].append(
                {
                    "sentence_id": sentence.get("sentence_id"),
                    "query_position": {
                        "index": query_pos,
                        "token": str_tokens[query_pos],
                    },
                    "top_increases": increases,
                    "top_decreases": decreases,
                }
            )

    summary: dict = {
        "target_head": args.target_head,
        "receiver_inputs": receiver_inputs,
        "query_position_spec": query_position_spec,
        "top_k": top_k,
        "scenarios": [],
    }
    if query_position_example is not None:
        summary["query_position_example"] = query_position_example

    for scenario in scenarios:
        state = scenario_metadata[scenario.label]
        if state["count"] == 0:
            continue
        mean_diff = state["sum_diff"] / state["count"]
        mean_clean = state["sum_clean"] / state["count"]
        mean_patched = state["sum_patched"] / state["count"]

        k_local = min(top_k, mean_diff.numel())
        top_values, top_indices = torch.topk(mean_diff, k_local)
        bottom_values, bottom_indices = torch.topk(mean_diff, k_local, largest=False)

        ref_tokens = state["reference_tokens"]

        mean_increases = [
            {
                "position": int(idx.item()),
                "token": ref_tokens[int(idx.item())],
                "delta": float(val.item()),
                "clean": float(mean_clean[int(idx.item())].item()),
                "patched": float(mean_patched[int(idx.item())].item()),
            }
            for val, idx in zip(top_values, top_indices)
        ]

        mean_decreases = [
            {
                "position": int(idx.item()),
                "token": ref_tokens[int(idx.item())],
                "delta": float(val.item()),
                "clean": float(mean_clean[int(idx.item())].item()),
                "patched": float(mean_patched[int(idx.item())].item()),
            }
            for val, idx in zip(bottom_values, bottom_indices)
        ]

        summary_entry = {
            "label": scenario.label,
            "sender_heads": [f"{layer}.{head}" for layer, head in scenario.sender_heads],
            "mean_top_increases": mean_increases,
            "mean_top_decreases": mean_decreases,
            "per_sentence": state["per_sentence"],
        }
        summary["scenarios"].append(summary_entry)

        print(f"Scenario [{scenario.label}] → heads {summary_entry['sender_heads']}")
        top_line = ", ".join(
            [
                f"pos {entry['position']} token '{entry['token']}' Δ={entry['delta']:.4f}"
                for entry in mean_increases
            ]
        )
        bottom_line = ", ".join(
            [
                f"pos {entry['position']} token '{entry['token']}' Δ={entry['delta']:.4f}"
                for entry in mean_decreases
            ]
        )
        print(f"  ↑ top increases: {top_line}")
        print(f"  ↓ top decreases: {bottom_line}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary to {args.save_json}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze attention changes for a target head under path patching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-head", default="10.7", help="Target head layer.head")
    parser.add_argument(
        "--receiver-input",
        default="v",
        help="Input stream(s) to patch: q, k, v, comma-separated, or 'all' for q,k,v",
    )
    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input-file",
        default="../../results/ioi/path_patching/structured_sentences_standard.jsonl",
        help="Structured sentences JSONL",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario definition label:head[,head] (repeatable). Defaults match project discussion.",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of sentences")
    parser.add_argument("--save-json", help="Optional path to save JSON summary")
    parser.add_argument(
        "--require-same-length",
        action="store_true",
        help="Require tokenized prompts to have the same length (filters to the modal length).",
    )
    parser.add_argument(
        "--query-position",
        default="end",
        help="Attention row to analyze: integer index (supports negatives) or one of end/bos/io/s1/s2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of tokens to report for the largest increases and decreases (per sentence and averaged).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> dict:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_analysis(args)


if __name__ == "__main__":
    main()
