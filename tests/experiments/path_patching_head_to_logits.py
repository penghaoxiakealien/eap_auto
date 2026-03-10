#!/usr/bin/env python
"""
Compute the direct effect on IOI logit difference for a single attention head.

Usage (example):
    python path_patch_single_head.py --sender_head 7.9

The script reproduces the path‑patching experiment from the Indirect Object Identification (IOI)
analysis, but only for the specified sender head. It prints the IOI metric variation (relative
change in logit difference) for that head.
"""

import argparse
import os
import re
from functools import partial
from typing import Callable, Tuple
import sys
sys.path.append("/data63/private/chensiyuan/EAP-IG/tests/experiments")
import torch as t
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table

from ioi_dataset import NAMES, IOIDataset
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint

# -----------------------------------------------------------------------------
# Configuration & model / dataset loading
# -----------------------------------------------------------------------------

# Use the HF mirror in mainland China (no‑op elsewhere)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2-small"

# Load model (with settings matching the IOI analysis)
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

# Build IOI + ABC datasets (25 prompts each by default)
N_PROMPTS = 25
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N_PROMPTS,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(DEVICE),
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

# -----------------------------------------------------------------------------
# Helper functions (mostly unchanged from the original notebook)
# -----------------------------------------------------------------------------

def format_prompt(sentence: str) -> str:
    """Underline the names when printed with rich."""
    return re.sub(
        "(" + "|".join(NAMES) + ")",
        lambda m: f"[u bold dark_orange]{m.group(0)}[/]",
        sentence,
    ) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)


def logits_to_ave_logit_diff(
    logits: t.Tensor, ioi_dataset: IOIDataset, per_prompt: bool = False
):
    """Return the (average) logit difference between IO and S tokens."""
    io_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
    ]
    s_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
    ]
    diff = io_logits - s_logits
    return diff if per_prompt else diff.mean()


# Pre‑compute the clean & corrupted baselines (IOI vs ABC prompts)
ioi_logits_clean, ioi_cache_clean = model.run_with_cache(ioi_dataset.toks)
abc_logits_corrupted, _ = model.run_with_cache(abc_dataset.toks)

clean_logit_diff = logits_to_ave_logit_diff(ioi_logits_clean, ioi_dataset).item()
corrupted_logit_diff = logits_to_ave_logit_diff(abc_logits_corrupted, ioi_dataset).item()


def ioi_metric(logits: t.Tensor) -> float:
    """Calibrated metric: 0 = clean (IOI), ‑1 = fully corrupted (ABC)."""
    patched_diff = logits_to_ave_logit_diff(logits, ioi_dataset)
    return (patched_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


# -----------------------------------------------------------------------------
# Path‑patching utilities
# -----------------------------------------------------------------------------

def patch_or_freeze_head_vectors(
    orig_head_vector: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
):
    """Freeze all heads except the one being patched."""
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def compute_sender_head_effect(layer: int, head: int) -> float:
    """Compute IOI metric variation for a single sender head."""
    model.reset_hooks()

    # Receiver = final residual stream (just after last layer norm)
    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name

    # Cache only the attention head outputs ("z")
    z_name_filter = lambda name: name.endswith("z")

    # Gather activations on ABC (new) & IOI (orig) prompts
    _, new_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter, return_type=None)
    _, orig_cache = model.run_with_cache(ioi_dataset.toks, names_filter=z_name_filter, return_type=None)

    # Patch the chosen head while freezing others
    hook_fn = partial(
        patch_or_freeze_head_vectors,
        new_cache=new_cache,
        orig_cache=orig_cache,
        head_to_patch=(layer, head),
    )
    model.add_hook(z_name_filter, hook_fn)

    # Forward pass with patched activations & grab final resid_post
    _, patched_cache = model.run_with_cache(
        ioi_dataset.toks, names_filter=resid_post_name_filter, return_type=None
    )
    patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

    return ioi_metric(patched_logits).item()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Path‑patch a single sender head and report its IOI metric variation."
    )
    parser.add_argument(
        "--sender_head",
        type=str,
        required=True,
        help="Sender head in the format <layer>.<head>, e.g. 7.9",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        layer_str, head_str = args.sender_head.split(".")
        layer, head = int(layer_str), int(head_str)
    except ValueError as e:
        raise ValueError("--sender_head must be in the format <layer>.<head>, e.g. 7.9") from e

    if not (0 <= layer < model.cfg.n_layers and 0 <= head < model.cfg.n_heads):
        raise ValueError(
            f"Head out of range: model has {model.cfg.n_layers} layers and {model.cfg.n_heads} heads per layer."
        )

    metric = compute_sender_head_effect(layer, head)
    percent_change = metric * 100
    print(f"Head {layer}.{head}: IOI metric variation = {percent_change:.2f}%")
    return metric

if __name__ == "__main__":
    main()
