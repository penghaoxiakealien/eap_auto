#!/usr/bin/env python
"""
Compute the direct effect on IOI logit difference for a single attention head or a group of heads.

Usage (examples):
    python path_patching_head_to_logits.py --sender_head 7.9
    python path_patching_head_to_logits.py --sender_heads 9.6 9.9 10.0

The script reproduces the path‑patching experiment from the Indirect Object Identification (IOI)
analysis, but only for the specified sender head(s). It prints the IOI metric variation (relative
change in logit difference) for that head or group of heads.
"""

import argparse
import os
import re
from functools import partial
from typing import Callable, Tuple, List
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


def patch_or_freeze_multiple_heads(
    orig_head_vector: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    heads_to_patch: List[Tuple[int, int]],
):
    """Freeze all heads except the ones being patched. 保持原有patch逻辑不变。"""
    # 第1步：先把所有头都冻结到原始状态
    orig_head_vector[...] = orig_cache[hook.name][...]
    
    # 第2步：把指定的头patch为corrupted状态
    current_layer = hook.layer()
    for layer, head in heads_to_patch:
        if layer == current_layer:
            orig_head_vector[:, :, head] = new_cache[hook.name][:, :, head]
    
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


def compute_multiple_heads_effect(heads_to_patch: List[Tuple[int, int]]) -> float:
    """Compute IOI metric variation for multiple sender heads."""
    model.reset_hooks()

    # Receiver = final residual stream (just after last layer norm)
    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name

    # Cache only the attention head outputs ("z")
    z_name_filter = lambda name: name.endswith("z")

    # Gather activations on ABC (new) & IOI (orig) prompts
    _, new_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name_filter, return_type=None)
    _, orig_cache = model.run_with_cache(ioi_dataset.toks, names_filter=z_name_filter, return_type=None)

    # Patch the chosen heads while freezing others
    hook_fn = partial(
        patch_or_freeze_multiple_heads,
        new_cache=new_cache,
        orig_cache=orig_cache,
        heads_to_patch=heads_to_patch,
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
        description="Path‑patch a single sender head or multiple heads and report IOI metric variation."
    )
    
    # 互斥选项：要么单个头，要么多个头
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sender_head",
        type=str,
        help="Single sender head in the format <layer>.<head>, e.g. 7.9",
    )
    group.add_argument(
        "--sender_heads",
        nargs='+',
        type=str,
        help="Multiple sender heads in the format <layer>.<head>, e.g. 9.6 9.9 10.0",
    )
    
    return parser.parse_args()


def parse_head_string(head_str: str) -> Tuple[int, int]:
    """Parse a head string like '9.6' into (layer, head) tuple."""
    try:
        layer_str, head_str = head_str.split(".")
        layer, head = int(layer_str), int(head_str)
    except ValueError as e:
        raise ValueError(f"Head must be in the format <layer>.<head>, got: {head_str}") from e
    
    if not (0 <= layer < model.cfg.n_layers and 0 <= head < model.cfg.n_heads):
        raise ValueError(
            f"Head {layer}.{head} out of range: model has {model.cfg.n_layers} layers and {model.cfg.n_heads} heads per layer."
        )
    
    return layer, head


def main():
    args = parse_args()
    
    # 显示基准信息
    print(f"\n=== 📊 基准性能 ===")
    print(f"干净数据集 (IOI) logit差值: {clean_logit_diff:.4f}")
    print(f"损坏数据集 (ABC) logit差值: {corrupted_logit_diff:.4f}")
    print(f"基准差距: {clean_logit_diff - corrupted_logit_diff:.4f}")
    
    if args.sender_head:
        # 单个头模式
        layer, head = parse_head_string(args.sender_head)
        print(f"\n=== 🔍 单个头分析 ===")
        print(f"正在分析头 {layer}.{head}...")
        
        metric = compute_sender_head_effect(layer, head)
        percent_change = metric * 100
        
        print(f"\n=== 📈 结果 ===")
        print(f"头 {layer}.{head}: IOI指标变化 = {percent_change:.2f}%")
        
        # 解释结果
        if percent_change < -50:
            interpretation = "🔥 极其重要 - 对IOI任务至关重要"
        elif percent_change < -20:
            interpretation = "⚡ 很重要 - 显著影响IOI任务"
        elif percent_change < -5:
            interpretation = "📉 有一定作用"
        elif percent_change > 5:
            interpretation = "📈 可能是抑制性头"
        else:
            interpretation = "🤷 影响较小"
            
        print(f"解释: {interpretation}")
        
        return metric
        
    else:
        # 多个头模式
        heads_to_patch = [parse_head_string(h) for h in args.sender_heads]
        print(f"\n=== 🎯 多头联合分析 ===")
        print(f"正在分析头组合: {[f'{l}.{h}' for l, h in heads_to_patch]}")
        
        # 先分析各个头的单独效果
        print(f"\n--- 各头单独效果 ---")
        individual_effects = {}
        individual_sum = 0
        
        for layer, head in heads_to_patch:
            metric = compute_sender_head_effect(layer, head)
            individual_effects[f"{layer}.{head}"] = metric
            individual_sum += metric
            print(f"头 {layer}.{head}: {metric * 100:.2f}%")
        
        # 再分析联合效果
        print(f"\n--- 联合效果分析 ---")
        combined_metric = compute_multiple_heads_effect(heads_to_patch)
        combined_percent = combined_metric * 100
        
        print(f"联合patch结果: {combined_percent:.2f}%")
        print(f"各头单独效果之和: {individual_sum * 100:.2f}%")
        
        # 协同效应分析
        synergy = combined_metric - individual_sum
        synergy_percent = synergy * 100
        
        print(f"\n=== ⚡ 协同效应分析 ===")
        print(f"协同效应: {synergy_percent:.2f}%")
        
        if abs(synergy_percent) > 10:
            if synergy_percent > 0:
                print("🔥 存在负协同效应 - 联合使用时相互抵消部分破坏力")
            else:
                print("⚡ 存在正协同效应 - 联合使用时相互增强破坏力")
        else:
            print("📊 协同效应不明显 - 基本是线性叠加")
        
        # 总体解释
        print(f"\n=== 🎊 结论 ===")
        if combined_percent < -100:
            conclusion = "🚨 CRITICAL: 联合patch效果比完全corrupted还要严重！"
        elif combined_percent < -80:
            conclusion = "💥 SEVERE: 联合patch几乎完全破坏了IOI任务性能"
        elif combined_percent < -50:
            conclusion = "⚠️ SIGNIFICANT: 联合patch严重影响了性能"
        elif combined_percent < -20:
            conclusion = "📉 MODERATE: 联合patch有明显影响"
        else:
            conclusion = "🤔 MILD: 联合patch影响相对较小"
            
        print(conclusion)
        
        return combined_metric


if __name__ == "__main__":
    main()