#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group patching (terminal, to logits) for agr_gender.

Goal:
  For each group in a grouped graph, take the subset of heads which have a direct
  head->logits entry in joint_p_qkv.json, patch those heads' z activations from
  the corrupted prompts into the clean prompts (simultaneously), and measure the
  change in gender logit-diff.

This is analogous to IOI "group_path_patching" but only for direct effects on logits.

Output:
  JSON with per-group mean delta and basic stats.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from agr_gender_dataset import AgrGenderDataset


LOCAL_MODEL_DIR = "/home/wangziran/gpt2"


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int

    @classmethod
    def from_name(cls, name: str) -> "HeadRef":
        name = name.strip()
        if name.startswith("a") and ".h" in name:
            l, h = name.split(".")
            return cls(layer=int(l[1:]), head=int(h[1:]))
        if "." in name:
            l, h = name.split(".", 1)
            return cls(layer=int(l), head=int(h))
        raise ValueError(f"Unrecognized head name: {name}")

    def as_name(self) -> str:
        return f"a{self.layer}.h{self.head}"


def load_model(device: str) -> HookedTransformer:
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        model = HookedTransformer.from_pretrained("gpt2", device=device, cache_dir=LOCAL_MODEL_DIR)
    else:
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.cfg.use_attn_result = True
    return model


def load_group_members(group_graph: Path) -> Dict[str, List[str]]:
    g = json.loads(group_graph.read_text())
    members = g.get("group_members")
    if isinstance(members, dict):
        out: Dict[str, List[str]] = {}
        for gid, hs in members.items():
            if isinstance(hs, list):
                out[str(gid)] = [str(x) for x in hs]
        return out
    groups = g.get("groups")
    if isinstance(groups, list):
        out = {}
        for grp in groups:
            if not isinstance(grp, dict):
                continue
            gid = grp.get("id") or grp.get("group_id")
            hs = grp.get("heads") or grp.get("members") or []
            if gid and isinstance(hs, list):
                out[str(gid)] = [str(x) for x in hs]
        return out
    raise ValueError("Unsupported grouped graph schema: missing group_members/groups")


def load_terminal_heads_from_joint(joint_path: Path, min_abs: float) -> Dict[str, float]:
    """
    Return mapping head_name (aL.hH) -> delta_logit_diff for head->logits edges.
    """
    data = json.loads(joint_path.read_text())
    edges = data.get("edges")
    if not isinstance(edges, list):
        raise ValueError("joint_p_qkv.json missing edges list")
    out: Dict[str, float] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("dst") != "logits":
            continue
        src = e.get("src")
        d = e.get("delta_logit_diff")
        if not isinstance(src, str) or not isinstance(d, (int, float)):
            continue
        if abs(float(d)) < min_abs:
            continue
        out[src] = float(d)
    return out


def logit_diff_batch(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    labels: torch.Tensor,
    he_id: int,
    she_id: int,
) -> torch.Tensor:
    # logits: [B, seq, vocab]
    pos = input_lengths - 1
    pos_logits = logits[torch.arange(logits.size(0), device=logits.device), pos]
    probs = torch.softmax(pos_logits, dim=-1)
    he = probs[:, he_id]
    she = probs[:, she_id]
    return torch.where(labels == 0, he - she, she - he)


def patch_group_z(
    head_out: torch.Tensor,
    hook: HookPoint,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    patch_map: Dict[int, List[int]],
) -> torch.Tensor:
    # Restore clean first
    head_out[...] = clean_cache[hook.name][...]
    layer = hook.layer()
    heads = patch_map.get(layer)
    if heads:
        for h in heads:
            head_out[:, :, h] = corrupted_cache[hook.name][:, :, h]
    return head_out


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    n = len(xs)
    def q(p: float) -> float:
        idx = int(round((n - 1) * p))
        return float(xs[max(0, min(n - 1, idx))])
    return {
        "min": float(xs[0]),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "max": float(xs[-1]),
        "mean": float(sum(xs) / n),
        "mean_abs": float(sum(abs(v) for v in xs) / n),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Group patch (terminal to logits) for agr_gender.")
    p.add_argument("--group-graph", type=Path, required=True)
    p.add_argument("--joint", type=Path, required=True, help="joint_p_qkv.json")
    p.add_argument("--data-path", type=Path, required=True, help="CSV (recommended) with clean/corrupted/label")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset-size", type=int, default=200, help="How many samples to use (0=all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-abs-logits-edge", type=float, default=0.0, help="Filter terminal heads by |delta| in joint")
    p.add_argument("--only-groups", type=str, default="", help="Comma-separated group ids to run (optional)")
    args = p.parse_args()

    members = load_group_members(args.group_graph)
    terminal = load_terminal_heads_from_joint(args.joint, args.min_abs_logits_edge)
    terminal_heads = set(terminal.keys())
    only_groups = {s.strip() for s in args.only_groups.split(",") if s.strip()}

    # choose groups that contain at least one terminal head
    groups: List[Tuple[str, List[str]]] = []
    for gid, hs in members.items():
        if only_groups and gid not in only_groups:
            continue
        patched = [h for h in hs if h in terminal_heads]
        if patched:
            groups.append((gid, patched))

    groups.sort(key=lambda x: x[0])
    print(f"Found {len(groups)} terminal groups (contain head->logits edges).")

    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = AgrGenderDataset(tokenizer=tok, data_path=args.data_path, device=args.device)
    if args.dataset_size and args.dataset_size > 0:
        n = min(int(args.dataset_size), ds.N)
        ds.samples = ds.samples[:n]
        ds.prompts = ds.prompts[:n]
        ds.sentences = ds.sentences[:n]
        if hasattr(ds, "word_idx") and isinstance(getattr(ds, "word_idx"), list):
            ds.word_idx = ds.word_idx[:n]
        ds.N = n
        ds.toks = ds.toks[:n]
        ds.corrupted_toks = ds.corrupted_toks[:n]
        ds.input_lengths = ds.input_lengths[:n]
        ds.pos_token_ids = ds.pos_token_ids[:n]
        ds.neg_token_ids = ds.neg_token_ids[:n]

    model = load_model(args.device)
    z_filter = lambda name: name.endswith("z")

    # Prepare label tensor for the (possibly truncated) dataset
    labels = torch.tensor([s.label for s in ds.samples], device=args.device, dtype=torch.long)
    he_id = int(ds.he_token_id)
    she_id = int(ds.she_token_id)

    # Baseline per-sample diff on clean (for reuse)
    with torch.no_grad():
        clean_logits = model(ds.toks)
    base_per = logit_diff_batch(clean_logits, ds.input_lengths, labels, he_id, she_id).detach().cpu().tolist()
    base_mean = float(sum(base_per) / len(base_per)) if base_per else 0.0

    out_rows: List[Dict[str, Any]] = []

    for gid, patched_heads in groups:
        patch_map: Dict[int, List[int]] = {}
        for hname in patched_heads:
            hr = HeadRef.from_name(hname)
            patch_map.setdefault(hr.layer, []).append(hr.head)
        for k in list(patch_map.keys()):
            patch_map[k] = sorted(set(patch_map[k]))

        deltas: List[float] = []
        patched_per: List[float] = []

        for start in tqdm(range(0, ds.N, args.batch_size), desc=f"group {gid}"):
            end = min(ds.N, start + args.batch_size)
            toks = ds.toks[start:end]
            corr = ds.corrupted_toks[start:end]
            lens = ds.input_lengths[start:end]
            lab = labels[start:end]

            if toks.shape != corr.shape:
                raise ValueError(f"Token shape mismatch clean vs corrupted for batch {start}:{end}: {toks.shape} vs {corr.shape}")

            with torch.no_grad():
                _, clean_cache = model.run_with_cache(toks, names_filter=z_filter, return_type=None)
                _, corr_cache = model.run_with_cache(corr, names_filter=z_filter, return_type=None)

            hook_fn = lambda tensor, hook: patch_group_z(tensor, hook, clean_cache, corr_cache, patch_map)
            model.add_hook(z_filter, hook_fn, level=1)
            with torch.no_grad():
                logits_patched = model(toks)
            model.reset_hooks()

            per = logit_diff_batch(logits_patched, lens, lab, he_id, she_id).detach().cpu().tolist()
            patched_per.extend(per)
            # deltas vs baseline for this slice
            for i_local, v in enumerate(per):
                deltas.append(float(v) - float(base_per[start + i_local]))

        mean_delta = float(sum(deltas) / len(deltas)) if deltas else 0.0
        mean_patched = float(sum(patched_per) / len(patched_per)) if patched_per else 0.0
        hurt = sum(1 for d in deltas if d < 0)
        help_ = sum(1 for d in deltas if d > 0)
        neutral = sum(1 for d in deltas if d == 0)

        out_rows.append(
            {
                "group_id": gid,
                "n_patched_heads": len(patched_heads),
                "patched_heads": patched_heads,
                "baseline_mean_logit_diff": base_mean,
                "patched_mean_logit_diff": mean_patched,
                "mean_delta_logit_diff": mean_delta,
                "delta_stats": summarize(deltas),
                "sign_counts": {"hurt(delta<0)": hurt, "help(delta>0)": help_, "zero": neutral},
            }
        )

    payload = {
        "meta": {
            "group_graph": str(args.group_graph),
            "joint": str(args.joint),
            "data_path": str(args.data_path),
            "dataset_size": int(ds.N),
            "batch_size": int(args.batch_size),
            "min_abs_logits_edge": float(args.min_abs_logits_edge),
            "baseline_mean_logit_diff": base_mean,
        },
        "groups": out_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote group patch results to {args.output}")


if __name__ == "__main__":
    main()
