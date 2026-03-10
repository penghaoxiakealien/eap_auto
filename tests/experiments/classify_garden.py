#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern mining for garden NPZ v-trans (mod) using word_idx positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch


def load_nodes_to_heads(p: Path) -> List[Tuple[int, int]]:
    d = json.loads(p.read_text())
    nodes = d.get("nodes")
    heads: List[Tuple[int, int]] = []
    if isinstance(nodes, list):
        names = nodes
    elif isinstance(nodes, dict):
        names = [n for n, info in nodes.items() if (info or {}).get("in_graph", False)]
    else:
        raise ValueError(f"无法识别 nodes 结构: {type(nodes)}")
    for n in names:
        if isinstance(n, str) and n.startswith("a") and (".h" in n):
            try:
                Ls, Hs = n.split(".")
                heads.append((int(Ls[1:]), int(Hs[1:])))
            except Exception:
                continue
    return sorted(set(heads))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Garden attention pattern profiling.")
    p.add_argument("--collapsed-json", type=Path, required=True)
    p.add_argument("--standard-json", type=Path, required=True, help="standard_garden_data.json")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--model-path", type=Path, default=None, help="本地模型目录（优先于 --model）。")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--mask-self", action="store_true")
    p.add_argument(
        "--allow-future",
        action="store_true",
        help="允许 key 位置在 query 之后（默认不允许，遵循因果注意力）。",
    )
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument(
        "--query-positions",
        type=str,
        # default="SUBJ,VERB,OBJ_HEAD,REL_PRON,REL_VERB,END",
        default="END",
        help="要分析的 query 位置（逗号分隔）。",
    )
    p.add_argument(
        "--key-positions",
        type=str,
        default="SUBJ,VERB,OBJ_HEAD,REL_PRON,REL_VERB,END",
        help="候选 key 位置集合（逗号分隔）。",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def load_samples(standard_json: Path, sample_size: int) -> List[Dict[str, Any]]:
    data = json.loads(standard_json.read_text())
    if not isinstance(data, list):
        raise ValueError("standard json 必须是列表")
    return data[:sample_size]


def _load_model(model_name: str, model_path: Optional[Path], device: str) -> HookedTransformer:
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        hf_model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
        official_name = model_name if model_name != "gpt2-small" else "gpt2"
        return HookedTransformer.from_pretrained(
            official_name,
            device=device,
            tokenizer=tokenizer,
            hf_model=hf_model,
            local_files_only=True,
        )
    candidates = [model_name]
    if model_name == "gpt2-small":
        candidates.append("gpt2")
    last_err = None
    for name in candidates:
        try:
            return HookedTransformer.from_pretrained(name, device=device)
        except Exception as exc:
            last_err = exc
    raise last_err if last_err else RuntimeError("无法加载模型")


def mine_patterns(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    samples: List[Dict[str, Any]],
    window: int,
    mask_self: bool,
    topk: int,
    query_positions: List[str],
    key_positions: List[str],
    allow_future: bool,
) -> Dict[str, Any]:
    positions = [p.upper() for p in query_positions]
    key_roles = [p.upper() for p in key_positions]
    from collections import defaultdict, Counter
    pattern_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    layer_set = sorted({L for L, _ in heads})
    names = [f"blocks.{L}.attn.hook_pattern" for L in layer_set]

    def candidates(pos_map: Dict[str, int], qpos: int, W: int, max_idx: int, allow_future: bool):
        outs = []
        for role in key_roles:
            base = pos_map.get(role)
            if not isinstance(base, int):
                continue
            for off in range(-W, W + 1):
                idx = base + off
                if 0 <= idx < max_idx:
                    if not allow_future and idx > qpos:
                        continue
                    outs.append((role, off, idx))
        return outs

    for s in samples:
        sentence = s.get("text") or s.get("sentence")
        if not isinstance(sentence, str) or not sentence:
            continue
        pos_map: Dict[str, int] = {}
        wd = s.get("word_idx") or {}
        for key in ["subj", "verb", "obj_head", "rel_pron", "rel_verb"]:
            if key in wd and wd[key] is not None:
                pos_map[key.upper()] = int(wd[key])
        end_idx = s.get("end_idx")
        if isinstance(end_idx, int):
            pos_map["END"] = int(end_idx)

        toks = model.to_tokens(sentence, prepend_bos=False)
        if toks is None or toks.numel() == 0:
            continue
        toks = toks.to(model.cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names, return_type=None)
        seq_len = int(toks.shape[1])

        for L, H in heads:
            hook_name = f"blocks.{L}.attn.hook_pattern"
            if hook_name not in cache:
                continue
            pat = cache[hook_name]
            for qp in positions:
                qpos = pos_map.get(qp)
                if not isinstance(qpos, int) or qpos < 0 or qpos >= seq_len:
                    continue
                row = pat[0, H, qpos, :].detach().float()
                if mask_self:
                    if 0 <= qpos < row.shape[0]:
                        row[qpos] = 0.0
                ssum = float(row.sum().item())
                if ssum > 0:
                    row = row / ssum
                cands = candidates(pos_map, qpos, window, seq_len, allow_future=allow_future)
                if mask_self:
                    cands = [c for c in cands if c[2] != qpos]
                if not cands:
                    continue
                best = max(cands, key=lambda t: float(row[t[2]].item()))
                role, off, _ = best
                hn = f"{L}.{H}"
                sig = f"{qp}->{role}@{off:+d}"
                pattern_counts[hn][qp][sig] += 1
                totals[hn][qp] += 1

    per_head: Dict[str, Any] = {}
    sig_groups: Dict[str, List[str]] = {}
    for hn in sorted(pattern_counts.keys(), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1]))):
        per_pos = {}
        sig_parts = []
        for qp in positions:
            cnt = pattern_counts[hn].get(qp, Counter())
            tot = totals[hn].get(qp, 0)
            top_list = []
            if tot > 0 and cnt:
                for pat, c in cnt.most_common(topk):
                    top_list.append({"pattern": pat, "count": int(c), "total": int(tot)})
                sig_parts.append(cnt.most_common(1)[0][0])
            else:
                sig_parts.append(f"{qp}->NA")
            per_pos[qp] = top_list
        sig = " | ".join(sig_parts)
        per_head[hn] = {"positions": per_pos, "signature": sig}
        sig_groups.setdefault(sig, []).append(hn)

    return {
        "positions": positions,
        "key_positions": key_roles,
        "per_head": per_head,
        "groups_by_signature": sig_groups,
        "window": window,
        "mask_self": mask_self,
        "topk": topk,
    }


def main(argv=None):
    args = parse_args(argv)
    heads = load_nodes_to_heads(args.collapsed_json)
    if not heads:
        raise SystemExit("在 collapsed JSON 中没有发现注意力头（aL.hH）。")

    samples = load_samples(args.standard_json, args.sample_size)
    model = _load_model(args.model, args.model_path, args.device)
    model.eval()

    query_positions = [p.strip().upper() for p in args.query_positions.split(",") if p.strip()]
    key_positions = [p.strip().upper() for p in args.key_positions.split(",") if p.strip()]

    patterns = mine_patterns(
        model=model,
        heads=heads,
        samples=samples,
        window=args.window,
        mask_self=bool(args.mask_self),
        topk=args.topk,
        query_positions=query_positions,
        key_positions=key_positions,
        allow_future=bool(args.allow_future),
    )

    out = {
        "meta": {
            "source_collapsed_json": str(args.collapsed_json),
            "source_standard_json": str(args.standard_json),
            "model": args.model,
            "device": args.device,
            "heads": [f"{L}.{H}" for (L, H) in heads],
            "topk": args.topk,
            "mask_self": bool(args.mask_self),
            "window": args.window,
            "sample_size": args.sample_size,
            "query_positions": query_positions,
            "key_positions": key_positions,
            "model_path": str(args.model_path) if args.model_path else None,
        },
        "positions": patterns["positions"],
        "key_positions": patterns["key_positions"],
        "per_head": patterns["per_head"],
        "groups_by_signature": patterns["groups_by_signature"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"保存模式画像结果: {args.output}")


if __name__ == "__main__":
    main()
