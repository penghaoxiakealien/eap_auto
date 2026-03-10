#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽样查看 agr_gender 句子在指定位置（默认 END）上的真实注意力分布。

示例：
  python tests/experiments/debug_agr_gender_attention.py \\
    --standard-json results/agr_gender/standard_gender_data.json \\
    --heads 0.1,7.3,9.6 \\
    --n-sentences 30 \\
    --topk 8 \\
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_head_list(s: str) -> List[Tuple[int, int]]:
    heads: List[Tuple[int, int]] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "." not in part:
            raise ValueError(f"head 格式应为 L.H，例如 7.3；收到: {part}")
        L, H = part.split(".", 1)
        heads.append((int(L), int(H)))
    return heads


def load_standard(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("standard json 应为 list")
    return data


def pick_sentences(data: List[Dict], n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    if n <= 0 or n >= len(data):
        return data
    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    return [data[i] for i in idxs[:n]]


def get_position(sample: Dict, which: str) -> Optional[int]:
    which_u = which.upper()
    if which_u == "END":
        if isinstance(sample.get("end_idx"), int):
            return int(sample["end_idx"])
        wd = sample.get("word_idx") or {}
        v = wd.get("end")
        return int(v) if isinstance(v, int) else None
    if which_u == "VERB":
        wd = sample.get("word_idx") or {}
        v = wd.get("verb")
        return int(v) if isinstance(v, int) else None
    raise ValueError(f"未知位置: {which}")


def load_model(device: str, model_name: str, model_path: Optional[str]) -> HookedTransformer:
    if model_path:
        # transformer_lens 对本地路径支持比较有限，这里直接用官方名 + local_files_only 由环境决定
        # 需要本地路径时，建议用户设置 TRANSFORMERS_CACHE/HF_HOME，并用官方名加载。
        pass
    return HookedTransformer.from_pretrained(model_name, device=device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug agr_gender attention patterns at END/VERB")
    ap.add_argument("--standard-json", type=Path, required=True)
    ap.add_argument("--heads", type=str, required=True, help="逗号分隔: 7.3,9.6,...")
    ap.add_argument("--position", type=str, default="END", choices=["END", "VERB", "end", "verb"])
    ap.add_argument("--n-sentences", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model-name", type=str, default="gpt2")
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--output", type=Path, default=None, help="可选：输出 JSONL")
    args = ap.parse_args()

    heads = parse_head_list(args.heads)
    layers = sorted({L for L, _ in heads})
    hook_names = {f"blocks.{L}.attn.hook_pattern" for L in layers}

    data = load_standard(args.standard_json)
    samples = pick_sentences(data, args.n_sentences, args.seed)

    model = load_model(args.device, args.model_name, args.model_path)
    model.eval()

    out_lines = []
    for i, s in enumerate(samples):
        text = s.get("text") or s.get("sentence")
        if not isinstance(text, str) or not text:
            continue
        pos = get_position(s, args.position)
        toks = model.to_tokens(text, prepend_bos=False).to(args.device)
        seq_len = int(toks.shape[1])
        if pos is None or pos < 0 or pos >= seq_len:
            # 位置无效则跳过
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks,
                names_filter=lambda n: n in hook_names,
                return_type=None,
            )

        token_strs = model.to_str_tokens(toks[0])
        record = {
            "idx": i,
            "text": text,
            "position": args.position.upper(),
            "pos_idx": pos,
            "pos_token": token_strs[pos],
            "heads": {},
        }

        for L, H in heads:
            hook = f"blocks.{L}.attn.hook_pattern"
            if hook not in cache:
                continue
            pat = cache[hook]  # [batch, head, qpos, kpos]
            row = pat[0, H, pos, : pos + 1].detach().float().cpu()
            # topk key positions
            k = min(args.topk, row.numel())
            vals, idxs = torch.topk(row, k=k, largest=True)
            top = []
            for v, j in zip(vals.tolist(), idxs.tolist()):
                top.append({"kpos": int(j), "token": token_strs[j], "attn": float(v)})
            record["heads"][f"{L}.{H}"] = top

        out_lines.append(record)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in out_lines:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ wrote {len(out_lines)} records to {args.output}")
    else:
        # 简单打印
        for r in out_lines:
            print(f"\n=== sample {r['idx']} pos={r['pos_idx']} token={r['pos_token']!r} ===")
            print(r["text"])
            for hn, tops in r["heads"].items():
                tops_s = ", ".join(
                    [f"{t['token']!r}:{t['attn']:.3f}" for t in tops[: min(5, len(tops))]]
                )
                print(f"  head {hn}: {tops_s}")


if __name__ == "__main__":
    main()
