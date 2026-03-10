#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出指定注意力头在 S2 查询位置的注意力分布，方便逐样本核对。"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def load_samples(json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(json_path.read_text())
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"无法解析 IOI 数据集: {json_path}")

def parse_heads(values: List[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for v in values:
        try:
            L_str, H_str = v.split(".")
            out.append((int(L_str), int(H_str)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"非法 head 格式: {v} (应为 L.H)") from exc
    return out

def extract_positions(sample: Dict[str, Any]) -> Dict[str, int | None]:
    pos = sample.get("positions") or {}
    def get(name: str) -> int | None:
        v = pos.get(name) if name in pos else pos.get(name.lower())
        return int(v) if isinstance(v, int) else None
    return {"S2": get("s2"), "S1": get("s1"), "IO": get("io"), "END": get("end")}

def build_entry(
    sentence: str,
    tokens: List[str],
    attn_row: torch.Tensor,
    head_name: str,
    s2_pos: int,
    topk: int,
) -> Dict[str, Any]:
    attn = attn_row.detach().float()
    total = float(attn.sum().item())
    if total > 0:
        attn = attn / total
    attn_list: List[Dict[str, Any]] = []
    for idx in range(int(attn.numel())):
        attn_list.append({
            "index": idx,
            "token": tokens[idx] if idx < len(tokens) else "<unk>",
            "attention": float(attn[idx].item()),
        })
    topk = min(topk, len(attn_list))
    top_entries = sorted(attn_list, key=lambda x: x["attention"], reverse=True)[:topk]
    return {
        "head": head_name,
        "s2_index": s2_pos,
        "top_tokens": top_entries,
        "all_tokens": attn_list,
    }

def inspect(
    json_path: Path,
    model_name: str,
    device: str,
    heads: List[Tuple[int, int]],
    topk: int,
    output: Path,
    mask_self: bool,
) -> None:
    samples = load_samples(json_path)
    if not samples:
        raise SystemExit("数据集中没有样本。")

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    needed_layers = sorted({layer for layer, _ in heads})
    hook_names = [f"blocks.{layer}.attn.hook_pattern" for layer in needed_layers]

    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        record: Dict[str, Any] = {}
        clean = sample.get("clean") or {}
        sentence = clean.get("sentence") or sample.get("sentence")
        if not isinstance(sentence, str) or not sentence:
            continue
        positions = extract_positions(sample)
        s2_pos = positions.get("S2")
        if s2_pos is None:
            continue

        with torch.no_grad():
            toks = model.to_tokens(sentence, prepend_bos=False).to(device)
            if toks.numel() == 0:
                continue
            str_tokens = model.to_str_tokens(sentence, prepend_bos=False)
            _, cache = model.run_with_cache(toks, names_filter=lambda name: name in hook_names, return_type=None)

        entry_heads: List[Dict[str, Any]] = []
        for layer, head_idx in heads:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            if hook_name not in cache:
                continue
            pattern = cache[hook_name]  # shape: [1, n_heads, seq, seq]
            if s2_pos < 0 or s2_pos >= pattern.shape[2]:
                continue
            row = pattern[0, head_idx, s2_pos, : s2_pos + 1].clone()
            if mask_self and s2_pos < row.numel():
                row[s2_pos] = 0.0
            entry_heads.append(build_entry(sentence, str_tokens, row, f"{layer}.{head_idx}", int(s2_pos), topk))

        if not entry_heads:
            continue

        record["sample_index"] = idx
        record["sentence"] = sentence
        record["tokens"] = str_tokens
        record["heads"] = entry_heads
        results.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "meta": {
            "source_json": str(json_path),
            "model": model_name,
            "device": device,
            "heads": [f"{L}.{H}" for (L, H) in heads],
            "topk": topk,
            "mask_self": mask_self,
            "samples": len(results),
        },
        "samples": results,
    }, indent=2, ensure_ascii=False))

    print(f"已保存注意力分布: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出指定头在 S2 查询处的注意力分布")
    parser.add_argument("--json", type=Path, required=True, help="standard_ioi_data.json 路径")
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--heads", nargs="+", default=["5.8", "5.9"], help="格式 L.H，例如 5.8 5.9")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON 文件")
    parser.add_argument("--mask-self", action="store_true", help="是否去掉自注意 (S2->S2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads = parse_heads(args.heads)
    inspect(
        json_path=args.json,
        model_name=args.model,
        device=args.device,
        heads=heads,
        topk=args.topk,
        output=args.output,
        mask_self=bool(args.mask_self),
    )


if __name__ == "__main__":
    main()
