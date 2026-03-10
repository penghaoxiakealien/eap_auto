#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def parse_args():
    p = argparse.ArgumentParser(description="导出指定注意力头在指定查询位置的注意力行（逐样本），以自定义文本格式打印。")
    p.add_argument("--json", type=Path, required=True, help="standard_ioi_data.json（包含 samples 列表）")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--heads", type=str, default="7.9,8.6", help="形如 L.H，逗号分隔，如 7.9,8.6")
    p.add_argument("--positions", type=str, default="END,S2", help="查询位置，逗号分隔：END,S2,S1,IO")
    p.add_argument("--variant", type=str, default="clean", choices=["clean","abc_corrupted","swapped"], help="使用哪个视图")
    p.add_argument("--mask-self", action="store_true", help="去自注意（对角置零后重归一）")
    p.add_argument("--max-samples", type=int, default=0, help=">0 则只处理前 N 个样本")
    return p.parse_args()

def load_samples(path: Path) -> List[Dict[str, Any]]:
    d = json.loads(path.read_text())
    if not isinstance(d, dict) or "samples" not in d or not isinstance(d["samples"], list):
        raise ValueError(f"{path} 不包含 'samples' 列表")
    return d["samples"]

def extract_positions(sample: Dict[str, Any]) -> Dict[str, Optional[int]]:
    pos = sample.get("positions", {}) or {}
    def pick(k: str) -> Optional[int]:
        v = pos.get(k); return int(v) if isinstance(v, int) else None
    return {"END": pick("end"), "IO": pick("io"), "S1": pick("s1"), "S2": pick("s2")}

def ensure_libs():
    try:
        from transformer_lens import HookedTransformer  # noqa: F401
    except Exception as e:
        raise SystemExit("需要安装 transformer_lens：pip install transformer-lens --upgrade") from e
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except Exception as e:
        raise SystemExit("需要安装 transformers：pip install transformers --upgrade") from e

def sanitize(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ").replace("\t", " ")

def encode_with_offsets(hf_tok, text: str):
    enc = hf_tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    # 统一成一维
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    if isinstance(ids[0], list):
        ids = ids[0]; offs = offs[0]
    return ids, [(int(a), int(b)) for (a, b) in offs]

def main():
    args = parse_args()
    ensure_libs()
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    import torch

    # 模型与分词器（fast tokenizer 提供 offsets）
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    hf_tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    samples = load_samples(args.json)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    # 解析 heads 与 positions
    heads: List[Tuple[int,int,str]] = []
    for s in [x.strip() for x in args.heads.split(",") if x.strip()]:
        Ls, Hs = s.split("."); L, H = int(Ls), int(Hs)
        heads.append((L, H, f"{L}.{H}"))
    positions = [x.strip().upper() for x in args.positions.split(",") if x.strip()]
    for p in positions:
        if p not in ("END","S2","S1","IO"):
            raise SystemExit(f"不支持的位置: {p}（只支持 END/S2/S1/IO）")

    # 打开输出文件：每个 头×位置 一个 txt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fhs: Dict[Tuple[str,str], Any] = {}
    for _, _, hname in heads:
        for role in positions:
            path = args.output_dir / f"s_check_{hname}_{role}.txt"
            fhs[(hname, role)] = path.open("w", encoding="utf-8")

    # 统计
    written_blocks = 0
    skipped_no_pos = 0
    skipped_no_sentence = 0
    skipped_oob = 0

    for i, sample in enumerate(samples):
        pos_map = extract_positions(sample)
        if not any(isinstance(pos_map.get(role), int) for role in positions):
            skipped_no_pos += 1
            continue

        view = sample.get(args.variant, {}) or {}
        sentence = view.get("sentence")
        if not isinstance(sentence, str) or not sentence:
            skipped_no_sentence += 1
            continue

        # 用 HF fast tokenizer 获取 ids 与 offsets，再送入 TLens
        ids, offsets = encode_with_offsets(hf_tok, sentence)
        if not ids:
            skipped_no_sentence += 1
            continue
        toks = torch.tensor([ids], dtype=torch.long, device=args.device)  # [1, L]
        seq_len = int(toks.shape[1])

        for L, H, hname in heads:
            pat_name = f"blocks.{L}.attn.hook_pattern"
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=lambda n: n == pat_name, return_type=None)
            if pat_name not in cache:
                raise SystemExit(f"缓存缺少 {pat_name}，请检查层号与模型。")
            pattern = cache[pat_name]  # [1, n_heads, q_len, k_len]
            q_len = int(pattern.shape[2])

            for role in positions:
                qpos = pos_map.get(role)
                if not isinstance(qpos, int):
                    continue
                if qpos < 0 or qpos >= q_len or qpos >= seq_len:
                    skipped_oob += 1
                    continue

                row = pattern[0, H, qpos, : qpos + 1].detach().float().cpu()
                if args.mask_self:
                    row[qpos] = 0.0
                ssum = float(row.sum().item())
                if ssum > 0:
                    row = row / ssum

                # 写块：token 文本用 offsets 回抽，不再单字母
                fh = fhs[(hname, role)]
                fh.write(f"句子：\n{sanitize(sentence)}\n")
                fh.write("{\n")
                for ti in range(qpos + 1):
                    a, b = offsets[ti]
                    tok_text = sentence[a:b] if (0 <= a < b <= len(sentence)) else hf_tok.convert_ids_to_tokens(ids[ti])
                    tok_text = sanitize(tok_text)
                    score = float(row[ti].item())
                    fh.write(f"    token: {tok_text}, att_score: {score:.6f}\n")
                fh.write("}\n\n")
                written_blocks += 1

    for fh in fhs.values():
        fh.close()

    meta = {
        "model": args.model,
        "device": args.device,
        "heads": [h for _,_,h in heads],
        "positions": positions,
        "mask_self": bool(args.mask_self),
        "variant": args.variant,
        "total_samples": len(samples),
        "written_blocks": written_blocks,
        "skipped_no_positions": skipped_no_pos,
        "skipped_no_sentence": skipped_no_sentence,
        "skipped_qpos_oob": skipped_oob,
        "files": {f"{h}_{r}": str((args.output_dir / f"s_check_{h}_{r}.txt").resolve()) for _,_,h in heads for r in positions},
    }
    print(json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()