#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从折叠/筛选后的图(JSON)中抽取注意力头集合，调用 find_important_position.run 计算画像，
并挖掘“注意力模式签名”（END/S2 -> IO/S1/S2 的 ±window 目标），按签名给头分组，
写入输出 JSON 的 patterns 与 pattern_groups，供可视化标注。
"""
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_nodes_to_heads(p: Path) -> List[Tuple[int,int]]:
    d = json.loads(p.read_text())
    nodes = d.get("nodes")
    heads: List[Tuple[int,int]] = []
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
    p = argparse.ArgumentParser(description="对筛选/折叠图中的注意力头进行模式画像与分组")
    p.add_argument("--collapsed-json", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True, help="IOI 数据集 standard_ioi_data.json")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--model-path", type=Path, default=None, help="本地模型目录（优先于 --model）。")
    p.add_argument("--device", type=str, default="cuda")
    # 透传到画像
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--mask-self", action="store_true")
    p.add_argument("--entities-only", action="store_true")
    p.add_argument("--entity-window", type=int, default=0)
    p.add_argument("--classify-threshold", type=float, default=0.1)
    p.add_argument("--end-entity-threshold", type=float, default=0.2)
    # 位置与分组阈值
    p.add_argument("--group-positions", type=str, default="END,S2")
    p.add_argument("--group-threshold", type=float, default=0.4)
    # 模式挖掘位置（默认同 group-positions）
    p.add_argument("--pattern-positions", type=str, default="", help="留空=沿用 --group-positions")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)

# —— 新增：模式挖掘（END/S2 -> IO/S1/S2 的 ±window 目标） ——
def _load_samples(ioi_json: Path) -> List[Dict[str, Any]]:
    d = json.loads(ioi_json.read_text())
    if isinstance(d, dict) and isinstance(d.get("samples"), list):
        return d["samples"]
    if isinstance(d, list):
        return d
    return []

def _extract_positions(sample: Dict[str, Any]) -> Dict[str, int]:
    pos = (sample.get("positions") or {})
    def geti(k): 
        v = pos.get(k) if k in pos else pos.get(k.lower())
        return int(v) if isinstance(v, int) else None
    return {"END": geti("end"), "IO": geti("io"), "S1": geti("s1"), "S2": geti("s2")}

def _mine_patterns(ioi_json: Path, model_name: str, model_path: Optional[Path], device: str, heads: List[Tuple[int,int]], qposes: List[str], window: int, mask_self: bool) -> Dict[str, Any]:
    """
    注意：完全复用 find_important_position.run 的前端：HookedTransformer.to_tokens + run_with_cache，
    不再使用 AutoTokenizer，避免 gpt2-small 的 repo 名称问题。
    """
    try:
        from transformer_lens import HookedTransformer
        import torch
    except Exception:
        return {"per_head": {}, "groups_by_signature": {}, "window": window, "note": "缺少依赖，跳过模式挖掘"}

    samples = _load_samples(ioi_json)
    if not samples:
        return {"per_head": {}, "groups_by_signature": {}, "window": window, "note": "无样本，跳过模式挖掘"}

    def _load_model(primary: str, local_path: Optional[Path]) -> HookedTransformer:
        if local_path:
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
            hf_model = AutoModelForCausalLM.from_pretrained(str(local_path), local_files_only=True)
            official_name = primary if primary != "gpt2-small" else "gpt2"
            return HookedTransformer.from_pretrained(
                official_name,
                device=device,
                tokenizer=tokenizer,
                hf_model=hf_model,
                local_files_only=True,
            )

        candidates = [primary]
        if primary == "gpt2-small":
            candidates.append("gpt2")
        last_err: Optional[Exception] = None
        for name in candidates:
            try:
                return HookedTransformer.from_pretrained(name, device=device)
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    model = _load_model(model_name, model_path)
    model.eval()

    # 仅缓存需要的层，方式与 fip.run 保持一致
    layer_set = sorted({L for L, _ in heads})
    names = [f"blocks.{L}.attn.hook_pattern" for L in layer_set]

    from collections import defaultdict, Counter
    pattern_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def candidates(pos_map: Dict[str,int], qpos: int, W: int, max_idx: int):
        outs = []
        for role in ("IO","S1","S2"):
            base = pos_map.get(role)
            if not isinstance(base, int):
                continue
            for off in range(-W, W+1):
                idx = base + off
                if 0 <= idx <= qpos and idx < max_idx:
                    outs.append((role, off, idx))
        return outs

    for s in samples:
        view = (s.get("clean") or s.get("corrupted") or {})
        sentence = view.get("sentence") or s.get("sentence")
        if not isinstance(sentence, str) or not sentence:
            continue
        pos_map = _extract_positions(s)

        # 关键：与 fip.run 一致的分词与缓存
        toks = model.to_tokens(sentence, prepend_bos=False).to(device)  # [1, L]
        if toks is None or toks.numel() == 0:
            continue
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names, return_type=None)
        seq_len = int(toks.shape[1])

        for L, H in heads:
            hook_name = f"blocks.{L}.attn.hook_pattern"
            if hook_name not in cache:
                continue
            pat = cache[hook_name]
            for qp in qposes:
                qpos = pos_map.get(qp)
                if not isinstance(qpos, int) or qpos < 0 or qpos >= seq_len:
                    continue
                row = pat[0, H, qpos, : qpos + 1].detach().float()
                if mask_self:
                    row[qpos] = 0.0
                ssum = float(row.sum().item())
                if ssum > 0:
                    row = row / ssum

                cands = candidates(pos_map, qpos, window, seq_len)
                if not cands:
                    continue
                best = max(cands, key=lambda t: float(row[t[2]].item()))
                role, off, _ = best
                hn = f"{L}.{H}"
                sig = f"{qp}->{role}@{off:+d}"
                pattern_counts[hn][qp][sig] += 1
                totals[hn][qp] += 1

    def sig_for_head(hn: str) -> str:
        parts = []
        for qp in qposes:
            cnt = pattern_counts[hn].get(qp, Counter())
            tot = totals[hn].get(qp, 0)
            parts.append(cnt.most_common(1)[0][0] if (tot > 0 and cnt) else f"{qp}->NA")
        return " | ".join(parts)

    per_head: Dict[str, Any] = {}
    sig_groups: Dict[str, List[str]] = {}
    for hn in sorted(pattern_counts.keys(), key=lambda x: (int(x.split('.')[0]), int(x.split('.')[1]))):
        per_pos = {}
        for qp in qposes:
            cnt = pattern_counts[hn].get(qp, Counter())
            tot = totals[hn].get(qp, 0)
            top2 = []
            if tot > 0 and cnt:
                for pat, c in cnt.most_common(2):
                    top2.append({"pattern": pat, "count": int(c), "total": int(tot)})
            per_pos[qp] = top2
        sig = sig_for_head(hn)
        per_head[hn] = {"positions": per_pos, "signature": sig}
        sig_groups.setdefault(sig, []).append(hn)

    for k in list(sig_groups.keys()):
        sig_groups[k] = sorted(sig_groups[k], key=lambda x: (int(x.split('.')[0]), int(x.split('.')[1])))

    return {"per_head": per_head, "groups_by_signature": sig_groups, "window": window}
# —— 结束 模式挖掘 —— #

def main(argv=None):
    args = parse_args(argv)

    # 动态加载 find_important_position.run（保留你原流程）
    fip_path = (Path(__file__).parent / "find_important_position.py").resolve()
    import importlib.util
    spec = importlib.util.spec_from_file_location("find_important_position", str(fip_path))
    fip = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(fip)  # type: ignore

    heads = load_nodes_to_heads(args.collapsed_json)
    if not heads:
        raise SystemExit("在 collapsed JSON 中没有发现注意力头（aL.hH）。")

    # 画像（沿用原逻辑）
    out = fip.run(
        json_path=args.json,
        model_name=args.model,
        device=args.device,
        heads=heads,
        positions=["END", "S2", "S1", "IO"],
        topk=args.topk,
        classify_threshold=args.classify_threshold,
        mask_self=bool(args.mask_self),
        entities_only=bool(args.entities_only),
        entity_window=int(args.entity_window),
        end_entity_threshold=float(args.end_entity_threshold),
        model_path=args.model_path,
    )

    results: Dict[str, Any] = {
        "meta": {
            "source_collapsed_json": str(args.collapsed_json),
            "source_ioi_json": str(args.json),
            "model": args.model,
            "device": args.device,
            "heads": [f"{L}.{H}" for (L, H) in heads],
            "topk": args.topk,
            "mask_self": bool(args.mask_self),
            "entities_only": bool(args.entities_only),
            "entity_window": int(args.entity_window),
            "classify_threshold": args.classify_threshold,
            "end_entity_threshold": args.end_entity_threshold,
            "pattern_positions": (args.pattern_positions or args.group_positions or "END,S2,S1,IO"),
            "model_path": str(args.model_path) if args.model_path else None,
        },
        "profile": out["profile"],
        "end_entity_mass": out.get("end_entity_mass", {}),
        "role_mass_by_qpos": out.get("role_mass_by_qpos", {}),
        "hard_suggestions_from_run": out["suggestions"],
    }

    # 注意力“模式签名”分组（用 entity_window 看 IO±1/S1±1/S2±1）
    default_pattern_positions = args.pattern_positions or args.group_positions or "END,S2,S1,IO"
    pattern_positions = [p.strip().upper() for p in default_pattern_positions.split(",") if p.strip()]
    patterns = _mine_patterns(args.json, args.model, args.model_path, args.device, heads, pattern_positions, args.entity_window, args.mask_self)
    results["patterns"] = patterns
    if isinstance(patterns, dict):
        results["pattern_groups"] = patterns.get("groups_by_signature", {})
        results["head_patterns"] = patterns.get("per_head", {})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"保存模式画像结果: {args.output}")

if __name__ == "__main__":
    main()