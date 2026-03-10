#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, math, os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch as t
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
t.set_grad_enabled(False)

Role = str

def load_ioi_dataset(json_path: str | Path) -> dict:
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_positions(sample: Dict[str, Any]) -> Dict[Role, int | None]:
    pos = sample.get("positions", {}) or {}
    return {
        "END": pos.get("end"),
        "S1": pos.get("s1"),
        "S2": pos.get("s2"),
        "IO": pos.get("io"),
    }

def valid(sample: Dict[str, Any]) -> bool:
    clean = sample.get("clean", {})
    return isinstance(clean.get("sentence"), str) and len(clean.get("sentence")) > 0

def attn_metrics(attn_vec: t.Tensor, topk: int) -> Dict[str, float]:
    s = t.sum(attn_vec)
    if s.item() <= 0:
        p = t.full_like(attn_vec, 1.0 / max(1, attn_vec.numel()))
    else:
        p = attn_vec / s
    n = p.numel()
    eps = 1e-12
    H = -t.sum(p * (p + eps).log()).item()
    H_norm = (H / math.log(n)) if n > 1 else 0.0
    hhi = t.sum(p * p).item()
    top1 = float(t.topk(p, k=min(1, n)).values.sum().item() if n > 0 else 0.0)
    topk_mass_val = float(t.topk(p, k=min(topk, n)).values.sum().item() if n > 0 else 0.0)
    return {
        "entropy": H,
        "normalized_entropy": H_norm,
        "concentration": 1.0 - H_norm,
        "hhi": hhi,
        "eff_support": (1.0 / hhi) if hhi > 0 else float("inf"),
        "top1": top1,
        "topk": topk_mass_val,
    }

def parse_heads(heads_arg: List[str]) -> List[Tuple[int,int]]:
    heads = []
    for h in heads_arg:
        try:
            L, H = h.split(".")
            heads.append((int(L), int(H)))
        except Exception:
            raise ValueError(f"非法 head 格式: {h} (应为 L.H)")
    return heads

def build_entity_mask(qpos: int, pos_map: Dict[Role, int | None], window: int) -> t.Tensor:
    keep = set()
    for r in ("S1", "S2", "IO"):
        rp = pos_map.get(r)
        if rp is None:
            continue
        for off in range(-window, window + 1):
            kp = rp + off
            if 0 <= kp <= qpos:
                keep.add(kp)
    mask = t.zeros(qpos + 1, dtype=t.bool)
    for k in keep:
        mask[k] = True
    return mask

def run(
    json_path: Path,
    model_name: str,
    device: str,
    heads: List[Tuple[int,int]],
    positions: List[Role],
    topk: int,
    classify_threshold: float,
    mask_self: bool,
    entities_only: bool,
    entity_window: int,
    end_entity_threshold: float,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    data = load_ioi_dataset(json_path)
    samples = [s for s in data.get("samples", []) if valid(s)]
    if not samples:
        raise ValueError("样本为空或 JSON 结构缺少 clean.sentence。")

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
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True

    needed_layers = sorted(set(L for L,_ in heads))

    metrics_acc: Dict[str, Dict[Role, List[Dict[str, float]]]] = {
        f"{L}.{H}": {role: [] for role in positions} for (L,H) in heads
    }
    count_acc: Dict[str, Dict[Role, int]] = {
        f"{L}.{H}": {role: 0 for role in positions} for (L,H) in heads
    }
    skip_acc: Dict[str, Dict[Role, int]] = {
        f"{L}.{H}": {role: 0 for role in positions} for (L,H) in heads
    }
    end_role_mass: Dict[str, Dict[str, float]] = {
        f"{L}.{H}": {"IO": 0.0, "S1": 0.0, "S2": 0.0, "count": 0.0} for (L,H) in heads
    }
    pos_role_mass: Dict[str, Dict[str, Dict[str, float]]] = {
        f"{L}.{H}": {role: {"IO": 0.0, "S1": 0.0, "S2": 0.0, "count": 0.0} for role in positions}
        for (L, H) in heads
    }
    for s in samples:
        sent = s["clean"]["sentence"]
        pos_map = get_positions(s)
        toks = model.to_tokens(sent, prepend_bos=False).to(device)
        seq_len = toks.shape[1]

        def names_filter(n: str) -> bool:
            return n.endswith("hook_pattern") and any(
                n == f"blocks.{L}.attn.hook_pattern" for L in needed_layers
            )

        _, cache = model.run_with_cache(toks, names_filter=names_filter)

        for (L, H) in heads:
            head_name = f"{L}.{H}"
            pattern = cache[f"blocks.{L}.attn.hook_pattern"]  # [1, n_heads, seq, seq]
            for role in positions:
                qpos = pos_map.get(role)
                if qpos is None or qpos < 0 or qpos >= seq_len:
                    skip_acc[head_name][role] += 1
                    continue
                qpos = int(qpos)
                attn_vec = pattern[0, H, qpos, :qpos+1].clone()

                if mask_self:
                    attn_vec[qpos] = 0.0

                if entities_only:
                    emask = build_entity_mask(qpos, pos_map, window=entity_window)
                    if not emask.any():
                        skip_acc[head_name][role] += 1
                        continue
                    attn_vec = attn_vec.masked_fill(~emask.to(attn_vec.device), 0.0)

                m = attn_metrics(attn_vec, topk=topk)
                metrics_acc[head_name][role].append(m)
                count_acc[head_name][role] += 1

                v_cur = pattern[0, H, qpos, :qpos+1].clone()
                if mask_self:
                    v_cur[qpos] = 0.0
                total_cur = float(v_cur.sum().item()) or 1.0
                def pick_mass(pname: str) -> float:
                    kp = pos_map.get(pname)
                    if kp is None or kp > qpos or kp < 0:
                        return 0.0
                    return float(v_cur[kp].item()) / total_cur
                pos_role_mass[head_name][role]["IO"]   += pick_mass("IO")
                pos_role_mass[head_name][role]["S1"]   += pick_mass("S1")
                pos_role_mass[head_name][role]["S2"]   += pick_mass("S2")
                pos_role_mass[head_name][role]["count"]+= 1.0

                # 历史 END 字段保持
                if role == "END":
                    v_end = v_cur  # 已处理自注意
                    total = total_cur
                    def pick(pname: str) -> float:
                        kp = pos_map.get(pname)
                        if kp is None or kp > qpos or kp < 0:
                            return 0.0
                        return float(v_end[kp].item()) / total
                    end_role_mass[head_name]["IO"] += pick("IO")
                    end_role_mass[head_name]["S1"] += pick("S1")
                    end_role_mass[head_name]["S2"] += pick("S2")
                    end_role_mass[head_name]["count"] += 1.0

    def agg_stats(items: List[Dict[str, float]]) -> Dict[str, Any]:
        if not items:
            return {"count": 0}
        keys = items[0].keys()
        out = {"count": len(items)}
        for k in keys:
            vals = [it[k] for it in items]
            mean = float(sum(vals) / len(vals))
            var = float(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
            std = math.sqrt(var)
            out[k + "_mean"] = mean
            out[k + "_std"] = std
        return out

    profile: Dict[str, Any] = {}
    for head_name in metrics_acc:
        profile[head_name] = {}
        for role in positions:
            profile[head_name][role] = {
                "aggregated": agg_stats(metrics_acc[head_name][role]),
                "skipped": skip_acc[head_name][role],
            }

    # 分类建议
    suggestions: Dict[str, str] = {}
    for head_name in metrics_acc:
        c = max(1.0, end_role_mass[head_name]["count"])
        io_m = end_role_mass[head_name]["IO"] / c
        s1_m = end_role_mass[head_name]["S1"] / c
        s2_m = end_role_mass[head_name]["S2"] / c
        nm_score = (io_m + s1_m) - s2_m

        label = "mixed"
        if nm_score > end_entity_threshold:
            label = "name_mover_like"
        else:
            conc_end = profile[head_name].get("END", {}).get("aggregated", {}).get("concentration_mean", None)
            conc_s2  = profile[head_name].get("S2",  {}).get("aggregated", {}).get("concentration_mean", None)
            if (conc_end is not None) and (conc_s2 is not None) and ((conc_s2 - conc_end) > classify_threshold):
                label = "s_inhibition_like"
        suggestions[head_name] = label

    return {
        "meta": {
            "json": str(json_path),
            "model": model_name,
            "device": device,
            "heads": [f"{L}.{H}" for (L,H) in heads],
            "positions": positions,
            "topk": topk,
            "classify_threshold": classify_threshold,
            "mask_self": mask_self,
            "entities_only": entities_only,
            "entity_window": entity_window,
            "end_entity_threshold": end_entity_threshold,
            "model_path": str(model_path) if model_path else None,
        },
        "profile": profile,
        "suggestions": suggestions,
        "counts": count_acc,
        "skipped": skip_acc,
        "end_entity_mass": end_role_mass,
        "role_mass_by_qpos": pos_role_mass,   # 新增
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按角色位置(END/S2/S1/IO)评估注意力集中度（可去自注意/仅实体域），并给出分类建议。"
    )
    p.add_argument("--json", type=str, required=True, help="standard_ioi_data.json 路径")
    p.add_argument("--heads", nargs="+", required=True, help="L.H 列表，如 9.6 9.9 10.0 8.10 7.9")
    p.add_argument("--positions", nargs="+", default=["END", "S2", "S1", "IO"], help="对齐位置")
    p.add_argument("--topk", type=int, default=2, help="top-k 质量指标")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=0.1, help="conc(S2)-conc(END) 判 s-inhibition 的阈值")
    p.add_argument("--mask-self", action="store_true", help="去自注意(对角)")
    p.add_argument("--entities-only", action="store_true", help="仅在实体集合(S1,S2,IO)上计算指标")
    p.add_argument("--entity-window", type=int, default=0, help="实体邻域窗口（与 --entities-only 联用）")
    p.add_argument("--end-entity-threshold", type=float, default=0.2, help="判 NMH 的 END 实体质量阈值")
    p.add_argument("--output", type=str, required=True, help="输出 JSON 路径")
    p.add_argument("--model-path", type=str, default=None, help="本地模型目录（优先于 --model）")
    return p.parse_args()

def main():
    args = parse_args()
    heads = parse_heads(args.heads)
    out = run(
        json_path=Path(args.json),
        model_name=args.model,
        device=args.device,
        heads=heads,
        positions=args.positions,
        topk=args.topk,
        classify_threshold=args.threshold,
        mask_self=bool(args.mask_self),
        entities_only=bool(args.entities_only),
        entity_window=int(args.entity_window),
        end_entity_threshold=float(args.end_entity_threshold),
        model_path=Path(args.model_path) if args.model_path else None,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"保存: {out_path}")
    for hn, label in out["suggestions"].items():
        conc_end = out["profile"][hn].get("END",{}).get("aggregated",{}).get("concentration_mean","NA")
        conc_s2  = out["profile"][hn].get("S2",{}).get("aggregated",{}).get("concentration_mean","NA")
        em = out["end_entity_mass"][hn]; c = max(1.0, em["count"])
        io_m, s1_m, s2_m = em["IO"]/c, em["S1"]/c, em["S2"]/c
        print(f"[{hn}] -> {label} | conc_END={conc_end} conc_S2={conc_s2} | END mass IO={io_m:.3f} S1={s1_m:.3f} S2={s2_m:.3f}")

if __name__ == "__main__":
    main()