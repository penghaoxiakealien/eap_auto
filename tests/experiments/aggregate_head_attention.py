#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

import torch as t
from transformer_lens import HookedTransformer

# （可选）动态生成 IOI 数据
try:
    from ioi_dataset import IOIDataset
    HAS_IOI_DS = True
except ImportError:
    HAS_IOI_DS = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    p = argparse.ArgumentParser(description="聚合多个 heads 在 IOI 任务 end 位置的注意力 TopK token 分布，并计算相似度。")
    p.add_argument("--input-standard", type=Path, help="standard_ioi_data.json 路径（与 --generate 互斥）")
    p.add_argument("--generate", type=int, default=0, help="动态生成 N 条 IOI 样本（>0 生效）")
    p.add_argument("--prompt-type", type=str, default="mixed", help="生成 IOI 时的 prompt_type")
    p.add_argument("--seed", type=int, default=1, help="生成数据集种子")
    p.add_argument("--heads", nargs="+", required=True, help="要分析的 heads，格式 L.H，如 9.6 9.9 10.0 8.10 7.9")
    p.add_argument("--top-k", type=int, default=3, help="每条样本提取注意力最高的前 K 个 token（默认3）")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--output", type=Path, required=True, help="输出 JSON 文件")
    p.add_argument("--strip-space", action="store_true", help="去掉 token 前导空格（默认保留原始 BPE）")
    return p.parse_args()

def load_dataset_from_standard(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    samples = []
    for s in data["samples"]:
        clean = s.get("clean", {})
        sent = clean.get("sentence")
        if not sent:
            continue
        end_pos = s.get("positions", {}).get("end")
        if end_pos is None:
            continue
        samples.append({
            "sentence": sent,
            "end_pos": end_pos
        })
    return samples

def generate_ioi_samples(n: int, prompt_type: str, seed: int, model: HookedTransformer) -> List[Dict[str, Any]]:
    if not HAS_IOI_DS:
        raise RuntimeError("未找到 ioi_dataset.IOIDataset，无法动态生成。请安装或提供 --input-standard。")
    ds = IOIDataset(
        prompt_type=prompt_type,
        N=n,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device="cpu"  # 文本本身即可
    )
    samples = []
    # IOIDataset: ds.sentences / ds.word_idx["end"]
    end_idx = ds.word_idx["end"]
    for toks_str in ds.sentences:
        # 注意：positions.end 可能与 end_idx 一致（统一用 end_idx）
        samples.append({
            "sentence": toks_str,
            "end_pos": end_idx
        })
    return samples

def js_distance(p: List[float], q: List[float]) -> float:
    # Jensen-Shannon 距离 = sqrt(JS散度)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def kl(a, b):
        eps = 1e-12
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0:
                s += ai * math.log((ai + eps) / (bi + eps))
        return s
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return math.sqrt(max(jsd, 0.0))

def cosine_similarity(p: List[float], q: List[float]) -> float:
    num = sum(pi * qi for pi, qi in zip(p, q))
    dp = math.sqrt(sum(pi * pi for pi in p))
    dq = math.sqrt(sum(qi * qi for qi in q))
    if dp == 0 or dq == 0:
        return 0.0
    return num / (dp * dq)

def build_vectors(head_stats: Dict[str, Any], vocab: List[str]):
    # 返回：head -> (full_weight_vec, topk_count_vec)
    weight_vectors = {}
    count_vectors = {}
    for head_name, stats in head_stats.items():
        agg_w = stats["aggregated_weight_sum"]   # token -> weight sum (全部 token)
        topk_counts = stats["topk_token_counts"] # token -> 频次
        w_vec = [agg_w.get(tok, 0.0) for tok in vocab]
        c_vec = [topk_counts.get(tok, 0) for tok in vocab]
        # 归一化 weight 转概率
        w_sum = sum(w_vec)
        if w_sum > 0:
            w_vec_norm = [x / w_sum for x in w_vec]
        else:
            w_vec_norm = w_vec
        # 可选：Counts 也转概率（此处保留原值用于参考，可单独再做）
        weight_vectors[head_name] = w_vec_norm
        count_vectors[head_name] = c_vec
    return weight_vectors, count_vectors

def rbo(list1, list2, p=0.9):
    # Rank-Biased Overlap (简化版，假定无重复；有重复已很少)
    S, T = list1, list2
    if not S or not T:
        return 0.0
    k = min(len(S), len(T))
    overlap = 0
    rbo_ext = 0.0
    seenS = set()
    seenT = set()
    for d in range(1, k+1):
        seenS.add(S[d-1])
        seenT.add(T[d-1])
        overlap_d = len(seenS & seenT)
        rbo_ext += (overlap_d / d) * (p ** (d-1))
    return (1 - p) * rbo_ext

def ndcg_overlap(list1, list2):
    # 只对交集；相关性=1。DCG / 理论最大 DCG(k)
    if not list1 or not list2: return 0.0
    pos2 = {tok:i for i,tok in enumerate(list2)}
    gains = []
    for i,tok in enumerate(list1):
        if tok in pos2:
            gains.append(1 / math.log2(2 + i))  # list1 中位置权重
    if not gains: return 0.0
    # 最佳情况（前 |gains| 都匹配在最前）同样使用 list1 的排序框架，最大 DCG 为前 m 项权重之和
    m = len(gains)
    ideal = sum(1 / math.log2(2 + i) for i in range(m))
    return sum(gains) / ideal if ideal > 0 else 0.0

def main():
    args = parse_args()

    # 解析 heads
    heads: List[Tuple[int,int]] = []
    for h in args.heads:
        try:
            l, hi = h.split(".")
            heads.append((int(l), int(hi)))
        except:
            raise ValueError(f"非法 head 格式: {h} (应为 L.H)")
    needed_layers = sorted(set(l for l,_ in heads))

    print(f"[info] 目标 heads: {args.heads}")
    print(f"[info] 目标层集合: {needed_layers}")

    # 加载模型
    print(f"[info] 加载模型 {args.model} ...")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True

    # 准备样本
    if args.generate > 0:
        samples = generate_ioi_samples(args.generate, args.prompt_type, args.seed, model)
        print(f"[info] 动态生成 IOI 样本: {len(samples)} 条")
    else:
        if not args.input_standard:
            raise ValueError("需提供 --input-standard 或使用 --generate N")
        samples = load_dataset_from_standard(args.input_standard)
        print(f"[info] 从 standard_ioi_data 加载样本: {len(samples)} 条")

    if len(samples) == 0:
        raise ValueError("无有效样本。")

    # 统计结构
    head_stats: Dict[str, Any] = {}
    for (l,h) in heads:
        name = f"{l}.{h}"
        head_stats[name] = {
            "layer": l,
            "head": h,
            "total_samples": 0,
            "top1_token_counts": Counter(),
            "topk_token_counts": Counter(),
            "topk_weight_sum": Counter(),
            "aggregated_weight_sum": Counter(),  # 全部 token 的注意力权重总和
            "topk_unique_tokens": set(),         # 样本级 topK 汇总集合
        }

    per_sample_topk: List[Dict[str, List[str]]] = []  # 每样本: head -> 有序 topk token 列表

    for idx, sample in enumerate(samples):
        sent = sample["sentence"]
        toks = model.to_tokens(sent, prepend_bos=False)
        str_tokens = model.to_str_tokens(toks[0])
        seq_len = toks.shape[1]
        end_pos = min(sample["end_pos"], seq_len-1)
        def names_filter(n: str):
            return n.endswith("hook_pattern") and any(
                n == f"blocks.{L}.attn.hook_pattern" for L in needed_layers
            )
        with t.no_grad():
            _, cache = model.run_with_cache(toks.to(args.device), names_filter=names_filter)

        sample_head_topk = {}
        for (L,H) in heads:
            hn = f"{L}.{H}"
            pattern = cache[f"blocks.{L}.attn.hook_pattern"]
            attn_vec = pattern[0, H, end_pos, :end_pos+1]
            topk = min(args.top_k, attn_vec.shape[0])
            vals, inds = t.topk(attn_vec, k=topk)
            ordered = []
            for v,i_pos in zip(vals, inds):
                tok = str_tokens[int(i_pos)]
                if args.strip_space: tok = tok.lstrip()
                ordered.append(tok)
                # 累积统计（原逻辑）
                head_stats[hn]["topk_token_counts"][tok] += 1
                head_stats[hn]["topk_weight_sum"][tok] += float(v)
                head_stats[hn]["topk_unique_tokens"].add(tok)
            if topk > 0:
                top1_tok = ordered[0]
                head_stats[hn]["top1_token_counts"][top1_tok] += 1
            # 全量权重
            for pos in range(end_pos+1):
                tok_all = str_tokens[pos]
                if args.strip_space: tok_all = tok_all.lstrip()
                head_stats[hn]["aggregated_weight_sum"][tok_all] += float(attn_vec[pos])
            head_stats[hn]["total_samples"] += 1
            sample_head_topk[hn] = ordered
        per_sample_topk.append(sample_head_topk)

    # === 计算聚合派生（保持原逻辑）===
    for hn, stats in head_stats.items():
        ts = stats["total_samples"]
        stats["top1_token_freq"] = {k: v/ts for k,v in stats["top1_token_counts"].items()}
        stats["topk_token_freq"] = {k: v/ts for k,v in stats["topk_token_counts"].items()}
        total_w = sum(stats["aggregated_weight_sum"].values()) or 1.0
        stats["aggregated_weight_prob"] = {k: w/total_w for k,w in stats["aggregated_weight_sum"].items()}
        topk_w_sum = sum(stats["topk_weight_sum"].values()) or 1.0
        stats["topk_weight_prob"] = {k: w/topk_w_sum for k,w in stats["topk_weight_sum"].items()}
        stats["topk_unique_tokens"] = sorted(stats["topk_unique_tokens"])

    # === 新增：逐样本 pairwise 相似度 (平均) ===
    heads_order = list(head_stats.keys())
    per_sample_metrics = {
        "mean_jaccard_k": {},
        "mean_overlap_ratio": {},
        "mean_rbo": {},
        "mean_ndcg_overlap": {}
    }
    for hi in heads_order:
        per_sample_metrics["mean_jaccard_k"][hi] = {}
        per_sample_metrics["mean_overlap_ratio"][hi] = {}
        per_sample_metrics["mean_rbo"][hi] = {}
        per_sample_metrics["mean_ndcg_overlap"][hi] = {}
        for hj in heads_order:
            if hi == hj:
                per_sample_metrics["mean_jaccard_k"][hi][hj] = 1.0
                per_sample_metrics["mean_overlap_ratio"][hi][hj] = 1.0
                per_sample_metrics["mean_rbo"][hi][hj] = 1.0
                per_sample_metrics["mean_ndcg_overlap"][hi][hj] = 1.0
                continue
            j_sum = o_sum = rbo_sum = ndcg_sum = 0.0
            count = 0
            for sample_map in per_sample_topk:
                li = sample_map.get(hi, [])
                lj = sample_map.get(hj, [])
                if not li and not lj:
                    continue
                set_i, set_j = set(li), set(lj)
                inter = len(set_i & set_j)
                union = len(set_i | set_j) or 1
                k_ref = max(1, min(len(li), len(lj)))
                j_sum += inter / union
                o_sum += inter / k_ref
                rbo_sum += rbo(li, lj, p=0.9)
                ndcg_sum += ndcg_overlap(li, lj)
                count += 1
            if count == 0:
                per_sample_metrics["mean_jaccard_k"][hi][hj] = 0.0
                per_sample_metrics["mean_overlap_ratio"][hi][hj] = 0.0
                per_sample_metrics["mean_rbo"][hi][hj] = 0.0
                per_sample_metrics["mean_ndcg_overlap"][hi][hj] = 0.0
            else:
                per_sample_metrics["mean_jaccard_k"][hi][hj] = j_sum / count
                per_sample_metrics["mean_overlap_ratio"][hi][hj] = o_sum / count
                per_sample_metrics["mean_rbo"][hi][hj] = rbo_sum / count
                per_sample_metrics["mean_ndcg_overlap"][hi][hj] = ndcg_sum / count

    # === 原先全局分布相似度（保留）===
    vocab = sorted(set(tok for h in head_stats.values() for tok in h["aggregated_weight_prob"].keys()))
    weight_vectors, _ = build_vectors(head_stats, vocab)
    pairwise_global = {"js_distance":{}, "cosine_similarity":{}, "jaccard_topk":{}}
    topk_sets = {h:set(head_stats[h]["topk_unique_tokens"]) for h in heads_order}
    for i,hi in enumerate(heads_order):
        pairwise_global["js_distance"][hi] = {}
        pairwise_global["cosine_similarity"][hi] = {}
        pairwise_global["jaccard_topk"][hi] = {}
        for j,hj in enumerate(heads_order):
            if hi==hj:
                pairwise_global["js_distance"][hi][hj]=0.0
                pairwise_global["cosine_similarity"][hi][hj]=1.0
                pairwise_global["jaccard_topk"][hi][hj]=1.0
            elif j<i:
                pairwise_global["js_distance"][hi][hj]=pairwise_global["js_distance"][hj][hi]
                pairwise_global["cosine_similarity"][hi][hj]=pairwise_global["cosine_similarity"][hj][hi]
                pairwise_global["jaccard_topk"][hi][hj]=pairwise_global["jaccard_topk"][hj][hi]
            else:
                p = weight_vectors[hi]; q = weight_vectors[hj]
                jsd = js_distance(p,q); cos = cosine_similarity(p,q)
                jac = len(topk_sets[hi]&topk_sets[hj])/ (len(topk_sets[hi]|topk_sets[hj]) or 1)
                pairwise_global["js_distance"][hi][hj]=jsd
                pairwise_global["cosine_similarity"][hi][hj]=cos
                pairwise_global["jaccard_topk"][hi][hj]=jac

    out = {
        "meta":{
            "model": args.model,
            "heads": heads_order,
            "top_k": args.top_k,
            "num_samples": len(samples),
            "source": "generated" if args.generate>0 else "standard_file"
        },
        "per_head": head_stats,
        "pairwise_global": pairwise_global,
        "pairwise_per_sample_avg": per_sample_metrics,
        "vocab_size_used": len(vocab)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("[done] 已写入:", args.output)
    print("[hint] 可重点查看 pairwise.js_distance 与 jaccard_topk ：Name Mover 头应相似度高。")

if __name__ == "__main__":
    main()