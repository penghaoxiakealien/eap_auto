import os, json, argparse
from typing import Dict, List, Tuple
import torch as t
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from ioi_dataset import IOIDataset

# ------------------ 解析输入 heads ------------------
def parse_heads(head_specs: List[str]) -> List[Tuple[int,int]]:
    out=[]
    for hs in head_specs:
        try:
            L,H = hs.split(".")
            out.append((int(L), int(H)))
        except:
            raise ValueError(f"非法 head 规格: {hs} (应为形如 9.6)")
    return out

def extract_layer(hook_name: str) -> int | None:
    """
    从 TransformerLens hook 名称中提取层号。
    例: blocks.9.attn.hook_z -> 9
    """
    for part in hook_name.replace("/", ".").split("."):
        if part.isdigit():
            return int(part)
    return None

def show_sample_indices(dataset: IOIDataset, model: HookedTransformer, k=3):
    print("\n[调试] 采样检查 token 对齐：")
    for i in range(min(k, dataset.toks.size(0))):
        end_pos = dataset.word_idx["end"][i].item()
        toks = dataset.toks[i]
        def safe_decode(pos):
            if 0 <= pos < toks.size(0):
                return model.to_string(int(toks[pos]))
            return "<OOB>"
        print(f"[sample {i}] end={end_pos}  tok[end]={safe_decode(end_pos)}  tok[end-1]={safe_decode(end_pos-1)}  tok[end+1]={safe_decode(end_pos+1)}  IO={dataset.ioi_prompts[i]['IO']}  S={dataset.ioi_prompts[i]['S']}")
    print("[调试] 检查完毕。\n")
    
def select_positions(word_idx_end: t.Tensor, mode: str, seq_len: int) -> t.Tensor:
    if mode == "end":
        return word_idx_end
    if mode == "end_minus1":
        return t.clamp(word_idx_end - 1, min=0)
    if mode == "end_plus1":
        return t.clamp(word_idx_end + 1, max=seq_len - 1)
    raise ValueError(f"未知 position_index_mode: {mode}")

def compute_logit_diff_at_positions(
    logits: t.Tensor,
    dataset: IOIDataset,
    position_index_mode: str = "end",
) -> t.Tensor:
    """
    根据 position_index_mode 取位置：
      end: 原 word_idx['end']
      end_minus1: end-1
      end_plus1: end+1
    注意 GPT2 logits[i,pos] 预测的是 input[pos] 这个 token，若想预测下一 token 需 pos-1。
    """
    device = logits.device
    N = logits.size(0)
    # to tensor
    if not isinstance(dataset.io_tokenIDs, t.Tensor):
        io_ids = t.tensor(dataset.io_tokenIDs, device=device, dtype=t.long)
        s_ids  = t.tensor(dataset.s_tokenIDs, device=device, dtype=t.long)
    else:
        io_ids = dataset.io_tokenIDs.to(device)
        s_ids  = dataset.s_tokenIDs.to(device)

    base_pos = dataset.word_idx["end"].to(device)
    pos = select_positions(base_pos, position_index_mode, logits.size(1))
    idx = t.arange(N, device=device)
    io_logits = logits[idx, pos, io_ids]
    s_logits  = logits[idx, pos, s_ids]
    return io_logits - s_logits

# ------------------ 构造数据集 ------------------
def build_datasets(model, N:int, seed:int, prompt_type:str="mixed"):
    ioi = IOIDataset(
        prompt_type=prompt_type,
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=str(model.cfg.device),
    )
    abc = ioi.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    return ioi, abc

# ------------------ 计算 logit diff ------------------
def compute_logit_diff(logits: t.Tensor, dataset: IOIDataset) -> t.Tensor:
    """
    logits: [N, seq, vocab]
    返回每个样本 IO - S 的 logit 差 [N]
    """
    N = logits.size(0)
    device = logits.device

    # dataset.io_tokenIDs / s_tokenIDs 可能是 list，转成 tensor
    if not isinstance(dataset.io_tokenIDs, t.Tensor):
        io_ids = t.tensor(dataset.io_tokenIDs, device=device, dtype=t.long)
        s_ids  = t.tensor(dataset.s_tokenIDs, device=device, dtype=t.long)
    else:
        io_ids = dataset.io_tokenIDs.to(device)
        s_ids  = dataset.s_tokenIDs.to(device)

    end_pos = dataset.word_idx["end"].to(device)  # [N]
    idx = t.arange(N, device=device)
    io_logits = logits[idx, end_pos, io_ids]
    s_logits  = logits[idx, end_pos, s_ids]
    return io_logits - s_logits

# ------------------ 计算均值向量（一次性缓存） ------------------
def compute_mean_vectors(
    model: HookedTransformer,
    dataset: IOIDataset,
    heads: List[Tuple[int,int]],
    position_only: bool
) -> Dict[str, Dict[int, t.Tensor]]:
    print(f"计算均值: 来源={'abc' if dataset.has_been_flipped else 'ioi'}, position_only={position_only}")
    layer_set = sorted(set(L for L,_ in heads))
    head_map = {L: sorted([h for (L2,h) in heads if L2==L]) for L in layer_set}
    hook_names = [utils.get_act_name("z", L) for L in layer_set]

    def names_filter(name: str):
        return name in hook_names

    with t.no_grad():
        _, cache = model.run_with_cache(dataset.toks, names_filter=names_filter, return_type=None)

    end_pos = dataset.word_idx["end"].to(model.cfg.device)
    N = dataset.toks.size(0)
    means: Dict[str, Dict[int, t.Tensor]] = {}

    for hk in hook_names:
        act = cache[hk]   # [N, seq, n_heads, d_head]
        layer = extract_layer(hk)
        if layer is None or layer not in head_map:
            continue
        means[hk] = {}
        for h in head_map[layer]:
            head_act = act[:, :, h, :]                  # [N, seq, d_head]
            if position_only:
                vecs = head_act[t.arange(N, device=act.device), end_pos, :]   # [N, d_head]
            else:
                vecs = head_act.mean(dim=1)                                 # [N, d_head]
            mean_vec = vecs.mean(dim=0).detach().to(t.float32)
            means[hk][h] = mean_vec
    print("均值向量数量:", sum(len(v) for v in means.values()))
    return means

# ------------------ Hook 工厂 ------------------
def make_ablation_hook(
    heads: List[Tuple[int,int]],
    mean_cache: Dict[str, Dict[int, t.Tensor]],
    ablation_mode: str,
    end_pos: t.Tensor
):
    heads_by_layer = {}
    for L,h in heads:
        heads_by_layer.setdefault(L, []).append(h)

    def hook_fn(head_out: t.Tensor, hook: HookPoint):
        layer = extract_layer(hook.name)
        if layer is None or layer not in heads_by_layer:
            return head_out
        for h in heads_by_layer[layer]:
            if ablation_mode == "zero":
                if h < head_out.size(2):
                    head_out[:, :, h, :] = 0.
                continue
            if hook.name not in mean_cache or h not in mean_cache[hook.name]:
                continue
            mean_vec = mean_cache[hook.name][h].to(head_out.device).to(head_out.dtype)
            if ablation_mode == "position_only":
                idx = t.arange(head_out.size(0), device=head_out.device)
                head_out[idx, end_pos, h, :] = mean_vec
            elif ablation_mode == "sequence_mean":
                head_out[:, :, h, :] = mean_vec
        return head_out
    return hook_fn
# ------------------ 汇总指标 ------------------
def summarize(clean: t.Tensor, ablated: t.Tensor) -> Dict:
    drop = clean - ablated
    mean_clean = clean.mean().item()
    mean_abl   = ablated.mean().item()
    mean_drop  = drop.mean().item()
    rel = (mean_clean - mean_abl)/abs(mean_clean)*100 if abs(mean_clean)>1e-8 else 0.0
    improved = int((drop<0).sum().item())
    degraded = int((drop>0).sum().item())
    return {
        "n_samples": clean.numel(),
        "clean_mean_logit_diff": mean_clean,
        "ablated_mean_logit_diff": mean_abl,
        "mean_performance_drop": mean_drop,
        "overall_relative_drop_percent": rel,
        "accuracy_before": float((clean>0).float().mean().item()),
        "accuracy_after": float((ablated>0).float().mean().item()),
        "accuracy_delta": float((ablated>0).float().mean().item() - (clean>0).float().mean().item()),
        "count_improved": improved,
        "count_degraded": degraded,
        "count_unchanged": int(clean.numel() - improved - degraded),
        "std_clean": float(clean.std().item()),
        "std_ablated": float(ablated.std().item()),
        "std_drop": float(drop.std().item()),
    }

# ------------------ 主流程 ------------------
def main():
    parser = argparse.ArgumentParser(description="一次性全量 mean / position_only / zero ablation 实验 (无分批)")
    parser.add_argument("--heads", nargs="+", required=True, help="例如: 9.6 9.9 10.0")
    parser.add_argument("--N", type=int, default=400, help="生成 IOI & ABC 各 N 条 (IOI 用于评估, ABC/IOI 之一用于均值)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt_type", type=str, default="mixed")
    parser.add_argument("--mean_source", type=str, default="abc", choices=["abc","ioi"],
                        help="计算均值的数据集 (abc=腐蚀, ioi=干净). zero 模式忽略")
    parser.add_argument("--ablation_mode", type=str, default="sequence_mean",
                        choices=["sequence_mean","position_only","zero"],
                        help="sequence_mean: 全序列替换; position_only: 仅 end 位置; zero: 置零")
    parser.add_argument("--output", type=str, default="results/ioi/mean_ablation_all_in_one.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--position_index_mode", type=str, default="end",
                        choices=["end","end_minus1","end_plus1"],
                        help="logit diff 计算位置模式（用于对齐调试）")
    parser.add_argument("--show_indices", action="store_true",
                        help="打印前若干样本 end/相邻 token 以人工确认位置")
    parser.add_argument("--scan_single_heads", action="store_true",
                        help="逐头单独 zero ablation 探测符号（忽略 mean_source）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    heads = parse_heads(args.heads)
    print("加载模型...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True

    print("生成数据集...")
    ioi_ds, abc_ds = build_datasets(model, args.N, args.seed, args.prompt_type)
    source_ds = abc_ds if args.mean_source=="abc" else ioi_ds

    if args.show_indices:
        show_sample_indices(ioi_ds, model, k=5)
    # 计算 clean 基线
    with t.no_grad():
        clean_logits = model(ioi_ds.toks)
    clean_diff = compute_logit_diff_at_positions(clean_logits, ioi_ds, args.position_index_mode)

    if args.scan_single_heads:
        print("[扫描] 逐头 zero ablation：")
        scan_results = {}
        for (L,H) in heads:
            hook_fn = make_ablation_hook(
                heads=[(L,H)],
                mean_cache={},
                ablation_mode="zero",
                end_pos=ioi_ds.word_idx["end"].to(model.cfg.device),
            )
            z_filter = lambda name: name.endswith("hook_z") or name.endswith("attn.hook_z") or name.endswith(".hook_z")
            model.add_hook(z_filter, hook_fn)
            with t.no_grad():
                abl_logits = model(ioi_ds.toks)
            model.reset_hooks()
            abl_diff = compute_logit_diff_at_positions(abl_logits, ioi_ds, args.position_index_mode)
            scan_results[f"{L}.{H}"] = float((clean_diff - abl_diff).mean().item())
        print("[扫描结果] (clean - ablated) <0 表示 ablation 提升：")
        for k,v in scan_results.items():
            print(f"  {k}: {v:.4f}")
        # 仅扫描就退出
        return
    
    # 均值缓存
    if args.ablation_mode == "zero":
        mean_cache = {}
        position_only_flag = False
    else:
        position_only_flag = (args.ablation_mode == "position_only")
        mean_cache = compute_mean_vectors(
            model=model,
            dataset=source_ds,
            heads=heads,
            position_only=position_only_flag
        )

    # Ablation
    end_pos = ioi_ds.word_idx["end"].to(model.cfg.device)
    hook_fn = make_ablation_hook(
        heads=heads,
        mean_cache=mean_cache,
        ablation_mode=args.ablation_mode,
        end_pos=end_pos
    )
    z_filter = lambda name: name.endswith("hook_z") or name.endswith("attn.hook_z") or name.endswith(".hook_z")
    model.add_hook(z_filter, hook_fn)
    with t.no_grad():
        ablated_logits = model(ioi_ds.toks)
    model.reset_hooks()

    ablated_diff = compute_logit_diff_at_positions(ablated_logits, ioi_ds, args.position_index_mode)

    # 汇总
    summary = summarize(clean_diff, ablated_diff)
    print("\n=== 汇总 ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    # 保存
    out = {
        "config": {
            "heads": [f"{L}.{H}" for L,H in heads],
            "N": args.N,
            "seed": args.seed,
            "prompt_type": args.prompt_type,
            "mean_source": args.mean_source,
            "ablation_mode": args.ablation_mode,
            "mean_from_dataset": "abc" if source_ds.has_been_flipped else "ioi",
        },
        "summary": summary,
        "per_sample": {
            "clean_logit_diff": clean_diff.tolist(),
            "ablated_logit_diff": ablated_diff.tolist(),
            "performance_drop": (clean_diff - ablated_diff).tolist()
        }
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {args.output}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    main()