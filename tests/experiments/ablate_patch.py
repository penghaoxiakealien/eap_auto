#!/usr/bin/env python
"""
在 ablate（mean-ablation）掉指定头的基础上，
重新测试所有其他头的 path patching 效果（节省显存版 + 逻辑更严谨）

主要改动：
- 使用 mean-ablation（与论文一致），并保留可选的 zero-ablation。
- 所有 forward/run_with_cache 均在 torch.no_grad() 下执行。
- 运行得到的 activation caches 立即 .cpu() 化并释放 GPU 内存。
- 针对 hook 内的张量赋值做了显式的 shape/broadcast 处理与断言。
"""

import argparse
import os
import json
from itertools import product
from functools import partial
from tqdm import tqdm
import torch as t
import gc
from rich import print as rprint
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from plotly_utils import imshow
from ioi_dataset import IOIDataset

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(model_name, device=device)
# 保留你原来的 config 改动
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

# -----------------------------------------------------------------------------
# 工具函数（节省显存 & mean-ablate）
# -----------------------------------------------------------------------------
def cache_to_cpu(cache):
    """把 ActivationCache（或 dict）里的所有 tensor 迁移到 CPU 并 detach。
    返回 dict(name -> cpu_tensor)。
    """
    out = {}
    # ActivationCache 支持 dict-like 的 items()
    for k, v in cache.items():
        out[k] = v.detach().clone().cpu()
    return out

def compute_mean_cache_on_dataset(model, dataset, names_filter):
    """在 new_dataset (pABC) 上计算每个 hook 的 mean activation（沿 batch 维度求平均）。
    返回 dict: name -> mean_tensor (shape: [seq, ...], i.e. batch 维被消掉)（在 CPU 上）。
    """
    model.reset_hooks()
    with t.no_grad():
        _, cache = model.run_with_cache(dataset.toks, names_filter=names_filter, return_type=None)
    mean_cache = {}
    for k, v in cache.items():
        # v shape: [batch, seq, ...], 我们按 batch 求 mean -> [seq, ...]
        mean_cache[k] = v.mean(dim=0).detach().clone().cpu()
    # 释放 GPU 里的 cache 引用
    del cache
    t.cuda.empty_cache()
    gc.collect()
    return mean_cache

def mean_ablate_heads(head_output: t.Tensor, hook: HookPoint, heads_to_ablate, mean_cache):
    """hook：对指定 (layer, head) 用 mean_cache 替换 head 输出（与论文的 mean-ablation 对齐）。
    head_output: (batch, seq, n_heads, head_dim)
    mean_cache[name]: (seq, n_heads, head_dim)  —— 在 CPU 上
    """
    layer = hook.layer()
    name = hook.name
    if name not in mean_cache:
        return head_output  # no-op

    # mean_val: (seq, n_heads, head_dim) on CPU
    mean_val = mean_cache[name]

    # 把 mean_val 移到当前 head_output device（通常是 GPU），但不要在这里一次性把整个 big cache 放到 GPU 上
    # 而是将需要的 slice copy 到 GPU（按 head index）
    # head_output shape: [B, S, H, HD]
    B, S, H, HD = head_output.shape
    # sanity
    assert mean_val.ndim == 3 and mean_val.shape[0] == S and mean_val.shape[1] == H and mean_val.shape[2] == HD, \
        f"mean_val shape mismatch for {name}: got {mean_val.shape}, expected (seq, n_heads, head_dim) {(S,H,HD)}"

    # 只替换那些在头列表里的 head；in-place 改动以节省内存
    for (l, h) in heads_to_ablate:
        if l == layer:
            # move that specific [seq, head_dim] slice to device and broadcast across batch
            slice_cpu = mean_val[:, h, :]  # shape [S, HD] (CPU)
            slice_gpu = slice_cpu.to(head_output.device).unsqueeze(0)  # [1, S, HD]
            # assign across batch dim
            head_output[:, :, h, :] = slice_gpu  # broadcasting (1,S,HD) -> (B,S,HD)
    return head_output

def zero_ablate_heads(head_output: t.Tensor, hook: HookPoint, heads_to_ablate):
    """零 ablation（备选）：把对应 head 的输出置为 0"""
    layer = hook.layer()
    for (l, h) in heads_to_ablate:
        if l == layer:
            head_output[:, :, h, :] = 0.0
    return head_output

def logits_to_ave_logit_diff(logits, ioi_dataset, per_prompt=False):
    """
    计算 IO-logit 和 S-logit 的平均差值
    
    Args:
        logits: 模型输出的logits tensor
        ioi_dataset: IOI数据集实例
        per_prompt: 如果True返回每个prompt的差值，否则返回平均值
    """
    # 获取最后位置的 IO 和 S token 的 logits
    io_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
    ]
    s_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
    ]
    
    # 计算差值
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def compute_ablated_cache_mean(model, dataset, heads_to_ablate, mean_cache, z_name_filter):
    """在 orig_dataset 上运行一次 forward，但对指定 heads 用 mean_cache 替换输出，返回 ablated_cache（已 cpu 化）"""
    model.reset_hooks()
    ablate_hook = partial(mean_ablate_heads, heads_to_ablate=heads_to_ablate, mean_cache=mean_cache)
    model.add_hook(z_name_filter, ablate_hook)
    with t.no_grad():
        _, ablated_cache = model.run_with_cache(dataset.toks, names_filter=z_name_filter, return_type=None)
    # 把 ablated_cache 迁到 CPU（detach）
    ablated_cache_cpu = cache_to_cpu(ablated_cache)
    model.reset_hooks()
    del ablated_cache
    t.cuda.empty_cache()
    gc.collect()
    return ablated_cache_cpu

# -----------------------------------------------------------------------------
# 核心 path-patching (基于 ablated baseline)，但节省显存（cache 都在 CPU）
# -----------------------------------------------------------------------------
def get_path_patch_with_ablated_baseline(
    model: HookedTransformer,
    patching_metric,
    heads_to_ablate: list[tuple[int, int]],
    new_dataset: IOIDataset,
    orig_dataset: IOIDataset,
    use_mean_ablation: bool = True,
):
    """在 ablated baseline（mean-ablate）上测试所有其他头的 path patching 效果。
    返回 results tensor (n_layers, n_heads)（每项为 patching_metric 返回值）。
    """
    model.reset_hooks()
    results = t.full((model.cfg.n_layers, model.cfg.n_heads), float("nan"), device="cpu", dtype=t.float32)

    z_name_filter = lambda name: name.endswith("z")

    rprint("🔍 计算 clean / corrupted caches（仅保留 z 名称）并移动到 CPU ...")
    with t.no_grad():
        _, clean_cache = model.run_with_cache(orig_dataset.toks, names_filter=z_name_filter, return_type=None)
        _, corrupted_cache = model.run_with_cache(new_dataset.toks, names_filter=z_name_filter, return_type=None)
    clean_cache_cpu = cache_to_cpu(clean_cache)
    corrupted_cache_cpu = cache_to_cpu(corrupted_cache)
    del clean_cache, corrupted_cache
    t.cuda.empty_cache()
    gc.collect()

    # 计算 mean_cache（基于 new_dataset / pABC），用于 mean-ablation
    rprint("🔍 计算 mean_cache（在 new_dataset 上按 batch 平均）...")
    mean_cache = compute_mean_cache_on_dataset(model, new_dataset, names_filter=z_name_filter)

    # 计算 ablated_cache：在 orig_dataset 上，应用 mean-ablate（替换指定 heads 输出）
    rprint(f"🔥 对指定 heads 做 mean-ablation（作为 baseline）：{heads_to_ablate} ...")
    ablated_cache_cpu = compute_ablated_cache_mean(model, orig_dataset, heads_to_ablate, mean_cache, z_name_filter)

    rprint("🔍 开始在 ablated baseline 上测试其他头的 path patching 效果...")
    # 遍历 sender heads
    for sender_layer, sender_head in tqdm(list(product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):
        if (sender_layer, sender_head) in heads_to_ablate:
            results[sender_layer, sender_head] = float("nan")
            continue

        # 单次 run 的 hook：首先把所有 head 的输出冻结为 ablated_cache，
        # 然后把 sender head 的输出用 clean_cache 的那一份覆盖（即将 sender 替换为 clean）
        def single_run_hook(orig_head_vector, hook: HookPoint,
                            ablated_cache_local=ablated_cache_cpu,
                            clean_cache_local=clean_cache_cpu,
                            sender=(sender_layer, sender_head),
                            heads_to_ablate_local=heads_to_ablate):
            name = hook.name
            # 基础断言：cache 中要包含这个 hook 的名字
            if name not in ablated_cache_local or name not in clean_cache_local:
                # 如果缺失，原样返回（避免崩掉），但这个一般不该发生
                return orig_head_vector

            # ablated_cache_local[name] shape: [batch, seq, n_heads, head_dim] (在 CPU)
            ac = ablated_cache_local[name].to(orig_head_vector.device)  # move needed slice to GPU
            # 确认 seq/n_heads/head_dim 一致
            assert ac.shape[1:] == orig_head_vector.shape[1:], \
                f"Shape mismatch for {name}: ablated {ac.shape} vs orig {orig_head_vector.shape}"

            # freeze to ablated (in-place assign)
            orig_head_vector[...] = ac

            # 如果这个 hook 对应的 layer 与 sender 相同，并且 sender 不是被永久 ablate 的 head，
            # 将 sender 头替换为 clean cache 中对应的 head（clean_cache_local 包含 batch 维）
            if sender[0] == hook.layer() and sender not in heads_to_ablate_local:
                # clean_cache_local[name] shape: [batch, seq, n_heads, head_dim]
                clean_slice = clean_cache_local[name][:, :, sender[1], :].to(orig_head_vector.device)  # [B, S, HD]
                # assign to orig_head_vector[:, :, sender[1], :]
                orig_head_vector[:, :, sender[1], :] = clean_slice

            return orig_head_vector

        # 注册 hook，运行一次 forward，得到 logits
        model.reset_hooks()
        model.add_hook(z_name_filter, single_run_hook)
        with t.no_grad():
            patched_logits = model(orig_dataset.toks)  # logits 在 GPU，上面 hook 会把激活写回 GPU
        model.reset_hooks()

        # 计算 metric（注意：patching_metric 期望 logits）
        metric_val = float(patching_metric(patched_logits))
        results[sender_layer, sender_head] = metric_val

        # 释放临时对象并回收显存
        del patched_logits
        t.cuda.empty_cache()
        gc.collect()

    return results

# -----------------------------------------------------------------------------
# 主函数（整合以上）
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="在 ablate 指定头（mean-ablation）基础上重新测试 path patching（节省显存版）")
    parser.add_argument("--ablate_heads", nargs='+', default=["9.6", "9.9", "10.0"],
                        help="要 ablate 的头，格式: layer.head")
    parser.add_argument("--output_suffix", default="with_nmh_ablated",
                        help="输出文件名后缀")
    parser.add_argument("--N", type=int, default=100, help="每个 dataset 的样本数（调试时可设小）")
    parser.add_argument("--use_zero_ablation", action="store_true",
                        help="如果指定，则使用零 ablation（与原来脚本一致），否则使用 mean ablation（推荐，与论文一致）")
    args = parser.parse_args()

    # 转换 ablate heads
    heads_to_ablate = [tuple(map(int, h.split("."))) for h in args.ablate_heads]
    rprint(f"🎯 将要 ablate 的头: {heads_to_ablate}")

    # 数据集（注意：你原来是把 N=25 固定在文件头，这里用命令行参数覆盖）
    N_local = args.N
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N_local,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=1,
        device=str(device),
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

    # 基准：clean & corrupted baseline（在 no_grad 下；不用保存 cache）
    model.reset_hooks()
    with t.no_grad():
        ioi_logits_clean, _ = model.run_with_cache(ioi_dataset.toks) 
        abc_logits_corrupted, _ = model.run_with_cache(abc_dataset.toks) 
    clean_logit_diff = logits_to_ave_logit_diff(ioi_logits_clean, ioi_dataset).item()
    corrupted_logit_diff = logits_to_ave_logit_diff(abc_logits_corrupted, ioi_dataset).item()
    rprint(f"📊 原始基准 IOI = {clean_logit_diff:.4f}, ABC = {corrupted_logit_diff:.4f}")

    # 计算 mean_cache（基于 abc_dataset）并用它来生成 ablated baseline logits（或零 ablation）
    z_name_filter = lambda name: name.endswith("z")
    if args.use_zero_ablation:
        rprint("⚠️ 使用 zero ablation（非论文推荐）。若需论文一致行为，请不要传 --use_zero_ablation。")
        # 简单实现：把对应 heads 置 0（原来实现）
        ablate_hook = partial(zero_ablate_heads, heads_to_ablate=heads_to_ablate)
        model.add_hook(z_name_filter, ablate_hook)
        with t.no_grad():
            ablated_logits = model(ioi_dataset.toks)
        model.reset_hooks()
        ablated_logit_diff = logits_to_ave_logit_diff(ablated_logits, ioi_dataset).item()
    else:
        rprint("📌 使用 mean-ablation（与论文一致）。先计算 mean_cache（基于 abc_dataset）...")
        mean_cache = compute_mean_cache_on_dataset(model, abc_dataset, names_filter=z_name_filter)
        # 应用 mean_ablate hook，并运行一次获得 ablated_logits（用于后续 metric 标准化）
        ablate_hook = partial(mean_ablate_heads, heads_to_ablate=heads_to_ablate, mean_cache=mean_cache)
        model.add_hook(z_name_filter, ablate_hook)
        with t.no_grad():
            ablated_logits = model(ioi_dataset.toks)
        model.reset_hooks()
        ablated_logit_diff = logits_to_ave_logit_diff(ablated_logits, ioi_dataset).item()

    rprint(f"📊 Ablated baseline logit diff = {ablated_logit_diff:.4f}")

    # 定义 metric（相对 ablated baseline）
    def ablated_metric(logits):
        patched_diff = logits_to_ave_logit_diff(logits, ioi_dataset) 
        # 分母可能为 0（非常罕见），所以要保护
        denom = (clean_logit_diff - corrupted_logit_diff)
        if abs(denom) < 1e-8:
            return float('nan')
        return (patched_diff - ablated_logit_diff) / denom

    # Path patching（主实验）
    path_patch_results = get_path_patch_with_ablated_baseline(
        model=model,
        patching_metric=ablated_metric,
        heads_to_ablate=heads_to_ablate,
        new_dataset=abc_dataset,
        orig_dataset=ioi_dataset,
        use_mean_ablation=not args.use_zero_ablation,
    )

    # 保存结果（与原脚本一致）
    os.makedirs("results/ioi/path_patching", exist_ok=True)
    output_prefix = f"heads_effect_{args.output_suffix}"

    fig = imshow(
        100 * path_patch_results,
        title=f"Path Patching Results (with {args.ablate_heads} ablated)",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff variation (%)"},
        coloraxis=dict(colorbar_ticksuffix="%"),
        width=700,
        return_fig=True,
    )
    fig.write_image(f"results/ioi/path_patching/{output_prefix}.png")

    # 保存 JSON（把结果转成百分比及 ABLATED 标记）
    result = {}
    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            if (i, j) in heads_to_ablate:
                result[f"{i}.{j}"] = "ABLATED"
            else:
                val = path_patch_results[i, j].item()
                result[f"{i}.{j}"] = round(float(val) * 100, 2)

    with open(f"results/ioi/path_patching/{output_prefix}.json", "w") as f:
        json.dump({
            "ablated_heads": args.ablate_heads,
            "clean_baseline": clean_logit_diff,
            "corrupted_baseline": corrupted_logit_diff,
            "ablated_baseline": ablated_logit_diff,
            "results": result,
        }, f, indent=2)

    rprint(f"\n💾 结果已保存到 results/ioi/path_patching/{output_prefix}.png/.json")

if __name__ == "__main__":
    main()
