import torch as t
import json
import os
import sys
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Tuple, Any
import argparse

# 确保可以导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from ioi_dataset import IOIDataset

LOCAL_MODEL_DIR = "/data31/private/wangziran/eap-ig/gpt2"


def load_model(device: str = "cuda"):
    """优先以官方名称加载本地缓存的gpt2模型，若目录不存在则回退到gpt2-small。"""
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"🔥 正在从本地缓存加载模型: {LOCAL_MODEL_DIR}")
        return HookedTransformer.from_pretrained(
            "gpt2",
            device=device,
            cache_dir=LOCAL_MODEL_DIR
        )
    print("⚠️ 未找到本地模型目录，回退到默认的 gpt2-small。")
    return HookedTransformer.from_pretrained("gpt2-small", device=device)

def patch_or_freeze_head_vectors(
    orig_head_vector,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
):
    """
    Path patching的核心函数：冻结所有头到orig_cache的值，除了head_to_patch，它从new_cache中获取。
    """
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def analyze_single_sentence_effects(
    model: HookedTransformer,
    sample_data: Dict[str, Any],
    head_to_patch: Tuple[int, int],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    对单个样本进行简化的patching前后logit对比分析。
    """
    clean_sentence = sample_data["clean"]["sentence"]
    abc_sentence = sample_data["abc_corrupted"]["sentence"]
    
    io_token_id = sample_data["target_tokens"]["io_token_id"]
    s_token_id = sample_data["target_tokens"]["s_token_id"]
    end_pos = sample_data["positions"]["end"]
    
    if verbose:
        print(f"\n--- 分析样本 {sample_data['sample_id']}: {clean_sentence} ---")
        print(f"Patching head: {head_to_patch[0]}.{head_to_patch[1]}")

    # 1. Tokenize (单句处理)
    clean_tokens = model.to_tokens(clean_sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
    abc_tokens = model.to_tokens(abc_sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
    
    # 确保长度一致
    max_len = max(clean_tokens.shape[1], abc_tokens.shape[1])
    if clean_tokens.shape[1] < max_len:
        clean_tokens = t.cat([clean_tokens, clean_tokens[:, -1:].repeat(1, max_len - clean_tokens.shape[1])], dim=1)
    if abc_tokens.shape[1] < max_len:
        abc_tokens = t.cat([abc_tokens, abc_tokens[:, -1:].repeat(1, max_len - abc_tokens.shape[1])], dim=1)
    
    actual_end_pos = min(end_pos, clean_tokens.shape[1] - 1)
    
    # 2. Clean Run: 获取原始的logits
    model.reset_hooks()
    z_name_filter = lambda name: name.endswith("z")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=z_name_filter)
    clean_end_logits = clean_logits[0, actual_end_pos]

    # 3. Corrupted Run: 获取用于patching的激活
    model.reset_hooks()
    _, abc_cache = model.run_with_cache(abc_tokens, names_filter=z_name_filter)

    # 4. Patched Run: 执行path patching
    hook_fn = partial(
        patch_or_freeze_head_vectors,
        new_cache=abc_cache,
        orig_cache=clean_cache,
        head_to_patch=head_to_patch,
    )
    model.add_hook(z_name_filter, hook_fn)
    
    # 只需要logits，不再需要cache
    patched_logits = model(clean_tokens)
    patched_end_logits = patched_logits[0, actual_end_pos]
    model.reset_hooks()

    # 5. 分析Logit变化 (***修改点***)
    # 只关注句子中出现的token
    sentence_tokens_ids = t.unique(clean_tokens[0]).cpu()
    
    sentence_token_logits = []
    for token_id in sentence_tokens_ids:
        sentence_token_logits.append({
            "token": model.to_string(token_id),
            "clean_logit": clean_end_logits[token_id].item(),
            "patched_logit": patched_end_logits[token_id].item(),
            "change": (patched_end_logits[token_id] - clean_end_logits[token_id]).item()
        })
    
    # 按logit变化量排序，方便查看
    sentence_token_logits.sort(key=lambda x: x["change"], reverse=True)

    logit_analysis = {
        "sentence_token_logits": sentence_token_logits,
    }

    # 6. 计算IOI Metric
    clean_logit_diff_val = (clean_end_logits[io_token_id] - clean_end_logits[s_token_id]).item()
    patched_logit_diff_val = (patched_end_logits[io_token_id] - patched_end_logits[s_token_id]).item()
    
    abc_logits = model(abc_tokens)
    abc_end_logits = abc_logits[0, actual_end_pos]
    abc_logit_diff_val = (abc_end_logits[io_token_id] - abc_end_logits[s_token_id]).item()
    
    if abs(clean_logit_diff_val - abc_logit_diff_val) < 1e-8:
        ioi_metric = 0.0
    else:
        ioi_metric = (patched_logit_diff_val - abc_logit_diff_val) / (clean_logit_diff_val - abc_logit_diff_val)

    delta_logit_diff_val = patched_logit_diff_val - clean_logit_diff_val
    direction = "increase" if delta_logit_diff_val > 0 else "decrease"
    return {
        "sample_id": sample_data["sample_id"],
        "sentence_text": clean_sentence,
        "tokens": model.to_str_tokens(clean_tokens[0]),
        "head_patched": f"{head_to_patch[0]}.{head_to_patch[1]}",
        "ioi_metric": ioi_metric,
        "clean_logit_diff": clean_logit_diff_val,
        "patched_logit_diff": patched_logit_diff_val,
        "abc_logit_diff": abc_logit_diff_val,
        "delta_logit_diff": delta_logit_diff_val,
        "direction": direction,
        "logit_analysis_at_end_pos": logit_analysis,
    }

def main():
    parser = argparse.ArgumentParser(description="分析单个头patching后的logit变化 (V3-Final)")
    parser.add_argument("--input_file", type=str, default="/data31/private/wangziran/eap_auto/results/ioi/path_patching/standard_ioi_data.json", help="标准IOI数据文件")
    parser.add_argument("--output_dir", type=str, default="/data31/private/wangziran/eap_auto/results/ioi/path_patching/NMH", help="输出目录")
    parser.add_argument("--head_to_patch", type=str, default="9.6", help="要patch的头 (格式: layer.head)")
    parser.add_argument("--max_samples", type=int, default=0, help="处理的最大样本数；<=0 表示使用全部样本")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    layer, head = map(int, args.head_to_patch.split('.'))
    head_to_patch = (layer, head)
    
    print(f"--- 开始分析, Patching Head: {args.head_to_patch} (V3: 仅Logit分析) ---")
    
    print("🔥 加载模型...")
    model = load_model()
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    
    print(f"📊 加载标准IOI数据: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        ioi_data = json.load(f)
    
    all_samples = ioi_data["samples"]
    if args.max_samples and args.max_samples > 0:
        samples = all_samples[:args.max_samples]
    else:
        samples = all_samples
    print(f"将处理 {len(samples)} 个样本")
    
    all_results = []
    for sample in tqdm(samples, desc="分析句子"):
        try:
            result = analyze_single_sentence_effects(
                model=model,
                sample_data=sample,
                head_to_patch=head_to_patch,
                verbose=args.verbose
            )
            all_results.append(result)
        except Exception as e:
            print(f"处理样本 {sample['sample_id']} 时出错: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
            
    output_filename = f"final_logit_effects_head_{layer}_{head}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, output_filename)
    
    output_data = {
        "experiment_info": {
            "head_patched": f"{layer}.{head}",
            "total_samples": len(all_results),
            "model_name": "gpt2-small",
            "analysis_type": "focused_logit_values_before_and_after_patch"
        },
        "results": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 分析完成，最终简化结果已保存到: {output_file}")

    if all_results:
        print("\n--- 样本0结果预览 (最终简化版) ---")
        sample_result = all_results[0]
        print(f"句子: {sample_result['sentence_text']}")
        print(f"IOI Metric: {sample_result['ioi_metric']:.4f}")
        
        print("\n句子内Token的Logit对比 (按变化量排序):")
        print(f"{'Token':<15} | {'Clean Logit':<15} | {'Patched Logit':<15} | {'Change':<15}")
        print("-" * 65)
        for item in sample_result['logit_analysis_at_end_pos']['sentence_token_logits']:
            print(f"{item['token']:<15} | {item['clean_logit']:<15.4f} | {item['patched_logit']:<15.4f} | {item['change']:<15.4f}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
