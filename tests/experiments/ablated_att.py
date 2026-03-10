#!/usr/bin/env python
"""
single_sentence_attention_analysis.py

单句版本的attention分析：
1. 支持mean ablation或path patching
2. 分析指定head在end位置的attention分布
3. 使用标准IOI数据，逐句处理避免批处理问题
"""

import torch as t
import json
import os
import sys
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Tuple, Any, Optional
import argparse

sys.path.append(os.path.dirname(__file__))
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint

def mean_ablate_heads_single(
    head_output: t.Tensor,  # [1, seq, n_heads, d_head]
    hook: HookPoint,
    heads_to_ablate: List[Tuple[int, int]],
    mean_cache: Dict[str, t.Tensor]
) -> t.Tensor:
    """单句版本的mean ablation"""
    layer = hook.layer()
    hook_name = hook.name
    
    if hook_name not in mean_cache:
        return head_output
    
    # mean_cache[hook_name] shape: [seq, n_heads, d_head]
    mean_val = mean_cache[hook_name].to(head_output.device)
    
    for (l, h) in heads_to_ablate:
        if l == layer:
            # 调整mean_val的序列长度以匹配当前输入
            current_seq_len = head_output.shape[1]
            
            if mean_val.shape[0] >= current_seq_len:
                # 如果mean cache的序列长度足够，直接截取
                mean_slice = mean_val[:current_seq_len, h, :].unsqueeze(0)  # [1, seq, d_head]
            else:
                # 如果不够，用最后一个值填充
                base_slice = mean_val[:, h, :]  # [mean_seq, d_head]
                padding_len = current_seq_len - mean_val.shape[0]
                last_val = base_slice[-1:].repeat(padding_len, 1)  # [padding_len, d_head]
                extended_slice = t.cat([base_slice, last_val], dim=0)  # [current_seq_len, d_head]
                mean_slice = extended_slice.unsqueeze(0)  # [1, seq, d_head]
            
            head_output[:, :, h, :] = mean_slice
    
    return head_output

def patch_heads_single(
    head_output: t.Tensor,
    hook: HookPoint,
    heads_to_patch: List[Tuple[int, int]],
    corrupted_cache: ActivationCache,
    clean_cache: ActivationCache
) -> t.Tensor:
    """单句版本的path patching"""
    layer = hook.layer()
    hook_name = hook.name
    
    # 先用clean填充
    head_output[...] = clean_cache[hook_name][...]
    
    # 然后将指定heads替换为corrupted版本
    for (l, h) in heads_to_patch:
        if l == layer:
            head_output[:, :, h, :] = corrupted_cache[hook_name][:, :, h, :]
    
    return head_output

def compute_mean_cache_from_sentences(
    model: HookedTransformer,
    abc_sentences: List[str],
    heads_to_analyze: List[Tuple[int, int]]
) -> Dict[str, t.Tensor]:
    """从句子列表计算mean cache"""
    
    print("📊 计算mean cache（基于ABC sentences）...")
    
    # 收集所有相关层的激活
    all_activations = {}  # {hook_name: [list of tensors]}
    
    z_name_filter = lambda name: name.endswith("z")
    
    for sentence in tqdm(abc_sentences, desc="处理ABC句子"):
        tokens = model.to_tokens(sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
        
        with t.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=z_name_filter, return_type=None)
        
        for hook_name, activations in cache.items():
            layer = int(hook_name.split('.')[1])
            
            # 检查是否是我们需要的层
            if any(l == layer for l, h in heads_to_analyze):
                if hook_name not in all_activations:
                    all_activations[hook_name] = []
                
                # activations shape: [1, seq, n_heads, d_head]
                all_activations[hook_name].append(activations[0].cpu())  # [seq, n_heads, d_head]
    
    # 计算mean - 需要处理不同序列长度
    mean_cache = {}
    for hook_name, activation_list in all_activations.items():
        # 找到最大序列长度
        max_seq_len = max(act.shape[0] for act in activation_list)
        
        # 填充所有激活到相同长度
        padded_activations = []
        for act in activation_list:
            if act.shape[0] < max_seq_len:
                # 用最后一个值填充
                padding = act[-1:].repeat(max_seq_len - act.shape[0], 1, 1)
                padded_act = t.cat([act, padding], dim=0)
            else:
                padded_act = act
            padded_activations.append(padded_act)
        
        # 计算平均值
        stacked_activations = t.stack(padded_activations, dim=0)  # [n_samples, seq, n_heads, d_head]
        mean_activation = stacked_activations.mean(dim=0)  # [seq, n_heads, d_head]
        mean_cache[hook_name] = mean_activation
    
    print(f"✅ Mean cache计算完成，包含 {len(mean_cache)} 个hook")
    return mean_cache

def analyze_single_sentence_attention(
    model: HookedTransformer,
    sample_data: Dict[str, Any],
    target_head: Tuple[int, int],
    intervention_type: str = "mean_ablation",  # "mean_ablation" or "path_patch"
    heads_to_intervene: List[Tuple[int, int]] = None,
    mean_cache: Optional[Dict[str, t.Tensor]] = None,
    abc_sentence: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    【最终修正版】分析单个句子的attention pattern
    使用数据文件中提供的精确io_pos和s1_pos。
    """
    
    clean_sentence = sample_data["clean"]["sentence"]
    io_token_str_from_data = sample_data["clean"]["io_token"]
    s_token_str_from_data = sample_data["clean"]["s_token"]
    end_pos = sample_data["positions"]["end"]
    
    # --- 核心修正：直接使用数据中提供的精确位置 ---
    io_pos = sample_data["positions"]["io"]
    # --- 最终修正：使用 's1' 而不是 's' ---
    s_pos = sample_data["positions"]["s1"]
    
    if verbose:
        print(f"分析样本 {sample_data['sample_id']}: {clean_sentence}")
        print(f"目标head: {target_head[0]}.{target_head[1]}, 干预方式: {intervention_type}")
        print(f"使用精确位置: IO at pos {io_pos}, S at pos {s_pos}")

    # 1. Tokenize
    clean_tokens = model.to_tokens(clean_sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
    actual_end_pos = min(end_pos, clean_tokens.shape[1] - 1)
    seq_len = clean_tokens.shape[1]
    
    # 获取token strings
    token_strs = [model.tokenizer.decode([int(tok)]) for tok in clean_tokens[0]]
    
    # 2. 设置intervention hooks
    model.reset_hooks()
    
    if intervention_type == "mean_ablation" and heads_to_intervene and mean_cache:
        z_name_filter = lambda name: name.endswith("z")
        ablation_hook = partial(
            mean_ablate_heads_single,
            heads_to_ablate=heads_to_intervene,
            mean_cache=mean_cache
        )
        model.add_hook(z_name_filter, ablation_hook)
        
    elif intervention_type == "path_patch" and heads_to_intervene and abc_sentence:
        abc_tokens = model.to_tokens(abc_sentence, prepend_bos=False)[0].unsqueeze(0).to("cuda")
        
        max_len = max(clean_tokens.shape[1], abc_tokens.shape[1])
        if clean_tokens.shape[1] < max_len:
            pad_len = max_len - clean_tokens.shape[1]
            clean_tokens = t.cat([clean_tokens, clean_tokens[:, -1:].repeat(1, pad_len)], dim=1)
        if abc_tokens.shape[1] < max_len:
            pad_len = max_len - abc_tokens.shape[1]
            abc_tokens = t.cat([abc_tokens, abc_tokens[:, -1:].repeat(1, pad_len)], dim=1)
        
        z_name_filter = lambda name: name.endswith("z")
        with t.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=z_name_filter, return_type=None)
            _, corrupted_cache = model.run_with_cache(abc_tokens, names_filter=z_name_filter, return_type=None)
        
        patch_hook = partial(
            patch_heads_single,
            heads_to_patch=heads_to_intervene,
            corrupted_cache=corrupted_cache,
            clean_cache=clean_cache
        )
        model.add_hook(z_name_filter, patch_hook)
        
        seq_len = clean_tokens.shape[1]
        token_strs = [model.tokenizer.decode([int(tok)]) for tok in clean_tokens[0]]
        actual_end_pos = min(end_pos, seq_len - 1)
    
    # 3. 获取attention pattern
    target_layer, target_head_idx = target_head
    attn_hook_name = f"blocks.{target_layer}.attn.hook_pattern"
    attn_filter = lambda name: name == attn_hook_name
    
    with t.no_grad():
        _, cache = model.run_with_cache(
            clean_tokens,
            names_filter=attn_filter,
            return_type=None
        )
    
    model.reset_hooks()
    
    # 4. 提取attention分布
    attention_patterns = cache[attn_hook_name][0, target_head_idx, actual_end_pos, :]
    attention_scores = attention_patterns.cpu().tolist()
    
    # 5. 构建token-level分析
    token_analysis = []
    for pos in range(seq_len):
        token_analysis.append({
            "position": pos,
            "token_id": int(clean_tokens[0, pos]),
            "token_text": token_strs[pos],
            "attention_score": attention_scores[pos],
            "is_end_position": pos == actual_end_pos
        })
    
    # 6. 找出attention最高的位置
    sorted_positions = sorted(range(seq_len), key=lambda x: attention_scores[x], reverse=True)
    top_attended = [
        {
            "position": pos,
            "token_text": token_strs[pos],
            "attention_score": attention_scores[pos]
        }
        for pos in sorted_positions[:5]
    ]
    
    # 7. --- 核心修正：直接使用精确位置获取特殊token的attention ---
    io_attention = None
    s_attention = None

    if io_pos < seq_len:
        io_attention = {
            "position": io_pos,
            "token_text": token_strs[io_pos],
            "attention_score": attention_scores[io_pos]
        }

    if s_pos < seq_len:
        s_attention = {
            "position": s_pos,
            "token_text": token_strs[s_pos],
            "attention_score": attention_scores[s_pos]
        }
    
    return {
        "sample_id": sample_data["sample_id"],
        "sentence_text": clean_sentence,
        "template_type": sample_data["template_type"],
        "target_head": f"{target_head[0]}.{target_head[1]}",
        "intervention_type": intervention_type,
        "heads_intervened": [f"{l}.{h}" for l, h in (heads_to_intervene or [])],
        
        "attention_analysis": {
            "end_position": actual_end_pos,
            "sequence_length": seq_len,
            "all_tokens": token_analysis,
            "top_5_attended": top_attended,
            "io_attention": io_attention,
            "s_attention": s_attention,
            "io_vs_s_ratio": (io_attention["attention_score"] / s_attention["attention_score"]) if (io_attention and s_attention and s_attention["attention_score"] > 1e-6) else None
        },
        
        "context_info": {
            "io_token": io_token_str_from_data,
            "s_token": s_token_str_from_data,
            "original_end_pos": end_pos,
            "io_pos": io_pos,
            "s_pos": s_pos
        }
    }

def main():
    parser = argparse.ArgumentParser(description="单句attention分析")
    parser.add_argument("--input_file", type=str, required=True, help="标准IOI数据文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--target_head", type=str, required=True, help="要分析的head (格式: layer.head)")
    parser.add_argument("--intervention_type", type=str, choices=["none", "mean_ablation", "path_patch"], default="none")
    parser.add_argument("--heads_to_intervene", nargs='+', help="要干预的heads (格式: layer.head)")
    parser.add_argument("--max_samples", type=int, default=100, help="处理的最大样本数")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 解析target head
    target_layer, target_head_idx = map(int, args.target_head.split('.'))
    target_head = (target_layer, target_head_idx)
    
    # 解析要干预的heads
    heads_to_intervene = []
    if args.heads_to_intervene:
        for head_str in args.heads_to_intervene:
            layer, head = map(int, head_str.split('.'))
            heads_to_intervene.append((layer, head))
    
    print(f"🧠 单句attention分析")
    print(f"  目标head: {args.target_head}")
    print(f"  干预方式: {args.intervention_type}")
    if heads_to_intervene:
        print(f"  干预heads: {[f'{l}.{h}' for l, h in heads_to_intervene]}")
    
    # 1. 加载模型
    print("🔥 加载GPT-2模型...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    
    # 2. 加载数据
    print(f"📊 加载标准IOI数据: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        ioi_data = json.load(f)
    
    samples = ioi_data["samples"][:args.max_samples]
    print(f"将处理 {len(samples)} 个样本")
    
    # 3. 预计算mean cache（如果需要）
    mean_cache = None
    if args.intervention_type == "mean_ablation":
        print("📊 预计算mean cache...")
        abc_sentences = [s["abc_corrupted"]["sentence"] for s in samples]
        mean_cache = compute_mean_cache_from_sentences(model, abc_sentences, heads_to_intervene + [target_head])
    
    # 4. 逐句分析
    print("🧠 开始逐句attention分析...")
    all_results = []
    
    for sample in tqdm(samples, desc="分析attention"):
        try:
            abc_sentence = sample["abc_corrupted"]["sentence"] if args.intervention_type == "path_patch" else None
            
            result = analyze_single_sentence_attention(
                model=model,
                sample_data=sample,
                target_head=target_head,
                intervention_type=args.intervention_type,
                heads_to_intervene=heads_to_intervene if heads_to_intervene else None,
                mean_cache=mean_cache,
                abc_sentence=abc_sentence,
                verbose=args.verbose
            )
            
            all_results.append(result)
            
        except Exception as e:
            print(f"处理样本 {sample['sample_id']} 时出错: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    print(f"✅ 成功处理了 {len(all_results)}/{len(samples)} 个样本")
    
    # 5. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建输出文件名
    intervention_str = f"_{args.intervention_type}" if args.intervention_type != "none" else ""
    heads_str = "_".join([f"{l}_{h}" for l, h in heads_to_intervene]) if heads_to_intervene else ""
    if heads_str:
        heads_str = f"_intervene_{heads_str}"
    
    output_file = os.path.join(args.output_dir, f"attention_analysis_head_{target_layer}_{target_head_idx}{intervention_str}{heads_str}.json")
    
    output_data = {
        "experiment_info": {
            "target_head": f"{target_layer}.{target_head_idx}",
            "intervention_type": args.intervention_type,
            "heads_intervened": [f"{l}.{h}" for l, h in heads_to_intervene] if heads_to_intervene else [],
            "total_samples": len(all_results),
            "model_name": "gpt2-small",
            "method": "single_sentence_attention_analysis"
        },
        "results": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存到: {output_file}")
    
    # 6. 显示统计信息
    if all_results:
        print("\n📈 统计信息:")
        
        # IO vs S attention统计
        io_vs_s_ratios = [r['attention_analysis']['io_vs_s_ratio'] for r in all_results if r['attention_analysis']['io_vs_s_ratio'] is not None]
        if io_vs_s_ratios:
            print(f"IO vs S attention ratio统计:")
            print(f"  平均比例: {sum(io_vs_s_ratios)/len(io_vs_s_ratios):.4f}")
            print(f"  最大比例: {max(io_vs_s_ratios):.4f}")
            print(f"  最小比例: {min(io_vs_s_ratios):.4f}")
            print(f"  IO>S的样本: {sum(1 for r in io_vs_s_ratios if r > 1)}/{len(io_vs_s_ratios)}")
        
        # 显示示例
        print("\n📝 示例结果:")
        for i, result in enumerate(all_results[:2]):
            att_info = result['attention_analysis']
            print(f"\n示例 {i+1}:")
            print(f"  句子: {result['sentence_text']}")
            print(f"  End位置关注的前3个token:")
            for j, item in enumerate(att_info['top_5_attended'][:3]):
                print(f"    {j+1}. 位置{item['position']}: '{item['token_text']}' ({item['attention_score']:.4f})")
            if att_info['io_attention'] and att_info['s_attention']:
                print(f"  IO attention: {att_info['io_attention']['attention_score']:.4f}")
                print(f"  S attention: {att_info['s_attention']['attention_score']:.4f}")
                print(f"  IO/S比例: {att_info['io_vs_s_ratio']:.4f}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()