#!/usr/bin/env python
"""
generate_standard_ioi_data.py

使用标准IOI dataset生成方法，创建句子集和对应信息，然后保存
"""

import torch as t
import json
import os
import sys
from typing import Dict, List, Any
import argparse

sys.path.append(os.path.dirname(__file__))
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset

def generate_and_save_ioi_data(
    model: HookedTransformer,
    N: int = 100,
    output_file: str = "standard_ioi_data.json",
    seed: int = 1
):
    """
    使用标准IOI方法生成数据集并保存完整信息
    """
    print(f"🎯 生成标准IOI数据集，样本数: {N}")
    
    # 1. 生成标准IOI数据集
    ioi_dataset = IOIDataset(
        prompt_type="mixed",  # 混合ABBA和BABA模板
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device="cuda"
    )
    
    # 2. 生成ABC corrupted数据集（标准方法）
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    
    # 3. 生成其他类型的corrupted数据集
    swapped_dataset = ioi_dataset.gen_swapped_s_io_prompts() if hasattr(ioi_dataset, 'gen_swapped_s_io_prompts') else None
    
    # 4. 收集完整信息
    data_collection = []
    
    for i in range(N):
        # 基本信息
        clean_prompt = ioi_dataset.ioi_prompts[i]
        abc_prompt = abc_dataset.ioi_prompts[i]
        
        # Token信息
        clean_tokens = ioi_dataset.toks[i].cpu().tolist()
        abc_tokens = abc_dataset.toks[i].cpu().tolist()
        
        # Token文本
        clean_token_strs = [model.tokenizer.decode([tok]) for tok in clean_tokens if tok != model.tokenizer.pad_token_id]
        abc_token_strs = [model.tokenizer.decode([tok]) for tok in abc_tokens if tok != model.tokenizer.pad_token_id]
        
        # 位置信息
        end_pos = int(ioi_dataset.word_idx["end"][i])
        io_pos = int(ioi_dataset.word_idx["IO"][i])
        s1_pos = int(ioi_dataset.word_idx["S1"][i])
        s2_pos = int(ioi_dataset.word_idx["S2"][i])
        
        sample_data = {
            "sample_id": i,
            "template_type": ioi_dataset.templates_by_prompt[i],  # "ABBA" 或 "BABA"
            
            # Clean版本信息
            "clean": {
                "sentence": clean_prompt["text"],
                "io_token": clean_prompt["IO"],
                "s_token": clean_prompt["S"],
                "tokens": clean_token_strs,
                "token_ids": clean_tokens[:len(clean_token_strs)],  # 去掉padding
            },
            
            # ABC corrupted版本信息
            "abc_corrupted": {
                "sentence": abc_prompt["text"],
                "io_token": abc_prompt["IO"],
                "s_token": abc_prompt["S"],
                "tokens": abc_token_strs,
                "token_ids": abc_tokens[:len(abc_token_strs)],
            },
            
            # 位置信息
            "positions": {
                "end": end_pos,
                "io": io_pos,
                "s1": s1_pos,
                "s2": s2_pos,
                "sequence_length": len(clean_token_strs)
            },
            
            # Token ID信息（用于logit分析）
            "target_tokens": {
                "io_token_id": int(ioi_dataset.io_tokenIDs[i]),
                "s_token_id": int(ioi_dataset.s_tokenIDs[i])
            }
        }
        
        # 如果有swapped数据集，也加入
        if swapped_dataset is not None:
            swapped_prompt = swapped_dataset.ioi_prompts[i]
            swapped_tokens = swapped_dataset.toks[i].cpu().tolist()
            swapped_token_strs = [model.tokenizer.decode([tok]) for tok in swapped_tokens if tok != model.tokenizer.pad_token_id]
            
            sample_data["swapped"] = {
                "sentence": swapped_prompt["text"],
                "io_token": swapped_prompt["IO"],
                "s_token": swapped_prompt["S"],
                "tokens": swapped_token_strs,
                "token_ids": swapped_tokens[:len(swapped_token_strs)]
            }
        
        data_collection.append(sample_data)
    
    # 5. 保存完整数据
    output_data = {
        "dataset_info": {
            "total_samples": N,
            "seed": seed,
            "prompt_type": "mixed",
            "model_name": "gpt2-small",
            "generation_method": "standard_ioi_dataset"
        },
        "samples": data_collection
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 标准IOI数据已保存到: {output_file}")
    print(f"📊 样本统计:")
    print(f"  总样本数: {N}")
    print(f"  平均序列长度: {sum(s['positions']['sequence_length'] for s in data_collection) / N:.1f}")
    
    # 显示示例
    if data_collection:
        sample = data_collection[0]
        print(f"\n📝 第一个样本示例:")
        print(f"  模板类型: {sample['template_type']}")
        print(f"  Clean句子: {sample['clean']['sentence']}")
        print(f"  ABC句子: {sample['abc_corrupted']['sentence']}")
        print(f"  IO: {sample['clean']['io_token']} -> {sample['abc_corrupted']['io_token']}")
        print(f"  S: {sample['clean']['s_token']} -> {sample['abc_corrupted']['s_token']}")
    
    return output_file, ioi_dataset, abc_dataset

def main():
    parser = argparse.ArgumentParser(description="生成标准IOI数据集")
    parser.add_argument("--N", type=int, default=100, help="样本数量")
    parser.add_argument("--output_file", type=str, default="standard_ioi_data.json", help="输出文件")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    
    args = parser.parse_args()
    
    # 初始化模型
    print("🔥 加载GPT-2模型...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    
    # 生成并保存数据
    generate_and_save_ioi_data(
        model=model,
        N=args.N,
        output_file=args.output_file,
        seed=args.seed
    )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()