#!/usr/bin/env python
"""
debug_clean_logits.py

检查干净模型的基础logit值，看看是哪里出了问题
"""

import torch as t
import json
import os
import sys
from typing import Dict, List

# 导入依赖
sys.path.append(os.path.dirname(__file__))
from transformer_lens import HookedTransformer

def analyze_clean_logits(
    model: HookedTransformer,
    sentence_info: Dict[str, str]
):
    """分析干净模型的logit值"""
    
    sentence_text = sentence_info["sentence_text"]
    io_token = sentence_info["io_token"].strip()
    s_token = sentence_info["s_token"].strip()
    
    print(f"\n=== 分析句子 ===")
    print(f"句子: {sentence_text}")
    print(f"IO token: '{io_token}'")
    print(f"S token: '{s_token}'")
    
    # Tokenize
    tokens = model.to_tokens(sentence_text, prepend_bos=False)[0].to("cuda")
    token_strs = model.to_str_tokens(sentence_text)
    
    print(f"Tokens: {token_strs}")
    print(f"Token shape: {tokens.shape}")
    
    # 获取logits
    with t.no_grad():
        logits = model(tokens.unsqueeze(0))  # [1, seq, vocab]
    
    # 最后一个位置的logits
    end_pos = tokens.shape[0] - 1
    end_logits = logits[0, end_pos]  # [vocab]
    
    print(f"\nEnd position: {end_pos} (token: '{token_strs[end_pos]}')")
    
    # 获取IO和S的token IDs和logit值
    try:
        io_id = model.to_single_token(io_token)
        s_id = model.to_single_token(s_token)
        
        print(f"\nIO token '{io_token}' -> ID: {io_id}")
        print(f"S token '{s_token}' -> ID: {s_id}")
        
        io_logit = end_logits[io_id].item()
        s_logit = end_logits[s_id].item()
        logit_diff = io_logit - s_logit
        
        print(f"\nLogit values at end position:")
        print(f"  IO logit: {io_logit:.4f}")
        print(f"  S logit: {s_logit:.4f}")
        print(f"  Logit diff (IO - S): {logit_diff:.4f}")
        
        # 找出最高的logits看看正常情况下会预测什么
        top_logits, top_indices = t.topk(end_logits, k=10)
        print(f"\nTop 10 predicted tokens:")
        for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
            token_str = model.tokenizer.decode([idx]).strip()
            print(f"  {i+1}. '{token_str}' (ID: {idx}): {logit:.4f}")
            
        # 看看IO和S在排名中的位置
        sorted_logits, sorted_indices = t.sort(end_logits, descending=True)
        
        io_rank = (sorted_indices == io_id).nonzero(as_tuple=True)[0].item() + 1
        s_rank = (sorted_indices == s_id).nonzero(as_tuple=True)[0].item() + 1
        
        print(f"\nToken rankings:")
        print(f"  IO '{io_token}' rank: {io_rank}")
        print(f"  S '{s_token}' rank: {s_rank}")
        
        return {
            "io_logit": io_logit,
            "s_logit": s_logit,
            "logit_diff": logit_diff,
            "io_rank": io_rank,
            "s_rank": s_rank,
            "io_id": io_id,
            "s_id": s_id
        }
        
    except Exception as e:
        print(f"Error getting token IDs: {e}")
        
        # 尝试用空格版本
        try:
            io_id_space = model.to_single_token(f" {io_token}")
            s_id_space = model.to_single_token(f" {s_token}")
            
            print(f"Trying with spaces:")
            print(f"  ' {io_token}' -> ID: {io_id_space}")
            print(f"  ' {s_token}' -> ID: {s_id_space}")
            
            io_logit_space = end_logits[io_id_space].item()
            s_logit_space = end_logits[s_id_space].item()
            logit_diff_space = io_logit_space - s_logit_space
            
            print(f"\nLogit values with spaces:")
            print(f"  IO logit: {io_logit_space:.4f}")
            print(f"  S logit: {s_logit_space:.4f}")
            print(f"  Logit diff (IO - S): {logit_diff_space:.4f}")
            
            return {
                "io_logit": io_logit_space,
                "s_logit": s_logit_space, 
                "logit_diff": logit_diff_space,
                "io_id": io_id_space,
                "s_id": s_id_space
            }
            
        except Exception as e2:
            print(f"Error with spaced tokens too: {e2}")
            return None

def main():
    print("加载模型...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    # 加载一些句子进行测试
    input_file = "/data31/private/wangziran/eap_auto/results/ioi/path_patching/structured_sentences.jsonl"
    sentences_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            sentences_data.append(json.loads(line))
            if i >= 5:  # 只测试前3个句子
                break
    
    print(f"测试 {len(sentences_data)} 个句子")
    
    all_results = []
    for i, sentence_info in enumerate(sentences_data):
        print(f"\n{'='*60}")
        print(f"句子 {i+1}/{len(sentences_data)}")
        result = analyze_clean_logits(model, sentence_info)
        if result:
            all_results.append(result)
    
    # 总结统计
    print(f"\n{'='*60}")
    print("=== 总结统计 ===")
    
    positive_diffs = [r for r in all_results if r["logit_diff"] > 0]
    negative_diffs = [r for r in all_results if r["logit_diff"] < 0]
    
    print(f"正logit差异: {len(positive_diffs)}/{len(all_results)}")
    print(f"负logit差异: {len(negative_diffs)}/{len(all_results)}")
    
    if all_results:
        avg_diff = sum(r["logit_diff"] for r in all_results) / len(all_results)
        print(f"平均logit差异: {avg_diff:.4f}")
        
        avg_io_logit = sum(r["io_logit"] for r in all_results) / len(all_results)
        avg_s_logit = sum(r["s_logit"] for r in all_results) / len(all_results)
        
        print(f"平均IO logit: {avg_io_logit:.4f}")
        print(f"平均S logit: {avg_s_logit:.4f}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()