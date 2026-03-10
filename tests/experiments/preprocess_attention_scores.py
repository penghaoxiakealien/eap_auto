import json
import argparse
import os
from collections import defaultdict

def add_suffixes_to_tokens(tokens: list[str]) -> list[str]:
    """
    为token列表中的重复项添加后缀 (e.g., "a_1", "a_2")。
    """
    counts = defaultdict(int)
    total_counts = defaultdict(int)
    for token in tokens:
        total_counts[token] += 1
    
    suffixed_tokens = []
    for token in tokens:
        if total_counts[token] > 1:
            counts[token] += 1
            suffixed_tokens.append(f"{token}_{counts[token]}")
        else:
            suffixed_tokens.append(token)
    return suffixed_tokens

def preprocess_attention_for_nmh(input_file: str, output_file: str, top_k: int = 5):
    """
    【最终修正版】
    读取 preprocessed_for_sampling.jsonl 文件，并生成 auto_NMH.py 所需的、
    用于采样和注意力评估的最终预处理文件 (preprocessed_attention_scores.json)。
    """
    print(f"--- 最终注意力预处理步骤 ---")
    print(f"读取可靠的注意力数据源: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析输入文件: {e}")
        return

    processed_for_nmh = []
    print(f"开始为 auto_NMH.py 转换 {len(source_data)} 条数据...")

    for i, item in enumerate(source_data):
        sentence_text = item.get("original_sentence")
        attention_scores = item.get("attention_scores")
        sentence_id = item.get("sentence_id", str(i))

        if not all([sentence_text, attention_scores is not None]):
            print(f"警告: 跳过缺少 'original_sentence' 或 'attention_scores' 的条目 (行号约 {i+1})")
            continue

        # 1. 从 attention_scores 中提取 top_k 个 token 字符串
        #    注意：原始文件中的 token 字段可能带前导空格，需要去除
        top_tokens_plain = [
            score_info.get("token", "").strip() 
            for score_info in attention_scores[:top_k]
        ]
        
        # 2. 为这些 top_k token 添加后缀以处理重复项
        final_top_k_tokens = add_suffixes_to_tokens(top_tokens_plain)

        # 3. 构建 auto_NMH.py 期望的最终格式
        processed_item = {
            "sentence_id": str(sentence_id),
            "sentence_text": sentence_text,
            "top_k_tokens": final_top_k_tokens,
        }
        if "indirect_object" in item:
            processed_item["indirect_object"] = item.get("indirect_object")
        processed_for_nmh.append(processed_item)

    print(f"处理完成。正在将 {len(processed_for_nmh)} 条记录写入: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_for_nmh, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功生成 auto_NMH.py 所需的预处理注意力文件！已保存到: {output_file}")
    except Exception as e:
        print(f"错误: 无法写入输出文件: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 auto_NMH.py 生成最终的预处理注意力分数文件。")
    parser.add_argument("--input", type=str, required=True, help="Path to the reliable attention scores JSONL file (e.g., preprocessed_for_sampling.jsonl).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final preprocessed JSON file for auto_NMH.py (e.g., preprocessed_attention_scores.json).")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top attended tokens to include for sampling diversity.")
    args = parser.parse_args()
    
    preprocess_attention_for_nmh(args.input, args.output, args.top_k)
'''
import json
import argparse
import os
from collections import defaultdict
import re

def preprocess_attention_scores(input_file: str, output_file: str, top_k: int = 2):
    """
    【最终修复版】读取原始的注意力分数文件(s2_attention_analysis.json)，
    根据其真实结构计算Ground Truth，并保存到新的JSON文件中。
    这个版本直接使用文件内提供的token字符串，不再进行任何有风险的分词。
    """
    print(f"正在读取原始注意力数据: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 假设输入文件是一个JSON列表
            original_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析输入文件: {e}")
        return

    processed_results = []
    print(f"开始处理 {len(original_data)} 条数据...")

    for item in original_data:
        sid = item.get("sentence_id")
        sentence_text = item.get("original_sentence")
        top_tokens_data = item.get("top_5_attended_tokens")

        if not all([sid, sentence_text, top_tokens_data]):
            print(f"警告: 跳过缺少 sentence_id, original_sentence, 或 top_5_attended_tokens 的条目: {sid or 'Unknown'}")
            continue

        # 1. 【修复核心】直接从 top_5_attended_tokens 的数据中提取token字符串
        #    不再依赖任何外部或内部的分词和位置查找，因为它们是错误的根源。
        #    我们相信文件中的 'token' 字段本身就是正确的。
        
        # 按分数排序（尽管它通常已经是排序的，但为了保险起见）
        sorted_top_tokens = sorted(top_tokens_data, key=lambda x: x.get('score', 0), reverse=True)
        
        # 2. 提取Top-K的token字符串
        top_k_token_strings = [token_info.get("token", "").strip() for token_info in sorted_top_tokens[:top_k]]
        
        # 3. 为重复的token添加后缀
        #    这个逻辑现在是独立的，只处理这Top-K个token，不再关心它们在完整句子中的位置。
        final_tokens = []
        counts = defaultdict(int)
        # 先统计这几个token的出现次数
        for token in top_k_token_strings:
            counts[token] += 1
        
        # 重新遍历，如果一个token出现多次，则添加后缀
        running_counts = defaultdict(int)
        for token in top_k_token_strings:
            if counts[token] > 1:
                running_counts[token] += 1
                final_tokens.append(f"{token}_{running_counts[token]}")
            else:
                final_tokens.append(token)

        processed_results.append({
            "sentence_id": sid,
            "sentence_text": sentence_text,
            "top_k_tokens": final_tokens
        })

    print(f"处理完成。正在将 {len(processed_results)} 条结果写入: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)
        print(f"注意力分数预处理成功！已生成 {len(processed_results)} 条记录。")
    except Exception as e:
        print(f"错误: 无法写入输出文件: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess attention scores to generate a ground truth file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raw attention scores JSON file (e.g., s2_attention_analysis_0.1.json).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed attention ground truth JSON file.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top tokens to extract.")
    args = parser.parse_args()
    
    preprocess_attention_scores(args.input, args.output, args.top_k)
'''
