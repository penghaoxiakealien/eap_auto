import json
import argparse
import os

def preprocess_causal_for_evaluation(input_file: str, output_file: str):
    """
    读取最终的logit效应分析文件，并将其转换为 auto_NMH.py 的因果评估输入格式。
    """
    print(f"--- 第3步: 预处理Logit变化用于因果评估 ---")
    print(f"读取原始Logit效应数据: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            logit_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析输入文件: {e}")
        return

    results = logit_data.get("results", [])
    processed_for_causal_eval = []
    print(f"开始处理 {len(results)} 个样本...")

    for item in results:
        sample_id = item.get("sample_id")
        sentence_text = item.get("sentence_text")
        logit_analysis = item.get("logit_analysis_at_end_pos", {}).get("sentence_token_logits", [])

        if sample_id is None or str(sample_id).strip() == "" or not sentence_text:
            print(f"警告: 跳过缺少必要信息的条目: sample_id={sample_id!r}")
            continue

        delta = item.get("delta_logit_diff")
        if delta is None:
            clean = item.get("clean_logit_diff")
            patched = item.get("patched_logit_diff")
            if clean is None or patched is None:
                print(f"警告: 缺少 logit diff 信息，跳过样本 {sample_id}")
                continue
            delta = patched - clean
        direction = "increase" if delta > 0 else "decrease"
        processed_item = {
            "sentence_id": str(sample_id),
            "sentence_text": sentence_text,
            "ground_truth": {
                "direction": direction
            }
        }
        processed_for_causal_eval.append(processed_item)

    print(f"处理完成。正在将 {len(processed_for_causal_eval)} 条记录写入: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_for_causal_eval, f, indent=2, ensure_ascii=False)
        print(f"✅ 因果评估所需的预处理文件生成成功！已保存到: {output_file}")
    except Exception as e:
        print(f"错误: 无法写入输出文件: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将Logit效应文件转换为 auto_NMH.py 的因果评估输入格式。")
    parser.add_argument("--input", type=str, required=True, help="Path to the final logit effects JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final preprocessed file for causal evaluation.")
    # *** 核心修改：将默认值改为1 ***
    args = parser.parse_args()

    preprocess_causal_for_evaluation(args.input, args.output)
'''
import json
import argparse
import os
from collections import defaultdict

def get_suffixed_map_from_tokens(tokens: list[str]) -> dict[int, str]:
    """
    【已修复】根据与diff向量严格对应的tokens列表，为每个位置生成带后缀的token。
    这个函数现在是健壮的，因为它使用与diff向量完全相同的数据源。
    """
    # 1. 全局扫描，确定哪些词是重复的
    global_counts = defaultdict(int)
    for token in tokens:
        # 假设token已经是基本单位，直接使用
        norm_token = token.strip().lower()
        global_counts[norm_token] += 1

    # 2. 遍历并生成带后缀的映射
    running_counts = defaultdict(int)
    suffixed_map = {}
    for i, token in enumerate(tokens):
        norm_token = token.strip().lower()
        # 只有在全局计数大于1时才需要后缀
        if global_counts[norm_token] > 1:
            running_counts[norm_token] += 1
            suffixed_map[i] = f"{token.strip()}_{running_counts[norm_token]}"
        else:
            suffixed_map[i] = token.strip()
    return suffixed_map

def preprocess_effects(input_file: str, output_file: str, top_k: int = 1):
    """
    【已修复】读取原始的因果效应分析文件，计算Ground Truth，并保存到新的JSON文件中。
    现在使用与diff向量严格对应的tokens列表进行所有操作。
    """
    print(f"正在读取原始数据: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载或解析输入文件: {e}")
        return

    processed_results = []
    print(f"开始处理 {len(original_data)} 条数据...")

    for item in original_data:
        sid = item.get("sentence_id")
        # 【修复核心】: 不再使用sentence_text进行分词，直接使用tokens列表
        tokens = item.get("tokens")
        diff_vector = item.get("end_token_diff_attention")
        
        # 检查核心数据是否存在且长度匹配
        if not all([sid, tokens, diff_vector]) or len(tokens) != len(diff_vector):
            print(f"警告: 跳过不完整或长度不匹配的条目 {sid or 'Unknown'}")
            continue

        # 1. 【已修复】根据正确的tokens列表生成 "位置 -> 带后缀token" 映射
        suffixed_map = get_suffixed_map_from_tokens(tokens)
        
        # 2. 将diff值与它们的原始位置(index)绑定并排序
        diff_with_pos = sorted(
            [(diff_vector[i], i) for i in range(len(diff_vector))],
            key=lambda x: x[0], 
            reverse=True
        )
        
        # 3. 根据排序后的diff值，从suffixed_map中查找正确位置上的、带后缀的token
        suppress_set = {suffixed_map.get(pos) for _, pos in diff_with_pos[:top_k] if suffixed_map.get(pos) is not None}
        promote_set = {suffixed_map.get(pos) for _, pos in diff_with_pos[-top_k:] if suffixed_map.get(pos) is not None}

        processed_results.append({
            "sentence_id": sid,
            "sentence_text": item.get("sentence_info", {}).get("sentence_text", ""), # 仍然保存原始文本以供参考
            "ground_truth": {
                "increase": list(suppress_set), # increase in attention -> suppress effect
                "decrease": list(promote_set)  # decrease in attention -> promote effect
            }
        })

    print(f"处理完成。正在将 {len(processed_results)} 条结果写入: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)
        print("预处理成功！")
    except Exception as e:
        print(f"错误: 无法写入输出文件: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess causal effects data to generate a ground truth file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raw analysis JSON file (e.g., analysis_A0.1_to_B7.3_on_C9.6.json).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed ground truth JSON file.")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top tokens to consider for promote/suppress.")
    args = parser.parse_args()
    
    preprocess_effects(args.input, args.output, args.top_k)
'''
