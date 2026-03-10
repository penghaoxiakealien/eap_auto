import argparse
import json
import os
import sys
import torch as t
from transformer_lens import HookedTransformer
from collections import defaultdict

# 确保可以从父目录导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置镜像，以便在国内环境顺利下载模型
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def find_token_indices(token_list, target_token_str):
    """
    在分词后的列表中查找特定字符串的所有出现位置。
    会处理分词器可能添加的前缀'Ġ'。
    """
    indices = []
    # 标准化目标token
    clean_target = target_token_str.strip()
    for i, token in enumerate(token_list):
        # 标准化当前token
        clean_token = token.replace('Ġ', '').strip()
        if clean_token == clean_target:
            indices.append(i)
    return indices

def main():
    parser = argparse.ArgumentParser(description="Calculate and analyze the attention pattern from the S2 position for a given head.")
    parser.add_argument("--layer", type=int, required=True, help="The head's layer.")
    parser.add_argument("--head", type=int, required=True, help="The head's number.")
    parser.add_argument("--input_file", type=str, 
                        default="../../results/ioi/path_patching/structured_sentences.jsonl",
                        help="Path to the structured sentences file.")
    parser.add_argument("--output_dir", type=str, 
                        default="../../results/ioi/s2_attention_analysis",
                        help="Directory to save the output analysis file.")
    args = parser.parse_args()

    head_str = f"{args.layer}.{args.head}"
    print(f"--- 开始为头 {head_str} 分析S2位置的注意力模式 ---")

    # 1. 加载模型
    model_name = 'gpt2-small'
    try:
        model = HookedTransformer.from_pretrained(model_name, device='cuda' if t.cuda.is_available() else 'cpu')
        model.eval()
        print(f"模型 '{model_name}' 加载成功。")
    except Exception as e:
        print(f"错误: 无法加载模型 '{model_name}': {e}")
        return

    # 2. 加载句子数据
    try:
        with open(args.input_file, 'r') as f:
            sentences_data = [json.loads(line) for line in f]
        print(f"成功从 {args.input_file} 加载了 {len(sentences_data)} 个句子。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析JSON文件失败: {args.input_file}")
        return

    # 3. 循环处理每个句子
    all_results = []
    for data in sentences_data:
        sentence_id = data["sentence_id"]
        sentence_text = data["sentence_text"]
        s_token_str = data["s_token"]
        io_token_str = data["io_token"]

        tokens = model.to_str_tokens(sentence_text)
        s_indices = find_token_indices(tokens, s_token_str)
        io_indices = find_token_indices(tokens, io_token_str)

        if len(s_indices) < 2:
            print(f"警告: 在句子 {sentence_id} 中未找到S2，已跳过。")
            continue
        
        s1_pos, s2_pos = s_indices[0], s_indices[1]
        io_pos = io_indices[0] if io_indices else -1

        _, cache = model.run_with_cache(sentence_text, names_filter=f"blocks.{args.layer}.attn.hook_pattern")
        attention_matrix = cache[f'blocks.{args.layer}.attn.hook_pattern'][0, args.head].detach().cpu()

        s2_attention_vector = attention_matrix[s2_pos]
        top_k = 5
        top_values, top_indices = t.topk(s2_attention_vector, top_k)

        # 存储结果
        result = {
            "sentence_id": sentence_id,
            "head": head_str,
            # 【已添加】: 保存原始句子和关键token信息
            "original_sentence": sentence_text,
            "io_token_str": io_token_str,
            "s_token_str": s_token_str,
            # -----------------------------------------
            "s2_position": s2_pos,
            "s2_token": tokens[s2_pos],
            "attention_from_s2_to_s1": {
                "position": s1_pos,
                "token": tokens[s1_pos],
                "score": round(s2_attention_vector[s1_pos].item(), 4)
            },
            "attention_from_s2_to_io": {
                "position": io_pos,
                "token": tokens[io_pos] if io_pos != -1 else "N/A",
                "score": round(s2_attention_vector[io_pos].item(), 4) if io_pos != -1 else 0.0
            },
            "top_5_attended_tokens": [
                {"position": idx.item(), "token": tokens[idx.item()], "score": round(val.item(), 4)}
                for val, idx in zip(top_values, top_indices)
            ]
        }
        all_results.append(result)
        print(f"已处理句子: {sentence_id}")

    # 4. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"s2_attention_analysis_{args.layer}.{args.head}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n--- 分析完成 ---")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()