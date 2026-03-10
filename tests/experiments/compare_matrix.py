import json
import numpy as np
import os

def analyze_attention_changes_at_end(json_file_path):
    """
    分析 causal_attention_effects.json 文件。

    对于每个sender head，找出它影响不同receiver head时，
    在END位置（最后一个名字）的注意力增加最多和减少最多的token。
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的JSON格式。")
        return

    # 遍历每个 sender head
    for sender_key, receivers in data.items():
        if not receivers:  # 跳过空的sender
            continue
            
        print("-" * 60)
        print(f"分析 Sender Head: {sender_key}")
        print("-" * 60)

        # 遍历该 sender 影响的所有 receiver head
        for receiver_key, path_data in receivers.items():
            
            # 检查数据是否完整
            if "diff_matrix" not in path_data or "tokens" not in path_data:
                print(f"  -> Receiver {receiver_key}: 数据不完整，跳过。")
                continue

            diff_matrix = np.array(path_data["diff_matrix"])
            tokens = path_data["tokens"]

            # 检查矩阵和token列表是否有效
            if diff_matrix.size == 0 or not tokens:
                print(f"  -> Receiver {receiver_key}: diff_matrix或tokens为空，跳过。")
                continue
            
            # 确定END位置的索引。在IOI数据集中，这通常是<|endoftext|>前的token。
            try:
                end_token_idx = tokens.index("<|endoftext|>") - 1
                if end_token_idx < 0:
                    # 如果<|endoftext|>是第一个token，则无法分析
                    end_token_idx = len(tokens) - 1
            except ValueError:
                # 如果没有<|endoftext|>，则假定最后一个token是END
                end_token_idx = len(tokens) - 1
            
            end_token = tokens[end_token_idx]

            # 提取END位置query对所有key的注意力变化
            # diff_matrix[end_token_idx] 是一个向量
            attention_diffs_at_end = diff_matrix[end_token_idx, :len(tokens)]

            # 找出注意力增加/减少最多的token的索引
            # 使用 np.nanargmax/np.nanargmin 来处理可能的NaN值
            max_increase_idx = np.nanargmax(attention_diffs_at_end)
            max_decrease_idx = np.nanargmin(attention_diffs_at_end)

            # 获取对应的token和分数值
            max_increase_token = tokens[max_increase_idx]
            max_increase_value = attention_diffs_at_end[max_increase_idx]
            
            max_decrease_token = tokens[max_decrease_idx]
            max_decrease_value = attention_diffs_at_end[max_decrease_idx]

            print(f"  -> Receiver {receiver_key}:")
            print(f"     在 '{end_token}' (pos {end_token_idx}) 位置:")
            print(f"       - 注意力增加最多: '{max_increase_token}' (pos {max_increase_idx}), by {max_increase_value:+.4f}")
            print(f"       - 注意力减少最多: '{max_decrease_token}' (pos {max_decrease_idx}), by {max_decrease_value:+.4f}")
        
        print("\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    json_path = "../../results/ioi/path_patching/causal_attention_effects.json"
    
    if not os.path.exists(json_path):
        json_path = "causal_attention_effects.json"

    analyze_attention_changes_at_end(json_path)
