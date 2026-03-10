from attention_score_by_head import run
import asyncio
import sys
import json
import os
import re
import math
import argparse
    

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Select top hypothesis.")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--head", type=int, required=True, help="Head number")
    parser.add_argument("--top_k", type=int, required=True, help="Top k results")
    parser.add_argument("--rounds", type=int, required=True, help="Round number")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    layer = args.layer
    head = args.head
    top_k = args.top_k
    rounds = args.rounds
    output_dir = args.output_dir
    input_file = os.path.join(output_dir, f"results_{rounds}.json")
    output_file = os.path.join(output_dir, "top_hypothesis.jsonl")
    with open(input_file, "r") as f:
        try:
            data = json.load(f)  # 读取整个文件作为 JSON 数组
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {input_file}: {e}")
            return

    # =================================================================
    # 核心修改：增加基于 logit_accuracy 的前置过滤
    # =================================================================
    logit_accuracy_threshold = 0.8  # 设置阈值，例如0.9，意味着至少90%的因果判断是正确的
    
    print(f"Initial number of candidates: {len(data)}")
    
    # 第一步：过滤掉 logit_accuracy 不达标的假设
    filtered_by_logit = [
        item for item in data 
        if item.get("scores", {}).get("logit_accuracy", 0) >= logit_accuracy_threshold
    ]
    
    print(f"Number of candidates after logit_accuracy filtering (>= {logit_accuracy_threshold}): {len(filtered_by_logit)}")

    # 如果过滤后没有候选者，可以决定是报错还是使用原始数据
    if not filtered_by_logit:
        print("Warning: No hypotheses met the logit_accuracy threshold. Sorting based on original data.")
        # 如果希望在没有达标项时依然产生输出，可以使用原始的`data`进行排序
        # 如果希望在这种情况下不产生输出，可以提前返回
        # return 
        target_data = data
    else:
        target_data = filtered_by_logit
    # =================================================================

    # 第二步：在通过筛选的候选中，按原有的注意力模式分数进行排序
    sorted_data = sorted(
        target_data, # 使用过滤后的数据进行排序
        key=lambda x: (
            x["scores"].get("accuracy", 0),
            x["scores"].get("ndcg pre", 0),
            x["scores"].get("Kendall Tau pre", 0)
        ),
        reverse=True
    )

    # 如果排序后没有数据，则直接退出
    if not sorted_data:
        print("No valid hypothesis found after filtering and sorting. Exiting.")
        return

    # 取前 top_k 个
    filtered_data = sorted_data[:top_k]
    
    # 把上面结果包装成一个json，把这个json和round再包装成一个json
    if top_k>=2:
        result = {
            "round": rounds,
            "top_k_hypothesis": filtered_data
        }
    elif top_k==1:
        result = {
            "round": rounds,
            "top_k_hypothesis": {
                "iteration": filtered_data[0]["iteration"],
                "hypothesis": filtered_data[0]["hypothesis"],
                "scores": filtered_data[0]["scores"],
                "hypothesis_analysis": filtered_data[0]["hypothesis_analysis"]
            }
        }
        output_file = os.path.join(output_dir, f"best_result.jsonl")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                # best_result.jsonl 可能包含多个JSON对象，需要特殊处理
                content = f.read().strip()
                if content:
                    # 假设文件内容是一个JSON数组
                    existing_data = json.loads(content)
                else:
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # 将新结果追加到现有数据中
    existing_data.append(result)

    # 保存到输出文件
    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results for round {rounds} saved to {output_file}")
    
if __name__ == "__main__":
    main()
