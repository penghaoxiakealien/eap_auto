import json
import os
import argparse

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Select the top hypothesis from a full run log.")
    parser.add_argument("--layer", type=int, required=True, help="The sender head's layer.")
    parser.add_argument("--head", type=int, required=True, help="The sender head's number.")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top hypotheses to select.")
    parser.add_argument("--rounds", type=int, required=True, help="The current round number.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the log results and to save outputs.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # --- 修改点 1: 输入文件路径更新为新的日志文件 ---
    log_file = os.path.join(args.output_dir, f"full_run_log_{args.rounds}.json")
    top_hypothesis_file = os.path.join(args.output_dir, "top_hypothesis.jsonl")
    best_result_file = os.path.join(args.output_dir, "best_result.jsonl")

    if not os.path.exists(log_file):
        print(f"错误: 输入文件 {log_file} 未找到。")
        return

    try:
        with open(log_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"错误: 无法读取或解析 {log_file}: {e}")
        return

    if not data or not isinstance(data, list):
        print(f"警告: {log_file} 为空或格式不正确。跳过本轮筛选。")
        return

    # --- 修改点 2: 从完整的日志中只提取 'iteration' 类型的数据进行处理 ---
    iteration_data = [item for item in data if item.get("type") == "iteration"]

    if not iteration_data:
        print("警告: 在日志文件中未找到任何迭代数据。")
        return

    f1_score_threshold = 0.8
    
    print(f"Initial number of iteration candidates: {len(iteration_data)}")
    
    candidates = [item for item in iteration_data if item.get("f1_score", 0) >= f1_score_threshold]
    
    print(f"Number of candidates after f1_score filtering (>= {f1_score_threshold}): {len(candidates)}")

    if not candidates:
        print("Warning: No hypotheses met the f1_score threshold. Sorting based on all iteration data.")
        candidates = iteration_data

    sorted_data = sorted(
        candidates,
        key=lambda x: x.get("f1_score", 0),
        reverse=True
    )

    if not sorted_data:
        print("No valid hypothesis found after filtering and sorting. Exiting.")
        return

    top_k_hypotheses = sorted_data[:args.top_k]

    with open(top_hypothesis_file, "a") as f:
        for item in top_k_hypotheses:
            entry = {
                "round": args.rounds,
                "f1_score": item.get("f1_score"),
                "hypothesis": item.get("hypothesis")
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"Top {len(top_k_hypotheses)} hypothesis for round {args.rounds} appended to {top_hypothesis_file}")

    all_top_hypotheses = []
    if os.path.exists(top_hypothesis_file):
        with open(top_hypothesis_file, "r") as f:
            for line in f:
                try:
                    all_top_hypotheses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if all_top_hypotheses:
        overall_best = max(all_top_hypotheses, key=lambda x: x.get("f1_score", 0))
        
        with open(best_result_file, "w") as f:
            json.dump(overall_best, f, indent=4)
        print(f"Overall best hypothesis updated in {best_result_file}")
    else:
        print("Warning: No top hypotheses found yet to determine the overall best.")

if __name__ == "__main__":
    main()