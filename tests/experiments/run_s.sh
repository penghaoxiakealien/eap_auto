#!/bin/bash

# 脚本出错时立即退出
set -e
set -o pipefail

# 实验参数设置
layer=7
head=9
rounds=5
top_k=1
typename="s_inhibition_head"
output_dir="$(dirname "$0")/../../results/ioi/hypothesis/${layer}.${head}_ss"

# 确保输出目录存在
mkdir -p "$output_dir"

echo "Cleaning up old result files..."
> "$output_dir/top_hypothesis.jsonl"
> "$output_dir/best_result.jsonl"
# 移除上一轮的完整日志文件
find "$output_dir" -name "full_run_log_*.json" -delete
echo "Cleanup complete."

# 运行自动化假设生成与验证循环
for round in $(seq 1 $rounds); do
    echo "--- Starting Round $round/$rounds ---"
    
    echo "Running auto_logit_s.py for round $round..."
    CUDA_VISIBLE_DEVICES=6 python auto_logit_s.py \
        --layer $layer \
        --head $head \
        --rounds $round \
        --typename $typename \
        --output_dir "$output_dir"
    echo "Finished auto_logit_s.py for round $round."

    # 自动筛选本轮的最佳假设
    echo "Selecting top $top_k hypothesis for round $round..."
    python select_top_hypo_s.py \
        --layer $layer \
        --head $head \
        --top_k $top_k \
        --rounds $round \
        --output_dir "$output_dir"
    echo "Finished selecting hypothesis for round $round."
done

echo -e "\nAll rounds finished. Best hypotheses saved in $output_dir/best_result.jsonl"