#!/bin/bash

set -e
set -o pipefail

# 定义变量
heads=("9.9")
type="name_mover_head"  # 头部类型
# 拼接输出目录
base_dir="$(dirname "$0")/../../results/ioi/hypothesis"
output_dir="$(dirname "$0")/../../results/ioi/hypothesis/${type}"
mkdir -p "$output_dir"
# 加载句子
# echo "Loading sentences..."
# CUDA_VISIBLE_DEVICES=6 python tests/experiments/attention_score_on_validation_examples.py --heads "${heads[*]}"  --typename $type
# echo "Finishing loading sentences..."


# 遍历每个 head
for head in "${heads[@]}"; do
    # 拼接输出目录并自动创建文件夹
    output_head_dir="$output_dir/$head"
    mkdir -p "$output_head_dir"
    # 在output_head_dir下新建一个空的文件training_sentences.jsonl用于储存每轮用来训练的例句，如果已经存在则清空
    if [ ! -f "$output_head_dir/validation_sentences.jsonl" ]; then
    touch "$output_head_dir/validation_sentences.jsonl"
    else
    > "$output_head_dir/validation_sentences.jsonl"
    fi
    # 运行分析
    echo "Running automatic analysis for head $head..."
    CUDA_VISIBLE_DEVICES=6 python tests/experiments/auto_test.py --head $head --type $type
    echo "Finishing automatic analysis for head $head..."
done

# 程序结束后输出总结信息
echo "All heads have been processed successfully!"