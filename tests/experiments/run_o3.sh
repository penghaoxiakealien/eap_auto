#!/bin/bash

set -e
set -o pipefail

# 定义变量
layer=10
head=0
top_k=1
typename="name_mover_head"
suffix="_l"

# 拼接输出目录
output_dir="$(dirname "$0")/../../results/ioi/hypothesis/${layer}.${head}"
# 创建输出目录
mkdir -p "$output_dir"
# # # 执行分析
echo "Running attention analysis for layer $layer, head $head"
CUDA_VISIBLE_DEVICES=6 python attention_score_by_head.py --layer $layer --head $head --output_dir "$output_dir" 
echo "Finishing attention analysis for layer $layer, head $head"

# # # 加载句子
echo "Loading sentences..."
CUDA_VISIBLE_DEVICES=6 python attention_score_on_examples.py --layer $layer --head $head --output_dir "$output_dir"
echo "Finishing loading sentences..."
# 在output_dir下新建一个空的文件top_hypothesis.jsonl用于储存粗选的hypothesis，如果已经存在则清空
if [ ! -f "$output_dir/top_hypothesis.jsonl" ]; then
    touch "$output_dir/top_hypothesis.jsonl"
else
    > "$output_dir/top_hypothesis.jsonl"
fi
# 在output_dir下新建一个空的文件training_sentences.jsonl用于储存每轮用来训练的例句，如果已经存在则清空
if [ ! -f "$output_dir/training_sentences.jsonl" ]; then
    touch "$output_dir/training_sentences.jsonl"
else
    > "$output_dir/training_sentences.jsonl"
fi

for round in {1..5}; do
    # 运行分析
    echo "Running automatic analysis for round $round..."
    CUDA_VISIBLE_DEVICES=6 python auto_backup.py --layer $layer --head $head --rounds $round --typename $typename
    echo "Finishing automatic analysis for round $round..."

    # 把outputdir/results.json中scores里所有value相加，选出最大的top_k个保存在新文件中
    echo "Selecting top $top_k scores..."
    python select_top_hypothesis.py --layer $layer --head $head --top_k $top_k --rounds $round --output_dir "$output_dir"
    echo "Finishing selecting top $top_k scores..."
done
