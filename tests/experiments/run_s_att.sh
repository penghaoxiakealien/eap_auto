#!/bin/bash

# 脚本出错时立即退出
set -e
set -o pipefail

# --- 实验参数配置 ---
LAYER=7
HEAD=3
TYPENAME="s_inhibition_head"
ROUNDS=1 # 【已修正】现在可以配置运行多轮

# --- 路径配置 ---
# 获取脚本所在目录
SCRIPT_DIR="$(dirname "$0")"
# 基础结果目录
BASE_RESULTS_DIR="$SCRIPT_DIR/../../results/ioi/hypothesis"
# 本次运行的基础输出目录
OUTPUT_DIR_BASE="$BASE_RESULTS_DIR/${LAYER}.${HEAD}_s_atg"
# 预计算脚本路径
PRECOMPUTE_SCRIPT="$SCRIPT_DIR/precompute_attention_scores.py"
# 主逻辑脚本路径
MAIN_SCRIPT="$SCRIPT_DIR/auto_s_att.py"
# 句子数据源
SENTENCE_SOURCE_FILE="$BASE_RESULTS_DIR/../path_patching/structured_sentences.jsonl"

# --- 主流程循环 ---
# 【已修正】使用for循环来执行指定轮次的独立实验
for (( r=1; r<=ROUNDS; r++ ))
do
    echo "========================================================"
    echo "  开始第 $r / $ROUNDS 轮自动化假设生成与两阶段精炼"
    echo "  目标头: ${LAYER}.${HEAD} (${TYPENAME})"
    echo "========================================================"
    
    # 为每一轮创建一个独立的子目录
    OUTPUT_DIR="$OUTPUT_DIR_BASE/round_$r"
    mkdir -p "$OUTPUT_DIR"
    echo "[步骤 1/3] 输出目录已创建: $OUTPUT_DIR"

    # 2. 预计算直接注意力的基准真相 (Ground Truth)
    echo -e "\n[步骤 2/3] 开始预计算直接注意力分数的基准..."
    GROUND_TRUTH_FILE="$OUTPUT_DIR/attention_scores_ground_truth.jsonl"
    if [ -f "$GROUND_TRUTH_FILE" ]; then
        echo "基准文件 '$GROUND_TRUTH_FILE' 已存在，跳过预计算。"
    else
        if [ ! -f "$SENTENCE_SOURCE_FILE" ]; then
            echo "错误: 句子源文件 '$SENTENCE_SOURCE_FILE' 未找到！"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=6 python "$PRECOMPUTE_SCRIPT" \
            --layer "$LAYER" \
            --head "$HEAD" \
            --input_file "$SENTENCE_SOURCE_FILE" \
            --output_dir "$OUTPUT_DIR"
        
        echo "预计算完成。"
    fi

    # 3. 运行主自动化脚本
    echo -e "\n[步骤 3/3] 开始运行两阶段假设精炼主程序 (第 $r 轮)..."
    CUDA_VISIBLE_DEVICES=6 python "$MAIN_SCRIPT" \
        --layer "$LAYER" \
        --head "$HEAD" \
        --rounds "$r" \
        --typename "$TYPENAME" \
        --output_dir "$OUTPUT_DIR"

    echo -e "\n--- 第 $r / $ROUNDS 轮运行完毕！ ---"
done

echo "========================================================"
echo "所有轮次运行完毕！"
echo "最终结果和日志已保存至: ${OUTPUT_DIR_BASE}"
echo "========================================================"