#!/bin/bash

set -e
set -o pipefail

LAYER=5
HEAD=5
TYPENAME="induction_head"
ROUNDS=1 

# --- 路径配置 ---
# 获取脚本所在目录
SCRIPT_DIR="$(dirname "$0")"
# 基础结果目录
BASE_RESULTS_DIR="$SCRIPT_DIR/../../results/ioi/hypothesis/Induction_Head"
# 本次运行的基础输出目录
OUTPUT_DIR_BASE="$BASE_RESULTS_DIR/${LAYER}.${HEAD}_i_2"
# 主逻辑脚本路径
MAIN_SCRIPT="$SCRIPT_DIR/auto_i.py"

PREPROCESSED_CAUSAL_GT_FILE="$SCRIPT_DIR/../../results/ioi/path_patching/IH/causal_effects_summary_ih_to_s.json"
PREPROCESSED_ATTENTION_GT_FILE="$SCRIPT_DIR/../../results/ioi/path_patching/IH/attention_vectors_summary.json"

RAW_CAUSAL_EFFECTS_FILE="$SCRIPT_DIR/../../results/ioi/path_patching/IH/analysis_A${LAYER}.${HEAD}_to_B7.3_on_C9.6.json" # 复用一个包含完整句子的文件即可

# --- 主流程循环 ---
for (( r=1; r<=ROUNDS; r++ ))
do
    echo "========================================================"
    echo "  开始第 $r / $ROUNDS 轮自动化实验 (感应头)"
    echo "  目标头: ${LAYER}.${HEAD} (${TYPENAME})"
    echo "========================================================"
    
    # --- 步骤 1: 创建独立的输出子目录 ---
    OUTPUT_DIR="$OUTPUT_DIR_BASE/round_$r"
    mkdir -p "$OUTPUT_DIR"
    echo "[步骤 1/2] 输出目录已创建: $OUTPUT_DIR"

    # --- 步骤 2: 运行主自动化脚本 ---
    echo -e "\n[步骤 2/2] 开始运行主程序 (第 $r 轮)..."
    echo "将使用以下预处理好的答案文件:"
    echo "  - 因果效应答案: $PREPROCESSED_CAUSAL_GT_FILE"
    echo "  - 直接注意力答案: $PREPROCESSED_ATTENTION_GT_FILE"
    
    # 检查文件是否存在
    if [ ! -f "$PREPROCESSED_CAUSAL_GT_FILE" ] || [ ! -f "$PREPROCESSED_ATTENTION_GT_FILE" ]; then
        echo "错误: 一个或多个预处理的答案文件未找到！请确保路径正确。"
        exit 1
    fi

    # 直接调用主脚本，传入所有必需的文件路径
    CUDA_VISIBLE_DEVICES=6 python "$MAIN_SCRIPT" \
        --layer "$LAYER" \
        --head "$HEAD" \
        --rounds "$r" \
        --typename "$TYPENAME" \
        --output_dir "$OUTPUT_DIR" \
        --causal_effects_file "$RAW_CAUSAL_EFFECTS_FILE" \
        --ground_truth_file "$PREPROCESSED_CAUSAL_GT_FILE" \
        --attention_ground_truth_file "$PREPROCESSED_ATTENTION_GT_FILE"

    echo -e "\n--- 第 $r / $ROUNDS 轮运行完毕！ ---"
done

echo "========================================================"
echo "所有轮次运行完毕！"
echo "最终结果和日志已保存至: ${OUTPUT_DIR_BASE}"
echo "========================================================"
