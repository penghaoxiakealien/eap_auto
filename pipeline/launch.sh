#!/bin/bash

set -e

BASE_DIR="/home/wangziran/eap_auto"
TASK_NAME="garden_npz_v_trans_mod"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FIND_BEST_SCRIPT="${BASE_DIR}/pipeline/findbestedge.py"
RUN_SINGLE_SCRIPT="${BASE_DIR}/pipeline/run_single.py"

FINAL_DATASET="${BASE_DIR}/datasets/garden/garden_npz_v_trans_mod.csv"
RESULTS_DIR="${BASE_DIR}/results/garden/${TASK_NAME}_${TIMESTAMP}"

export EAP_ROOT="/home/wangziran/eap-ig"

echo "STEP 1: 运行 findbestedge.py 寻找最佳边数..."
# 创建结果目录
mkdir -p ${RESULTS_DIR}

FIND_BEST_OUTPUT=$(python ${FIND_BEST_SCRIPT} \
    --data_file ${FINAL_DATASET} \
    --output_dir ${RESULTS_DIR} | tee /dev/tty)

echo "-> 寻找最佳边数完成。结果保存在: ${RESULTS_DIR}"
echo ""

echo "STEP 2: 使用找到的最佳边数进行最终的单次运行..."

BEST_EDGE=$(echo "${FIND_BEST_OUTPUT}" | grep "推荐的最佳边数:" | awk '{print $NF}')

if [[ -z "$BEST_EDGE" ]]; then
    echo "警告: 未能从 findbestedge.py 的输出中自动确定最佳边数。"
    echo "跳过最终的单次运行。"
else
    echo "-> 找到的最佳边数为: ${BEST_EDGE}"
    echo "-> 使用此边数运行一次详细分析..."

    python ${RUN_SINGLE_SCRIPT} \
        --data_file ${FINAL_DATASET} \
        --output_dir "${RESULTS_DIR}/final_run_edge_${BEST_EDGE}" \
        --n_edge ${BEST_EDGE}
    
    echo "-> 最终分析完成。结果保存在: ${RESULTS_DIR}/final_run_edge_${BEST_EDGE}"
fi

echo ""
echo "全部执行完毕！"
