#!/usr/bin/env bash
# 简化版 agr_gender 流水线（不影响 IOI 脚本）：
# 1) 运行 EAP-IG 生成原始图（可通过 --skip-eap 跳过）
# 2) 折叠图（若 run_agr_gender_gpt2 已输出则直接复用）
# 3) 泛化 edge path patching 生成 joint_p.json
# 4) 按百分比筛边生成子图
#
# 说明：
# - 未包含 IOI 专用的 classify/dominant/path-patch 可视化步骤，后续可再扩展。
# - 默认模型 gpt2；需保证数据集路径正确（默认为 eap-ig 仓库下的 agr_gender CSV）。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/results/agr_gender_gpt2"
DATA_PATH="${PROJECT_ROOT}/../eap-ig/datasets/agr_gender_eap_data.csv"
DEVICE="${DEVICE:-cuda}"
TOPN="${TOPN:-2100}"
IG_STEPS="${IG_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
MODEL_PATH="${MODEL_PATH:-}"

# 筛边参数
TOP_PERCENT="${TOP_PERCENT:-70}"
MIN_EDGES="${MIN_EDGES:-1}"
SCORE_KEY="${SCORE_KEY:-delta_logit_diff}"
USE_ABS="${USE_ABS:-}"   # 为空=按绝对值

SKIP_EAP="${SKIP_EAP:-}"

echo "[1/4] 切换目录并设置 PYTHONPATH"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SKIP_EAP}" ]]; then
  echo "[2/4] 运行 EAP-IG (agr_gender)"
  EAP_CMD=(
    python "${PROJECT_ROOT}/run_agr_gender_gpt2.py"
    --data-path "${DATA_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --topn "${TOPN}"
    --ig-steps "${IG_STEPS}"
    --device "${DEVICE}"
    --model-name "${MODEL_NAME}"
  )
  if [[ -n "${MODEL_PATH}" ]]; then
    EAP_CMD+=(--model-path "${MODEL_PATH}")
  fi
  "${EAP_CMD[@]}"
else
  echo "[2/4] 跳过 EAP-IG（SKIP_EAP 已设置）"
fi

GRAPH_JSON="${OUTPUT_DIR}/graph.json"
COLLAPSED_JSON="${OUTPUT_DIR}/graph_collapsed.json"
[[ -f "${GRAPH_JSON}" ]] || { echo "未找到原始图 ${GRAPH_JSON}，请先跑 run_agr_gender_gpt2.py"; exit 2; }
[[ -f "${COLLAPSED_JSON}" ]] || { echo "未找到折叠图 ${COLLAPSED_JSON}，请确认 run_agr_gender_gpt2.py 输出"; exit 2; }

echo "[3/4] edge path patching (agr_gender, q-channel)"
JOINT_JSON="${OUTPUT_DIR}/joint_p.json"
python "${PROJECT_ROOT}/tests/experiments/edge_path_patching_generic.py" \
  --collapsed-graph "${COLLAPSED_JSON}" \
  --output "${JOINT_JSON}" \
  --task agr_gender \
  --data-path "${DATA_PATH}" \
  --device "${DEVICE}"

echo "[4/4] 按百分比筛选前 ${TOP_PERCENT}% 边并生成子图"
FILTER_PREFIX="${OUTPUT_DIR}/filtered/graph_top${TOP_PERCENT}"
mkdir -p "$(dirname "${FILTER_PREFIX}")"
python "${PROJECT_ROOT}/filter_edges_by_percent.py" \
  --edge-weights "${JOINT_JSON}" \
  --output-prefix "${FILTER_PREFIX}" \
  --top-percent "${TOP_PERCENT}" \
  --min-edges "${MIN_EDGES}" \
  --score-key "${SCORE_KEY}" \
  ${USE_ABS}

echo "完成。主要输出："
echo "  原始图: ${GRAPH_JSON}"
echo "  折叠图: ${COLLAPSED_JSON}"
echo "  joint_p: ${JOINT_JSON}"
echo "  筛选子图: ${FILTER_PREFIX}.json (及 csv/png 若脚本生成)"
