#!/usr/bin/env bash
# filepath: /data31/private/wangziran/eap_auto/run_eap_head_classification.sh
set -euo pipefail

# === 路径与参数（按需修改）===
PROJECT_ROOT="/data31/private/wangziran/eap_auto"
OUTPUT_DIR="${PROJECT_ROOT}/results/ioi_gpt2_800"
IOI_JSON="${PROJECT_ROOT}/results/ioi/path_patching/standard_ioi_data.json"   # classify 的 IOI 样本
DATASET_CSV="${PROJECT_ROOT}/ioi_gpt2.csv"                                   # run_ioi_gpt2 的数据
DEVICE="${DEVICE:-cuda}"
TOPN="${TOPN:-200}"
IG_STEPS="${IG_STEPS:-5}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
MODEL_PATH="${MODEL_PATH:-}"

MODEL_ARGS=()
if [[ -n "${MODEL_PATH}" ]]; then
  MODEL_ARGS+=(--model-path "${MODEL_PATH}")
fi

# 必需：edge_path_patching 的 joint_p.json（本脚本强制要求存在）
JOINT_WEIGHTS_JSON="${PROJECT_ROOT}/results/ioi_gpt2/joint_p.json"

# 筛选参数：保留前 60%
TOP_PERCENT="70"                   # [1,100]
MIN_EDGES="1"
SCORE_KEY="delta_logit_diff"      # joint_p.json 中用于排序的字段
USE_ABS=""                        # 为空表示按绝对值排序；如要原值排序，改为：USE_ABS="--no-abs-score"

# 分类参数（使用“去自注意 + 仅实体域”）
TOPK="3"
TOPK_TAG="top${TOPK}"
THRESH_CONC="0.05"                # conc(S2)-conc(END) 的参考阈值
END_ENTITY_TH="0.2"               # (IO+S1-S2) 的参考阈值

echo "[1/8] 切换工程目录并设置 PYTHONPATH"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[2/8] 运行 EAP-IG（生成原始图 JSON）"
RUN_CMD=(
  python "${PROJECT_ROOT}/run_ioi_gpt2.py"
  --dataset "${DATASET_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size 8
  --topn "${TOPN}"
  --ig-steps "${IG_STEPS}"
  --device "${DEVICE}"
  --model-name "${MODEL_NAME}"
)
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  RUN_CMD+=("${MODEL_ARGS[@]}")
fi
"${RUN_CMD[@]}"

# 猜测原始图路径（如你的 run_ioi_gpt2.py 输出不同文件名，改这里）
GRAPH_JSON_CANDIDATES=(
  "${OUTPUT_DIR}/graph.json"
  "${OUTPUT_DIR}/graph_full.json"
)
GRAPH_JSON=""
for p in "${GRAPH_JSON_CANDIDATES[@]}"; do
  if [[ -f "$p" ]]; then GRAPH_JSON="$p"; break; fi
done
if [[ -z "${GRAPH_JSON}" ]]; then
  echo "错误：未找到原始图 JSON（尝试于 ${OUTPUT_DIR}/graph*.json）。请检查 run_ioi_gpt2.py 的输出文件名。" >&2
  exit 2
fi
echo "原始图: ${GRAPH_JSON}"

echo "[3/8] 折叠原始图为 head-only（便于对照）"
COLLAPSE_PREFIX="${OUTPUT_DIR}/graph_collapsed"
python "${PROJECT_ROOT}/graph_collapse.py" \
  "${GRAPH_JSON}" \
  --output-prefix "${COLLAPSE_PREFIX}"
COLLAPSED_JSON="${COLLAPSE_PREFIX}.json"
[[ -f "${COLLAPSED_JSON}" ]] || { echo "错误：未生成折叠图 ${COLLAPSED_JSON}"; exit 2; }
echo "折叠图: ${COLLAPSED_JSON}"

echo "[4/8] 基于 joint_p.json 筛选前 ${TOP_PERCENT}% 边并重建子图（强制执行）"
if [[ ! -f "${JOINT_WEIGHTS_JSON}" ]]; then
  echo "未找到 joint_p.json，自动调用 edge_path_patching 生成..."
  EP_OUTPUT_DIR=$(dirname "${JOINT_WEIGHTS_JSON}")
  mkdir -p "${EP_OUTPUT_DIR}"
  python "${PROJECT_ROOT}/tests/experiments/edge_path_patching.py" \
    --collapsed-graph "${COLLAPSED_JSON}" \
    --output "${JOINT_WEIGHTS_JSON}" \
    --channels q k v \
    --joint-patch \
    --dataset-size 64 \
    --prompt-type mixed \
    --device "${DEVICE}"
fi
FILTER_OUT_PREFIX="${OUTPUT_DIR}/filtered/graph_top${TOP_PERCENT}"
mkdir -p "$(dirname "${FILTER_OUT_PREFIX}")"
python "${PROJECT_ROOT}/filter_edges_by_percent.py" \
  --edge-weights "${JOINT_WEIGHTS_JSON}" \
  --output-prefix "${FILTER_OUT_PREFIX}" \
  --top-percent "${TOP_PERCENT}" \
  --min-edges "${MIN_EDGES}" \
  --score-key "${SCORE_KEY}" \
  ${USE_ABS}
FILTERED_JSON="${FILTER_OUT_PREFIX}.json"
[[ -f "${FILTERED_JSON}" ]] || { echo "错误：未生成筛选子图 ${FILTERED_JSON}"; exit 4; }
echo "筛选子图: ${FILTERED_JSON}"

echo "[5/8] 对筛选子图进行软分类（主路径，使用 60% 子图）"
CLASSIFY_OUT_FILTERED="${OUTPUT_DIR}/head_patterns_filtered_${TOP_PERCENT}pct_${TOPK_TAG}.json"
CLASSIFY_CMD=(
  python "${PROJECT_ROOT}/tests/experiments/classify.py"
  --collapsed-json "${FILTERED_JSON}"
  --json "${IOI_JSON}"
  --model "${MODEL_NAME}"
  --device "${DEVICE}"
  --topk "${TOPK}"
  --mask-self
  --entities-only
  --entity-window 1
  --classify-threshold "${THRESH_CONC}"
  --end-entity-threshold "${END_ENTITY_TH}"
  --pattern-positions END,S2,S1,IO
  --output "${CLASSIFY_OUT_FILTERED}"
)
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  CLASSIFY_CMD+=("${MODEL_ARGS[@]}")
fi
"${CLASSIFY_CMD[@]}"

PNG_BASE="${OUTPUT_DIR}/graph_top${TOP_PERCENT}_patterns_${TOPK_TAG}"

echo "[6/8] 提炼主导模式（80% / 75%）"
DOMINANT_PREFIX="${OUTPUT_DIR}/dominant/head_patterns_filtered_${TOP_PERCENT}pct_${TOPK_TAG}"
DOMINANT_CMD=(
  python "${PROJECT_ROOT}/tests/experiments/extract_dominant_patterns.py"
  --patterns-json "${CLASSIFY_OUT_FILTERED}"
  --output-prefix "${DOMINANT_PREFIX}"
  --threshold 0.80
  --threshold 0.75
  --min-top-ratio 0.7
  --positions END,S2,S1,IO
  --graph-json "${FILTERED_JSON}"
  --visual-prefix "${PNG_BASE}"
  --visualize
)
"${DOMINANT_CMD[@]}"
DOMINANT_SOFT_THR75="${DOMINANT_PREFIX}_thr75_soft.json"
DOMINANT_SOFT_THR80="${DOMINANT_PREFIX}_thr80_soft.json"

# 渲染：改为“模式模式”，节点标签直接标注签名（如 S2->IO@+1 | END->S1@0）
PNG_OUT="${PNG_BASE}.png"
python "${PROJECT_ROOT}/tests/experiments/visualize.py" \
  --graph-json "${FILTERED_JSON}" \
  --soft-json "${CLASSIFY_OUT_FILTERED}" \
  --output "${PNG_OUT}" \
  --layout dot \
  --color-mode pattern
echo "  模式标注图: ${PNG_OUT}"

echo "[7/8] 基于路径插补细分 thr75 类别并执行分组路径修补"
PATH_PATCH_JSON="${OUTPUT_DIR}/analysis/path_patch_thr75.json"
UPDATED_SOFT_THR75="${DOMINANT_PREFIX}_thr75_soft_pathpatch.json"
GROUP_PATCH_JSON="${OUTPUT_DIR}/analysis/group_path_patch_thr75.json"
GROUP_PATCH_CMD=(
  python "${PROJECT_ROOT}/tests/experiments/group_path_patch_pipeline.py"
  --graph-json "${FILTERED_JSON}"
  --dominant-soft "${DOMINANT_SOFT_THR75}"
  --classification-json "${PATH_PATCH_JSON}"
  --updated-soft "${UPDATED_SOFT_THR75}"
  --group-patch-json "${GROUP_PATCH_JSON}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
  --n-prompts 100
)
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  GROUP_PATCH_CMD+=("${MODEL_ARGS[@]}")
fi
"${GROUP_PATCH_CMD[@]}"

PATH_PATCH_PNG="${PNG_BASE}_thr75_pathpatch.png"
python "${PROJECT_ROOT}/tests/experiments/visualize.py" \
  --graph-json "${FILTERED_JSON}" \
  --soft-json "${UPDATED_SOFT_THR75}" \
  --output "${PATH_PATCH_PNG}" \
  --layout dot \
  --color-mode pattern
echo "  路径插补细分图: ${PATH_PATCH_PNG}"

echo "[8/8] 分组路径修补完成"

# 如需对折叠图也分类，对比效果，可取消以下注释
# echo "[可选] 对折叠图进行软分类（对照）"
# CLASSIFY_OUT_COLLAPSED="${OUTPUT_DIR}/head_patterns_collapsed_${TOPK_TAG}.json"
# python "${PROJECT_ROOT}/tests/experiments/classify.py" \
#   --collapsed-json "${COLLAPSED_JSON}" \
#   --json "${IOI_JSON}" \
#   --model "gpt2-small" \
#   --device "${DEVICE}" \
#   --topk "${TOPK}" \
#   --mask-self \
#   --entities-only \
#   --entity-window 0 \
#   --classify-threshold "${THRESH_CONC}" \
#   --end-entity-threshold "${END_ENTITY_TH}" \
#   --output "${CLASSIFY_OUT_COLLAPSED}"

echo "完成。输出："
echo "  筛选图 CSV: ${FILTER_OUT_PREFIX}.csv"
echo "  筛选图 JSON: ${FILTERED_JSON}"
echo "  软分类结果(60%, ${TOPK_TAG}): ${CLASSIFY_OUT_FILTERED}"
echo "  主导模式摘要(80%): ${DOMINANT_PREFIX}_thr80.json"
echo "  主导模式摘要(75%): ${DOMINANT_PREFIX}_thr75.json"
echo "  主导模式图(80%): ${PNG_BASE}_thr80.png"
echo "  主导模式图(75%): ${PNG_BASE}_thr75.png"
echo "  路径插补结果: ${PATH_PATCH_JSON}"
echo "  路径插补 soft: ${UPDATED_SOFT_THR75}"
echo "  路径插补图: ${PATH_PATCH_PNG}"
echo "  分组路径插补结果: ${GROUP_PATCH_JSON}"
# echo "  折叠图软分类: ${CLASSIFY_OUT_COLLAPSED}"  # 若开启需同步加上 ${TOPK_TAG}
# echo "  折叠图 JSON: ${COLLAPSED_JSON}"
# echo "  折叠图软分类: ${CLASSIFY_OUT_COLLAPSED}"
