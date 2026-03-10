#!/bin/bash
set -euo pipefail

# Run terminal group patching (to logits) for agr_gender.
# This patches (simultaneously) the subset of heads in each group that have direct head->logits edges.

CONDA_ENV="eap-ig"
DEVICE="cuda"
DATA_PATH="results/agr_gender/standard_gender_data.csv"
GROUP_GRAPH="results/agr_gender_gpt2_new/group_graph_thr0_001_thr75_layered_nocycle.json"
JOINT="results/agr_gender_gpt2_new/joint_p_qkv.json"
OUT="results/agr_gender_gpt2_new/analysis/group_patch_to_logits_terminal.json"
DATASET_SIZE=200
BATCH_SIZE=16
MIN_ABS=0.0
ONLY_GROUPS=""

usage() {
  cat <<'EOF'
用法: run_group_patch_to_logits_agr_gender.sh [选项]
  --data-path PATH          CSV 数据集（含 clean/corrupted/label）
  --group-graph PATH        分组图 JSON（layered_nocycle）
  --joint PATH              joint_p_qkv.json
  --output PATH             输出 JSON
  --dataset-size N          样本数（默认 200，0=全部）
  --batch-size N            batch 大小（默认 16）
  --min-abs-logits-edge X   只 patch |delta|>=X 的 head->logits 头（默认 0）
  --only-groups LIST        仅跑指定 group_id（逗号分隔，可选）
  --device DEV              cuda/cpu
  --conda-env NAME          conda 环境名（默认 eap-ig）
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2;;
    --group-graph) GROUP_GRAPH="$2"; shift 2;;
    --joint) JOINT="$2"; shift 2;;
    --output) OUT="$2"; shift 2;;
    --dataset-size) DATASET_SIZE="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --min-abs-logits-edge) MIN_ABS="$2"; shift 2;;
    --only-groups) ONLY_GROUPS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1" >&2; usage; exit 1;;
  esac
done

CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$OUT")"

python tests/experiments/group_patch_to_logits_agr_gender.py \
  --group-graph "$GROUP_GRAPH" \
  --joint "$JOINT" \
  --data-path "$DATA_PATH" \
  --output "$OUT" \
  --device "$DEVICE" \
  --dataset-size "$DATASET_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --min-abs-logits-edge "$MIN_ABS" \
  --only-groups "$ONLY_GROUPS"

echo "✅ Done: $OUT"

