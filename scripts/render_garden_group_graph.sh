#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${REPO_ROOT}/results/garden_0412"
DATA_PATH="${REPO_ROOT}/datasets/garden/garden_npz_v_trans_mod.csv"
STANDARD_JSON="${REPO_ROOT}/results/garden_0131/standard_garden_data.json"
MODEL_NAME="gpt2"
MODEL_PATH="/home/wangziran/gpt2"
DEVICE="cuda"

TOPN="800"
IG_STEPS="5"
BATCH_SIZE="8"
LIMIT="300"
DROP_INPUT="0"
DROP_LOGITS="0"
FIND_BEST="0"

CLASSIFY_TOPK="2"
CLASSIFY_SAMPLE_SIZE="100"
CLASSIFY_QUERY_POSITIONS="END"
CLASSIFY_KEY_POSITIONS="SUBJ,VERB,OBJ_HEAD,REL_PRON,REL_VERB,END"
CLASSIFY_WINDOW="0"
CLASSIFY_MASK_SELF="1"

DOMINANT_THRESHOLDS="0.75,0.80"
DOMINANT_POSITIONS=""

LAYER_BUCKETS="0-4,5-8,9-11"
GROUP_MIN_EDGE_ABS_SCORE="0.0"
GROUP_LAYOUT="dot"

JOINT_JSON=""
AUTO_JOINT="0"
JOINT_OUTPUT_NAME="joint_p.json"
JOINT_DATASET_SIZE="64"
JOINT_SEED="1"
JOINT_CHANNELS="q,k,v"
JOINT_PATCH="1"
JOINT_ONLY="0"
BREAK_AGG="mean"
BREAK_MIN_ABS_WEIGHT="0.0"
BREAK_MAX_DELETIONS="2000"
PRUNE_MIN_ABS_WEIGHT="0.001"
PRUNE_EDGE_SCOPE="internal"
DROP_ISOLATED="0"

PP_GROUPS_PROMPTS="50"
PP_GROUPS_SEED="1"

usage() {
  cat <<EOF
Usage: bash scripts/render_garden_group_graph.sh [options]

Core options:
  --results-root PATH
  --data-path PATH
  --standard-json PATH
  --model-name NAME
  --model-path PATH
  --device DEVICE

Graph build:
  --topn N
  --ig-steps N
  --batch-size N
  --limit N
  --find-best
  --drop-input
  --drop-logits

Classify / dominant:
  --classify-topk N
  --classify-sample-size N
  --classify-query-positions CSV
  --classify-key-positions CSV
  --classify-window N
  --no-mask-self
  --dominant-thresholds CSV
  --dominant-positions CSV

Group graph:
  --layer-buckets CSV
  --group-min-edge-abs-score FLOAT
  --layout NAME

Optional delta-weighted postprocess:
  --joint-json PATH
  --auto-joint
  --joint-output-name NAME
  --joint-dataset-size N
  --joint-seed N
  --joint-channels CSV
  --no-joint-patch
  --joint-only
  --break-agg mean|max_abs
  --break-min-abs-weight FLOAT
  --break-max-deletions N
  --prune-min-abs-weight FLOAT
  --prune-edge-scope internal|all
  --drop-isolated
  --pp-groups-prompts N
  --pp-groups-seed N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --standard-json) STANDARD_JSON="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --topn) TOPN="$2"; shift 2 ;;
    --ig-steps) IG_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --find-best) FIND_BEST="1"; shift ;;
    --drop-input) DROP_INPUT="1"; shift ;;
    --drop-logits) DROP_LOGITS="1"; shift ;;
    --classify-topk) CLASSIFY_TOPK="$2"; shift 2 ;;
    --classify-sample-size) CLASSIFY_SAMPLE_SIZE="$2"; shift 2 ;;
    --classify-query-positions) CLASSIFY_QUERY_POSITIONS="$2"; shift 2 ;;
    --classify-key-positions) CLASSIFY_KEY_POSITIONS="$2"; shift 2 ;;
    --classify-window) CLASSIFY_WINDOW="$2"; shift 2 ;;
    --no-mask-self) CLASSIFY_MASK_SELF="0"; shift ;;
    --dominant-thresholds) DOMINANT_THRESHOLDS="$2"; shift 2 ;;
    --dominant-positions) DOMINANT_POSITIONS="$2"; shift 2 ;;
    --layer-buckets) LAYER_BUCKETS="$2"; shift 2 ;;
    --group-min-edge-abs-score) GROUP_MIN_EDGE_ABS_SCORE="$2"; shift 2 ;;
    --layout) GROUP_LAYOUT="$2"; shift 2 ;;
    --joint-json) JOINT_JSON="$2"; shift 2 ;;
    --auto-joint) AUTO_JOINT="1"; shift ;;
    --joint-output-name) JOINT_OUTPUT_NAME="$2"; shift 2 ;;
    --joint-dataset-size) JOINT_DATASET_SIZE="$2"; shift 2 ;;
    --joint-seed) JOINT_SEED="$2"; shift 2 ;;
    --joint-channels) JOINT_CHANNELS="$2"; shift 2 ;;
    --no-joint-patch) JOINT_PATCH="0"; shift ;;
    --joint-only) JOINT_ONLY="1"; shift ;;
    --break-agg) BREAK_AGG="$2"; shift 2 ;;
    --break-min-abs-weight) BREAK_MIN_ABS_WEIGHT="$2"; shift 2 ;;
    --break-max-deletions) BREAK_MAX_DELETIONS="$2"; shift 2 ;;
    --prune-min-abs-weight) PRUNE_MIN_ABS_WEIGHT="$2"; shift 2 ;;
    --prune-edge-scope) PRUNE_EDGE_SCOPE="$2"; shift 2 ;;
    --drop-isolated) DROP_ISOLATED="1"; shift ;;
    --pp-groups-prompts) PP_GROUPS_PROMPTS="$2"; shift 2 ;;
    --pp-groups-seed) PP_GROUPS_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export EAP_ROOT="${EAP_ROOT:-/home/wangziran/eap-ig}"

mkdir -p "${RESULTS_ROOT}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Dataset not found: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${STANDARD_JSON}" ]]; then
  echo "standard_garden_data.json not found, generating at ${STANDARD_JSON}"
  python "${REPO_ROOT}/tests/experiments/generate_garden_standard.py" \
    --data-path "${DATA_PATH}" \
    --output "${STANDARD_JSON}"
fi

GRAPH_DIR="${RESULTS_ROOT}/graph_build"
mkdir -p "${GRAPH_DIR}"

BEST_EDGE="${TOPN}"
if [[ "${FIND_BEST}" == "1" ]]; then
  echo "[1/7] Running findbestedge.py"
  FIND_LOG="${RESULTS_ROOT}/findbestedge.log"
  python "${REPO_ROOT}/pipeline/findbestedge.py" \
    --data_file "${DATA_PATH}" \
    --output_dir "${RESULTS_ROOT}/findbestedge" | tee "${FIND_LOG}"
  BEST_EDGE="$(python - <<'PY' "${FIND_LOG}"
import re, sys
text = open(sys.argv[1]).read()
m = re.findall(r"推荐的最佳边数:\s*([0-9]+)", text)
print(m[-1] if m else "")
PY
)"
  if [[ -z "${BEST_EDGE}" ]]; then
    echo "Failed to parse recommended edge count from ${FIND_LOG}" >&2
    exit 1
  fi
fi

GRAPH_JSON="${GRAPH_DIR}/graph.json"
COLLAPSED_JSON="${GRAPH_DIR}/graph_collapsed.json"

if [[ ! -f "${GRAPH_JSON}" || ! -f "${COLLAPSED_JSON}" ]]; then
  echo "[2/7] Building graph and collapsed graph (topn=${BEST_EDGE})"
  CMD=(
    python "${REPO_ROOT}/run_garden_gpt2.py"
    --dataset "${DATA_PATH}"
    --output-dir "${GRAPH_DIR}"
    --batch-size "${BATCH_SIZE}"
    --topn "${BEST_EDGE}"
    --ig-steps "${IG_STEPS}"
    --device "${DEVICE}"
    --model-name "${MODEL_NAME}"
    --model-path "${MODEL_PATH}"
  )
  if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi
  if [[ "${DROP_INPUT}" == "1" ]]; then
    CMD+=(--drop-input)
  fi
  if [[ "${DROP_LOGITS}" == "1" ]]; then
    CMD+=(--drop-logits)
  fi
  "${CMD[@]}"
else
  echo "[2/7] Reusing existing graph files in ${GRAPH_DIR}"
fi

CLASSIFY_JSON="${RESULTS_ROOT}/classify_garden.json"
echo "[3/7] Running classify_garden.py"
CLASSIFY_CMD=(
  python "${REPO_ROOT}/tests/experiments/classify_garden.py"
  --collapsed-json "${COLLAPSED_JSON}"
  --standard-json "${STANDARD_JSON}"
  --model "${MODEL_NAME}"
  --model-path "${MODEL_PATH}"
  --device "${DEVICE}"
  --topk "${CLASSIFY_TOPK}"
  --window "${CLASSIFY_WINDOW}"
  --sample-size "${CLASSIFY_SAMPLE_SIZE}"
  --query-positions "${CLASSIFY_QUERY_POSITIONS}"
  --key-positions "${CLASSIFY_KEY_POSITIONS}"
  --output "${CLASSIFY_JSON}"
)
if [[ "${CLASSIFY_MASK_SELF}" == "1" ]]; then
  CLASSIFY_CMD+=(--mask-self)
fi
"${CLASSIFY_CMD[@]}"

echo "[4/7] Extracting dominant patterns"
DOM_PREFIX="${RESULTS_ROOT}/dominant/head_patterns"
mkdir -p "${RESULTS_ROOT}/dominant"
DOM_CMD=(
  python "${REPO_ROOT}/tests/experiments/extract_dominant_patterns_garden.py"
  --patterns-json "${CLASSIFY_JSON}"
  --output-prefix "${DOM_PREFIX}"
)
IFS=',' read -r -a _thr_list <<< "${DOMINANT_THRESHOLDS}"
for thr in "${_thr_list[@]}"; do
  DOM_CMD+=(--threshold "${thr}")
done
if [[ -n "${DOMINANT_POSITIONS}" ]]; then
  DOM_CMD+=(--positions "${DOMINANT_POSITIONS}")
fi
"${DOM_CMD[@]}"

PRIMARY_THR="$(python - <<'PY' "${DOMINANT_THRESHOLDS}"
import sys
parts=[p.strip() for p in sys.argv[1].split(",") if p.strip()]
thr=float(parts[0])
print(f"thr{int(round(thr*100)):02d}")
PY
)"
DOM_SOFT="${DOM_PREFIX}_${PRIMARY_THR}_soft.json"
PP_GROUPS_JSON="${RESULTS_ROOT}/path_patch_${PRIMARY_THR}.json"
PP_UPDATED_SOFT="${DOM_PREFIX}_${PRIMARY_THR}_soft_pathpatch.json"

if [[ -z "${JOINT_JSON}" && "${AUTO_JOINT}" == "1" ]]; then
  JOINT_JSON="${RESULTS_ROOT}/${JOINT_OUTPUT_NAME}"
fi

echo "[5/7] Running dominant-group path patch classification"
python "${REPO_ROOT}/tests/experiments/path_patch_dominant_groups_garden.py" \
  --dominant-soft "${DOM_SOFT}" \
  --output-json "${PP_GROUPS_JSON}" \
  --updated-soft "${PP_UPDATED_SOFT}" \
  --data-path "${DATA_PATH}" \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --n-prompts "${PP_GROUPS_PROMPTS}" \
  --seed "${PP_GROUPS_SEED}"

GROUP_JSON="${RESULTS_ROOT}/group_graph_${PRIMARY_THR}.json"
GROUP_PNG="${RESULTS_ROOT}/group_graph_${PRIMARY_THR}.png"

echo "[5b/7] Rendering grouped graph"
GROUP_CMD=(
  python "${REPO_ROOT}/tests/experiments/group_graph_garden.py"
  --graph-json "${GRAPH_JSON}"
  --dominant-soft "${PP_UPDATED_SOFT}"
  --min-edge-abs-score "${GROUP_MIN_EDGE_ABS_SCORE}"
  --layer-buckets "${LAYER_BUCKETS}"
  --output-json "${GROUP_JSON}"
  --output-png "${GROUP_PNG}"
  --layout "${GROUP_LAYOUT}"
)
if [[ -n "${JOINT_JSON}" ]]; then
  GROUP_CMD+=(--edge-weights-json "${JOINT_JSON}")
fi
"${GROUP_CMD[@]}"

if [[ -n "${JOINT_JSON}" && ! -f "${JOINT_JSON}" ]]; then
  echo "[6/7] Generating joint edge path patch weights"
  JOINT_CMD=(
    python "${REPO_ROOT}/tests/experiments/edge_path_patching_garden.py"
    --collapsed-graph "${COLLAPSED_JSON}"
    --output "${JOINT_JSON}"
    --data-path "${DATA_PATH}"
    --dataset-size "${JOINT_DATASET_SIZE}"
    --seed "${JOINT_SEED}"
    --device "${DEVICE}"
  )
  IFS=',' read -r -a _joint_channels <<< "${JOINT_CHANNELS}"
  if [[ "${JOINT_ONLY}" != "1" ]]; then
    JOINT_CMD+=(--channels "${_joint_channels[@]}")
  else
    JOINT_CMD+=(--channels q k v)
  fi
  if [[ "${JOINT_PATCH}" == "1" ]]; then
    JOINT_CMD+=(--joint-patch)
  fi
  if [[ "${JOINT_ONLY}" == "1" ]]; then
    JOINT_CMD+=(--joint-only)
  fi
  "${JOINT_CMD[@]}"
fi

if [[ -n "${JOINT_JSON}" ]]; then
  echo "[6/7] Breaking cycles with delta weights"
  DAG_JSON="${RESULTS_ROOT}/group_graph_${PRIMARY_THR}_dagish.json"
  DAG_PNG="${RESULTS_ROOT}/group_graph_${PRIMARY_THR}_dagish.png"
  python "${REPO_ROOT}/tests/experiments/break_cycles_group_graph_by_delta.py" \
    --group-json "${GROUP_JSON}" \
    --joint-json "${JOINT_JSON}" \
    --output-json "${DAG_JSON}" \
    --output-png "${DAG_PNG}" \
    --agg "${BREAK_AGG}" \
    --min-abs-weight "${BREAK_MIN_ABS_WEIGHT}" \
    --max-deletions "${BREAK_MAX_DELETIONS}" \
    --layout "${GROUP_LAYOUT}"

  echo "[7/7] Pruning grouped graph by delta"
  PRUNED_JSON="${RESULTS_ROOT}/group_graph_${PRIMARY_THR}_dagish_pruned.json"
  PRUNED_PNG="${RESULTS_ROOT}/group_graph_${PRIMARY_THR}_dagish_pruned.png"
  PRUNE_CMD=(
    python "${REPO_ROOT}/tests/experiments/prune_group_graph_by_delta.py"
    --group-json "${DAG_JSON}"
    --joint-json "${JOINT_JSON}"
    --output-json "${PRUNED_JSON}"
    --output-png "${PRUNED_PNG}"
    --min-abs-weight "${PRUNE_MIN_ABS_WEIGHT}"
    --edge-scope "${PRUNE_EDGE_SCOPE}"
  )
  if [[ "${DROP_ISOLATED}" == "1" ]]; then
    PRUNE_CMD+=(--drop-isolated)
  fi
  "${PRUNE_CMD[@]}"
else
  echo "[6/7] Skipping cycle-break and prune because --joint-json was not provided"
  echo "[7/7] Done"
fi

echo
echo "Graph build dir: ${GRAPH_DIR}"
echo "Classify JSON: ${CLASSIFY_JSON}"
echo "Dominant soft: ${DOM_SOFT}"
echo "Path-patch group classification: ${PP_GROUPS_JSON}"
echo "Updated dominant soft: ${PP_UPDATED_SOFT}"
echo "Grouped graph JSON: ${GROUP_JSON}"
echo "Grouped graph PNG: ${GROUP_PNG}"
if [[ -n "${JOINT_JSON}" ]]; then
  echo "Joint edge weights: ${JOINT_JSON}"
  echo "DAG-ish grouped graph: ${RESULTS_ROOT}/group_graph_${PRIMARY_THR}_dagish.json"
  echo "Pruned grouped graph: ${RESULTS_ROOT}/group_graph_${PRIMARY_THR}_dagish_pruned.json"
fi
