#!/bin/bash
set -euo pipefail

LAYER=0
HEAD=0
ROUNDS=5
TYPENAME="gender_terminal_head"
CONDA_ENV="eap-ig"
RESULTS_ROOT="/data31/private/wangziran/eap_auto/results/agr_gender"
STANDARD_JSON=""
ATTENTION_POSITION="end"
DATA_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
DEVICE="cuda"
DATASET_SIZE=200
FORCE=0
OUTPUT_FAMILY="Terminal_2"

usage() {
  cat <<'EOF'
用法: run_gender_terminal_head_token.sh [选项]
  --layer L                 sender layer
  --head H                  sender head
  --rounds N                迭代轮数(默认5)
  --typename NAME           typename 传参
  --results-root PATH       results/agr_gender 根目录
  --standard-json PATH      standard_gender_data.json 路径
  --attention-position POS  end/verb/a1/b/a2（按图标注选择）
  --dataset-size N          计算logit effects的样本数(默认200, 0=全部)
  --data-dir PATH           覆写 path_patching 子目录
  --output-dir PATH         覆写输出目录
  --output-family NAME      输出子目录名(默认 Terminal_2)
  --device DEV              cuda/cpu
  --force                   忽略缓存，强制重新计算所有步骤
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layer) LAYER="$2"; shift 2;;
    --head) HEAD="$2"; shift 2;;
    --rounds) ROUNDS="$2"; shift 2;;
    --typename) TYPENAME="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --standard-json) STANDARD_JSON="$2"; shift 2;;
    --attention-position) ATTENTION_POSITION="$2"; shift 2;;
    --dataset-size) DATASET_SIZE="$2"; shift 2;;
    --data-dir) DATA_DIR_OVERRIDE="$2"; shift 2;;
    --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
    --output-family) OUTPUT_FAMILY="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --force) FORCE=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1" >&2; usage; exit 1;;
  esac
done

REPO_ROOT="/data31/private/wangziran/eap_auto"
BASE_RESULTS_DIR="$RESULTS_ROOT"
if [[ -z "$STANDARD_JSON" ]]; then
  STANDARD_JSON="$BASE_RESULTS_DIR/standard_gender_data.json"
fi
head_str="${LAYER}.${HEAD}"

if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
  DATA_SOURCE_DIR="$DATA_DIR_OVERRIDE"
else
  DATA_SOURCE_DIR="$BASE_RESULTS_DIR/path_patching/${OUTPUT_FAMILY}/${LAYER}_${HEAD}"
fi
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
  OUTPUT_DIR="$BASE_RESULTS_DIR/hypothesis/${OUTPUT_FAMILY}/${LAYER}.${HEAD}_$(date +%Y%m%d_%H%M%S)"
fi

RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
ATTENTION_GT_FILE="$OUTPUT_DIR/attention_scores_ground_truth.jsonl"
LOGIT_OUTPUT_FILE="$DATA_SOURCE_DIR/final_logit_effects_head_${LAYER}_${HEAD}.json"
CAUSAL_OUTPUT_FILE="$DATA_SOURCE_DIR/causal_effects_${LAYER}_${HEAD}.json"
RESULT_FILE="$OUTPUT_DIR/final_result_round_${ROUNDS}.json"
BEST_SUMMARY_FILE="$OUTPUT_DIR/best_hypothesis.json"
LEGACY_DATA_SOURCE_DIR="$BASE_RESULTS_DIR/path_patching/Terminal/${LAYER}_${HEAD}"
LEGACY_RAW_ATTENTION_FILE="$LEGACY_DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"

CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

file_ready() { [ -s "$1" ]; }

causal_ids_ready() {
  # Returns 0 iff causal_effects JSON list contains at least one non-empty sentence_id.
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
if not isinstance(data, list):
    sys.exit(1)
for item in data:
    sid = str(item.get("sentence_id") or "").strip()
    if sid:
        sys.exit(0)
sys.exit(1)
PY
}

echo "=== Step 1: precompute raw attention for head ${head_str} (${ATTENTION_POSITION}) ==="
if [ "$FORCE" -eq 0 ] && file_ready "$RAW_ATTENTION_FILE"; then
  echo "↪ 已存在，跳过: $RAW_ATTENTION_FILE"
else
  # Reuse legacy Terminal raw attention if it matches the requested attention position.
  if [ "$FORCE" -eq 0 ] && file_ready "$LEGACY_RAW_ATTENTION_FILE"; then
    if python - "$LEGACY_RAW_ATTENTION_FILE" "$ATTENTION_POSITION" <<'PY'
import json, sys
path, want = sys.argv[1], sys.argv[2]
data = json.loads(open(path, "r", encoding="utf-8").read())
pos = None
if isinstance(data, list) and data:
    pos = data[0].get("attention_position")
print("OK" if (pos == want) else "NO")
PY
    then
      echo "↪ 复用旧 Terminal 注意力缓存(同位置): $LEGACY_RAW_ATTENTION_FILE"
      mkdir -p "$DATA_SOURCE_DIR"
      cp -f "$LEGACY_RAW_ATTENTION_FILE" "$RAW_ATTENTION_FILE"
    fi
  fi
  if [ "$FORCE" -eq 0 ] && file_ready "$RAW_ATTENTION_FILE"; then
    echo "↪ 已就绪，跳过计算: $RAW_ATTENTION_FILE"
  else
  python "$REPO_ROOT/tests/experiments/precompute_attention_scores_agr_gender.py" \
    --head "$head_str" \
    --input_json "$STANDARD_JSON" \
    --output_file "$RAW_ATTENTION_FILE" \
    --attention-position "$ATTENTION_POSITION" \
    --device "$DEVICE" \
    --topk 10
  fi
fi

echo "=== Step 2: export attention_scores_ground_truth.jsonl ==="
if [ "$FORCE" -eq 0 ] && file_ready "$ATTENTION_GT_FILE"; then
  echo "↪ 已存在，跳过: $ATTENTION_GT_FILE"
else
  python - "$RAW_ATTENTION_FILE" "$ATTENTION_GT_FILE" <<'PY'
import json, pathlib, sys
raw = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = []
for item in raw:
    sid = str(item.get("sample_id", ""))
    out.append({
        "key": sid,
        "original_sentence": item.get("sentence_text",""),
        "attention_scores": item.get("top_attended_tokens", []),
    })
pathlib.Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
PY
fi

echo "=== Step 3: precompute IOI-style token logit effects ==="
if [ "$FORCE" -eq 0 ] && file_ready "$LOGIT_OUTPUT_FILE"; then
  echo "↪ 已存在，跳过: $LOGIT_OUTPUT_FILE"
else
  python "$REPO_ROOT/tests/experiments/precompute_gender_logit_effects.py" \
    --input_file "$STANDARD_JSON" \
    --output_file "$LOGIT_OUTPUT_FILE" \
    --head_to_patch "$head_str" \
    --max_samples "$DATASET_SIZE" \
    --device "$DEVICE"
fi

echo "=== Step 4: preprocess causal effects (increase/decrease tokens) ==="
if [ "$FORCE" -eq 0 ] && file_ready "$CAUSAL_OUTPUT_FILE"; then
  echo "↪ 已存在，跳过: $CAUSAL_OUTPUT_FILE"
else
  python "$REPO_ROOT/tests/experiments/preprocess_causal_effects.py" \
    --input "$LOGIT_OUTPUT_FILE" \
    --output "$CAUSAL_OUTPUT_FILE" \
    --top_k 1
fi

# Safety: older cached causal files may have empty sentence_id (no overlap with attention keys).
if [ "$FORCE" -eq 0 ] && file_ready "$CAUSAL_OUTPUT_FILE" && ! causal_ids_ready "$CAUSAL_OUTPUT_FILE"; then
  echo "⚠️ 检测到 $CAUSAL_OUTPUT_FILE 的 sentence_id 全为空，自动强制重算 Step 3-4 以修复对齐问题。"
  python "$REPO_ROOT/tests/experiments/precompute_gender_logit_effects.py" \
    --input_file "$STANDARD_JSON" \
    --output_file "$LOGIT_OUTPUT_FILE" \
    --head_to_patch "$head_str" \
    --max_samples "$DATASET_SIZE" \
    --device "$DEVICE"
  python "$REPO_ROOT/tests/experiments/preprocess_causal_effects.py" \
    --input "$LOGIT_OUTPUT_FILE" \
    --output "$CAUSAL_OUTPUT_FILE" \
    --top_k 1
fi

echo "=== Step 5: run auto_gender_terminal_token.py ==="
if [ "$FORCE" -eq 0 ] && file_ready "$RESULT_FILE"; then
  echo "↪ 已存在，跳过: $RESULT_FILE"
else
  python "$REPO_ROOT/tests/experiments/auto_gender_terminal_token.py" \
    --layer "$LAYER" \
    --head "$HEAD" \
    --rounds "$ROUNDS" \
    --typename "$TYPENAME" \
    --output_dir "$OUTPUT_DIR" \
    --data_source_dir "$DATA_SOURCE_DIR" \
    --max-iters "$ROUNDS"
fi

if [ -f "$RESULT_FILE" ]; then
  python "$REPO_ROOT/tests/experiments/save_best_hypothesis.py" \
    --result_file "$RESULT_FILE" \
    --output_file "$BEST_SUMMARY_FILE"
else
  echo "⚠️ 未找到结果文件 $RESULT_FILE，跳过最佳假设摘要。"
fi

echo "✅ agr_gender terminal (token-level) pipeline finished. Results saved to $OUTPUT_DIR"
