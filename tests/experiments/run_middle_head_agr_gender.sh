#!/bin/bash
set -euo pipefail

#############################################
# agr_gender middle head pipeline
#  - direct attention ground truth for sender head
#  - sender->receiver diff_vectors causal dataset
#  - run auto_gender_s_att.py (Claude)
#############################################

LAYER=0
HEAD=0
RECEIVER_HEADS=""
TARGET_HEADS=""
ROUNDS=5
TYPENAME="middle_head_agr_gender"
CONDA_ENV="eap-ig"
RESULTS_ROOT="/data31/private/wangziran/eap_auto/results/agr_gender"
STANDARD_JSON=""
ATTENTION_POSITION="end"
RECEIVER_DESC_FILES=""
TARGET_DESC_FILES=""
DATA_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
DEVICE="cuda"
FORCE=0
USE_ABC=0
GROUP_PATCH=1
GROUP_AVG_ONLY=0

usage() {
  cat <<'EOF'
用法: run_middle_head_agr_gender.sh [选项]
  --layer L                 发送头所在层
  --head H                  发送头编号
  --receiver-heads LIST     逗号分隔的下游头列表
  --target-heads LIST       逗号分隔的目标头列表（A->B->C 用）
  --rounds N                auto_s_att 迭代轮次 (默认: 5)
  --typename NAME           typename 传参
  --conda-env NAME          Conda 环境名
  --results-root PATH       results/agr_gender 根目录
  --standard-json PATH      standard_gender_data.json 路径
  --attention-position POS  query 位置 (end/verb/a1/b/a2)
  --receiver-desc FILES     head:path,head:path 的 best_hypothesis.json 列表（可选）
  --target-desc FILES       head:path,head:path 的 target best_hypothesis.json 列表（可选）
  --data-dir PATH           覆写 path_patching 子目录
  --output-dir PATH         覆写输出目录
  --device DEV              cuda/cpu (默认 cuda)
  --force                   忽略缓存，强制重新计算所有步骤
  --use-abc                 使用 A->B->C 的 diff_vectors_abc 进行因果评估
  --group-patch             一次性 patch 整个 receiver 组（默认）
  --group-avg-only          只保存组平均 diff（不保存每个 receiver）
  -h, --help                显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layer) LAYER="$2"; shift 2;;
    --head) HEAD="$2"; shift 2;;
    --receiver-heads) RECEIVER_HEADS="$2"; shift 2;;
    --target-heads) TARGET_HEADS="$2"; shift 2;;
    --rounds) ROUNDS="$2"; shift 2;;
    --typename) TYPENAME="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --standard-json) STANDARD_JSON="$2"; shift 2;;
    --attention-position) ATTENTION_POSITION="$2"; shift 2;;
    --receiver-desc) RECEIVER_DESC_FILES="$2"; shift 2;;
    --target-desc) TARGET_DESC_FILES="$2"; shift 2;;
    --data-dir) DATA_DIR_OVERRIDE="$2"; shift 2;;
    --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --force) FORCE=1; shift 1;;
    --use-abc) USE_ABC=1; shift 1;;
    --group-patch) GROUP_PATCH=1; shift 1;;
    --group-avg-only) GROUP_AVG_ONLY=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "${RECEIVER_HEADS}" ]]; then
  echo "❌ --receiver-heads 不能为空" >&2
  exit 1
fi

REPO_ROOT="/data31/private/wangziran/eap_auto"
BASE_RESULTS_DIR="$RESULTS_ROOT"
if [[ -z "$STANDARD_JSON" ]]; then
  STANDARD_JSON="$BASE_RESULTS_DIR/standard_gender_data.json"
fi

head_str="${LAYER}.${HEAD}"
if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
  DATA_SOURCE_DIR="$DATA_DIR_OVERRIDE"
else
  DATA_SOURCE_DIR="$BASE_RESULTS_DIR/path_patching/Middle_Head/${LAYER}_${HEAD}"
fi
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
  OUTPUT_DIR="$BASE_RESULTS_DIR/hypothesis/Middle_Head/${LAYER}.${HEAD}_$(date +%Y%m%d_%H%M)"
fi

RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
PREPROCESSED_SAMPLING_FILE="$DATA_SOURCE_DIR/preprocessed_for_sampling.jsonl"
PREPROCESSED_ATTENTION_FILE="$DATA_SOURCE_DIR/preprocessed_attention_scores.json"
MIDDLE_DIFF_FILE="$DATA_SOURCE_DIR/causal_dataset_${LAYER}_${HEAD}.json"
ATTENTION_GT_FILE="$OUTPUT_DIR/attention_scores_ground_truth.jsonl"
RECEIVER_DESC_OUTPUT="$DATA_SOURCE_DIR/receiver_descriptions.json"
RESULT_FILE="$OUTPUT_DIR/final_result_round_${ROUNDS}.json"
BEST_SUMMARY_FILE="$OUTPUT_DIR/best_hypothesis.json"

CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

file_ready() {
  [ -s "$1" ]
}

echo "=== Step 1: precompute raw attention for head ${head_str} (${ATTENTION_POSITION}) ==="
if [ "$FORCE" -eq 0 ] && file_ready "$RAW_ATTENTION_FILE"; then
  echo "↪ 已存在，跳过: $RAW_ATTENTION_FILE"
else
  python "$REPO_ROOT/tests/experiments/precompute_attention_scores_agr_gender.py" \
    --head "$head_str" \
    --input_json "$STANDARD_JSON" \
    --output_file "$RAW_ATTENTION_FILE" \
    --attention-position "$ATTENTION_POSITION" \
    --device "$DEVICE" \
    --topk 10
fi

echo "=== Step 2: convert raw attention to sampling jsonl ==="
if [ "$FORCE" -eq 0 ] && file_ready "$PREPROCESSED_SAMPLING_FILE"; then
  echo "↪ 已存在，跳过: $PREPROCESSED_SAMPLING_FILE"
else
  python - "$RAW_ATTENTION_FILE" "$PREPROCESSED_SAMPLING_FILE" <<'PY'
import json, pathlib, sys
raw_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])
data = json.loads(raw_path.read_text())
with dst_path.open("w", encoding="utf-8") as out_f:
    for item in data:
        out_f.write(json.dumps({
            "sentence_id": str(item.get("sample_id", "")),
            "original_sentence": item["sentence_text"],
            "number_of_important_tokens": len(item.get("top_attended_tokens", [])),
            "attention_scores": [
                {"token": tok.get("token", "").strip(), "position": tok.get("position", -1), "score": tok.get("score", 0.0)}
                for tok in item.get("top_attended_tokens", [])
            ],
        }, ensure_ascii=False) + "\n")
PY
fi

echo "=== Step 3: build preprocessed_attention_scores.json ==="
if [ "$FORCE" -eq 0 ] && file_ready "$PREPROCESSED_ATTENTION_FILE"; then
  echo "↪ 已存在，跳过: $PREPROCESSED_ATTENTION_FILE"
else
  python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
    --input "$PREPROCESSED_SAMPLING_FILE" \
    --output "$PREPROCESSED_ATTENTION_FILE" \
    --top_k 5
fi

echo "=== Step 4: export attention_scores_ground_truth.jsonl for auto_s_att ==="
if [ "$FORCE" -eq 0 ] && file_ready "$ATTENTION_GT_FILE"; then
  echo "↪ 已存在，跳过: $ATTENTION_GT_FILE"
else
  python - "$PREPROCESSED_SAMPLING_FILE" "$ATTENTION_GT_FILE" <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
records = []
with src.open() as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        key = item.get("sentence_id") or str(idx)
        records.append({
            "key": key,
            "original_sentence": item["original_sentence"],
            "attention_scores": item["attention_scores"],
        })
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
PY
fi

if [ -n "${RECEIVER_DESC_FILES}" ]; then
  echo "=== Step 4b: assemble receiver descriptions ==="
  python - "$RECEIVER_DESC_FILES" "$RECEIVER_DESC_OUTPUT" <<'PY'
import json, sys, pathlib
mapping = {}
pairs = [p.strip() for p in sys.argv[1].split(",") if p.strip()]
for pair in pairs:
    if ":" not in pair:
        continue
    head, path = pair.split(":", 1)
    head = head.strip()
    path = pathlib.Path(path.strip())
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "best_hypothesis" in data:
            mapping[head] = str(data.get("best_hypothesis", "")).strip()
    except Exception as e:
        print(f"Warning: failed to load receiver description from {path}: {e}")
if mapping:
    out_path = pathlib.Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    print("Warning: no receiver descriptions gathered.")
PY
else
  RECEIVER_DESC_OUTPUT=""
fi

echo "=== Step 5: compute sender->receiver diff vectors (agr_gender) ==="
if [ "$FORCE" -eq 0 ] && file_ready "$MIDDLE_DIFF_FILE"; then
  echo "↪ 已存在，跳过: $MIDDLE_DIFF_FILE"
else
  python "$REPO_ROOT/tests/experiments/precompute_middle_head_agr_gender.py" \
    --sender_head "$head_str" \
    --receiver_heads "$RECEIVER_HEADS" \
    ${TARGET_HEADS:+--target_heads "$TARGET_HEADS"} \
    ${GROUP_PATCH:+--group_patch} \
    ${GROUP_AVG_ONLY:+--group_avg_only} \
    --data_json "$STANDARD_JSON" \
    --attention_position "$ATTENTION_POSITION" \
    --output_file "$MIDDLE_DIFF_FILE" \
    --receiver_input "q" \
    --device "$DEVICE"
fi

RECEIVER_DESC_FLAG=""
RECEIVER_DESC_VALUE=""
TARGET_DESC_FLAG=""
TARGET_DESC_VALUE=""
if [ -n "${RECEIVER_DESC_FILES}" ] && [ -s "$RECEIVER_DESC_OUTPUT" ]; then
  RECEIVER_DESC_FLAG="--receiver_descriptions_file"
  RECEIVER_DESC_VALUE="$RECEIVER_DESC_OUTPUT"
fi

TARGET_DESC_OUTPUT="$DATA_SOURCE_DIR/target_descriptions.json"
if [ -n "${TARGET_DESC_FILES}" ]; then
  echo "=== Step 4c: assemble target descriptions ==="
  python - "$TARGET_DESC_FILES" "$TARGET_DESC_OUTPUT" <<'PY'
import json, sys, pathlib
mapping = {}
pairs = [p.strip() for p in sys.argv[1].split(",") if p.strip()]
for pair in pairs:
    if ":" not in pair:
        continue
    head, path = pair.split(":", 1)
    head = head.strip()
    path = pathlib.Path(path.strip())
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "best_hypothesis" in data:
            mapping[head] = str(data.get("best_hypothesis", "")).strip()
    except Exception as e:
        print(f"Warning: failed to load target description from {path}: {e}")
if mapping:
    out_path = pathlib.Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    print("Warning: no target descriptions gathered.")
PY
fi
if [ -s "$TARGET_DESC_OUTPUT" ]; then
  TARGET_DESC_FLAG="--target_descriptions_file"
  TARGET_DESC_VALUE="$TARGET_DESC_OUTPUT"
fi

echo "=== Step 6: run auto_gender_s_att.py for ${head_str} ==="
if [ "$FORCE" -eq 0 ] && file_ready "$RESULT_FILE"; then
  echo "↪ 已存在，跳过: $RESULT_FILE"
else
  ARGS=(
    "$REPO_ROOT/tests/experiments/auto_gender_s_att.py"
    --layer "$LAYER"
    --head "$HEAD"
    --rounds "$ROUNDS"
    --typename "$TYPENAME"
    --output_dir "$OUTPUT_DIR"
    --causal_dataset "$MIDDLE_DIFF_FILE"
    --receiver_heads "$RECEIVER_HEADS"
  )
  if [ "$USE_ABC" -eq 1 ]; then
    ARGS+=(--use-abc)
  fi
  if [ -n "$TARGET_HEADS" ]; then
    ARGS+=(--target_heads "$TARGET_HEADS")
  fi
  if [ -n "$RECEIVER_DESC_FLAG" ]; then
    ARGS+=("$RECEIVER_DESC_FLAG" "$RECEIVER_DESC_VALUE")
  fi
  if [ -n "$TARGET_DESC_FLAG" ]; then
    ARGS+=("$TARGET_DESC_FLAG" "$TARGET_DESC_VALUE")
  fi
  python "${ARGS[@]}"
fi

if [ -f "$RESULT_FILE" ]; then
  python "$REPO_ROOT/tests/experiments/save_best_hypothesis.py" \
    --result_file "$RESULT_FILE" \
    --output_file "$BEST_SUMMARY_FILE"
else
  echo "⚠️ 未找到结果文件 $RESULT_FILE，跳过最佳假设摘要。"
fi

echo "✅ agr_gender middle head pipeline finished. Results saved to $OUTPUT_DIR"
