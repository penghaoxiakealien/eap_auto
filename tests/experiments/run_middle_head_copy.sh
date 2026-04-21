#!/bin/bash
set -euo pipefail

#############################################
# 可配置参数
#############################################
LAYER=0
HEAD=1
RECEIVER_HEADS="6.9"
ROUNDS=1
CONDA_ENV="eap-ig"
# 可选：为下游头提供描述（格式 head:path,head:path）
RECEIVER_DESC_FILES="6.9:/home/wangziran/eap_auto/results/ioi/hypothesis/Middle_Head/6.9_20251117_0912/final_result_round_1.json"

#############################################
# 路径设定
#############################################
REPO_ROOT="/home/wangziran/eap_auto"
BASE_RESULTS_DIR="$REPO_ROOT/results/ioi"
STRUCTURED_SENTENCE_FILE="$BASE_RESULTS_DIR/path_patching/structured_sentences.jsonl"
ATTN_SOURCE_FILE="$STRUCTURED_SENTENCE_FILE"
head_str="${LAYER}.${HEAD}"

DATA_SOURCE_DIR="$BASE_RESULTS_DIR/path_patching/Middle_Head/${LAYER}_${HEAD}"
OUTPUT_DIR="$BASE_RESULTS_DIR/hypothesis/Middle_Head/${LAYER}.${HEAD}_$(date +%Y%m%d_%H%M)"

RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
PREPROCESSED_SAMPLING_FILE="$DATA_SOURCE_DIR/preprocessed_for_sampling.jsonl"
PREPROCESSED_ATTENTION_FILE="$DATA_SOURCE_DIR/preprocessed_attention_scores.json"
MIDDLE_DIFF_FILE="$DATA_SOURCE_DIR/causal_dataset_${LAYER}_${HEAD}.json"
ATTENTION_GT_FILE="$OUTPUT_DIR/attention_scores_ground_truth.jsonl"
RECEIVER_DESC_OUTPUT="$DATA_SOURCE_DIR/receiver_descriptions.json"
RESULT_FILE="$OUTPUT_DIR/final_result_round_${ROUNDS}.json"
BEST_SUMMARY_FILE="$OUTPUT_DIR/best_hypothesis.json"

#############################################
# 环境准备
#############################################
CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p /tmp
export TMPDIR=/tmp
mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

#############################################
# Step 1: 预计算直接注意力 (middle head)
#############################################
echo "=== Step 1: precompute raw attention for head ${head_str} ==="
python "$REPO_ROOT/tests/experiments/precompute_attention_scores.py" \
    --head "$head_str" \
    --input_file "$ATTN_SOURCE_FILE" \
    --output_file "$RAW_ATTENTION_FILE"

#############################################
# Step 2: 生成 sampling/attention 所需的预处理文件
#############################################
echo "=== Step 2: convert raw attention to sampling jsonl ==="
python - "$RAW_ATTENTION_FILE" "$PREPROCESSED_SAMPLING_FILE" <<'PY'
import json, pathlib, sys
raw_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])
with raw_path.open() as f:
    data = json.load(f)
with dst_path.open("w", encoding="utf-8") as out_f:
    for item in data:
        out_f.write(json.dumps({
            "sentence_id": str(item.get("sample_id", "")),
            "original_sentence": item["sentence_text"],
            "indirect_object": item.get("io_token", ""),
            "number_of_important_tokens": len(item.get("top_attended_tokens", [])),
            "attention_scores": [
                {
                    "token": tok.get("token", "").strip(),
                    "position": tok.get("position", -1),
                    "score": tok.get("score", 0.0)
                }
                for tok in item.get("top_attended_tokens", [])
            ]
        }, ensure_ascii=False) + "\n")
PY

echo "=== Step 3: build preprocessed_attention_scores.json ==="
python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
    --input "$PREPROCESSED_SAMPLING_FILE" \
    --output "$PREPROCESSED_ATTENTION_FILE" \
    --top_k 5

echo "=== Step 4: export attention_scores_ground_truth.jsonl for auto_s_att ==="
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
        key = item.get("sentence_id")
        if not key:
            key = str(idx)
        records.append({
            "key": key,
            "original_sentence": item["original_sentence"],
            "attention_scores": item["attention_scores"],
        })
with dst.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
PY

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
        with path.open() as f:
            data = json.load(f)
        best = max(data, key=lambda item: (item.get("scores", {}).get("causal_f1", 0), item.get("scores", {}).get("attention_f1", 0)))
        mapping[head] = best.get("hypothesis", "").strip()
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
    REVISED_DESC_PATH=""
fi

#############################################
# Step 5: 计算 sender->receiver 路径差值
#############################################
echo "=== Step 5: compute sender->receiver diff vectors ==="
python "$REPO_ROOT/tests/experiments/precompute_middle_head.py" \
    --sender_head "$head_str" \
    --receiver_heads "$RECEIVER_HEADS" \
    --sentences_file "$STRUCTURED_SENTENCE_FILE" \
    --output_file "$MIDDLE_DIFF_FILE"

RECEIVER_DESC_ARG=()
if [ -n "${RECEIVER_DESC_FILES}" ] && [ -s "$RECEIVER_DESC_OUTPUT" ]; then
    RECEIVER_DESC_ARG=(--receiver_descriptions_file "$RECEIVER_DESC_OUTPUT")
fi

#############################################
# Step 6: 运行 auto_s_att.py
#############################################
echo "=== Step 6: run auto_s_att.py for ${head_str} ==="
python "$REPO_ROOT/tests/experiments/auto_s_att.py" \
    --layer "$LAYER" \
    --head "$HEAD" \
    --rounds "$ROUNDS" \
    --output_dir "$OUTPUT_DIR" \
    --causal_dataset "$MIDDLE_DIFF_FILE" \
    --receiver_heads "$RECEIVER_HEADS" \
    "${RECEIVER_DESC_ARG[@]}"

if [ -f "$RESULT_FILE" ]; then
    python "$REPO_ROOT/tests/experiments/save_best_hypothesis.py" \
        --result_file "$RESULT_FILE" \
        --output_file "$BEST_SUMMARY_FILE"
else
    echo "⚠️ 未找到结果文件 $RESULT_FILE，跳过最佳假设摘要。"
fi

echo "✅ Middle head pipeline finished. Results saved to $OUTPUT_DIR"
