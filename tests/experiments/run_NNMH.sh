#!/bin/bash
set -euo pipefail

#############################################
# 可配置参数
#############################################
LAYER=10
HEAD=7
ROUNDS=1
CONDA_ENV="eap-ig"

#############################################
# 路径设定
#############################################
REPO_ROOT="/home/wangziran/eap_auto"
SCRIPT_PATH="$REPO_ROOT/tests/experiments/auto_NMH.py"
BASE_RESULTS_DIR="$REPO_ROOT/results/ioi"
DATA_SOURCE_DIR="$BASE_RESULTS_DIR/path_patching/NNMH_${LAYER}_${HEAD}"
OUTPUT_DIR="$BASE_RESULTS_DIR/hypothesis/Negative_Name_Mover_Head/${LAYER}.${HEAD}_$(date +%Y%m%d_%H%M)"
STANDARD_IOI_FILE="$BASE_RESULTS_DIR/path_patching/standard_ioi_data.json"
LOGIT_OUTPUT_FILE="$DATA_SOURCE_DIR/final_logit_effects_head_${LAYER}_${HEAD}.json"
CAUSAL_OUTPUT_FILE="$DATA_SOURCE_DIR/causal_effects_${LAYER}_${HEAD}.json"
RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
PREPROCESSED_SAMPLING_FILE="$DATA_SOURCE_DIR/preprocessed_for_sampling.jsonl"
PREPROCESSED_ATTENTION_FILE="$DATA_SOURCE_DIR/preprocessed_attention_scores.json"

#############################################
# 环境准备
#############################################
CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p /tmp
export TMPDIR=/tmp
mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

head_str="${LAYER}.${HEAD}"

#############################################
# 第1步：预计算原始注意力分数
#############################################
echo "=== Step 1: precompute raw attention for head ${head_str} ==="
python "$REPO_ROOT/tests/experiments/precompute_attention_scores.py" \
    --head "$head_str" \
    --input_file "$STANDARD_IOI_FILE" \
    --output_file "$RAW_ATTENTION_FILE"

#############################################
# 第2步：raw -> preprocessed_for_sampling.jsonl
#############################################
echo "=== Step 2: convert raw attention to sampling jsonl ==="
python - "$RAW_ATTENTION_FILE" "$PREPROCESSED_SAMPLING_FILE" <<'PY'
import json
import pathlib
import sys

raw_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])

with raw_path.open() as f:
    data = json.load(f)

with dst_path.open("w", encoding="utf-8") as out_f:
    for item in data:
        out_f.write(json.dumps({
            "original_sentence": item["sentence_text"],
            "indirect_object": item["io_token"],
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

#############################################
# 第3步：生成 preprocessed_attention_scores.json
#############################################
echo "=== Step 3: preprocess attention scores for sampling ==="
python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
    --input "$PREPROCESSED_SAMPLING_FILE" \
    --output "$PREPROCESSED_ATTENTION_FILE" \
    --top_k 2

#############################################
# 第4步：预计算 logit 效应
#############################################
echo "=== Step 4: precompute logit effects ==="
python "$REPO_ROOT/tests/experiments/precompute_logit_effects.py" \
    --head_to_patch "$head_str" \
    --output_dir "$DATA_SOURCE_DIR"

#############################################
# 第5步：生成因果真值
#############################################
echo "=== Step 5: preprocess causal effects ==="
python "$REPO_ROOT/tests/experiments/preprocess_causal_effects.py" \
    --input "$LOGIT_OUTPUT_FILE" \
    --output "$CAUSAL_OUTPUT_FILE"

#############################################
# 第6步：运行 auto_NMH.py
#############################################
echo "=== Step 6: run auto_NMH.py for ${head_str} ==="
python "$SCRIPT_PATH" \
    --layer "$LAYER" \
    --head "$HEAD" \
    --rounds "$ROUNDS" \
    --output_dir "$OUTPUT_DIR" \
    --data_source_dir "$DATA_SOURCE_DIR"

echo "✅ 完成：结果保存在 $OUTPUT_DIR"
