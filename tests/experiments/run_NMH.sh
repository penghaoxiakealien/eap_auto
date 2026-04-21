#!/bin/bash
set -euo pipefail

#############################################
# 默认参数（可通过 CLI 覆写）
#############################################
LAYER=10
HEAD=0
ROUNDS=1
CONDA_ENV="eap-ig"
RESULTS_ROOT="/home/wangziran/eap_auto/results/ioi"
STANDARD_IOI_FILE=""
TASK_PREFIX="NMH"
OUTPUT_FAMILY="Name_Mover_Head"
DATA_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
MODEL="gpt-5-chat"
OPTIMIZE_ONLY="dual"
VALIDATE_EVERY=2
VALIDATION_SAMPLE_SIZE=0
TEST_ALL_VALIDATIONS=0
TEST_SAMPLE_SIZE=0

usage() {
    cat <<'EOF'
用法: run_NMH.sh [选项]
  --layer L             指定层号 (默认: 10)
  --head H              指定头号 (默认: 0)
  --rounds N            仅用于结果文件命名（训练固定 10 epoch）
  --conda-env NAME      Conda 环境 (默认: eap-ig)
  --results-root PATH   结果根目录 (默认: /home/wangziran/eap_auto/results/ioi)
  --standard-file PATH  standard_ioi_data.json 的路径；未提供则使用 results-root 默认位置
  --task-prefix NAME    path_patching 子目录前缀 (默认: NMH)
  --output-family NAME  hypothesis 子目录名 (默认: Name_Mover_Head)
  --output-dir PATH     指定输出目录（默认自动时间戳）
  --data-dir PATH       指定 path_patching 数据目录
  --model NAME          OpenRouter 模型名 (默认: gpt-5-chat)
  --optimize-only MODE  精炼阶段仅使用 causal/attention/dual
  --validate-every N    每 N 轮做一次 validation (默认: 2)
  --validation-sample-size N  每次 validation 使用句子数（0=全量 validation 集）
  --test-all-validations      将每个 validation checkpoint 都在 test 集上评估一次
  --test-sample-size N        final test 使用句子数（0=全量 test 集）
  -h, --help            显示本帮助
EOF
}

#############################################
# 解析命令行参数
#############################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --standard-file)
            STANDARD_IOI_FILE="$2"
            shift 2
            ;;
        --task-prefix)
            TASK_PREFIX="$2"
            shift 2
            ;;
        --output-family)
            OUTPUT_FAMILY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --optimize-only)
            OPTIMIZE_ONLY="$2"
            shift 2
            ;;
        --validate-every)
            VALIDATE_EVERY="$2"
            shift 2
            ;;
        --validation-sample-size)
            VALIDATION_SAMPLE_SIZE="$2"
            shift 2
            ;;
        --test-all-validations)
            TEST_ALL_VALIDATIONS=1
            shift 1
            ;;
        --test-sample-size)
            TEST_SAMPLE_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

#############################################
# 路径设定
#############################################
REPO_ROOT="/home/wangziran/eap_auto"
SCRIPT_PATH="$REPO_ROOT/tests/experiments/auto_NMH.py"
BASE_RESULTS_DIR="$RESULTS_ROOT"
if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
    DATA_SOURCE_DIR="$DATA_DIR_OVERRIDE"
else
    DATA_SOURCE_DIR="$BASE_RESULTS_DIR/path_patching/${TASK_PREFIX}_${LAYER}_${HEAD}"
fi
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
    OUTPUT_DIR="$BASE_RESULTS_DIR/hypothesis/${OUTPUT_FAMILY}/${LAYER}.${HEAD}_$(date +%Y%m%d_%H%M)"
fi
if [[ -z "$STANDARD_IOI_FILE" ]]; then
    STANDARD_IOI_FILE="$BASE_RESULTS_DIR/path_patching/standard_ioi_data.json"
fi
LOGIT_OUTPUT_FILE="$DATA_SOURCE_DIR/final_logit_effects_head_${LAYER}_${HEAD}.json"
CAUSAL_OUTPUT_FILE="$DATA_SOURCE_DIR/causal_effects_${LAYER}_${HEAD}.json"
RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
PREPROCESSED_SAMPLING_FILE="$DATA_SOURCE_DIR/preprocessed_for_sampling.jsonl"
PREPROCESSED_ATTENTION_FILE="$DATA_SOURCE_DIR/preprocessed_attention_scores.json"
RESULT_FILE="$OUTPUT_DIR/${LAYER}.${HEAD}_${ROUNDS}.json"
BEST_SUMMARY_FILE="$OUTPUT_DIR/best_hypothesis.json"
FINAL_SUMMARY_FILE="$OUTPUT_DIR/final_hypothesis.json"
TEST_RESULT_FILE="$OUTPUT_DIR/test_results.json"

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
if [ -f "$RAW_ATTENTION_FILE" ]; then
    echo "=== Step 1: found existing raw attention, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/precompute_attention_scores.py" \
        --head "$head_str" \
        --input_file "$STANDARD_IOI_FILE" \
        --output_file "$RAW_ATTENTION_FILE"
fi

#############################################
# 第2步：raw -> preprocessed_for_sampling.jsonl
#############################################
echo "=== Step 2: convert raw attention to sampling jsonl ==="
if [ -f "$PREPROCESSED_SAMPLING_FILE" ]; then
    echo "=== Step 2: found existing sampling jsonl, skipping ==="
else
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
fi

#############################################
# 第3步：生成 preprocessed_attention_scores.json
#############################################
echo "=== Step 3: preprocess attention scores for sampling ==="
if [ -f "$PREPROCESSED_ATTENTION_FILE" ]; then
    echo "=== Step 3: found existing preprocessed attention, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
        --input "$PREPROCESSED_SAMPLING_FILE" \
        --output "$PREPROCESSED_ATTENTION_FILE" \
        --top_k 2
fi

#############################################
# 第4步：预计算 logit 效应
#############################################
echo "=== Step 4: precompute logit effects ==="
if [ -f "$LOGIT_OUTPUT_FILE" ]; then
    echo "=== Step 4: found existing logit effects, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/precompute_logit_effects.py" \
        --input_file "$STANDARD_IOI_FILE" \
        --head_to_patch "$head_str" \
        --max_samples 0 \
        --output_dir "$DATA_SOURCE_DIR"
fi

#############################################
# 第5步：生成因果真值
#############################################
echo "=== Step 5: preprocess causal effects ==="
if [ -f "$CAUSAL_OUTPUT_FILE" ]; then
    echo "=== Step 5: found existing causal effects, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/preprocess_causal_effects.py" \
        --input "$LOGIT_OUTPUT_FILE" \
        --output "$CAUSAL_OUTPUT_FILE"
fi

#############################################
# 第6步：运行 auto_NMH.py
#############################################
echo "=== Step 6: run auto_NMH.py for ${head_str} ==="
python "$SCRIPT_PATH" \
  --layer "$LAYER" \
  --head "$HEAD" \
  --rounds "$ROUNDS" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --data_source_dir "$DATA_SOURCE_DIR" \
  --optimize-only "$OPTIMIZE_ONLY" \
  --validate-every "$VALIDATE_EVERY" \
  --validation-sample-size "$VALIDATION_SAMPLE_SIZE" \
  --test-sample-size "$TEST_SAMPLE_SIZE" \
  $([ "$TEST_ALL_VALIDATIONS" -eq 1 ] && echo "--test-all-validations")

if [ -f "$BEST_SUMMARY_FILE" ]; then
    echo "✅ 已存在最佳假设摘要: $BEST_SUMMARY_FILE"
else
    # Retry briefly in case the file is written slightly later.
    retries=5
    while [ $retries -gt 0 ] && [ ! -f "$RESULT_FILE" ]; do
        sleep 1
        retries=$((retries-1))
    done
    if [ -f "$RESULT_FILE" ]; then
        python "$REPO_ROOT/tests/experiments/save_best_from_iterations.py" \
            --input_file "$RESULT_FILE" \
            --output_file "$BEST_SUMMARY_FILE" \
            --head "$LAYER.$HEAD"
    else
        fallback_file=$(ls -1 "$OUTPUT_DIR"/${LAYER}.${HEAD}_*.json 2>/dev/null | head -n 1 || true)
        if [ -n "$fallback_file" ]; then
            python "$REPO_ROOT/tests/experiments/save_best_from_iterations.py" \
                --input_file "$fallback_file" \
                --output_file "$BEST_SUMMARY_FILE" \
                --head "$LAYER.$HEAD"
        else
            echo "⚠️ 未找到结果文件 $RESULT_FILE，跳过最佳假设摘要。"
        fi
    fi
fi

if [ -f "$TEST_RESULT_FILE" ]; then
    python "$REPO_ROOT/tests/experiments/save_final_hypothesis.py" \
        --test_file "$TEST_RESULT_FILE" \
        --output_file "$FINAL_SUMMARY_FILE" \
        --head "$LAYER.$HEAD" \
        --typename "$OUTPUT_FAMILY"
else
    echo "⚠️ 未找到测试结果 $TEST_RESULT_FILE，跳过 final_hypothesis 生成。"
fi

echo "✅ 完成：结果保存在 $OUTPUT_DIR"
