#!/bin/bash
set -euo pipefail

#############################################
# 默认参数（可通过 CLI 覆写）
#############################################
LAYER=0
HEAD=1
RECEIVER_HEADS="5.5"
ROUNDS=5
CONDA_ENV="eap-ig"
RESULTS_ROOT="/data31/private/wangziran/eap_auto/results/ioi"
STRUCTURED_SENTENCE_FILE=""
STANDARD_IOI_FILE=""
ATTENTION_POSITION="end"
RECEIVER_DESC_FILES=""
DATA_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
OPTIMIZE_ONLY="dual"
MODEL="gpt-5-chat"
STRICT_CAUSAL_BATCH=0
VALIDATE_EVERY=2
VALIDATION_SAMPLE_SIZE=0
TEST_SAMPLE_SIZE=0
TEST_ALL_VALIDATIONS=0

usage() {
    cat <<'EOF'
用法: run_middle_head.sh [选项]
  --layer L                 发送头所在层
  --head H                  发送头编号
  --receiver-heads LIST     逗号分隔的下游头列表
  --rounds N                (Deprecated) 仅为兼容保留，不再用于命名
  --conda-env NAME          Conda 环境名
  --results-root PATH       results/ioi 根目录
  --structured-file PATH    structured_sentences.jsonl 路径
  --standard-file PATH      standard_ioi_data.json 路径（优先于 structured-file）
  --attention-position POS  注意力位置 (end/s2 等)
  --receiver-desc FILES     必填，head:path,head:path 形式的下游 head 描述文件列表
  --data-dir PATH           指定 path_patching/Middle_Head 子目录
  --output-dir PATH         指定输出目录
  --optimize-only MODE      精炼阶段仅使用 causal/attention/dual
  --model NAME              OpenRouter 模型名（默认: gpt-5-chat）
  --strict-causal-batch     causal 批量预测严格模式（不重试、失败即退出）
  --validate-every N        每 N 轮做一次 validation (默认: 2)
  --validation-sample-size N  每次 validation 使用句子数（0=全量 validation 集）
  --test-sample-size N      final test 使用句子数（0=全量 test 集）
  --test-all-validations    将每个 validation checkpoint 都在 test 集上评估一次
  -h, --help                显示此帮助
EOF
}

#############################################
# 解析命令行参数
#############################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer) LAYER="$2"; shift 2;;
        --head) HEAD="$2"; shift 2;;
        --receiver-heads) RECEIVER_HEADS="$2"; shift 2;;
        --rounds) ROUNDS="$2"; shift 2;;
        --conda-env) CONDA_ENV="$2"; shift 2;;
        --results-root) RESULTS_ROOT="$2"; shift 2;;
        --structured-file) STRUCTURED_SENTENCE_FILE="$2"; shift 2;;
        --standard-file) STANDARD_IOI_FILE="$2"; shift 2;;
        --attention-position) ATTENTION_POSITION="$2"; shift 2;;
        --receiver-desc) RECEIVER_DESC_FILES="$2"; shift 2;;
        --data-dir) DATA_DIR_OVERRIDE="$2"; shift 2;;
        --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
        --optimize-only) OPTIMIZE_ONLY="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --strict-causal-batch) STRICT_CAUSAL_BATCH=1; shift 1;;
        --validate-every) VALIDATE_EVERY="$2"; shift 2;;
        --validation-sample-size) VALIDATION_SAMPLE_SIZE="$2"; shift 2;;
        --test-sample-size) TEST_SAMPLE_SIZE="$2"; shift 2;;
        --test-all-validations) TEST_ALL_VALIDATIONS=1; shift 1;;
        -h|--help) usage; exit 0;;
        *) echo "未知参数: $1" >&2; usage; exit 1;;
    esac
done

if [[ -z "$RECEIVER_DESC_FILES" ]]; then
    echo "错误: --receiver-desc 为必填参数。请先提供下游 NMH/NNMH 的 best_hypothesis.json 映射。" >&2
    usage
    exit 1
fi

#############################################
# 路径设定
#############################################
REPO_ROOT="/data31/private/wangziran/eap_auto"
BASE_RESULTS_DIR="$RESULTS_ROOT"
if [[ -n "$STANDARD_IOI_FILE" ]]; then
    STRUCTURED_SENTENCE_FILE="$STANDARD_IOI_FILE"
elif [[ -z "$STRUCTURED_SENTENCE_FILE" ]]; then
    STRUCTURED_SENTENCE_FILE="$BASE_RESULTS_DIR/path_patching/structured_sentences.jsonl"
fi
ATTN_SOURCE_FILE="$STRUCTURED_SENTENCE_FILE"
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
RESULT_FILE="$OUTPUT_DIR/final_result_all_rounds.json"
BEST_SUMMARY_FILE="$OUTPUT_DIR/best_hypothesis.json"
FINAL_SUMMARY_FILE="$OUTPUT_DIR/final_hypothesis.json"
TEST_RESULT_FILE="$OUTPUT_DIR/test_results.json"

#############################################
# 环境准备
#############################################
CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p /data31/private/wangziran/tmp
export TMPDIR=/data31/private/wangziran/tmp
mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

#############################################
# Step 1: 预计算直接注意力 (middle head)
#############################################
echo "=== Step 1: precompute raw attention for head ${head_str} ==="
if [ -f "$RAW_ATTENTION_FILE" ]; then
    echo "=== Step 1: found existing raw attention, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/precompute_attention_scores.py" \
        --head "$head_str" \
        --input_file "$ATTN_SOURCE_FILE" \
        --output_file "$RAW_ATTENTION_FILE"
fi

#############################################
# Step 2: 生成 sampling/attention 所需的预处理文件
#############################################
echo "=== Step 2: convert raw attention to sampling jsonl ==="
if [ -f "$PREPROCESSED_SAMPLING_FILE" ]; then
    echo "=== Step 2: found existing sampling jsonl, skipping ==="
else
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
fi

echo "=== Step 3: build preprocessed_attention_scores.json ==="
if [ -f "$PREPROCESSED_ATTENTION_FILE" ]; then
    echo "=== Step 3: found existing preprocessed attention, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
        --input "$PREPROCESSED_SAMPLING_FILE" \
        --output "$PREPROCESSED_ATTENTION_FILE" \
        --top_k 5
fi

echo "=== Step 4: export attention_scores_ground_truth.jsonl for auto_s_att ==="
if [ -f "$ATTENTION_GT_FILE" ]; then
    echo "=== Step 4: found existing attention ground truth, skipping ==="
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
            mapping[head] = data.get("best_hypothesis", "").strip()
        elif isinstance(data, list):
            best = max(
                data,
                key=lambda item: (
                    item.get("scores", {}).get("causal_f1", 0),
                    item.get("scores", {}).get("attention_f1", 0),
                ),
            )
            mapping[head] = best.get("hypothesis", "").strip()
        else:
            print(f"Warning: unsupported receiver description format at {path}")
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

#############################################
# Step 5: 计算 sender->receiver 路径差值
#############################################
echo "=== Step 5: compute sender->receiver diff vectors ==="
if [ -f "$MIDDLE_DIFF_FILE" ]; then
    echo "=== Step 5: found existing causal dataset, skipping ==="
else
    python "$REPO_ROOT/tests/experiments/precompute_middle_head.py" \
        --sender_head "$head_str" \
        --receiver_heads "$RECEIVER_HEADS" \
        --sentences_file "$STRUCTURED_SENTENCE_FILE" \
        --output_file "$MIDDLE_DIFF_FILE" \
        --attention_position "$ATTENTION_POSITION"
fi

RECEIVER_DESC_ARG=()
if [ -n "${RECEIVER_DESC_FILES}" ] && [ -s "$RECEIVER_DESC_OUTPUT" ]; then
    RECEIVER_DESC_ARG=(--receiver_descriptions_file "$RECEIVER_DESC_OUTPUT")
fi

if [ ${#RECEIVER_DESC_ARG[@]} -eq 0 ]; then
    echo "错误: 未能生成有效的 receiver descriptions 文件，请检查 --receiver-desc 输入。" >&2
    exit 1
fi

STRICT_CAUSAL_ARG=()
if [ "$STRICT_CAUSAL_BATCH" -eq 1 ]; then
    STRICT_CAUSAL_ARG=(--strict-causal-batch)
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
    --attention-position "$ATTENTION_POSITION" \
    --receiver_heads "$RECEIVER_HEADS" \
    --optimize-only "$OPTIMIZE_ONLY" \
    --model "$MODEL" \
    --validate-every "$VALIDATE_EVERY" \
    --validation-sample-size "$VALIDATION_SAMPLE_SIZE" \
    --test-sample-size "$TEST_SAMPLE_SIZE" \
    $([ "$TEST_ALL_VALIDATIONS" -eq 1 ] && echo "--test-all-validations") \
    "${STRICT_CAUSAL_ARG[@]}" \
    "${RECEIVER_DESC_ARG[@]}"

if [ -f "$RESULT_FILE" ]; then
    python "$REPO_ROOT/tests/experiments/save_best_hypothesis.py" \
        --result_file "$RESULT_FILE" \
        --output_file "$BEST_SUMMARY_FILE"
else
    echo "⚠️ 未找到结果文件 $RESULT_FILE，跳过最佳假设摘要。"
fi

if [ -f "$TEST_RESULT_FILE" ]; then
    python "$REPO_ROOT/tests/experiments/save_final_hypothesis.py" \
        --test_file "$TEST_RESULT_FILE" \
        --output_file "$FINAL_SUMMARY_FILE" \
        --head "$LAYER.$HEAD" \
        --typename "Middle_Head"
else
    echo "⚠️ 未找到测试结果 $TEST_RESULT_FILE，跳过 final_hypothesis 生成。"
fi

echo "✅ Middle head pipeline finished. Results saved to $OUTPUT_DIR"
