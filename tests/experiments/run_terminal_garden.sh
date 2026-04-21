#!/bin/bash
set -euo pipefail

#############################################
# Default params (override via CLI)
#############################################
LAYER=8
HEAD=10
ROUNDS=5
CONDA_ENV="eap-ig"
RESULTS_ROOT="/home/wangziran/eap_auto/results/garden/garden_npz_v_trans_mod_run"
DATA_PATH="/home/wangziran/eap_auto/datasets/garden/garden_npz_v_trans_mod.csv"
STANDARD_GARDEN_JSON="/home/wangziran/eap_auto/results/garden/standard_garden_data.json"
MODEL_PATH="/home/wangziran/gpt2"
MODEL="gpt-5.2-2025-12-11"
OUTPUT_FAMILY="Terminal"
DATA_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
ATTN_BATCH_SIZE=1
STRICT_ALIGN=0
ATTENTION_POSITION="end"
REQUIRE_REASONING=0
MAX_SAMPLES=300
VALIDATE_EVERY=2
VALIDATION_SAMPLE_SIZE=25
TEST_SAMPLE_SIZE=50

usage() {
  cat <<'EOF'
Usage: run_terminal_garden.sh [options]
  --layer L             Layer (default: 8)
  --head H              Head (default: 10)
  --rounds N            auto_terminal rounds (default: 5)
  --conda-env NAME      Conda env (default: eap-ig)
  --results-root PATH   Results root (default: garden run dir)
  --data-path PATH      Garden CSV path
  --standard-json PATH  standard_garden_data.json path
  --model-path PATH     Local model path (optional)
  --model NAME          OpenRouter model (default: gpt-5.2-2025-12-11)
  --output-family NAME  hypothesis subdir name (default: Terminal)
  --data-dir PATH       Override data_source_dir
  --output-dir PATH     Override output dir
  --attention-batch-size N  Batch size for raw attention (default: 1)
  --strict-attention-align  Convert attention tokens to strict word-level tokens
  --attention-position POS  Query position (default: end)
  --max-samples N       Max garden samples to use (default: 300)
  --validate-every N    Validation cadence for auto_terminal (default: 2)
  --validation-sample-size N  Validation sample size (default: 25)
  --test-sample-size N  Test sample size (default: 50)
  --with-reasoning      Require [REASONING] blocks
  --no-reasoning         Do not require [REASONING] blocks from LLM
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layer) LAYER="$2"; shift 2 ;;
    --head) HEAD="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --standard-json) STANDARD_GARDEN_JSON="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-family) OUTPUT_FAMILY="$2"; shift 2 ;;
    --data-dir) DATA_DIR_OVERRIDE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --attention-batch-size) ATTN_BATCH_SIZE="$2"; shift 2 ;;
    --strict-attention-align) STRICT_ALIGN=1; shift 1 ;;
    --attention-position) ATTENTION_POSITION="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --validate-every) VALIDATE_EVERY="$2"; shift 2 ;;
    --validation-sample-size) VALIDATION_SAMPLE_SIZE="$2"; shift 2 ;;
    --test-sample-size) TEST_SAMPLE_SIZE="$2"; shift 2 ;;
    --with-reasoning) REQUIRE_REASONING=1; shift 1 ;;
    --no-reasoning) REQUIRE_REASONING=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

REPO_ROOT="/home/wangziran/eap_auto"
SCRIPT_PATH="$REPO_ROOT/tests/experiments/garden/auto_terminal.py"

if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
  DATA_SOURCE_DIR="$DATA_DIR_OVERRIDE"
else
  DATA_SOURCE_DIR="$RESULTS_ROOT/path_patching/${OUTPUT_FAMILY}/${LAYER}_${HEAD}"
fi

if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
  OUTPUT_DIR="$RESULTS_ROOT/hypothesis/${OUTPUT_FAMILY}/${LAYER}_${HEAD}"
fi

CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

head_str="${LAYER}.${HEAD}"
RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
PREPROCESSED_SAMPLING_FILE="$DATA_SOURCE_DIR/preprocessed_for_sampling.jsonl"
PREPROCESSED_ATTENTION_FILE="$DATA_SOURCE_DIR/preprocessed_attention_scores.json"
LOGIT_EFFECT_FILE="$DATA_SOURCE_DIR/heads_direct_effect_on_logit_difference.json"

echo "=== Step 1: raw attention ==="
python "$REPO_ROOT/tests/experiments/precompute_attention_scores_garden.py" \
  --standard-json "$STANDARD_GARDEN_JSON" \
  --output-file "$RAW_ATTENTION_FILE" \
  --head "$head_str" \
  --batch-size "$ATTN_BATCH_SIZE" \
  --attention-position "$ATTENTION_POSITION" \
  --max-samples "$MAX_SAMPLES"

echo "=== Step 2: raw -> preprocessed_for_sampling.jsonl ==="
python - "$RAW_ATTENTION_FILE" "$PREPROCESSED_SAMPLING_FILE" <<'PY'
import json, pathlib, sys
raw_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])
data = json.loads(raw_path.read_text())
with dst_path.open("w", encoding="utf-8") as out_f:
    for item in data:
        out_f.write(json.dumps({
            "sentence_id": item.get("sample_id", ""),
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

if [ "$STRICT_ALIGN" -eq 1 ]; then
  echo "=== Step 3: strict align attention tokens ==="
  python "$REPO_ROOT/tests/experiments/convert_attention_tokens_to_words.py" \
    --input-jsonl "$PREPROCESSED_SAMPLING_FILE" \
    --output-preprocessed "$PREPROCESSED_ATTENTION_FILE" \
    --output-ground-truth "$OUTPUT_DIR/attention_scores_ground_truth.jsonl" \
    --top-k 2
else
  echo "=== Step 3: build preprocessed_attention_scores.json ==="
  python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
    --input "$PREPROCESSED_SAMPLING_FILE" \
    --output "$PREPROCESSED_ATTENTION_FILE" \
    --top_k 2
fi

echo "=== Step 4: compute head logit effect ==="
python "$REPO_ROOT/tests/experiments/precompute_logit_effects_garden.py" \
  --data-path "$DATA_PATH" \
  --head "$head_str" \
  --model-path "$MODEL_PATH" \
  --output-file "$LOGIT_EFFECT_FILE" \
  --max-samples "$MAX_SAMPLES"

echo "=== Step 5: run auto_terminal ==="
python "$SCRIPT_PATH" \
  --layer "$LAYER" \
  --head "$HEAD" \
  --rounds "$ROUNDS" \
  --output_dir "$OUTPUT_DIR" \
  --data_source_dir "$DATA_SOURCE_DIR" \
  --model "$MODEL" \
  --validate-every "$VALIDATE_EVERY" \
  --validation-sample-size "$VALIDATION_SAMPLE_SIZE" \
  --test-sample-size "$TEST_SAMPLE_SIZE" \
  $( [[ "$REQUIRE_REASONING" -eq 1 ]] && echo "--with-reasoning" || echo "--no-reasoning" )
