#!/bin/bash
set -euo pipefail

#############################################
# Default params (override via CLI)
#############################################
LAYER=0
HEAD=1
RECEIVER_HEADS="10.7"
ROUNDS=5
TYPENAME="garden_middle_head"
CONDA_ENV="eap-ig"
RESULTS_ROOT="/data31/private/wangziran/eap_auto/results/garden/garden_npz_v_trans_mod_run"
STANDARD_GARDEN_JSON="/data31/private/wangziran/eap_auto/results/garden/standard_garden_data.json"
ATTENTION_POSITION="end"
RECEIVER_ATTENTION_POSITION=""
RECEIVER_DESC_FILES=""
DATA_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
OUTPUT_FAMILY="Middle_Head"
TOP_K=2
ATTN_BATCH_SIZE=1
STRICT_ALIGN=0
INTERMEDIATE_HEADS=""
TARGET_HEAD=""
RECEIVER_INPUTS="q,k,v"
RECEIVER_GROUP_ID=""
RECEIVER_GROUP_SIGNATURE=""
RECEIVER_GROUP_HEADS=""

usage() {
  cat <<'EOF'
Usage: run_middle_head_garden.sh [options]
  --layer L                 Sender layer
  --head H                  Sender head index
  --receiver-heads LIST     Comma-separated receiver heads
  --rounds N                auto_middle rounds (default: 5)
  --typename NAME           Typename passed to auto_middle
  --conda-env NAME          Conda env (default: eap-ig)
  --results-root PATH       Results root (default: garden run dir)
  --standard-json PATH      standard_garden_data.json path
  --attention-position POS  Attention row (default: end)
  --receiver-attention-position POS  Receiver query position for diff vectors (default: attention-position)
  --receiver-desc FILES     head:path,head:path receiver descriptions
  --data-dir PATH           Override data_source_dir
  --output-dir PATH         Override output dir
  --output-family NAME      hypothesis subdir name (default: Middle_Head)
  --top-k N                 Top-k tokens for preprocessed_attention_scores (default: 2)
  --attention-batch-size N  Batch size for raw attention (default: 1)
  --strict-attention-align  Convert attention tokens to strict word-level tokens
  --intermediate-heads LIST Comma-separated intermediate heads (A->B->C)
  --target-head H           Target head for A->B->C
  --receiver-inputs LIST    Receiver inputs to patch (default: q,k,v)
  --receiver-group-id ID    Optional receiver group id
  --receiver-group-signature SIG  Optional receiver group signature
  --receiver-group-heads LIST     Optional receiver group heads (comma-separated)
  -h, --help                Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layer) LAYER="$2"; shift 2 ;;
    --head) HEAD="$2"; shift 2 ;;
    --receiver-heads) RECEIVER_HEADS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --typename) TYPENAME="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --standard-json) STANDARD_GARDEN_JSON="$2"; shift 2 ;;
    --attention-position) ATTENTION_POSITION="$2"; shift 2 ;;
    --receiver-attention-position) RECEIVER_ATTENTION_POSITION="$2"; shift 2 ;;
    --receiver-desc) RECEIVER_DESC_FILES="$2"; shift 2 ;;
    --data-dir) DATA_DIR_OVERRIDE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --output-family) OUTPUT_FAMILY="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --attention-batch-size) ATTN_BATCH_SIZE="$2"; shift 2 ;;
    --strict-attention-align) STRICT_ALIGN=1; shift 1 ;;
    --intermediate-heads) INTERMEDIATE_HEADS="$2"; shift 2 ;;
    --target-head) TARGET_HEAD="$2"; shift 2 ;;
    --receiver-inputs) RECEIVER_INPUTS="$2"; shift 2 ;;
    --receiver-group-id) RECEIVER_GROUP_ID="$2"; shift 2 ;;
    --receiver-group-signature) RECEIVER_GROUP_SIGNATURE="$2"; shift 2 ;;
    --receiver-group-heads) RECEIVER_GROUP_HEADS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

REPO_ROOT="/data31/private/wangziran/eap_auto"
SCRIPT_PATH="$REPO_ROOT/tests/experiments/garden/auto_middle.py"

if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
  OUTPUT_DIR="$RESULTS_ROOT/hypothesis/${OUTPUT_FAMILY}/${LAYER}.${HEAD}_$(date +%Y%m%d_%H%M)"
fi

if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
  DATA_SOURCE_DIR="$DATA_DIR_OVERRIDE"
else
  DATA_SOURCE_DIR="$OUTPUT_DIR"
fi

CONDA_BASE="/home/wangziran/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "$DATA_SOURCE_DIR" "$OUTPUT_DIR"

head_str="${LAYER}.${HEAD}"
RAW_ATTENTION_FILE="$DATA_SOURCE_DIR/raw_attention_head_${LAYER}_${HEAD}.json"
PREPROCESSED_SAMPLING_FILE="$DATA_SOURCE_DIR/preprocessed_for_sampling.jsonl"
PREPROCESSED_ATTENTION_FILE="$DATA_SOURCE_DIR/preprocessed_attention_scores.json"
ATTENTION_GT_FILE="$DATA_SOURCE_DIR/attention_scores_ground_truth.jsonl"
MIDDLE_DIFF_FILE="$DATA_SOURCE_DIR/causal_dataset.json"
RECEIVER_DESC_OUTPUT="$DATA_SOURCE_DIR/receiver_descriptions.json"
META_FILE="$OUTPUT_DIR/receiver_group_meta.json"

if [[ ! -f "$META_FILE" ]]; then
  META_FILE="$META_FILE" \
  META_SENDER="$head_str" \
  META_RECEIVER_HEADS="$RECEIVER_HEADS" \
  META_GROUP_ID="$RECEIVER_GROUP_ID" \
  META_GROUP_SIGNATURE="$RECEIVER_GROUP_SIGNATURE" \
  META_GROUP_HEADS="$RECEIVER_GROUP_HEADS" \
  META_ATTENTION_POSITION="$ATTENTION_POSITION" \
  META_RECEIVER_ATTENTION_POSITION="${RECEIVER_ATTENTION_POSITION:-$ATTENTION_POSITION}" \
  META_RECEIVER_INPUTS="$RECEIVER_INPUTS" \
  META_INTERMEDIATE="$INTERMEDIATE_HEADS" \
  META_TARGET="$TARGET_HEAD" \
  python -c "import json, os; meta={'run_type':'middle_plus' if os.environ.get('META_INTERMEDIATE') and os.environ.get('META_TARGET') else 'middle','sender_head':os.environ.get('META_SENDER',''),'receiver_heads':[h for h in os.environ.get('META_RECEIVER_HEADS','').split(',') if h],'receiver_group_id':os.environ.get('META_GROUP_ID') or None,'receiver_group_signature':os.environ.get('META_GROUP_SIGNATURE') or None,'receiver_group_heads':[h for h in os.environ.get('META_GROUP_HEADS','').split(',') if h],'attention_position':os.environ.get('META_ATTENTION_POSITION',''),'receiver_attention_position':os.environ.get('META_RECEIVER_ATTENTION_POSITION',''),'receiver_inputs':[h for h in os.environ.get('META_RECEIVER_INPUTS','').split(',') if h]}; meta.update({'intermediate_heads':[h for h in os.environ.get('META_INTERMEDIATE','').split(',') if h]}) if os.environ.get('META_INTERMEDIATE') else None; meta.update({'target_head':os.environ.get('META_TARGET')}) if os.environ.get('META_TARGET') else None; open(os.environ['META_FILE'],'w',encoding='utf-8').write(json.dumps(meta,ensure_ascii=False,indent=2))"
fi

echo "=== Step 1: precompute raw attention for head ${head_str} ==="
python "$REPO_ROOT/tests/experiments/precompute_attention_scores_garden.py" \
  --standard-json "$STANDARD_GARDEN_JSON" \
  --output-file "$RAW_ATTENTION_FILE" \
  --head "$head_str" \
  --batch-size "$ATTN_BATCH_SIZE" \
  --attention-position "$ATTENTION_POSITION"

echo "=== Step 2: raw -> preprocessed_for_sampling.jsonl ==="
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
            "indirect_object": item.get("io_token", ""),
            "number_of_important_tokens": len(item.get("top_attended_tokens", [])),
            "attention_scores": [
                {
                    "token": tok.get("token", "").strip(),
                    "position": tok.get("position", -1),
                    "score": tok.get("score", 0.0),
                }
                for tok in item.get("top_attended_tokens", [])
            ],
        }, ensure_ascii=False) + "\n")
PY

if [ "$STRICT_ALIGN" -eq 1 ]; then
  echo "=== Step 3: strict align attention tokens ==="
  python "$REPO_ROOT/tests/experiments/convert_attention_tokens_to_words.py" \
    --input-jsonl "$PREPROCESSED_SAMPLING_FILE" \
    --output-preprocessed "$PREPROCESSED_ATTENTION_FILE" \
    --output-ground-truth "$ATTENTION_GT_FILE" \
    --top-k "$TOP_K"
else
  echo "=== Step 3: build preprocessed_attention_scores.json ==="
  python "$REPO_ROOT/tests/experiments/preprocess_attention_scores.py" \
    --input "$PREPROCESSED_SAMPLING_FILE" \
    --output "$PREPROCESSED_ATTENTION_FILE" \
    --top_k "$TOP_K"

  echo "=== Step 4: export attention_scores_ground_truth.jsonl for auto_middle ==="
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
dst.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
PY
fi

if [ -n "${RECEIVER_DESC_FILES}" ]; then
  echo "=== Step 4b: assemble receiver descriptions ==="
  python - "$RECEIVER_DESC_FILES" "$RECEIVER_DESC_OUTPUT" <<'PY'
import json, sys, pathlib
mapping = {}
arg = sys.argv[1].strip()
out_path = pathlib.Path(sys.argv[2])
if ":" not in arg and arg:
    try:
        data = json.loads(pathlib.Path(arg).read_text())
        if isinstance(data, dict):
            mapping.update({k: (v or "").strip() for k, v in data.items()})
    except Exception as e:
        print(f"Warning: failed to load receiver mapping from {arg}: {e}")
else:
    pairs = [p.strip() for p in arg.split(",") if p.strip()]
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
        except Exception as e:
            print(f"Warning: failed to load receiver description from {path}: {e}")
if mapping:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    print("Warning: no receiver descriptions gathered.")
PY
else
  RECEIVER_DESC_OUTPUT=""
fi

echo "=== Step 5: compute sender->receiver diff vectors ==="
if [[ -n "$INTERMEDIATE_HEADS" && -n "$TARGET_HEAD" ]]; then
  python "$REPO_ROOT/tests/experiments/precompute_sender_target_via_intermediates.py" \
    --sender_head "$head_str" \
    --intermediate_heads "$INTERMEDIATE_HEADS" \
    --target_head "$TARGET_HEAD" \
    --standard-json "$STANDARD_GARDEN_JSON" \
    --output_file "$MIDDLE_DIFF_FILE" \
    --attention_position "${RECEIVER_ATTENTION_POSITION:-$ATTENTION_POSITION}"
else
  python "$REPO_ROOT/tests/experiments/precompute_middle_head_garden.py" \
    --sender_head "$head_str" \
    --receiver_heads "$RECEIVER_HEADS" \
    --standard-json "$STANDARD_GARDEN_JSON" \
    --output_file "$MIDDLE_DIFF_FILE" \
    --receiver-attention-position "${RECEIVER_ATTENTION_POSITION:-$ATTENTION_POSITION}" \
    --receiver-inputs "$RECEIVER_INPUTS"
fi

RECEIVER_DESC_ARG=()
if [ -n "$RECEIVER_DESC_OUTPUT" ]; then
  RECEIVER_DESC_ARG=(--receiver_descriptions_file "$RECEIVER_DESC_OUTPUT")
fi

echo "=== Step 6: run auto_middle ==="
python "$SCRIPT_PATH" \
  --layer "$LAYER" \
  --head "$HEAD" \
  --rounds "$ROUNDS" \
  --typename "$TYPENAME" \
  --output_dir "$OUTPUT_DIR" \
  --data-source-dir "$DATA_SOURCE_DIR" \
  --receiver_heads "${TARGET_HEAD:-$RECEIVER_HEADS}" \
  "${RECEIVER_DESC_ARG[@]}"
