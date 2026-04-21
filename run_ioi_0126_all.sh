#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="/home/wangziran/eap_auto/results/ioi_0126"

RUN_NMH="$REPO_ROOT/tests/experiments/run_NMH.sh"
RUN_MIDDLE="$REPO_ROOT/tests/experiments/run_middle_head.sh"
AUTO_D="$REPO_ROOT/tests/experiments/auto_d.py"

# Heads from existing ioi_0126 hypothesis directories
get_heads() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    return
  fi
  ls -1 "$dir" \
    | sed 's#/##' \
    | awk -F'_' '{print $1}' \
    | sort -u
}

NMH_HEADS=$(get_heads "$RESULTS_ROOT/hypothesis/Name_Mover_Head")
NNMH_HEADS=$(get_heads "$RESULTS_ROOT/hypothesis/Negative_Name_Mover_Head")
SIH_HEADS=$(get_heads "$RESULTS_ROOT/hypothesis/SIH")
DTH_HEADS=$(get_heads "$RESULTS_ROOT/hypothesis/DTH")

if [ -z "$NMH_HEADS$NNMH_HEADS$SIH_HEADS$DTH_HEADS" ]; then
  echo "No heads found under $RESULTS_ROOT/hypothesis"
  exit 1
fi

echo "NMH heads: $NMH_HEADS"
echo "NNMH heads: $NNMH_HEADS"
echo "SIH heads: $SIH_HEADS"
echo "DTH heads: $DTH_HEADS"

echo "=== Running NMH ==="
for head in $NMH_HEADS; do
  layer="${head%%.*}"
  h="${head##*.}"
  bash "$RUN_NMH" \
    --layer "$layer" \
    --head "$h" \
    --results-root "$RESULTS_ROOT" \
    --task-prefix "NMH" \
    --output-family "Name_Mover_Head"
done

echo "=== Running NNMH ==="
for head in $NNMH_HEADS; do
  layer="${head%%.*}"
  h="${head##*.}"
  bash "$RUN_NMH" \
    --layer "$layer" \
    --head "$h" \
    --results-root "$RESULTS_ROOT" \
    --task-prefix "NNMH" \
    --output-family "Negative_Name_Mover_Head"
done

echo "=== Running SIH ==="
for head in $SIH_HEADS; do
  layer="${head%%.*}"
  h="${head##*.}"
  out_dir="$RESULTS_ROOT/hypothesis/SIH/${head}_$(date +%Y%m%d_%H%M)"
  bash "$RUN_MIDDLE" \
    --layer "$layer" \
    --head "$h" \
    --receiver-heads "9.6,9.9,10.0" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$out_dir"
done

echo "=== Running DTH ==="
for head in $DTH_HEADS; do
  layer="${head%%.*}"
  h="${head##*.}"
  out_dir="$RESULTS_ROOT/hypothesis/DTH/${head}_$(date +%Y%m%d_%H%M)"
  mkdir -p "$out_dir"

  causal_file="$REPO_ROOT/results/ioi/path_patching/DTH/analysis_A${layer}.${h}_to_B7.3_on_C9.6.json"
  gt_file="$REPO_ROOT/results/ioi/path_patching/DTH/preprocessed_ground_truth.json"
  attn_gt_file="$REPO_ROOT/results/ioi/path_patching/DTH/preprocessed_attention_ground_truth.json"

  if [ ! -f "$causal_file" ]; then
    echo "Missing causal effects file: $causal_file"
    exit 1
  fi
  if [ ! -f "$gt_file" ] || [ ! -f "$attn_gt_file" ]; then
    echo "Missing DTH ground truth files under results/ioi/path_patching/DTH"
    exit 1
  fi

  python "$AUTO_D" \
    --layer "$layer" \
    --head "$h" \
    --rounds "1" \
    --typename "d_transcription_head" \
    --output_dir "$out_dir" \
    --causal_effects_file "$causal_file" \
    --ground_truth_file "$gt_file" \
    --attention_ground_truth_file "$attn_gt_file"
done

echo "✅ All IOI_0126 runs finished."
