#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="/home/wangziran/eap_auto/results/ioi_0126"

RUN_NMH="$REPO_ROOT/tests/experiments/run_NMH.sh"
RUN_MIDDLE="$REPO_ROOT/tests/experiments/run_middle_head.sh"
RUN_MIDDLE_PLUS="$REPO_ROOT/tests/experiments/run_middle_head_plus.sh"

NMH_HEADS=("9.6" "9.9" "10.0")
NNMH_HEADS=("10.7" "11.10")
SIH_HEADS=("7.9" "8.6" "8.10")
DTH_HEADS=("0.1" "0.10" "3.0")

MODES=("all")
mode_to_optimize() {
  case "$1" in
    all) echo "dual";;
    causal) echo "causal";;
    att) echo "attention";;
    *) echo "dual";;
  esac
}

receiver_heads="9.6,9.9,10.0"
intermediate_heads="7.9,8.6,8.10"

stamp=$(date +%Y%m%d_%H%M)
FAMILIES="NMH,NNMH,SIH,DTH,DTH_single"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --families)
      FAMILIES="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1" >&2
      echo "用法: $0 [--families NMH,NNMH,SIH,DTH,DTH_single]" >&2
      exit 1
      ;;
  esac
done

has_family() {
  local target="$1"
  [[ ",$FAMILIES," == *",$target,"* ]]
}

resolve_receiver_desc_files() {
  local heads_csv="$1"
  local mapping=""
  local IFS=','
  read -r -a heads <<< "$heads_csv"
  for h in "${heads[@]}"; do
    local p
    p=$(ls -td "$RESULTS_ROOT/hypothesis/Name_Mover_Head/${h}"_*_all 2>/dev/null | head -n 1 || true)
    if [[ -z "$p" ]]; then
      p=$(ls -td "$RESULTS_ROOT/hypothesis/Negative_Name_Mover_Head/${h}"_*_all 2>/dev/null | head -n 1 || true)
    fi
    if [[ -z "$p" || ! -f "$p/best_hypothesis.json" ]]; then
      echo "错误: 未找到 receiver $h 的 best_hypothesis.json，请先跑 NMH/NNMH。" >&2
      exit 1
    fi
    if [[ -n "$mapping" ]]; then
      mapping+=","
    fi
    mapping+="${h}:$p/best_hypothesis.json"
  done
  echo "$mapping"
}

RECEIVER_DESC_FILES="$(resolve_receiver_desc_files "$receiver_heads")"

run_nmh_family() {
  local family="$1"
  shift
  local heads=("$@")
  local task_prefix="$family"
  local output_family
  if [ "$family" = "NMH" ]; then
    output_family="Name_Mover_Head"
  else
    output_family="Negative_Name_Mover_Head"
  fi

  for head in "${heads[@]}"; do
    local layer="${head%%.*}"
    local h="${head##*.}"
    for mode in "${MODES[@]}"; do
      local opt
      opt=$(mode_to_optimize "$mode")
      local out_dir="$RESULTS_ROOT/hypothesis/${output_family}/${head}_${stamp}_${mode}"
      bash "$RUN_NMH" \
        --layer "$layer" \
        --head "$h" \
        --results-root "$RESULTS_ROOT" \
        --task-prefix "$task_prefix" \
        --output-family "$output_family" \
        --output-dir "$out_dir" \
        --optimize-only "$opt"
    done
  done
}

run_middle_family() {
  local family="$1"
  shift
  local heads=("$@")
  for head in "${heads[@]}"; do
    local layer="${head%%.*}"
    local h="${head##*.}"
    for mode in "${MODES[@]}"; do
      local opt
      opt=$(mode_to_optimize "$mode")
      local out_dir="$RESULTS_ROOT/hypothesis/${family}/${head}_${stamp}_${mode}"
      if [ "$family" = "DTH" ]; then
        bash "$RUN_MIDDLE_PLUS" \
          --layer "$layer" \
          --head "$h" \
          --intermediate-heads "$intermediate_heads" \
          --target-heads "$receiver_heads" \
          --receiver-desc "$RECEIVER_DESC_FILES" \
          --results-root "$RESULTS_ROOT" \
          --output-dir "$out_dir" \
          --optimize-only "$opt"
      else
        bash "$RUN_MIDDLE" \
          --layer "$layer" \
          --head "$h" \
          --receiver-heads "$receiver_heads" \
          --receiver-desc "$RECEIVER_DESC_FILES" \
          --results-root "$RESULTS_ROOT" \
          --output-dir "$out_dir" \
          --optimize-only "$opt"
      fi
    done
  done
}

if has_family NMH; then
  echo "=== NMH ==="
  run_nmh_family NMH "${NMH_HEADS[@]}"
fi

if has_family NNMH; then
  echo "=== NNMH ==="
  run_nmh_family NNMH "${NNMH_HEADS[@]}"
fi

if has_family SIH; then
  echo "=== SIH ==="
  run_middle_family SIH "${SIH_HEADS[@]}"
fi

if has_family DTH; then
  echo "=== DTH ==="
  run_middle_family DTH "${DTH_HEADS[@]}"
fi

if has_family DTH_single; then
  echo "=== DTH_single extra (0.1->7.9->9.6, 0.1->7.3->9.6) ==="
  base_out="$RESULTS_ROOT/hypothesis/DTH_single"
  opt=$(mode_to_optimize "all")

  bash "$RUN_MIDDLE_PLUS" \
    --layer 0 \
    --head 1 \
    --intermediate-heads "7.9" \
    --target-heads "9.6" \
    --receiver-desc "$RECEIVER_DESC_FILES" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$base_out/0.1_7.9_9.6_${stamp}_all" \
    --optimize-only "$opt"

  bash "$RUN_MIDDLE_PLUS" \
    --layer 0 \
    --head 1 \
    --intermediate-heads "7.3" \
    --target-heads "9.6" \
    --receiver-desc "$RECEIVER_DESC_FILES" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$base_out/0.1_7.3_9.6_${stamp}_all" \
    --optimize-only "$opt"
fi

echo "✅ Done."
