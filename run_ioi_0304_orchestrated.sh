#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0304}"
STANDARD_FILE="${STANDARD_FILE:-${RESULTS_ROOT}/path_patching/standard_ioi_data.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

NMH_SCRIPT="${REPO_ROOT}/run_ioi_nmh_nnmh_all_causal_att.sh"
SIH_DTH_SCRIPT="${REPO_ROOT}/run_ioi_sih_dth_all_causal_att.sh"

LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "$LOG_DIR"
NMH_LOG="${LOG_DIR}/nmh_nnmh_${STAMP}.log"
SIH_DTH_LOG="${LOG_DIR}/sih_dth_${STAMP}.log"
ORCH_LOG="${LOG_DIR}/orchestrator_${STAMP}.log"
touch "$NMH_LOG" "$SIH_DTH_LOG" "$ORCH_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"
}

check_dependency() {
  if [[ ! -f "$STANDARD_FILE" ]]; then
    log "ERROR: missing dataset file: $STANDARD_FILE"
    exit 1
  fi
  local sample_count
  sample_count="$(python - <<'PY' "$STANDARD_FILE"
import json,sys
p=sys.argv[1]
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
if isinstance(d, dict) and "samples" in d:
    print(len(d.get("samples", [])))
elif isinstance(d, list):
    print(len(d))
else:
    print(0)
PY
)"
  log "Dataset ready: ${STANDARD_FILE} (samples=${sample_count})"
}

latest_all_test_result() {
  local head="$1"
  local pattern="${RESULTS_ROOT}/hypothesis/Name_Mover_Head/${head}_*_all/test_results.json"
  ls -1t $pattern 2>/dev/null | head -n 1 || true
}

all_nmh_all_done() {
  local h p run_dir token_usage
  for h in 9.6 9.9 10.0; do
    p="$(latest_all_test_result "$h")"
    if [[ -z "$p" ]]; then
      return 1
    fi
    run_dir="$(dirname "$p")"
    token_usage="${run_dir}/token_usage_summary.json"
    if [[ ! -f "$token_usage" ]]; then
      return 1
    fi
  done
  return 0
}

check_dependency
log "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "Using RESULTS_ROOT=${RESULTS_ROOT}"
log "Logs: $NMH_LOG | $SIH_DTH_LOG | $ORCH_LOG"

log "Start NMH/NNMH pipeline in background..."
(
  RESULTS_ROOT="$RESULTS_ROOT" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  bash "$NMH_SCRIPT"
) >"$NMH_LOG" 2>&1 &
NMH_PID=$!
log "NMH/NNMH pid=${NMH_PID}"

log "Live logs (Ctrl-C stops tail only; jobs keep running):"
tail -f "$NMH_LOG" "$SIH_DTH_LOG" &
TAIL_PID=$!

SIH_STARTED=0
SIH_PID=""

cleanup() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "$TAIL_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

while true; do
  # If NMH script exits before SIH trigger, still do one final check.
  if all_nmh_all_done && [[ "$SIH_STARTED" -eq 0 ]]; then
    log "Detected NMH _all ready for 9.6/9.9/10.0. Start SIH/DTH pipeline..."
    (
      RESULTS_ROOT="$RESULTS_ROOT" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      bash "$SIH_DTH_SCRIPT"
    ) >"$SIH_DTH_LOG" 2>&1 &
    SIH_PID=$!
    SIH_STARTED=1
    log "SIH/DTH pid=${SIH_PID}"
  fi

  if ! kill -0 "$NMH_PID" >/dev/null 2>&1; then
    if wait "$NMH_PID"; then
      NMH_EXIT=0
    else
      NMH_EXIT=$?
    fi
    log "NMH/NNMH process exited with code ${NMH_EXIT}"
    # ensure trigger opportunity even if loop ticks late
    if all_nmh_all_done && [[ "$SIH_STARTED" -eq 0 ]]; then
      log "NMH exited and _all artifacts exist. Start SIH/DTH now..."
      (
        RESULTS_ROOT="$RESULTS_ROOT" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        bash "$SIH_DTH_SCRIPT"
      ) >"$SIH_DTH_LOG" 2>&1 &
      SIH_PID=$!
      SIH_STARTED=1
      log "SIH/DTH pid=${SIH_PID}"
    fi
    break
  fi
  sleep 30
done

if [[ "$SIH_STARTED" -eq 1 ]]; then
  if wait "$SIH_PID"; then
    SIH_EXIT=0
  else
    SIH_EXIT=$?
  fi
  log "SIH/DTH process exited with code ${SIH_EXIT}"
else
  log "WARNING: SIH/DTH was not started (NMH _all not detected)."
fi

log "Done."
