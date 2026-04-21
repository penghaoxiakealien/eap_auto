#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${REPO_ROOT}/results/garden/garden_npz_v_trans_mod_run"
GROUP_GRAPH_JSON="${RESULTS_ROOT}/analysis/group_graph_thr75_dagish.json"
STANDARD_JSON="${REPO_ROOT}/results/garden/standard_garden_data.json"
LOG_DIR="${RESULTS_ROOT}/logs_middle"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" \
python "${REPO_ROOT}/tests/experiments/garden/run_middle_from_group_graph.py" \
  --group-graph-json "${GROUP_GRAPH_JSON}" \
  --results-root "${RESULTS_ROOT}" \
  --standard-json "${STANDARD_JSON}" \
  --attention-batch-size 32 \
  --strict-attention-align \
  --require-all-receivers \
  --max-passes 999 \
  --use-log-dir \
  --log-dir "${LOG_DIR}" \
  --resume
