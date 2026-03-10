#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0305}"

python "$REPO_ROOT/tests/experiments/eval_initial_hypothesis_on_test.py" \
  --results-root "$RESULTS_ROOT" \
  --families "Name_Mover_Head,Negative_Name_Mover_Head"

echo "✅ Batch1 done (NMH/NNMH initial test eval)."

