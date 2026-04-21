#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="/home/wangziran/eap_auto/results/ioi_0312"

DTH_DESC="7.9:/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Middle_Head/7.9_20260313_0152/best_hypothesis.json,8.6:/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Middle_Head/8.6_20260313_1418/best_hypothesis.json,8.10:/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Middle_Head/8.10_20260312_1631/best_hypothesis.json,9.6:/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Name_Mover_Head/9.6_20260315_0536/best_hypothesis.json,9.9:/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Name_Mover_Head/9.9_20260315_0726/best_hypothesis.json,10.0:/home/wangziran/eap_auto/results/ioi_0312/hypothesis/Name_Mover_Head/10.0_20260315_0752/best_hypothesis.json"

bash "$REPO_ROOT/tests/experiments/run_middle_head_plus.sh" \
  --layer 3 --head 0 \
  --intermediate-heads 7.9,8.6,8.10 \
  --target-heads 9.6,9.9,10.0 \
  --receiver-desc "$DTH_DESC" \
  --attention-position s2 \
  --results-root "$RESULTS_ROOT" \
  --model gpt-5.2 \
  --optimize-only dual \
  --validate-every 2 \
  --validation-sample-size 25 \
  --test-sample-size 50
