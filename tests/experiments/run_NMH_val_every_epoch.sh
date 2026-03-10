#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 版本A：每轮都做 validation（validate_every=1），
# 并保留“每个 validation checkpoint 都在 test 集上评估”的输出。
bash "$SCRIPT_DIR/run_NMH.sh" \
  --validate-every 1 \
  --validation-sample-size 10 \
  --test-sample-size 20 \
  --test-all-validations \
  "$@"
