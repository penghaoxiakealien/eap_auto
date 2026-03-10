#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 统一策略：每2轮validation一次，每次validation 25句，final test 50句。
bash "$SCRIPT_DIR/run_middle_head.sh" \
  --validate-every 2 \
  --validation-sample-size 25 \
  --test-sample-size 50 \
  "$@"
