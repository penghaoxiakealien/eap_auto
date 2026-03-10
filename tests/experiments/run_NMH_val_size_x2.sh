#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 版本B：每2轮 validation 一次；validation 25句；final test 50句。
bash "$SCRIPT_DIR/run_NMH.sh" \
  --validate-every 2 \
  --validation-sample-size 25 \
  --test-sample-size 50 \
  "$@"
