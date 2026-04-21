#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_single_middle_head_plus.sh" \
  --validate-every 2 \
  --validation-sample-size 25 \
  --test-sample-size 50 \
  "$@"
