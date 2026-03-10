#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_terminal_subset_parallel.sh [options]
  --heads-file PATH        File with one head per line (format: L.H)
  --group-heads-file PATH  File with group_id<TAB>head per line (auto position)
  --results-root PATH      Results root (default: garden run dir)
  --data-path PATH         Dataset CSV path
  --standard-json PATH     standard_garden_data.json path
  --model-path PATH        Model path (gpt2)
  --rounds N               Number of rounds (default: 5)
  --max-parallel N         Max parallel jobs (default: 3)
  --cuda-devices IDS       CUDA_VISIBLE_DEVICES (default: 7)
  --attention-position POS Query position (default: end)
  -h, --help               Show help
EOF
}

HEADS_FILE=""
GROUP_HEADS_FILE=""
RESULTS_ROOT="/data31/private/wangziran/eap_auto/results/garden/garden_npz_v_trans_mod_run"
DATA_PATH="/data31/private/wangziran/eap_auto/datasets/garden/garden_npz_v_trans_mod.csv"
STANDARD_JSON="/data31/private/wangziran/eap_auto/results/garden/standard_garden_data.json"
MODEL_PATH="/data31/private/wangziran/eap-ig/gpt2"
ROUNDS=5
MAX_PARALLEL=3
CUDA_DEVICES="7"
ATTENTION_POSITION="end"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --heads-file) HEADS_FILE="$2"; shift 2 ;;
    --group-heads-file) GROUP_HEADS_FILE="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --standard-json) STANDARD_JSON="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --cuda-devices) CUDA_DEVICES="$2"; shift 2 ;;
    --attention-position) ATTENTION_POSITION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$HEADS_FILE" && -z "$GROUP_HEADS_FILE" ]]; then
  echo "Error: --heads-file or --group-heads-file is required." >&2
  usage
  exit 1
fi

if [[ -n "$HEADS_FILE" && ! -f "$HEADS_FILE" ]]; then
  echo "Error: heads file not found: $HEADS_FILE" >&2
  exit 1
fi

if [[ -n "$GROUP_HEADS_FILE" && ! -f "$GROUP_HEADS_FILE" ]]; then
  echo "Error: group-heads file not found: $GROUP_HEADS_FILE" >&2
  exit 1
fi

RUN_SCRIPT="/data31/private/wangziran/eap_auto/tests/experiments/run_terminal_garden.sh"

TMP_LIST="$(mktemp /tmp/terminal_heads_to_run.XXXXXX)"
trap 'rm -f "$TMP_LIST"' EXIT

if [[ -n "$GROUP_HEADS_FILE" ]]; then
  python - "$GROUP_HEADS_FILE" "$RESULTS_ROOT" "$TMP_LIST" <<'PY'
import sys, pathlib

def infer_pos(group_id: str) -> str:
    text = (group_id or "").split("|", 1)[0]
    if ":" in text:
        text = text.split(":", 1)[1]
    prefix = text.split("->", 1)[0].strip().lower()
    if prefix in {"subj","verb","obj_head","rel_pron","rel_verb","end"}:
        return prefix.upper()
    if prefix.startswith("s1"):
        return "S1"
    if prefix.startswith("s2"):
        return "S2"
    if prefix.startswith("io"):
        return "IO"
    return "END"

group_file = pathlib.Path(sys.argv[1])
results_root = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])
lines = []
for raw in group_file.read_text().splitlines():
    raw = raw.strip()
    if not raw:
        continue
    if "\t" in raw:
        gid, head = raw.split("\t", 1)
    elif " " in raw:
        gid, head = raw.split(" ", 1)
    else:
        continue
    head = head.strip()
    if "." not in head:
        continue
    layer, hid = head.split(".", 1)
    best = results_root / "hypothesis" / "Terminal" / f"{layer}_{hid}" / "best_hypothesis.json"
    if best.exists():
        print(f"[skip] {head} already has best_hypothesis")
        continue
    pos = infer_pos(gid)
    lines.append(f"{layer} {hid} {pos}")
out_path.write_text("\\n".join(lines) + ("\\n" if lines else ""))
PY
else
  while read -r head; do
    head="$(echo "$head" | tr -d '[:space:]')"
    [[ -z "$head" ]] && continue
    if [[ "$head" != *.* ]]; then
      echo "Skip invalid head format: $head" >&2
      continue
    fi
    layer="${head%.*}"
    hid="${head#*.}"
    best_file="${RESULTS_ROOT}/hypothesis/Terminal/${layer}_${hid}/best_hypothesis.json"
    if [[ -f "$best_file" ]]; then
      echo "[skip] ${head} already has best_hypothesis"
      continue
    fi
    echo "${layer} ${hid} ${ATTENTION_POSITION}" >> "$TMP_LIST"
  done < "$HEADS_FILE"
fi

if [[ ! -s "$TMP_LIST" ]]; then
  echo "No heads to run."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

cat "$TMP_LIST" | xargs -n 3 -P "$MAX_PARALLEL" bash -lc '
  layer="$0"
  hid="$1"
  pos="$2"
  echo "[RUN] ${layer}.${hid} (pos=${pos})"
  bash "'"$RUN_SCRIPT"'" \
    --layer "$layer" \
    --head "$hid" \
    --rounds "'"$ROUNDS"'" \
    --results-root "'"$RESULTS_ROOT"'" \
    --data-path "'"$DATA_PATH"'" \
    --standard-json "'"$STANDARD_JSON"'" \
    --model-path "'"$MODEL_PATH"'" \
    --attention-position "$pos"
'
