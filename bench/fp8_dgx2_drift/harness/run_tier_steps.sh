#!/usr/bin/env bash
# Run N opencode probes sequentially against the currently-running Atlas
# container and score each one. Intended for statistical comparison of
# Atlas drift-mitigation tiers (N≥10 per tier required to overcome the
# FP8 per-run variance).
#
# Usage:
#   ./run_tier.sh <tier-name> <N> [--container atlas-qwen-final]
#
# Outputs:
#   bench/fp8_dgx2_drift/harness/runs/run_<tier>_<i>.json   (per run)
#   bench/fp8_dgx2_drift/harness/reports/<tier>.csv         (aggregated)
#
# Each probe is the SAME prompt and target template; the target path
# carries the tier+run index so concurrent storage of artifacts is clean.

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tier-name> <N> [--container <name>]" >&2
  exit 2
fi

TIER="$1"
N="$2"
shift 2

CONTAINER="atlas-qwen-final"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${HARNESS_DIR}/runs"
mkdir -p "${RUNS_DIR}"

# Verify atlas container is up + responsive before burning time on probes.
if ! sudo docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q "${CONTAINER}"; then
  echo "FATAL: container '${CONTAINER}' is not running" >&2
  exit 3
fi
if ! curl -sS -m 5 http://localhost:8888/v1/models >/dev/null 2>&1; then
  echo "FATAL: atlas /v1/models not responding on localhost:8888" >&2
  exit 3
fi

# Capture the atlas startup-log marker so per-run windows can offset
# correctly. Each probe captures docker logs --since the start ts of
# that probe.

PROMPT_TEMPLATE='Please create a pure rust Axum project inside __TARGET__. Just have a ping/pong endpoint. Add tests, run them and prove all tests pass, then run the server and use curl to prove it works. Finally, tear down the server.'

echo "=== tier=${TIER} runs=${N} container=${CONTAINER} ===" >&2
echo "harness: ${HARNESS_DIR}" >&2

for i in $(seq 1 "${N}"); do
  TARGET="/tmp/harness-${TIER}-r${i}"
  OC_JSON="/tmp/harness-${TIER}-r${i}.json"
  OC_ERR="/tmp/harness-${TIER}-r${i}.err"
  ATLAS_LOG="/tmp/harness-${TIER}-r${i}.atlas.log"
  OUT_JSON="${RUNS_DIR}/run_${TIER}_${i}.json"

  rm -rf "${TARGET}" "${OC_JSON}" "${OC_ERR}" "${ATLAS_LOG}"
  mkdir -p "/tmp/harness-${TIER}-r${i}-cwd"
  cd "/tmp/harness-${TIER}-r${i}-cwd"

  PROMPT="${PROMPT_TEMPLATE//__TARGET__/${TARGET}}"

  echo "--- run ${i}/${N} target=${TARGET} ---" >&2

  START_TS=$(date +%s.%N)
  # opencode has its own internal timeout; we cap at 6 min as a hard ceiling.
  timeout 360 opencode run --dangerously-skip-permissions --agent harness --format json \
    "${PROMPT}" > "${OC_JSON}" 2> "${OC_ERR}" || true
  END_TS=$(date +%s.%N)

  # Atlas log window for THIS run only. Docker logs --since accepts
  # epoch-seconds (truncate decimals).
  START_TS_INT=${START_TS%.*}
  sudo docker logs "${CONTAINER}" --since "${START_TS_INT}" 2>&1 > "${ATLAS_LOG}" || true

  python3 "${HARNESS_DIR}/score_run.py" \
    --tier "${TIER}" \
    --run "${i}" \
    --target "${TARGET}" \
    --opencode-json "${OC_JSON}" \
    --opencode-stderr "${OC_ERR}" \
    --atlas-log-window "${ATLAS_LOG}" \
    --probe-start-ts "${START_TS}" \
    --probe-end-ts "${END_TS}" \
    --out "${OUT_JSON}"

  # Brief per-run summary so the harness operator can spot regressions
  # in real time.
  files_count=$(jq -r '.filesystem.files_count' "${OUT_JSON}")
  cargo_ok=$(jq -r '.cargo.cargo_toml_valid' "${OUT_JSON}")
  drift_lean=$(jq -r '.drift.write_content_starts_with_lean' "${OUT_JSON}")
  drift_empty=$(jq -r '.drift.write_empty_path' "${OUT_JSON}")
  drift_pathdrift=$(jq -r '.drift.write_path_drift_from_target' "${OUT_JSON}")
  wall=$(jq -r '.wall_time_s' "${OUT_JSON}")
  echo "    files=${files_count} cargo_valid=${cargo_ok} lean=${drift_lean} empty_path=${drift_empty} path_drift=${drift_pathdrift} wall=${wall}s" >&2
done

echo "=== tier ${TIER} complete (N=${N}). Run aggregate.py next. ===" >&2
