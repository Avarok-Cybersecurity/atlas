#!/usr/bin/env bash
# Variant of run_tier.sh for running against a REMOTE vLLM endpoint
# (no local docker container). Used to A/B vLLM-FP8 on dgx2 vs Atlas
# on dgx1 with the same N=10 cargo_valid harness, and to validate
# the new harness infrastructure (warm-up halt + webserver test).
#
# Skips:
#   - sudo docker ps container check (vLLM is on dgx2 via SSH tunnel)
#   - per-run docker logs window (no atlas-side metrics to gather)
#
# Reads:
#   - API at $API_BASE (default http://localhost:8889/v1 -- tunnel to dgx2:8888)
#   - XDG_CONFIG_HOME_OVERRIDE (default /tmp/oc-tunnel-config) — opencode
#     config pointing at the tunnel.
#
# Usage:
#   API_BASE=http://localhost:8889/v1 ./run_tier_vllm.sh <tier-name> <N>

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tier-name> <N>" >&2
  exit 2
fi

TIER="$1"
N="$2"

API_BASE="${API_BASE:-http://localhost:8889/v1}"

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${HARNESS_DIR}/runs"
mkdir -p "${RUNS_DIR}"

# Verify the vLLM endpoint is reachable.
if ! curl -sS -m 5 "${API_BASE}/models" >/dev/null 2>&1; then
  echo "FATAL: vLLM /v1/models not responding at ${API_BASE}" >&2
  exit 3
fi

# ── Warm-up sanity check (HALTS on failure) ───────────────────────
warmup_endpoint() {
  local api="$1"
  local label="$2"
  echo "[warmup] ${label} ${api} ..." >&2
  local body
  body=$(curl -sS -m 60 "${api}/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"What is 2+2? Respond with just the number."}],"max_tokens":80,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1)
  local merged
  merged=$(echo "${body}" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); m = d.get('choices',[{}])[0].get('message',{}); print((m.get('content','') or '') + ' ' + (m.get('reasoning_content','') or ''))" 2>/dev/null)
  if [[ -z "${merged}" ]]; then
    echo "[warmup] FATAL: ${label} returned no parseable response" >&2
    echo "[warmup] raw body (first 400 chars): ${body:0:400}" >&2
    exit 4
  fi
  if ! echo "${merged}" | grep -q '4'; then
    echo "[warmup] FATAL: ${label} did not emit '4' for '2+2' — catastrophic regression, halting" >&2
    echo "[warmup] response excerpt: ${merged:0:300}" >&2
    exit 4
  fi
  echo "[warmup] ${label} OK" >&2
}

warmup_endpoint "${API_BASE}" "vllm-remote"

# ── Prompt (SSOT: byte-identical to run_tier.sh; uses opencode --dir so the
#    prompt carries no per-run path — the model's cwd IS the scored target) ──
PROMPT='Please create a pure rust Axum project here in the current working directory. Just have a ping/pong endpoint. The server MUST bind to the port from the ATLAS_HARNESS_PORT env var (default 3001) — use `let port: u16 = std::env::var("ATLAS_HARNESS_PORT").unwrap_or_else(|_| "3001".to_string()).parse().unwrap();` then bind to `0.0.0.0:port`. Add tests, run them and prove all tests pass, then run the server and use curl to prove it works. Finally, tear down the server.'

echo "=== tier=${TIER} runs=${N} api=${API_BASE} ===" >&2
echo "harness: ${HARNESS_DIR}" >&2

for i in $(seq 1 "${N}"); do
  TARGET="/tmp/harness-${TIER}-r${i}"
  OC_JSON="/tmp/harness-${TIER}-r${i}.json"
  OC_ERR="/tmp/harness-${TIER}-r${i}.err"
  EMPTY_LOG="/tmp/harness-${TIER}-r${i}.atlas.log"
  OUT_JSON="${RUNS_DIR}/run_${TIER}_${i}.json"

  rm -rf "${TARGET}" "${OC_JSON}" "${OC_ERR}" "${EMPTY_LOG}"
  : > "${EMPTY_LOG}"          # empty atlas log window — score_run.py handles it
  mkdir -p "${TARGET}"        # opencode --dir == scored target (no cwd/target split)

  echo "--- run ${i}/${N} target=${TARGET} ---" >&2

  START_TS=$(date +%s.%N)
  ATLAS_HARNESS_PORT=3001 \
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME_OVERRIDE:-/tmp/oc-tunnel-config}" \
    timeout "${OC_TIMEOUT:-360}" opencode run --dangerously-skip-permissions --dir "${TARGET}" --format json \
    "${PROMPT}" > "${OC_JSON}" 2> "${OC_ERR}" || true
  END_TS=$(date +%s.%N)

  ATLAS_HARNESS_PORT=3001 \
    python3 "${HARNESS_DIR}/score_run.py" \
    --tier "${TIER}" \
    --run "${i}" \
    --target "${TARGET}" \
    --opencode-json "${OC_JSON}" \
    --opencode-stderr "${OC_ERR}" \
    --atlas-log-window "${EMPTY_LOG}" \
    --probe-start-ts "${START_TS}" \
    --probe-end-ts "${END_TS}" \
    --webserver-port 3001 \
    --out "${OUT_JSON}"

  files_count=$(jq -r '.filesystem.files_count' "${OUT_JSON}")
  cargo_ok=$(jq -r '.cargo.cargo_toml_valid' "${OUT_JSON}")
  webserver_ok=$(jq -r '.webserver.webserver_ok // false' "${OUT_JSON}")
  drift_lean=$(jq -r '.drift.write_content_starts_with_lean' "${OUT_JSON}")
  drift_empty=$(jq -r '.drift.write_empty_path' "${OUT_JSON}")
  drift_pathdrift=$(jq -r '.drift.write_path_drift_from_target' "${OUT_JSON}")
  wall=$(jq -r '.wall_time_s' "${OUT_JSON}")
  echo "    files=${files_count} cargo_valid=${cargo_ok} webserver_ok=${webserver_ok} lean=${drift_lean} empty_path=${drift_empty} path_drift=${drift_pathdrift} wall=${wall}s" >&2
done

echo "=== tier ${TIER} complete (N=${N}). Aggregating... ===" >&2
# Exit code = total cargo+webserver failure count (0 = all green).
python3 "${HARNESS_DIR}/aggregate.py" --tier "${TIER}" >&2
agg_rc=$?
echo "=== tier ${TIER}: exit code ${agg_rc} (total cargo+webserver failures) ===" >&2
exit "${agg_rc}"
