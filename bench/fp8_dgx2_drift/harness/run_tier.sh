#!/usr/bin/env bash
# Run N opencode probes sequentially against the currently-running Atlas
# container (and optionally a remote vLLM via SSH tunnel) and score each one.
# Intended for statistical comparison of Atlas drift-mitigation tiers
# (N≥10 per tier required to overcome the FP8 per-run variance).
#
# Usage:
#   ./run_tier.sh <tier-name> <N>
#     [--container <name>]          (default atlas-qwen-final)
#     [--split-dgx]                  (run N/2 locally + N/2 on dgx2 via tunnel)
#     [--remote-api <URL>]           (default http://localhost:8889/v1)
#     [--cosine-mode]                (use cosine_run.py diagnostic instead of opencode)
#     [--skip-warmup]                (skip the "What is 2+2?" sanity check)
#
# Outputs:
#   bench/fp8_dgx2_drift/harness/runs/run_<tier>_<i>.json   (per run)
#   bench/fp8_dgx2_drift/harness/reports/<tier>.csv         (aggregated)
#
# Each probe is the SAME prompt and target template; the target path
# carries the tier+run index so concurrent storage of artifacts is clean.
#
# Warm-up: before any harness iteration runs, a direct API probe
# ("What is 2+2?") asserts the model responds with "4". HALTS on
# failure — saves the operator from waiting 25 min on a catastrophic
# regression.

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tier-name> <N> [--container <name>] [--split-dgx] [--remote-api URL] [--cosine-mode] [--skip-warmup]" >&2
  exit 2
fi

TIER="$1"
N="$2"
shift 2

CONTAINER="atlas-qwen-final"
SPLIT_DGX=0
REMOTE_API="http://localhost:8889/v1"
COSINE_MODE=0
SKIP_WARMUP=0
REMOTE_ONLY=0
BAIL=0
# --claude-code: drive Claude Code (the `claude` CLI) against Atlas instead of
# opencode, via `sudo -u claude env ANTHROPIC_BASE_URL=... claude -p ...`.
# Reproduces the non-opencode-client looping/garbling failure. Defaults to
# plan mode (CC_PERMISSION_MODE), the regime in which the failure was reported.
CLAUDE_CODE=0
# --prompt-file PATH: read the agent task PROMPT from a file instead of the
# built-in Axum ping/pong default (overrides PROMPT). `--prompt-file -` reads
# stdin. Lets the graduated-difficulty ladder feed prompts via the CLI.
PROMPT_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="$2"; shift 2 ;;
    --split-dgx) SPLIT_DGX=1; shift ;;
    --remote-api) REMOTE_API="$2"; shift 2 ;;
    --cosine-mode) COSINE_MODE=1; shift ;;
    --skip-warmup) SKIP_WARMUP=1; shift ;;
    --remote-only) REMOTE_ONLY=1; shift ;;
    --bail) BAIL=1; shift ;;
    --claude-code) CLAUDE_CODE=1; shift ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${HARNESS_DIR}/runs"
mkdir -p "${RUNS_DIR}"

LOCAL_API="http://localhost:8888/v1"

# ── Cosine mode short-circuit ────────────────────────────────────
if [[ "${COSINE_MODE}" == "1" ]]; then
  echo "=== cosine-mode: running cosine_run.py (per-layer drift diagnostic) ===" >&2
  exec python3 "${HARNESS_DIR}/../cosine_run.py"
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
  # Inspect either content or reasoning_content (some configs route to reasoning).
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

if [[ "${SKIP_WARMUP}" == "0" ]]; then
  # Verify local container is up + responsive
  if ! sudo docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q "${CONTAINER}"; then
    echo "FATAL: container '${CONTAINER}' is not running" >&2
    exit 3
  fi
  if ! curl -sS -m 5 "${LOCAL_API}/models" >/dev/null 2>&1; then
    echo "FATAL: atlas /v1/models not responding on localhost:8888" >&2
    exit 3
  fi
  warmup_endpoint "${LOCAL_API}" "local-atlas"

  if [[ "${SPLIT_DGX}" == "1" ]]; then
    if ! curl -sS -m 5 "${REMOTE_API}/models" >/dev/null 2>&1; then
      echo "FATAL: remote vLLM/atlas /v1/models not responding at ${REMOTE_API}" >&2
      echo "       (expected an SSH tunnel: ssh -L 8889:localhost:8888 claude@10.10.10.2)" >&2
      exit 3
    fi
    warmup_endpoint "${REMOTE_API}" "remote-dgx2"
  fi
fi

# ── Prompt template ────────────────────────────────────────────────
# Constant across runs: the target directory is passed to opencode via
# `--dir` so the prompt itself carries no per-run path. This (a) gives a
# bit-identical token sequence for every run, enabling prefix-cache reuse,
# and (b) removes tokenization noise that would otherwise confound the
# A/B comparison between tiers.
PROMPT='Please create a pure rust Axum project here in the current working directory. Just have a ping/pong endpoint. The server MUST bind to the port from the ATLAS_HARNESS_PORT env var (default 3001) — use `let port: u16 = std::env::var("ATLAS_HARNESS_PORT").unwrap_or_else(|_| "3001".to_string()).parse().unwrap();` then bind to `0.0.0.0:port`. Add tests, run them and prove all tests pass, then run the server and use curl to prove it works. Finally, tear down the server.'

# CLI prompt override: --prompt-file PATH (or `-` for stdin). Enables the
# graduated-difficulty ladder to drive the harness via piped/file input.
if [[ -n "${PROMPT_FILE}" ]]; then
  if [[ "${PROMPT_FILE}" == "-" ]]; then
    PROMPT="$(cat)"
  else
    [[ -r "${PROMPT_FILE}" ]] || { echo "FATAL: --prompt-file '${PROMPT_FILE}' not readable" >&2; exit 2; }
    PROMPT="$(cat "${PROMPT_FILE}")"
  fi
  [[ -n "${PROMPT}" ]] || { echo "FATAL: prompt from '${PROMPT_FILE}' is empty" >&2; exit 2; }
fi

# ── Per-iteration runner ───────────────────────────────────────────
run_one() {
  local i="$1"
  local api="$2"
  local extra_env="$3"
  local label="$4"

  local TARGET="/tmp/harness-${TIER}-r${i}"
  local OC_JSON="/tmp/harness-${TIER}-r${i}.json"
  local OC_ERR="/tmp/harness-${TIER}-r${i}.err"
  local ATLAS_LOG="/tmp/harness-${TIER}-r${i}.atlas.log"
  local OUT_JSON="${RUNS_DIR}/run_${TIER}_${i}.json"

  rm -rf "${TARGET}" "${OC_JSON}" "${OC_ERR}" "${ATLAS_LOG}"
  : > "${ATLAS_LOG}"  # empty by default; populated below if local
  # opencode's --dir is the agent's cwd; we pre-create it so opencode can
  # write into it on the first tool call.
  mkdir -p "${TARGET}"

  echo "--- run ${i}/${N} [${label}] target=${TARGET} ---" >&2

  local START_TS END_TS START_TS_INT
  START_TS=$(date +%s.%N)
  # opencode has its own internal timeout; we cap at OC_TIMEOUT seconds as a
  # hard ceiling. Default 360 (6 min) is the established harness ceiling; the
  # OC_TIMEOUT env var overrides it for de-confounding experiments (e.g. slow
  # full-BF16 runs that need a longer agentic budget to finish).
  # ATLAS_HARNESS_PORT is exposed both to opencode (so the model can read it
  # to write port-reading Rust) AND to score_run.py (so it can curl the right port).
  # --dir sets opencode's working directory; the model sees only "current
  # working directory" in the prompt, never the absolute path.
  if [[ "${CLAUDE_CODE}" == "1" ]]; then
    # Drive Claude Code against Atlas. Runs as user `claude` with its real
    # ~/.claude config (model=claude-opus-4-8, alwaysThinking, effort=high) so
    # this faithfully reproduces the reported failure regime. ANTHROPIC_BASE_URL
    # routes to Atlas; cwd is the target dir (claude has no --dir flag). Default
    # plan mode (CC_PERMISSION_MODE) — the regime in which the loop was reported.
    # Prompt is piped via stdin (NOT a positional): claude's `--add-dir` is
    # variadic and would otherwise swallow a trailing prompt arg. cwd is the
    # target dir, so claude already has write access there (no --add-dir needed).
    # `timeout` runs INSIDE sudo (as user claude) so it directly parents the
    # claude process and reliably kills it on expiry — with `timeout` OUTSIDE
    # sudo, the SIGTERM hits sudo and the grandchild claude survives as an
    # orphan past the deadline (-k 10 sends SIGKILL 10s after SIGTERM).
    ( cd "${TARGET}" && \
      printf '%s' "${PROMPT}" | sudo -n -u claude \
        timeout -k 10 "${OC_TIMEOUT:-360}" \
        env \
          ANTHROPIC_BASE_URL=http://localhost:8888 \
          ANTHROPIC_AUTH_TOKEN=dummy \
          ATLAS_HARNESS_PORT=${HPORT:-3001} \
          /workspace/.local/bin/claude -p \
            --output-format stream-json --verbose \
            --permission-mode "${CC_PERMISSION_MODE:-plan}" \
        ) > "${OC_JSON}" 2> "${OC_ERR}" || true
  else
    # opencode: --dir sets opencode's working directory; the model sees only
    # "current working directory" in the prompt, never the absolute path.
    ATLAS_HARNESS_PORT=${HPORT:-3001} \
      env ${extra_env} \
      timeout "${OC_TIMEOUT:-360}" opencode run --dangerously-skip-permissions --dir "${TARGET}" --format json \
      "${PROMPT}" > "${OC_JSON}" 2> "${OC_ERR}" || true
  fi
  END_TS=$(date +%s.%N)

  # Reap any server the agent backgrounded (e.g. `cargo run &` to self-test its
  # /ping endpoint). On the opencode timeout SIGTERM such a process reparents to
  # init (PPID=1) and KEEPS HOLDING ITS PORT — across runs that leaked a zombie
  # the scorer's curl then hit, producing false positives/negatives. The scorer
  # now uses an ephemeral port (so it is already isolated), but we reap the leak
  # at the source too. Identify victims PRECISELY by working directory == this
  # run's target dir, so we never touch the Atlas container or anything else.
  # Same-user processes (opencode runs as us), no sudo needed.
  if [[ -n "${TARGET}" && -d "${TARGET}" ]]; then
    _tdir_real=$(readlink -f "${TARGET}" 2>/dev/null || echo "${TARGET}")
    for _pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
      _cwd=$(readlink -f "/proc/${_pid}/cwd" 2>/dev/null) || continue
      case "${_cwd}" in
        "${_tdir_real}"|"${_tdir_real}"/*) kill -9 "${_pid}" 2>/dev/null || true ;;
      esac
    done
  fi

  # Atlas log window for THIS run only (local only).
  if [[ "${label}" == "local" ]]; then
    START_TS_INT=${START_TS%.*}
    sudo docker logs "${CONTAINER}" --since "${START_TS_INT}" 2>&1 > "${ATLAS_LOG}" || true
  fi

  ATLAS_HARNESS_PORT=${HPORT:-3001} \
    python3 "${HARNESS_DIR}/score_run.py" \
    --tier "${TIER}" \
    --run "${i}" \
    --target "${TARGET}" \
    --opencode-json "${OC_JSON}" \
    --opencode-stderr "${OC_ERR}" \
    --atlas-log-window "${ATLAS_LOG}" \
    --probe-start-ts "${START_TS}" \
    --probe-end-ts "${END_TS}" \
    --webserver-port ${HPORT:-3001} \
    --out "${OUT_JSON}"

  local files_count cargo_ok drift_lean drift_empty drift_pathdrift wall webserver_ok
  files_count=$(jq -r '.filesystem.files_count' "${OUT_JSON}")
  cargo_ok=$(jq -r '.cargo.cargo_toml_valid' "${OUT_JSON}")
  drift_lean=$(jq -r '.drift.write_content_starts_with_lean' "${OUT_JSON}")
  drift_empty=$(jq -r '.drift.write_empty_path' "${OUT_JSON}")
  drift_pathdrift=$(jq -r '.drift.write_path_drift_from_target' "${OUT_JSON}")
  wall=$(jq -r '.wall_time_s' "${OUT_JSON}")
  webserver_ok=$(jq -r '.webserver.webserver_ok // false' "${OUT_JSON}")
  echo "    files=${files_count} cargo_valid=${cargo_ok} webserver_ok=${webserver_ok} lean=${drift_lean} empty_path=${drift_empty} path_drift=${drift_pathdrift} wall=${wall}s" >&2

  # Claude-Code confirm signal: plan mode writes no files, so the cargo/webserver
  # line is not the loop signal. Report (a) longest run of repeated lines in the
  # captured assistant text (degeneration fingerprint) and (b) Atlas-side
  # loop/repetition watchdog fires during this run's window.
  if [[ "${CLAUDE_CODE}" == "1" ]]; then
    local cc_rep cc_wd
    cc_rep=$(python3 - "${OC_JSON}" <<'PY' 2>/dev/null || echo "parse_error"
import json, sys, re
txt = []
for ln in open(sys.argv[1], errors="replace"):
    ln = ln.strip()
    if not ln:
        continue
    try:
        e = json.loads(ln)
    except Exception:
        continue
    # stream-json: assistant message events carry content blocks
    msg = e.get("message") if isinstance(e, dict) else None
    if isinstance(msg, dict):
        for blk in msg.get("content", []) or []:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                txt.append(blk["text"])
    if isinstance(e, dict) and isinstance(e.get("result"), str):
        txt.append(e["result"])
blob = "\n".join(txt)
lines = [l.strip() for l in blob.splitlines() if len(l.strip()) > 12]
# longest run of an identical (non-trivial) line repeating consecutively
best = 1; cur = 1
for a, b in zip(lines, lines[1:]):
    cur = cur + 1 if a == b else 1
    best = max(best, cur)
# also count total distinct vs duplicate lines (paraphrase loops collapse this)
dup = len(lines) - len(set(lines))
print(f"chars={len(blob)} lines={len(lines)} max_consecutive_repeat={best} dup_lines={dup}")
PY
)
    cc_wd=$(grep -ciE 'loop|repeat|watchdog|stuck|NoSsmSnapshot|fuzzy|simhash|attractor|degener' "${ATLAS_LOG}" 2>/dev/null || echo 0)
    echo "    [claude-code] ${cc_rep} | atlas_watchdog_hits=${cc_wd}  (raw: ${OC_JSON}, atlas-log: ${ATLAS_LOG})" >&2
  fi

  # --bail: exit immediately on the first failure (cargo_valid != true OR webserver_ok != true).
  if [[ "${BAIL}" == "1" ]] && { [[ "${cargo_ok}" != "true" ]] || [[ "${webserver_ok}" != "true" ]]; }; then
    echo "[bail] run ${i} failed cargo_valid=${cargo_ok} webserver_ok=${webserver_ok} — exiting early (--bail)" >&2
    exit 5
  fi
}

# ── Iteration loop ─────────────────────────────────────────────────
echo "=== tier=${TIER} runs=${N} container=${CONTAINER} split_dgx=${SPLIT_DGX} ===" >&2
echo "harness: ${HARNESS_DIR}" >&2

if [[ "${SPLIT_DGX}" == "1" ]]; then
  # Split N into two halves; run N/2 locally and N/2 against remote in parallel
  local_count=$(( N / 2 ))
  remote_count=$(( N - local_count ))
  echo "split: local=${local_count} remote=${remote_count} (remote=${REMOTE_API})" >&2

  # Local half: indices 1..local_count
  (
    for i in $(seq 1 "${local_count}"); do
      run_one "${i}" "${LOCAL_API}" "" "local"
    done
  ) &
  LOCAL_PID=$!

  # Remote half: indices (local_count+1)..N
  (
    for i in $(seq $((local_count + 1)) "${N}"); do
      # XDG_CONFIG_HOME swaps opencode config to the tunnel one (vLLM-side)
      run_one "${i}" "${REMOTE_API}" "XDG_CONFIG_HOME=${XDG_CONFIG_HOME_OVERRIDE:-/tmp/oc-tunnel-config}" "remote"
    done
  ) &
  REMOTE_PID=$!

  wait "${LOCAL_PID}" "${REMOTE_PID}"
elif [[ "${REMOTE_ONLY}" == "1" ]]; then
  echo "remote-only: all ${N} runs against ${REMOTE_API}" >&2
  for i in $(seq 1 "${N}"); do
    run_one "${i}" "${REMOTE_API}" "XDG_CONFIG_HOME=${XDG_CONFIG_HOME_OVERRIDE:-/tmp/oc-tunnel-config}" "remote"
  done
else
  for i in $(seq 1 "${N}"); do
    run_one "${i}" "${LOCAL_API}" "" "local"
  done
fi

echo "=== tier ${TIER} complete (N=${N}). Aggregating... ===" >&2
# Exit code = total failure count (cargo + webserver) for this tier, so a clean
# 10/10 run exits 0 and `run_tier.sh ... && echo PASS` gates correctly.
python3 "${HARNESS_DIR}/aggregate.py" --tier "${TIER}" >&2
agg_rc=$?
echo "=== tier ${TIER}: exit code ${agg_rc} (total cargo+webserver failures; 0 = all green) ===" >&2
exit "${agg_rc}"
