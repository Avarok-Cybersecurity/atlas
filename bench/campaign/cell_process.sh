#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Explicit process path for providers whose rental already is a container.
# shellcheck disable=SC2154,SC2034

serve_process() {
  python3 - "$PROCESS_RENDER_JSON" <<'PY'
import json, sys
doc = json.loads(sys.argv[1])
print('process_argv: ' + json.dumps(doc['argv']))
print('process_environment: ' + json.dumps(doc['environment'], sort_keys=True))
PY
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  if ! mkdir "$PROCESS_RUN_DIR"; then
    note_fail serve
    add_note "process run directory already exists; no existing owner record is this cell's"
    return 0
  fi
  PROCESS_DIR_RESERVED=1
  if ! python3 "$HERE/process_recipe.py" "${PROCESS_RENDER_ARGS[@]}" --stage "$OUT" \
      > "$OUT/process-render.log" 2>&1; then
    note_fail serve
    return 0
  fi
  if ! python3 - "$OUT/process-env.json" "$SERVE_ENV" <<'PY'
import json, pathlib, sys
env = json.loads(pathlib.Path(sys.argv[1]).read_text())
if any('\n' in value or '\r' in value for value in env.values()):
    raise SystemExit('multiline environment value cannot be recorded as KEY=value')
pathlib.Path(sys.argv[2]).write_text(''.join(f'{key}={value}\n' for key, value in sorted(env.items())))
PY
  then note_fail serve; return 0; fi
  if [ "$ENGINE" = "atlas" ]; then
    if ! run_child python3 "$HERE/process_exec.py" --argv-nul "$OUT/audit.argv" \
        --env-json "$OUT/process-env.json" > "$OUT/check-kernels.txt" 2>&1; then
      note_fail serve
      return 0
    fi
  fi
  START_EPOCH="$(date +%s)"
  if ! run_child python3 "$HERE/process_launch.py" start --record "$PROCESS_RECORD" \
      --evidence "$OUT/process-launch.json" --log "$OUT/serve.log" \
      --argv-nul "$SERVE_ARGV" --env-json "$OUT/process-env.json" \
      > "$OUT/process-start.log" 2>&1; then
    note_fail serve
  fi
}

capture_process_launch() {
  [ "$PROCESS_DIR_RESERVED" = "1" ] || return 0
  [ -f "$PROCESS_RECORD" ] || return 0
  if python3 "$HERE/process_launch.py" capture --record "$PROCESS_RECORD" \
      --evidence "$OUT/process-launch.json" > "$OUT/process-capture.log" 2>&1; then
    MODEL_LAUNCH_PROCESS_JSON="$OUT/process-launch.json"
    MODEL_LAUNCH_PROCESS_OWNER_JSON="$PROCESS_RECORD"
  else
    add_note "process launch capture failed; model revision remains unproven"
  fi
}

stop_process() {
  if [ "$DRY_RUN" = "1" ]; then
    show "python3 $HERE/process_launch.py stop --record $PROCESS_RECORD"
    return 0
  fi
  [ "$PROCESS_DIR_RESERVED" = "1" ] || return 0
  [ -f "$PROCESS_RECORD" ] || return 0
  if ! python3 "$HERE/process_launch.py" stop --record "$PROCESS_RECORD" \
      > "$OUT/process-stop.log" 2>&1; then
    note_fail teardown
    add_note "owned process cleanup could not be confirmed; inspect process-stop.log"
  fi
}
