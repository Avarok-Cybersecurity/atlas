#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Existing recipe launcher path, sourced by run_cell.sh.
# shellcheck disable=SC2154,SC2034
serve_recipe() {
START_EPOCH=""
if [ "$ENGINE" = "atlas" ]; then
  ATLAS_ENV=(
    "NGPUS=$NGPUS" "EP_SIZE=$EP_SIZE" "TP_SIZE=$TP_SIZE"
    "PORT_BASE=$PORT" "BIND=0.0.0.0" "NCCL_PROFILE=default"
    "BOOT_TIMEOUT_S=$BOOT_CAP" "EXTRA_ARGS=$EXTRA_ARGS"
    "ATLAS_NODE_RUN_DIR=$NODE_RUN_DIR"
  )
  [ -n "$WARMUP_PROMPT" ] && ATLAS_ENV+=( "WARMUP_PROMPT=$WARMUP_PROMPT" )
  [ -n "${SPARK_BIN:-}" ] && ATLAS_ENV+=( "SPARK_BIN=$SPARK_BIN" )
  [ -n "${IMAGE:-}" ] && ATLAS_ENV+=( "IMAGE=$IMAGE" )

  show "env ${ATLAS_ENV[*]} bash $LAUNCHER --check-kernels $HF_ID"
  show "env ${ATLAS_ENV[*]} bash $LAUNCHER $HF_ID   # backgrounded; time_to_ready.sh owns the boot clock"
  if [ "$DRY_RUN" = "1" ]; then
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --dry-run "$HF_ID" 2>&1 | sed 's/^/  | /' \
      || exit $?
  else
    # `mkdir`, deliberately without -p: this cell must CREATE the directory it
    # hands the launcher. A directory that already exists may hold an earlier
    # run's rank records, and the finalizer's --stop reaches whatever the
    # directory records -- which is how a launch that was refused before it
    # created anything went on to stop somebody else's ranks.
    if mkdir -p "$(dirname "$NODE_RUN_DIR")" 2>/dev/null \
       && mkdir "$NODE_RUN_DIR" 2>/dev/null; then
      NODE_RUN_DIR_RESERVED=1
      echo "run dir:     $NODE_RUN_DIR (reserved by this cell)"
    else
      echo "REFUSED: $NODE_RUN_DIR could not be reserved for this cell."
      echo "  It already exists, so whatever it records is not this invocation's"
      echo "  to stop later. Nothing is launched."
      note_fail serve
    fi
    printf '%s\n' "${ATLAS_ENV[@]}" > "$SERVE_ENV"
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --dry-run "$HF_ID" > "$OUT/serve-dryrun.txt" 2>&1
    python3 - "$OUT/serve-dryrun.txt" "$SERVE_ARGV" <<'PY'
import pathlib, shlex, sys
line = next((l for l in pathlib.Path(sys.argv[1]).read_text().splitlines()
             if l.startswith("rank0_command: ")), None)
argv = shlex.split(line[len("rank0_command: "):]) if line else []
pathlib.Path(sys.argv[2]).write_bytes(b"\0".join(a.encode() for a in argv))
PY
    if [ "$NODE_RUN_DIR_RESERVED" = "1" ]; then
      if run_child env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --check-kernels "$HF_ID" \
        > "$OUT/check-kernels.txt" 2>&1; then
        START_EPOCH="$(date +%s)"
        env "${ATLAS_ENV[@]}" bash "$LAUNCHER" "$HF_ID" > "$OUT/serve.log" 2>&1 &
        SERVE_PID=$!
        CHILD_PIDS="$CHILD_PIDS $SERVE_PID"
        # The launcher pid is this invocation's to KILL. What is this invocation's
        # to STOP is decided later, by what the launcher actually recorded in the
        # run directory above -- a launcher that refuses records nothing.
        echo "launcher pid $SERVE_PID; log $OUT/serve.log"
      else
        note_fail serve
      fi
    fi
  fi
else
  VC_ARGS=( "$MODEL" "$SKU" --spec "$SPEC" --label "$RUN_LABEL" )
  show "VLLM_CONTAINER=$CONTAINER bash $HERE/vllm_control.sh ${VC_ARGS[*]} --id-file $CONTAINER_ID_FILE --image-file $CONTAINER_IMAGE_FILE"
  if [ "$DRY_RUN" = "1" ]; then
    VLLM_CONTAINER="$CONTAINER" VLLM_RECIPES="$VLLM_RECIPES" \
      bash "$HERE/vllm_control.sh" "${VC_ARGS[@]}" --dry-run 2>&1 | sed 's/^/  | /' \
      || exit $?
  else
    printf 'VLLM_IMAGE_DIGEST=%s\n' "${VLLM_IMAGE_DIGEST:-}" > "$SERVE_ENV"
    # Ownership is written down BEFORE the create, not after it: from the next
    # line on, a container of this cell's may exist whether or not its ID ever
    # comes back, and the finalizer has to be able to find it either way.
    cat > "$OWNER_JSON" <<OWNER
{
  "cell_id": "$CELL_ID",
  "container_name": "$CONTAINER",
  "run_label": "$RUN_LABEL",
  "container_id_file": "$CONTAINER_ID_FILE",
  "runner_pid": $$
}
OWNER
    CREATE_ATTEMPTED=1
    START_EPOCH="$(date +%s)"
    run_child env VLLM_CONTAINER="$CONTAINER" VLLM_RECIPES="$VLLM_RECIPES" \
      bash "$HERE/vllm_control.sh" "${VC_ARGS[@]}" --id-file "$CONTAINER_ID_FILE" \
      --image-file "$CONTAINER_IMAGE_FILE" \
      > "$OUT/serve.log" 2>&1
    serve_rc=$?
    if [ "$serve_rc" -eq 125 ]; then
      echo "docker run exited 125: a container named $CONTAINER already exists and"
      echo "  was NOT created by this invocation. Nothing of it is stopped or removed;"
      echo "  the serve stage fails and the cell records it."
      note_fail serve
    elif [ "$serve_rc" -ne 0 ]; then
      note_fail serve
    fi
    if [ -s "$CONTAINER_ID_FILE" ]; then
      CONTAINER_ID="$(cat "$CONTAINER_ID_FILE")"
      echo "container id: $CONTAINER_ID"
    fi
    sed -n 's/^docker run /docker run /p' "$OUT/serve.log" | head -1 > "$OUT/serve-cmd.txt"
    python3 - "$OUT/serve-cmd.txt" "$SERVE_ARGV" <<'PY'
import pathlib, shlex, sys
text = pathlib.Path(sys.argv[1]).read_text().strip()
argv = shlex.split(text) if text else []
pathlib.Path(sys.argv[2]).write_bytes(b"\0".join(a.encode() for a in argv))
PY
  fi
fi

}
