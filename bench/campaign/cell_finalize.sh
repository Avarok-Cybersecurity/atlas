#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by run_cell.sh; functions share its per-cell state.
# shellcheck disable=SC2154,SC2034

record_cell_deadline() {
  [ "${CELL_DEADLINE_NOTED:-0}" != "1" ] || return 0
  [ -n "${CELL_DEADLINE_RECEIPT:-}" ] || return 1
  local detail=""
  detail="$(python3 - "$CELL_DEADLINE_RECEIPT" <<'PY'
import json, pathlib, sys
try:
    doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
    if doc.get('deadline_exceeded') is not True:
        raise ValueError('deadline has not expired')
    print(f"whole-cell deadline exceeded after {doc['timeout_s']:g}s; cleanup grace {doc['grace_s']:g}s")
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)
PY
  )" || return 1
  CELL_DEADLINE_NOTED=1
  note_fail "${CURRENT_STAGE:-preflight}"
  add_note "$detail during ${CURRENT_STAGE:-preflight}; receipt $CELL_DEADLINE_RECEIPT"
  return 0
}

cancel_cell_deadline() {
  [ -n "${CELL_DEADLINE_PID:-}" ] || return 0
  local watch_rc=0
  if python3 "$HERE/cell_deadline.py" cancel --receipt "$CELL_DEADLINE_RECEIPT" \
      --watchdog-pid "$CELL_DEADLINE_PID"; then
    # cancel waits on the pidfd. Reaping here therefore cannot wait for the
    # original whole-cell budget, including when the watchdog never armed.
    wait "$CELL_DEADLINE_PID" || watch_rc=$?
    if [ "$watch_rc" -ne 0 ]; then
      note_fail teardown
      add_note "deadline watchdog exited $watch_rc; inspect $CELL_DEADLINE_RECEIPT"
    fi
  else
    # An unacknowledged cancellation must not turn into an unbounded wait.
    # The watchdog observes the runner's exit through its target pidfd.
    note_fail teardown
    add_note "deadline watchdog cancellation unconfirmed; inspect $CELL_DEADLINE_RECEIPT and cell-deadline.log"
  fi
  CELL_DEADLINE_PID=""
}

# Idempotent: the EXIT trap fires after a signal handler and after normal
# completion, and neither must tear the same thing down twice.
finalize() {
  local rc=$? sig_rc=""
  if [ "$FINALIZED" = "1" ]; then return "$rc"; fi
  FINALIZED=1

  if [ -n "$INTERRUPT_SIG" ]; then
    case "$INTERRUPT_SIG" in
      INT) sig_rc=130 ;;
      HUP) sig_rc=129 ;;
      *)   sig_rc=143 ;;
    esac
    echo ""
    echo "!!! SIG$INTERRUPT_SIG during stage '${CURRENT_STAGE:-preflight}': releasing what"
    echo "    this cell created, then writing the artifact that says so."
    note_fail "${CURRENT_STAGE:-preflight}"
    add_note "interrupted by SIG$INTERRUPT_SIG during the ${CURRENT_STAGE:-preflight} stage: this cell was terminated before it finished, so its gates are unfinished rather than passed"
  fi

  # Before teardown, not after: teardown removes the container this cell's
  # identity is read of, and takes the launcher's record of it with it.
  capture_model_launch
  capture_engine_identity

  stage teardown "stage 6/7 teardown"
  teardown_owned

  stage validate "stage 7/7 assemble and validate"
  build_assemble
  show "${ASSEMBLE[*]}"
  show "python3 $HERE/validate_artifact.py $ARTIFACT"

  if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "dry-run: nothing launched, nothing written."
    exit "${sig_rc:-$rc}"
  fi

  local assembled_stage="$FAILING_STAGE" assembled_note="$EXTRA_NOTE"
  "${ASSEMBLE[@]}" || note_fail validate
  if [ ! -f "$ARTIFACT" ]; then
    echo "ERROR: the artifact was not written; there is nothing to validate." >&2
    cancel_cell_deadline
    exit "${sig_rc:-1}"
  fi
  python3 "$HERE/validate_artifact.py" "$ARTIFACT" || {
    note_fail validate
    # Re-assemble so the artifact's own verdict admits the validation failure
    # rather than claiming a verdict its shape does not support.
    build_assemble
    "${ASSEMBLE[@]}" --failing-stage validate >/dev/null
  }

  # TERM may arrive while an ordinary finalizer is already cleaning up or
  # validating. Keep its artifact correction inside the watchdog budget.
  if [ "$assembled_stage" != "$FAILING_STAGE" ] || [ "$assembled_note" != "$EXTRA_NOTE" ]; then
    assembled_stage="$FAILING_STAGE"; assembled_note="$EXTRA_NOTE"
    build_assemble
    "${ASSEMBLE[@]}" || note_fail validate
    python3 "$HERE/validate_artifact.py" "$ARTIFACT" || note_fail validate
  fi
  cancel_cell_deadline
  # Cancellation itself can fail, or observe expiry racing with the last
  # validation. This final file-only amendment records that outcome after
  # cancellation; engine cleanup is already complete and no work is launched.
  if [ "$assembled_stage" != "$FAILING_STAGE" ] || [ "$assembled_note" != "$EXTRA_NOTE" ]; then
    build_assemble
    "${ASSEMBLE[@]}" || note_fail validate
    python3 "$HERE/validate_artifact.py" "$ARTIFACT" || note_fail validate
  fi
  case "$INTERRUPT_SIG" in INT) sig_rc=130 ;; HUP) sig_rc=129 ;; TERM) sig_rc=143 ;; esac

  echo ""
  if [ -n "$sig_rc" ]; then
    echo "=== $CELL_ID: TERMINATED by SIG$INTERRUPT_SIG at stage '$FAILING_STAGE'; artifact written to $ARTIFACT ==="
    exit "$sig_rc"
  fi
  if [ -n "$FAILING_STAGE" ]; then
    echo "=== $CELL_ID: FAILED at stage '$FAILING_STAGE'; artifact written to $ARTIFACT ==="
    exit 1
  fi
  if [ "$MAIN_DONE" != "1" ]; then
    # An exit that came from neither the main flow nor a handled signal: the
    # cell is torn down and recorded, but the original status is its own.
    exit "$rc"
  fi
  echo "=== $CELL_ID: all gates passed; artifact written to $ARTIFACT ==="
  echo "    Verdict is PARTIAL until the paired cell from the other engine exists"
  echo "    within 24 h; re-run with --paired-artifact to promote it."
  exit 0
}

# A handler does none of the work itself: it records WHICH signal and goes
# through the same finalizer a clean finish does.
on_signal() {
  local deadline_was_noted="${CELL_DEADLINE_NOTED:-0}"
  [ -n "$INTERRUPT_SIG" ] || INTERRUPT_SIG="$1"
  if [ "$1" = "TERM" ] && record_cell_deadline; then
    if [ "$FINALIZED" = "1" ] && [ "$deadline_was_noted" != "1" ]; then
      # A new deadline during normal or operator-triggered finalization must
      # not skip cleanup; subsequent operator signals retain their semantics.
      # The watchdog retains its hard grace until cancellation at the end.
      return 0
    fi
  fi
  finalize
  # Reached only if the finalizer had already run (a second signal during
  # teardown). The process still has to end, and with this signal's status.
  case "$INTERRUPT_SIG" in INT) exit 130 ;; HUP) exit 129 ;; *) exit 143 ;; esac
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap finalize EXIT
# INT is trapped for the interactive Ctrl-C, which is delivered to the
# foreground process group. A shell STARTED in the background inherits SIGINT
# ignored, and a signal ignored on entry cannot be trapped -- bash's rule, not
# this script's. So a campaign driver that backgrounds its cells must terminate
# them with TERM, which is also what the regression in campaign_test.sh sends.
