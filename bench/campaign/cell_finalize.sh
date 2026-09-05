#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by run_cell.sh; functions share its per-cell state.
# shellcheck disable=SC2154,SC2034

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

  step "stage 6/7 teardown"
  teardown_owned

  step "stage 7/7 assemble and validate"
  build_assemble
  show "${ASSEMBLE[*]}"
  show "python3 $HERE/validate_artifact.py $ARTIFACT"

  if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "dry-run: nothing launched, nothing written."
    exit "${sig_rc:-$rc}"
  fi

  "${ASSEMBLE[@]}" || note_fail validate
  if [ ! -f "$ARTIFACT" ]; then
    echo "ERROR: the artifact was not written; there is nothing to validate." >&2
    exit "${sig_rc:-1}"
  fi
  python3 "$HERE/validate_artifact.py" "$ARTIFACT" || {
    note_fail validate
    # Re-assemble so the artifact's own verdict admits the validation failure
    # rather than claiming a verdict its shape does not support.
    build_assemble
    "${ASSEMBLE[@]}" --failing-stage validate >/dev/null
  }

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
  [ -n "$INTERRUPT_SIG" ] || INTERRUPT_SIG="$1"
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
