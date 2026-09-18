#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Refuse .block_on( / .block_in_place( under tui/ and recipe/.
#
# Extracted from tui-threading.yml so the standalone job and the batched
# `cheap checks` job run THE SAME CODE. The body below is main's verbatim,
# INCLUDING the existence assertion and the grep-status check: the first
# extraction of this step dropped both, and the certification self-test caught
# it — a grep over a missing directory exits 2, the `if` reads that as "no
# hits", and the gate goes green over a tree it never read.
#
# SCAN_DIRS is the SSOT for the trees this rule covers and is passed by the
# caller (the workflow sets it); the default here matches the workflow so the
# script is runnable by hand.
set -euo pipefail
: "${SCAN_DIRS:=crates/spark-server/src/tui/ crates/spark-server/src/recipe/}"
set -euo pipefail
# Only real call syntax counts -- comments explaining the rule are
# allowed to name it.
#
# ★ TEST FILES ARE EXCLUDED, and the distinction is the whole point of
# the rule rather than a loophole in it. What is forbidden is BLOCKING
# THE RENDER THREAD on a future: that freezes the dashboard and, since
# the TUI is the server's foreground, hides the server with it. A test
# that stands up its own runtime and drives an async fn to completion
# is not the render thread and cannot freeze anything -- it is the
# ordinary way to test async code, and forbidding it would push the
# tests toward worse shapes (sleep-and-poll) for no safety gained.
#
# The exclusion is by FILENAME (`*_tests.rs`), which is the repo's
# convention for `#[cfg(test)]` siblings mounted with `#[path]`. If
# production code is ever put in a file named that way, this check
# will not see it -- that is the cost, and it is smaller than the
# alternative.
# ★ TWO WAYS THIS CHECK WENT GREEN WITHOUT SCANNING, BOTH CLOSED.
# Each of the two branches that met here caught one and missed the
# other.
#
# (1) THE TREES MUST EXIST. `grep -r` on a missing directory exits 2,
# `2>/dev/null` hid the reason, and `if hits=$(...)` read any non-zero
# as "no hits" — so this required check printed OK against a tree with
# no `tui/` at all. Verified: the old block passed in an empty
# directory. A rename would have retired the rule silently.
#
# (2) EXIT 2 MEANS THE SCAN FAILED — a missing directory OR an
# unreadable file — and GNU grep returns 2 even when it ALSO found
# matches. Asserting the directories exist still lets an unreadable
# file report clean, so capture the status and refuse on >= 2.
#
# Stderr no longer goes to /dev/null: the reason belongs in the log of
# the job that refuses.
for d in $SCAN_DIRS; do
  [ -d "$d" ] || {
    echo "::error::$d does not exist, so this check scanned nothing."
    echo "The render-thread rule is pinned to these trees. If one moved,"
    echo "update SCAN_DIRS in this workflow in the same commit -- an"
    echo "unscanned tree is an unenforced rule, and it would have gone"
    echo "green without this line."
    exit 1
  }
done
rc=0
hits=$(grep -rnE '\.(block_on|block_in_place)\(' \
         --include='*.rs' --exclude='*_tests.rs' \
         $SCAN_DIRS) || rc=$?
if [ "$rc" -ge 2 ]; then
  echo "::error::The block_on scan did not run (grep exited $rc). This check cannot vouch for a tree it failed to read. The directories exist (asserted above), so this is most likely an unreadable file; grep's own message is in the log."
  exit 1
fi
if [ "$rc" -eq 0 ]; then
  echo "::error::The TUI render thread must never poll a future."
  echo "$hits"
  echo
  echo "The render loop's only interaction with async work is try_recv"
  echo "on a channel (see tui/chat.rs and tui/bench_preflight.rs for"
  echo "the sanctioned shape: spawn on the runtime, answer over an"
  echo "mpsc the tick drains). Blocking the render thread on a future"
  echo "freezes the dashboard and, because the TUI is the server's"
  echo "foreground, hides the server with it."
  exit 1
fi
echo "OK: no block_on/block_in_place under tui/ or recipe/"
