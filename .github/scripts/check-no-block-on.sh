#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Refuse .block_on( / .block_in_place( under the SCAN_DIRS trees.
#
# Extracted from tui-threading.yml so the standalone job and the batched
# `cheap checks` job run THE SAME CODE. Regenerated 2026-09-06 after
# main added an existence assertion for the scanned tree: an extracted
# copy goes stale the moment the original is improved, and taking the
# extraction side of that conflict would have silently reverted the fix.
#
# Expects SCAN_DIRS in the environment where the original used it.
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
# ★ The scanned trees must EXIST. `grep -r` on a missing directory
# exits 2, the `2>/dev/null` hides the reason, and `if hits=$(...)`
# reads any non-zero as "no hits" -- so this required check printed
# OK and exited 0 against a tree with no `tui/` at all. Verified:
# running the block below in an empty directory passes. A rename or a
# move of either tree would silently retire the rule while the check
# stayed green, which is the failure mode this whole gate exists to
# prevent. Assert the inputs before trusting the absence of output.
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
if hits=$(grep -rnE '\.(block_on|block_in_place)\(' \
            --include='*.rs' --exclude='*_tests.rs' \
            $SCAN_DIRS 2>/dev/null); then
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

