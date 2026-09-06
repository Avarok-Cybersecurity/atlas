#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Refuse .block_on( / .block_in_place( under tui/ and recipe/.
#
# Extracted verbatim from tui-threading.yml so the standalone job and the batched
# `cheap checks` job run THE SAME CODE during the transition. If these two
# ever disagree, the batch is not a faithful merge and the transition is not
# safe to complete.
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
if hits=$(grep -rnE '\.(block_on|block_in_place)\(' \
            --include='*.rs' --exclude='*_tests.rs' \
            crates/spark-server/src/tui/ \
            crates/spark-server/src/recipe/ 2>/dev/null); then
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

