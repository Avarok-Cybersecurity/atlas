#!/usr/bin/env bash
# queue-perf-pr.sh — the serialized-landing protocol for record-bearing
# performance PRs.
#
# WHY THIS EXISTS. Benchmark records cannot compose across PRs: a campaign
# that covered PR A's tree does not cover a tree with PR B's kernels placed
# under it — the interaction is unmeasured, and the gate refuses by design.
# The merge queue tests exactly such composed trees, so a perf PR can be
# fully green at its head and bounce in the queue every time another perf PR
# lands ahead of it. Observed 2026-08-23 (#731/#732/#733): three green PRs,
# repeated bounces, zero code defects.
#
# THE PROTOCOL: freeze -> campaign -> queue alone.
#   1. Update the branch to CURRENT main and push. This is the freeze; any
#      later main movement restarts the protocol.
#   2. Run all ten gates against exactly that sha (on a box with exclusive
#      GPU), commit the records, push.
#   3. Queue the PR with NO other performance PR ahead of it. Do not add a
#      second record-bearing PR to the queue until this one has landed.
#
# This script performs step 1 and prints the step-2 campaign command; the
# campaign itself needs a GPU box and is deliberately not run from here.
#
# THREE TRAPS the printed command now spells out, each of which cost hours on
# 2026-08-28:
#   * ATLAS_HOME must be writable. If it is not, every gate fails instantly
#     with "recipe ... is not in the local index (0 cached)" and never says
#     why.
#   * The TTFT gates need TWO runs each on a box with no stored baseline. The
#     first one only creates the baseline and records `info`, which the gate
#     does not accept — so a ten-gate campaign silently comes back eight.
#   * The exit code is not the evidence. A BFCL run generated all 995
#     responses, died in scoring, and correctly wrote NO record; a driver that
#     trusted `rc` called that a pass. Ask
#     `--pull-request-gate-check` what was actually recorded.
#
# Usage: scripts/queue-perf-pr.sh <branch>
set -euo pipefail
BRANCH="${1:?usage: queue-perf-pr.sh <branch>}"
git fetch origin main "$BRANCH"
WT="$(mktemp -d)/wt"
git worktree add "$WT" "origin/$BRANCH"
cd "$WT"
git merge --no-edit origin/main -m "Merge main — freeze for the serialized queue protocol (see scripts/queue-perf-pr.sh)"
SHA=$(git rev-parse --short=10 HEAD)
git push origin "HEAD:$BRANCH"
echo
echo "FROZEN: $BRANCH @ $SHA"
echo "Now run, on a GPU box with the repo checked out at exactly $SHA:"
echo
echo '  # ATLAS_HOME must be WRITABLE. The gates read the recipe index and the'
echo '  # same-box baselines from $ATLAS_HOME/.., default ~/.atlas. If that'
echo '  # directory cannot be created, every gate dies instantly with'
echo '  # "recipe ... is not in the local index (0 cached)" and the cause is'
echo '  # nowhere in the message.'
echo '  export ATLAS_HOME=${ATLAS_HOME:-$HOME/.atlas}'
echo
echo '  # The TTFT gates compare against a stored SAME-BOX baseline. On a box'
echo '  # that has none, run 1 only CREATES it and records verdict `info`,'
echo '  # which the gate does not accept. They need two runs each; listing'
echo '  # them twice is the whole fix.'
echo '  for g in decode-floor vision-fidelity video-fidelity \'
echo '           ssm-state-poisoning-gate concurrency-sweep agentic-webserver \'
echo '           ttft-cold-gate ttft-cold-gate ttft-warm-gate ttft-warm-gate \'
echo '           bfcl-subset bfcl-subset-echolp; do'
echo '    timeout 21600 ./target/release/spark benchmark run "$g" --pull-request-gate --yes'
echo '    rc=$?   # capture IMMEDIATELY: a $(date) in the next line resets it'
echo '    echo "$g rc=$rc"'
echo '  done'
echo
echo '  # rc is a hint, not the evidence. Check what was actually recorded:'
echo "  ./target/release/spark benchmark --pull-request-gate-check --pr <N>"
echo
echo "then commit .benchmarks/ and push, wait for green, and queue the PR alone."
