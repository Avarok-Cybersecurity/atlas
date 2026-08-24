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
echo '  for g in ttft-cold-gate ttft-warm-gate vision-fidelity ssm-state-poisoning-gate \'
echo '           decode-floor concurrency-sweep video-fidelity bfcl-subset \'
echo '           bfcl-subset-echolp agentic-webserver; do'
echo '    timeout 21600 ./target/release/spark benchmark run "$g" --pull-request-gate --yes'
echo '  done'
echo "then commit .benchmarks/ and push, wait for green, and queue the PR alone."
