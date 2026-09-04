#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Classify the change set of the current event, so a PR that only edits the web
# properties does not have to wait on the Rust and CUDA work.
#
# Emits to $GITHUB_OUTPUT:
#   web_only      true iff >=1 file changed AND every changed file is under
#                 site/ blog/ book/ web-shared/
#   web_touched   true iff any changed file is under those trees
#   blog_touched  true iff any changed file is under blog/
#
# ── Two invariants, and the reason for each ─────────────────────────────────
#
# FAIL OPEN. Any event this does not understand, and any error, must classify
# as "run everything". GitHub counts a SKIPPED required job as satisfied, so a
# wrong `true` here would not fail loudly — it would silently merge un-gated
# Rust changes. Consumers must additionally write their gate as
# `!cancelled() && ... != 'true'`, so a FAILED run of this script also runs
# everything. Both halves are needed; neither is decorative.
#
# NEVER filter the trigger. A `paths:` filter on `pull_request` or
# `merge_group` stops the workflow from running at all, so the required check
# is never CREATED and the PR waits on it forever. Five workflows in this repo
# carry that lesson in their comments. Job-level `if:` is the only safe skip.
#
# The allowlist must stay disjoint from every Rust build input. Nothing under
# crates/ embeds a file from these four trees today (no include_str!/
# include_bytes! points at them); anything that starts to must shrink this list.
set -euo pipefail

emit() {
  {
    echo "web_only=$1"
    echo "web_touched=$2"
    echo "blog_touched=$3"
    echo "builds_binaries=$4"
  } >>"${GITHUB_OUTPUT:-/dev/stdout}"
}

# Reading the list from stdin is a first-class mode, not a test hook: it is how
# you check a change set locally (`git diff --name-only main... | classify-diff.sh -`)
# and it is what lets the classification rules be exercised directly, without
# fabricating git refs to stand in for an event.
if [ "${1:-}" = "-" ]; then
  files=$(cat)
  classify_only=1
fi

case "${classify_only:+stdin}${GITHUB_EVENT_NAME:-}" in
  stdin*)
    ;;
  pull_request)
    # Three-dot: the diff against the MERGE BASE. The event payload's base sha
    # is the base tip at PR creation and goes stale, and a two-dot diff against
    # the moving base tip would count main's own commits as this PR's.
    files=$(git diff --name-only "origin/${PR_BASE_REF:?}...${PR_HEAD_SHA:?}")
    ;;
  merge_group)
    # Queue entry: base_sha is main plus every earlier entry, head_sha adds
    # this one. Two-dot is exact — precisely what this entry contributes on top
    # of what the queue has already validated.
    files=$(git diff --name-only "${MG_BASE_SHA:?}" "${MG_HEAD_SHA:?}")
    ;;
  *)
    # push, schedule, workflow_dispatch, workflow_call: never fast-path.
    emit false true true true
    echo "event '${GITHUB_EVENT_NAME:-?}' never classifies as web-only"
    exit 0
    ;;
esac

total=$(printf '%s\n' "$files" | grep -c . || true)
non_web=$(printf '%s\n' "$files" | grep -cvE '^(site|blog|book|web-shared)/' || true)
web=$(printf '%s\n' "$files" | grep -cE '^(site|blog|book|web-shared)/' || true)
blog=$(printf '%s\n' "$files" | grep -cE '^blog/' || true)

web_only=false
if [ "$total" -gt 0 ] && [ "$non_web" -eq 0 ]; then web_only=true; fi
web_touched=false; [ "$web" -gt 0 ] && web_touched=true
blog_touched=false; [ "$blog" -gt 0 ] && blog_touched=true

# Can this diff change a shipped binary?
#
# The nine cross-platform release-matrix builds are the most expensive thing a
# PR can trigger, and a diff that cannot alter a binary has no business
# triggering them. Measured 2026-09-04: a docs-and-CI-only PR released all nine
# on a queue where hosted runners were completing ~8 jobs an hour.
#
# FAIL-SAFE BY CONSTRUCTION: this asks whether every changed path is provably
# inert, and answers `true` (build) the moment one is not. A new top-level
# directory therefore builds until someone deliberately adds it here.
#
# `.github/` is inert with a CARVE-OUT, and the carve-out is the point.
# `certification-commands.yml` cannot change a binary; `release-build.yml`,
# `release.yml` and `.github/actions/**` decide how binaries are produced, so a
# change there must build even though it lives under `.github/`.
inert_re='^(site|blog|book|web-shared|docs|assets)/|^[^/]*\.md$|^\.github/'
build_re='^\.github/(workflows/(release|dev-release|kernel-compile)|actions/)'
non_inert=$(printf '%s\n' "$files" | grep -cvE "$inert_re" || true)
build_touched=$(printf '%s\n' "$files" | grep -cE "$build_re" || true)
builds_binaries=true
if [ "$total" -gt 0 ] && [ "$non_inert" -eq 0 ] && [ "$build_touched" -eq 0 ]; then
  builds_binaries=false
fi

emit "$web_only" "$web_touched" "$blog_touched" "$builds_binaries"
echo "changed=$total non_web=$non_web web=$web blog=$blog -> web_only=$web_only"
printf '%s\n' "$files" | sed 's/^/  /'
