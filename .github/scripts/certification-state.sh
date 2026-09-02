#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Resolve which certification state a pull request is in, and emit the name of
# the diagram that depicts it.
#
# Emits to $GITHUB_OUTPUT:
#   state     one of the names under docs/diagrams/states/
#   headline  one line for the comment, above the image
#
# ── Why the state is computed, never stored ─────────────────────────────────
#
# Everything below is read from GitHub at the moment the bot runs: check-run
# conclusions on the head sha, the PR's mergeable state, its merge-queue entry.
# Nothing is remembered in a file a job could write.
#
# That is the same property `atlas-governance` guards for the ledger — a verdict
# must not depend on something the PR's own CI can append to. A stamp stored in
# the repo would be settable by the branch it is meant to gate.
#
# The ONE exception is the previous state, which the bot reads back out of its
# own comment marker. That is display only: it decides whether to show "you were
# further along and a push knocked you back" instead of a bare stage card. It
# never decides whether anything may merge.
set -euo pipefail

PR="${1:?usage: certification-state.sh <pr-number>}"
REPO="${REPO:-$GITHUB_REPOSITORY}"
PREV="${PREV_STATE:-}"

emit() {
  # $1 state, $2 headline
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf 'state=%s\n' "$1" >>"$GITHUB_OUTPUT"
    printf 'headline=%s\n' "$2" >>"$GITHUB_OUTPUT"
  fi
  printf '%s\t%s\n' "$1" "$2"
}

pr_json=$(gh api "repos/$REPO/pulls/$PR" 2>/dev/null || echo '{}')
merged=$(printf '%s' "$pr_json" | jq -r '.merged // false')
head_sha=$(printf '%s' "$pr_json" | jq -r '.head.sha // empty')
mergeable_state=$(printf '%s' "$pr_json" | jq -r '.mergeable_state // "unknown"')

# ── Terminal and blocking states come first ────────────────────────────────
# Order matters: a merged PR is merged whatever its checks say, and a PR in
# conflict describes a tree nobody will merge, so its green checks are noise.
if [ "$merged" = "true" ]; then
  emit "pr-certification-merged" "Merged."
  exit 0
fi

if [ "$mergeable_state" = "dirty" ]; then
  emit "pr-certification-blocked" \
    "Blocked — this branch conflicts with main. Green checks here describe a tree nobody will merge."
  exit 0
fi

# `gh pr view` is the only surface that reports merge-queue membership.
in_queue=$(gh pr view "$PR" --repo "$REPO" --json isInMergeQueue \
  --jq '.isInMergeQueue // false' 2>/dev/null || echo false)
if [ "$in_queue" = "true" ]; then
  emit "pr-certification-queued" \
    "In the merge queue. The whole pipeline is running again against the queue's own merge commit."
  exit 0
fi

# ── The three signals that place a PR in a stage ───────────────────────────
# A check that has not reported is not a pass. `// empty` then the -z test keeps
# "absent" and "failed" distinct in the code even though both mean "not yet".
conclusion_of() {
  gh api "repos/$REPO/commits/$head_sha/check-runs?per_page=100" \
    --jq ".check_runs[] | select(.name == \"$1\") | .conclusion" 2>/dev/null |
    head -1
}

stamp=$(conclusion_of "Stamp")
seal=$(conclusion_of "Seal")
records=$(conclusion_of "PR Benchmark Certifications")

has_stamp=false; [ "$stamp" = "success" ] && has_stamp=true
has_seal=false;  [ "$seal" = "success" ] && has_seal=true
has_records=false; [ "$records" = "success" ] && has_records=true

# ── Demotion: only when we can name what knocked it back ───────────────────
# A demotion card is shown when the PREVIOUS state was further along than this
# one. Without that comparison the bot would render a bare stage card after a
# push, which is accurate and useless — it does not say why the seal vanished.
demoted_from_stage_2_or_3() {
  case "$PREV" in
    *stage-2*|*stage-3*|*queued*) return 0 ;;
    *) return 1 ;;
  esac
}

if demoted_from_stage_2_or_3 && [ "$has_seal" = false ]; then
  if [ "$has_records" = false ]; then
    emit "pr-certification-demoted-push-both" \
      "A new commit landed. The seal is void and the records no longer cover this tree — a perf path moved."
  else
    emit "pr-certification-demoted-push-seal" \
      "A new commit landed. The seal is void; the records still stand because no perf path moved."
  fi
  exit 0
fi

if demoted_from_stage_2_or_3 && [ "$has_records" = false ] && [ "$has_seal" = true ]; then
  emit "pr-certification-demoted-main-perf" \
    "The records no longer cover this tree. The seal survives — a sealer vouched for the diff, and the diff has not changed."
  exit 0
fi

# ── Stages ─────────────────────────────────────────────────────────────────
if [ "$has_stamp" = false ]; then
  emit "pr-certification-stage-1" \
    "Stage 1 — Verification. The expensive lane is held back until someone with write access comments \`/stamp\`."
  exit 0
fi

if [ "$has_seal" = true ] && [ "$has_records" = true ]; then
  emit "pr-certification-stage-3" "Stage 3 — Ready to merge. Nothing is waiting on a person."
elif [ "$has_seal" = true ]; then
  emit "pr-certification-stage-2-needs-records" \
    "Stage 2 — sealed, waiting on benchmark records. Run the campaign on a GPU box and commit the records."
elif [ "$has_records" = true ]; then
  emit "pr-certification-stage-2-needs-seal" \
    "Stage 2 — records cover this tree, waiting on the engineer's seal. A codeowner of every path in the diff comments \`/seal\`."
else
  emit "pr-certification-stage-2-both" \
    "Stage 2 — Certification. Waiting on both the engineer's seal and benchmark records."
fi
