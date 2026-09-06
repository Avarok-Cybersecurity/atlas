#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Refuse to push a stack layer whose head would be identical to its base.
#
# ★ WHY THIS EXISTS. On 2026-09-06 a bad force-push briefly gave #937, #938 and
# #939 heads equal to their own base branches. GitHub closed all three within
# five seconds. Restoring the refs did not undo it: a CLOSED pull request's
# head is FROZEN, so `gh pr reopen` and `PATCH /pulls/{n} -f state=open` both
# refuse with
#
#     state cannot be changed. There are no new commits between ...
#
# even after the branch is put back. The PRs were unrecoverable and had to be
# reopened as #948/#949/#950 under a new stack. The same mechanism had already
# destroyed #944 an hour earlier, by MERGE rather than by close: when a PR's
# commits land underneath its own base branch, GitHub marks it merged into that
# branch. Both are the same one-line mistake — pushing a layer that no longer
# differs from what it sits on — and both are irreversible after the fact.
#
# So this check runs BEFORE the push, not after. It is the only place it can
# work: every guard downstream of the push is reporting on a corpse.
#
# Usage:
#   assert-stack-layer-differs.sh <new-head-ish> <base-ref>   # one layer
#   assert-stack-layer-differs.sh --selftest                  # prove it fires
#
# Exit 0 = the layer has content of its own. Exit 1 = do not push.
set -uo pipefail

# A layer is safe iff its head is NOT an ancestor-or-equal of its base AND the
# tree differs. Both halves are needed and neither implies the other:
#
#   * equal trees, different shas  -> GitHub still sees "no new commits" and
#     closes the PR, so a sha comparison alone is not enough;
#   * head IS the base's ancestor  -> the classic "commits landed underneath
#     their own base" shape that merged #944, and its tree can differ.
check_layer() {
  local head="$1" base="$2" hs bs
  hs=$(git rev-parse --verify "$head^{commit}" 2>/dev/null) || {
    echo "REFUSE: '$head' is not a commit"; return 1; }
  bs=$(git rev-parse --verify "$base^{commit}" 2>/dev/null) || {
    echo "REFUSE: base '$base' is not a commit"; return 1; }

  if [ "$hs" = "$bs" ]; then
    echo "REFUSE: head and base are the same commit ($(git rev-parse --short "$hs"))."
    echo "        GitHub closes a PR whose head equals its base, and a closed PR's"
    echo "        head is frozen — restoring the branch will NOT reopen it."
    return 1
  fi
  if git merge-base --is-ancestor "$hs" "$bs"; then
    echo "REFUSE: head $(git rev-parse --short "$hs") is an ancestor of base $(git rev-parse --short "$bs")."
    echo "        The layer's commits sit UNDERNEATH its own base branch; GitHub"
    echo "        marks such a PR merged into that branch (this is what took #944)."
    return 1
  fi
  if [ "$(git rev-parse "$hs^{tree}")" = "$(git rev-parse "$bs^{tree}")" ]; then
    echo "REFUSE: head and base have identical trees, so the PR has an empty diff."
    echo "        Different shas do not save it — GitHub reads 'no new commits'."
    return 1
  fi
  echo "ok: $(git rev-parse --short "$hs") differs from base $(git rev-parse --short "$bs")"
  return 0
}

# ── self-test ──────────────────────────────────────────────────────────────
# ★ A GUARD THAT CANNOT FAIL IS WORSE THAN NO GUARD, because it reports safety.
# Each rule gets an input that MUST be refused and one that MUST pass, built in
# a throwaway repo so the rules are exercised rather than described.
selftest() {
  local tmp rc=0 out
  tmp=$(mktemp -d) || return 1
  trap 'rm -rf "$tmp"' RETURN
  (
    cd "$tmp" || exit 1
    git init -q . && git config user.email t@t && git config user.name t
    echo a > f && git add f && git commit -qm base
    BASE=$(git rev-parse HEAD)
    echo b > f && git add f && git commit -qm layer
    LAYER=$(git rev-parse HEAD)
    # a commit with the base's tree but a different sha
    git checkout -q --detach "$BASE" && git commit -q --allow-empty -m "empty on base"
    EMPTY=$(git rev-parse HEAD)
    printf '%s %s %s\n' "$BASE" "$LAYER" "$EMPTY"
  ) > "$tmp/shas" || { echo "selftest: fixture build failed"; return 1; }
  read -r BASE LAYER EMPTY < "$tmp/shas"

  # ★ EACH CONTROL ASSERTS WHICH RULE FIRED, NOT MERELY THAT SOMETHING DID.
  # The three refusals overlap: head==base ALSO has identical trees, and an
  # ancestor MAY. A control that checks only the exit code therefore stays
  # green when the rule it is named after is dead and a sibling catches the
  # input instead -- which is exactly what happened the first time this suite
  # was sabotaged for its own negative control.
  run() { # run <expect 0|1> <needle> <label> <head> <base>
    local want="$1" needle="$2" label="$3"; shift 3
    out=$(cd "$tmp" && check_layer "$@" 2>&1); local got=$?
    if [ "$got" -eq "$want" ] && printf '%s' "$out" | grep -qF -- "$needle"; then
      printf '  ok   %s\n' "$label"
    else
      printf '  FAIL %s (wanted rc=%s + %s, got rc=%s)\n     %s\n' \
        "$label" "$want" "\"$needle\"" "$got" "$out"
      rc=1
    fi
  }
  run 0 "differs from base"    "a real layer passes"                        "$LAYER" "$BASE"
  run 1 "same commit"          "control: head == base is refused"           "$BASE"  "$BASE"
  run 1 "is an ancestor of"    "control: head under its own base refused"   "$BASE"  "$LAYER"
  run 1 "identical trees"      "control: same tree, different sha, refused" "$EMPTY" "$BASE"
  run 1 "is not a commit"      "control: a nonexistent head is refused"     "no-such-ref" "$BASE"
  [ "$rc" -eq 0 ] && echo "selftest: 5 passed, 0 failed" || echo "selftest: FAILED"
  return "$rc"
}

case "${1:-}" in
  --selftest) selftest ;;
  "")         echo "usage: $0 <new-head-ish> <base-ref> | $0 --selftest" >&2; exit 2 ;;
  *)          check_layer "$1" "${2:?base ref required}" ;;
esac
