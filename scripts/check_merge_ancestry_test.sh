#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Unit tests for scripts/check_merge_ancestry.sh, run against scratch repos
# built from nothing — no network, no dependence on this repository's own
# history. Each case reproduces a real PR shape:
#
#   (a) normal descendant branch                  -> PASS, no warnings
#   (b) orphan snapshot branch (the PR #452 shape:
#       root commit carrying a full stale tree)   -> FAIL exit 1, NO MERGE BASE
#   (c) merely-behind branch                      -> PASS with behind ::warning
#   (d) branch that rewinds a file to a
#       pre-merge-base version                    -> PASS with revert ::warning
#   (e) bogus sha                                 -> exit 2 (infra error)
#
# CI runs this on every PR (see .github/workflows/merge-ancestry.yml) so a
# change to the guard script cannot land with the guard silently broken.
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/check_merge_ancestry.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

asserts=0
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { asserts=$((asserts + 1)); echo "  ok [$1] $2"; }

# ── Fixture repo: main = root A -> B -> C ("engine work") ──────────────────
repo="$tmp/repo"
git init -q -b main "$repo"
cd "$repo"
git config user.name "Test"
git config user.email "test@example.com"
git config commit.gpgsign false

printf 'engine v1\n' > engine.rs
printf 'lib v1\n' > lib.rs
git add -A && git commit -qm "A: root"
sha_a="$(git rev-parse HEAD)"

printf 'engine v2\n' > engine.rs
git commit -qam "B: engine work 1"
sha_b="$(git rev-parse HEAD)"

printf 'engine v3\n' > engine.rs
printf 'lib v2\n' > lib.rs
git commit -qam "C: engine work 2"
sha_c="$(git rev-parse HEAD)"

run() { # run <base> <head>; sets rc + out
  set +e
  out="$("$SCRIPT" "$1" "$2" main 2>&1)"
  rc=$?
  set -e
}

# ── (a) normal descendant branch off current base tip -> PASS, quiet ───────
git checkout -qb feat "$sha_c"
printf 'feature\n' > feature.rs
git add -A && git commit -qm "feat: new work"
head_feat="$(git rev-parse HEAD)"

run "$sha_c" "$head_feat"
[[ $rc -eq 0 ]] || fail a "descendant branch: expected exit 0, got $rc: $out"
grep -q '✓ merge-ancestry' <<<"$out" || fail a "missing PASS marker: $out"
grep -q 'NO MERGE BASE' <<<"$out" && fail a "spurious NO MERGE BASE: $out"
grep -q 'behind its base' <<<"$out" && fail a "spurious behind-warning: $out"
grep -q 'REVERT SIGNATURE' <<<"$out" && fail a "spurious revert-warning: $out"
ok a "descendant branch passes cleanly"

# ── (b) orphan snapshot: the exact #452 shape ──────────────────────────────
# Two commits; the FIRST is a repository root carrying the full tree as it
# stood at B (i.e. snapshotted before C landed). No ancestry with main.
git checkout -q "$sha_b"
git checkout -q --orphan snapshot452   # index/worktree = B's tree, no parent
git commit -qm "snapshot root (stale tree of B)"
printf 'the intended fix\n' > fix.rs
git add -A && git commit -qm "intended fix"
head_orphan="$(git rev-parse HEAD)"
git merge-base "$sha_c" "$head_orphan" >/dev/null 2>&1 \
  && fail b "fixture broken: orphan unexpectedly shares history with main"

run "$sha_c" "$head_orphan"
[[ $rc -eq 1 ]] || fail b "orphan branch: expected exit 1, got $rc: $out"
grep -q 'NO MERGE BASE' <<<"$out" || fail b "missing NO MERGE BASE: $out"
grep -q "$sha_c" <<<"$out" || fail b "message must name the base sha: $out"
grep -q "$head_orphan" <<<"$out" || fail b "message must name the head sha: $out"
grep -q 'force-with-lease' <<<"$out" || fail b "message must carry the remedy: $out"
grep -q 'allow-unrelated-histories' <<<"$out" || fail b "message must warn against --allow-unrelated-histories: $out"
ok b "orphan snapshot fails with NO MERGE BASE + shas + remedy"

# ── (c) merely-behind branch -> PASS with a behind ::warning ───────────────
git checkout -qb behind "$sha_b"       # branched before C
printf 'late feature\n' > late.rs
git add -A && git commit -qm "behind: new work"
head_behind="$(git rev-parse HEAD)"

run "$sha_c" "$head_behind"
[[ $rc -eq 0 ]] || fail c "behind branch: expected exit 0 (WARN not FAIL), got $rc: $out"
grep -q '::warning::PR is 1 commit(s) behind its base' <<<"$out" \
  || fail c "missing behind-warning: $out"
grep -q '✓ merge-ancestry' <<<"$out" || fail c "missing PASS marker: $out"
ok c "merely-behind branch warns (1 commit) and passes"

# ── (d) revert signature: file rewound to a pre-merge-base version ─────────
git checkout -qb rewind "$sha_c"
printf 'engine v1\n' > engine.rs       # byte-identical to engine.rs at A
git commit -qam "rewind engine.rs to v1"
head_rewind="$(git rev-parse HEAD)"

run "$sha_c" "$head_rewind"
[[ $rc -eq 0 ]] || fail d "rewind branch: expected exit 0 (WARN not FAIL), got $rc: $out"
grep -q '::warning::REVERT SIGNATURE' <<<"$out" || fail d "missing revert-warning: $out"
grep -q 'engine.rs' <<<"$out" || fail d "revert-warning must list the file: $out"
ok d "pre-merge-base rewind warns and lists engine.rs"

# ── (e) missing commit -> exit 2 infra error, never a silent pass ──────────
run "$sha_c" "0000000000000000000000000000000000000001"
[[ $rc -eq 2 ]] || fail e "bogus head sha: expected exit 2, got $rc: $out"
grep -q 'not present locally' <<<"$out" || fail e "missing infra-error message: $out"
ok e "missing commit is an infra error (exit 2)"

echo ""
echo "ALL $asserts assert groups passed."
