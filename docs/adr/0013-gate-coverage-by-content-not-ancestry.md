# ADR-0013: A gate record covers a commit by CONTENT, never by ancestry

**Status:** Accepted
**Date:** 2026-08-09
**Supersedes part of:** the coverage rule introduced alongside ADR-0012

## Context

A gate record answers one question: *was the perf-relevant code the same when
this benchmark was measured?* `record_covers` answered it in two steps —
first assert the record's commit is an ancestor of `HEAD`, then diff the two
trees and filter for `PERF_PATHS`.

The first step is unsound in this repository, and it took main down.

**Atlas squash-merges.** A record is always written on a PR branch, against a
commit on that branch — it cannot be written *at* `HEAD`, because committing
it moves `HEAD`. The squash then lands a brand-new commit on main with a
different sha and no parent link back to the branch. So every record a PR
paid GPU hours for stops being an ancestor of anything the moment it merges.

That is not a hypothetical. `.benchmarks/*/2026-08-09-b0be4ba0e6.json` are
five real **passing** records for PR #389 (`b0be4ba0e` being that branch's
merge of #417). #389 squash-landed as `dd2ac46d5`, and the gate then reported,
for all five:

```
NONE  agentic-webserver — latest record is for b0be4ba0e6 — it is not an
      ancestor of this commit
```

The trees were the same:

```
$ git diff --name-only b0be4ba0e origin/main
.benchmarks/…   ← 8 files, all of them records
```

`.benchmarks/` is deliberately outside `PERF_PATHS` — the records are the
verdict, not its subject. So: identical code, five discarded records, main red
for three commits, and every PR opened afterwards inheriting a demand for five
fresh GPU legs in order to fix a typo.

The failure mode is worse than its cost, because it is **not dischargeable by
work**. "This file changed" tells an author what to re-measure. "Not an
ancestor" tells them nothing they can act on, and no amount of benchmarking
fixes it — the next record they cut is orphaned by its own merge in exactly
the same way.

This is the same trap that had already been recorded on the human side of the
process: never ask `git merge-base --is-ancestor` whether a lever has landed,
because a squash gives it a different sha and the answer is a false negative.
The tooling repeated the mistake the humans had already learned.

## Decision

**Drop the ancestry precondition. Keep the tree diff.**

`git diff A B` compares trees. It is defined for any two commits in the
repository and requires no history relationship between them. It already
answered the question; ancestry only added an assumption about the *shape* of
history, and this repo's merge strategy violates that assumption by design.

The obvious objection — *then a record from an unrelated branch could cover
main* — is answered by the diff itself. An unrelated branch differs on the perf
paths and is rejected. If it does **not** differ, then it measured the same
code, and its record is valid. That is the whole content-not-ancestry
doctrine, and it is precisely why an identical squash lands covered.

The one thing ancestry incidentally caught was a commit missing from the clone
(a shallow fetch). `git diff` fails outright in that case, so it still returns
`None` and the caller still fails closed. The gate job checks out with
`fetch-depth: 0`.

## Consequences

**Measured on main, with the real binary:** `5x NONE` → `5x PASS`.

**The error text had to change too.** `"it is not an ancestor of this commit"`
was printed whenever the path list came back empty, which conflated *git could
not answer* with *nothing changed*. The unanswerable case is now its own arm
and says so, with the `fetch-depth: 0` hint attached.

**A merge still needs one fresh record.** Nothing here weakens the coverage
rule: the fix itself lands in `crates/`, which is a perf path, so main needed
one gate run after it merged. What changed is that the run is now *worth*
doing — before, the record it produced would have been orphaned by its own
merge, and the gate would have gone red again immediately.

**Bootstrapping is unavoidable and should be stated, not hidden.** There is no
way to fix the gate without touching the gate. Any future change to
`atlas-plugin`'s coverage logic will read red on its own PR for the same
reason. That is the rule working, not a defect in it.

**What this does not fix.** Coverage is still coarse for host code: a
`#[cfg(test)]`-only move inside `crates/` invalidates every record even though
the release binary is provably unchanged. ADR-0012's closure hash gives that
kind of proof for `kernels/`; there is no equivalent for Rust yet, and
inventing one on the basis of "this diff looks test-only" would be a static
analysis nobody could trust. The honest cost of a host-side change remains one
gate run.

## Tests

Three, on a real git fixture that reproduces the squash shape — and asserts
that it did, since a fixture where the record *is* an ancestor would pass
vacuously.

| test | with ancestry restored |
|---|---|
| `a_record_survives_its_pr_being_squash_merged` | **FAILS** |
| `an_unrelated_commit_that_differs_is_still_not_covered` | passes |
| `a_record_for_an_unknown_commit_is_not_covered` | passes |

The last two pass either way deliberately. They pin that "drop ancestry" did
not become "accept anything", and that the fail-closed arm survived — without
them, deleting the check outright would look like a passing fix.
