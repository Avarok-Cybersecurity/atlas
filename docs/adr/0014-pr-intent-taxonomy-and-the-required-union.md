# ADR-0014: PR intent is a descending tree, and it may only ADD benchmarks

**Status:** Accepted
**Date:** 2026-08-09
**Builds on:** ADR-0012 (closure hash), ADR-0013 (coverage by content)

## Context

ADR-0012 and ADR-0013 answer *"which gates does this diff invalidate?"* from
paths alone. Paths cannot answer a different question: **what is this change
for?** A scheduler edit under `crates/spark-server/` touches no `kernels/`, so
every target's closure hash is unchanged — yet it can move decode wall badly.

The first attempt at this emitted a single flat label from a hand-written list
and called a CI-only commit `performance`. A flat label also cannot express
*performance for which subsystem*, and it gave the classifier one shot at a
choice it had no way to reconsider.

## Decision

### 1. A tree, descended one level at a time

`.github/pr-taxonomy.json` holds six roots. The classifier is offered **only the
children of where it currently stands**, one closed-set call per level. Shape
rules are enforced by `pr_taxonomy::validate`, not by comment:

- keys are lowercase kebab-case, so a path is a safe label and anchor
- a node with two or more children is a choice; a node with exactly one is
  **rejected** — asking a model to pick from a set of one wastes a call and
  manufactures confidence, and `resolve` auto-follows such a step if the rule is
  ever relaxed
- every `_benches` entry must name a registered benchmark. A path selecting an
  unregistered id is a silent no-op, which is the worst kind of gate bug because
  it reads as coverage.

### 2. `_benches` may only ADD. It can never remove.

The required set is `path_derived ∪ intent_derived`, computed by
`gate::required`. The path-derived half stands on its own; nothing the model
says can shrink it.

**This is the entire reason a language model is allowed near a merge gate.** A
misclassification costs GPU minutes, never a missed regression. Invert it and
the classifier becomes a way to skip tests by writing a misleading PR title —
and the diff would not even have to lie, only the prose.

### 3. Classifications are unioned, never last-wins

The classifier is not stable: three live runs on one PR produced `tooling`,
`performance`, `tooling`. A gate whose demands change between re-runs is worse
than no gate. Every `EventKind::Category` recorded for a head sha counts, and
the ledger being grow-only and deduplicated-on-read (identity excludes `at`)
makes the union cheap, order-independent, and replay-stable.

## What this does NOT buy, stated plainly

**The union is very nearly a no-op today.** Two independent reasons:

1. `validate` pins `_benches ⊆ coverage::REQUIRED`, so `intent ⊆ REQUIRED`.
2. `PERF_PATHS` contains a bare `"crates"`, so any code change already
   invalidates all five gates.

So for a code PR the union adds nothing, and `benches_may_only_add` — the
property this ADR rests on — is true and **vacuous**. It is insurance for a
world that does not exist yet: the one ADR-0012 creates, where the closure hash
narrows `by_path` below "all five".

The one class where intent bites today is measured, not assumed: **`recipes/`
and `docker/` invalidate nothing.** A recipe sets the serve flags, and the whole
GB10 concurrency ladder is serve flags, so a recipe change can move decode wall
with an empty `by_path`. `required_tests::the_live_case_is_recipes` pins that,
and `intent_is_redundant_for_a_crates_change` pins the vacuity — when the second
one fails, the union has become load-bearing, and its doc comment says so, so
nobody "fixes" it by widening paths back.

**Neither half is wired into `check_gates` yet.** `required_for` is computed and
reported; it does not currently change any verdict. The gate stays advisory
until the union is proven stable, per the owner's decision to flip it to
required only once it is.

## The input surface is now cross-repo

The classifier reads PR title, body, changed paths, and — where
`GH_ORGANIZATION_READ_TOKEN` is available — recipe content from the separate
public `atlas-recipes` repository. That means **text in another repository can
influence this repository's gate**.

This is safe for exactly one reason, and it is the same reason as everything
else here: the influence is add-only. The worst a hostile or wrong recipe
description can do is buy more benchmark time. It can never subtract a gate.
Recipe fetch failure degrades to classifying without that context, never to
failing the job — a gate that goes red because another repo was briefly
unreachable trains people to ignore it.

## Consequences

- A misclassification is a cost, not a hole. Budget for it.
- The taxonomy is a reviewable artifact: adding a category is a diff on a JSON
  file with a test suite attached, not a prompt edit.
- Two implementations of `benches_for` must never coexist. One already caused a
  divergence (`_benches` as a bare string parsed as empty in Rust while jq read
  it fine — the Rust half failing in the *removing* direction). The Rust is
  authoritative; anything else calls it.
- `intent_only()` is retained separately from the union so the telemetry table
  can say *why* a gate is required. Collapsed into one set, "intent added this"
  becomes invisible, which is how the previous coverage gap survived.
