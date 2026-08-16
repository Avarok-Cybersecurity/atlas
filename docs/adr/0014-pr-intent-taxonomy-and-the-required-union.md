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

## Amendment (2026-08-16): three claims above went stale, and one was never true

Audits during the governance-harvest work (ADR-0015) refuted parts of this
record. Per this directory's append-only rule the original text stands;
read it against the following.

**1. "Any code change already invalidates all five gates" was never true.**
Reason 2 under *What this does NOT buy* claimed the union is vacuous for code
PRs because `PERF_PATHS` contains a bare `crates`. It does — but
`GATE_MACHINERY` excludes the whole `crates/atlas-plugin/src/gate` prefix
from **every** gate, so paths under it invalidate nothing and intent is their
only source of coverage. The union was live inside `crates/` from day one; it
was never waiting on the closure-hash narrowing.
`required_tests::crates_paths_split_into_fully_covered_and_not_covered_at_all`
pins both halves of the corrected claim.

**2. The `recipes/` "live case" is unreachable from this repository.** This
repo tracks zero `recipes/` files — they live in the separate `atlas-recipes`
repo, and `invalidating_paths` diffs *this* one, so a `recipes/` path can
never appear in the diff the gate filters. The reachable
off-the-floor classes are `docker/`, `docs/`, `.github/`, `scripts/`,
`bench/`, `kernels/**/BENCH.toml`, and the excluded `crates/` paths above.
The tests named in the original text are gone with the claim:
`the_live_case_is_recipes` became `intent_adds_where_the_paths_are_silent`
(which pins reachable paths only), and `intent_is_redundant_for_a_crates_change`
became the split test cited in point 1.

**3. "Five gates" is now ten.** `REQUIRED` grew to ten entries: the vision
and video fidelity gates, the echolp BFCL draw, the SSM state-poisoning gate,
and — promoted from candidates on 2026-08-15 once their calibration
preconditions were met — `decode-floor` and `concurrency-sweep`. Every count
above should be read against `coverage::REQUIRED`, which is the SSOT.

**4. The empty-`_benches` gap is closed.** The original tree left
`correctness/sampling`, `capability/adapters` and `capability/serving-api`
empty and named none of the later gates. The 2026-08-16 fill (ADR-0015)
populated every leaf that describes something a benchmark measures, and
pinned the deliberate emptiness of `infrastructure/*`, `documentation/*`
and `unknown` — with `_doc` notes in the JSON and a test over the `unknown`
subtree, so an abstention can never manufacture GPU spend.

Unchanged: the union is still advisory (`check_gates` iterates
`REQUIRED_GATES` unconditionally and the exit code cannot see intent), and
`_benches` still may only ADD — both by the same doctrine as before.
