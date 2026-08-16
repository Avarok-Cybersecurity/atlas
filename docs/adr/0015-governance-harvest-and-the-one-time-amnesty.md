# ADR-0015: The governance harvest, and a one-time content-pinned amnesty

**Status:** Accepted
**Date:** 2026-08-16
**Builds on:** ADR-0013 (coverage by content), ADR-0014 (intent taxonomy)

## Context

ADR-0014 shipped half a mechanism. The intent taxonomy, the descending
classifier, the grow-only ledger and `required_for = by_path ∪ by_intent` all
exist — and the intent half has been identically empty since the day it
landed, because nothing ever moves a PR's ledger lines from the CI artifact
into `governance/` on the default branch. `by_intent ≡ ∅`, forever, and the
categorizer's output gates nothing.

Fixing that means three changes, and one of them is expensive. The harvest
workflow and the `--pr` glue are off every invalidation path. But the
taxonomy's empty `_benches` leaves — `correctness/sampling`,
`capability/adapters`, `capability/serving-api`, and no leaf anywhere naming
the gates promoted on 2026-08-15 (`decode-floor`, `concurrency-sweep`) or the
SSM poison gate — live in `.github/pr-taxonomy.json`, which is a
`BOUNDARY_FILES` entry. So is `check.rs`, which any wiring must touch. A diff
touching either invalidates **every** standing gate record: ~4h19m of GPU to
re-earn ten gates, for an edit that only ever ADDS required benchmarks.

That cost is deliberate (coverage.rs says so in as many words: "a cheap edit
that quietly weakens the gate is worse than an expensive one that cannot").
But this particular edit strengthens the gate, every standing record was
freshly earned on 2026-08-16, and the owner authorized a bypass for this
landing alone — "just this one time".

## Decision

### 1. The taxonomy fill

Every empty leaf that describes something a benchmark measures gets
`_benches`; the promoted 2026-08-15 gates become reachable through intent
(`performance/decode` → `decode-floor`, `performance/scheduling` →
`concurrency-sweep`, and so on); `correctness/ssm-state` and
`correctness/kv-cache` gain `ssm-state-poisoning-gate`. `infrastructure/*`
and `documentation/*` stay empty **with `_doc` notes saying it is on
purpose** — tooling and prose cannot move an inference number.
`unknown` stays empty and a test pins its whole subtree, so an abstention can
never manufacture GPU spend.

### 2. A one-time, content-pinned amnesty — not a waiver

Rejected first, with the reasons on record:

- **Closure-hash refresh** (ADR-0012's rung): inapplicable — these are not
  `kernels/` paths, and no device-code hash can speak for a policy file.
- **Re-blessing the records** (rewriting their shas to the landing commit):
  forges provenance; every record would claim a commit it never measured.
- **Two-phase landing** (land the fill, then re-earn under it): the tree diff
  is history-independent, so records earned before the fill are invalidated
  by it no matter which order the commits land in.

Chosen: `crates/atlas-plugin/src/gate/amnesty.rs`, a table of exactly two
entries — `.github/pr-taxonomy.json` and `check.rs`, the two boundary files
this PR must touch — each pinning the **blob OID of the file as this PR lands
it**. `check.rs::invalidating_paths` drops a surviving path only when
`git rev-parse <head>:<path>` returns exactly the pinned 40-hex OID, and logs
a warning naming the path and the grant every time it does. Everything else —
an unlisted path, a later edit (different OID), any git failure — invalidates
exactly as before. The grant covers the reviewed bytes and nothing else:
there is no time window, no path-level waiver, and the second edit to either
file pays full price.

The two-entry shape is itself deliberate: the fill avoids touching
`coverage.rs`, `required.rs` or any other boundary file precisely so the
table cannot grow past the two files the change cannot avoid.

### 3. Self-limiting, self-expiring

- `the_table_is_exactly_the_2026_08_16_grant` pins the entry count, the two
  paths, and that the OIDs are real 40-hex blob names — it ships RED against
  placeholder OIDs, forcing the pin phase before the grant can arm.
- `amnesty_expires_once_every_gate_has_a_fresh_record` reads the real
  `.benchmarks/` tree and FAILS with removal instructions once every required
  gate's newest record postdates `AMNESTY_EPOCH` (end of 2026-08-16 UTC). The
  table cannot quietly outlive its purpose.
- Acceptance is empirical, both directions: gate-check on the landing branch
  prints PASS for all ten gates with the amnesty log lines visible, and with
  the table emptied all ten go Missing — proving the amnesty is live rather
  than vacuous.

## Residual risk, stated plainly

`amnesty.rs` cannot live in `BOUNDARY_FILES` without circularity — its own
landing would invalidate everything it exists to protect. It is therefore
covered only by `GATE_MACHINERY`'s cargo-test rationale, like the rest of the
gate bookkeeping. A future PR could widen the table with an ordinary
gate-directory edit. Compensations, none of them new machinery: the
pinned-table test makes that widening a visible test edit; every application
is logged in the gate output; CODEOWNERS review covers the gate directory;
and the gate already executes PR-checkout code, so a malicious PR could
always lie to itself — the merge queue's re-check on main is the backstop
either way. This is a reviewed two-file exception, not a new attack class.

## Consequences

- The intent half of ADR-0014 can finally carry data: the fill makes the
  promoted and poison gates reachable through intent, and the harvest
  workflow (landing separately) starts producing the abstain-rate evidence
  the enforcement decision was deferred on.
- Standing records survive a strengthening-only boundary edit, once, on the
  record, with the bypass reviewable in the diff rather than performed
  out-of-band.
- The amnesty's removal is a test failure away from mandatory — the cheapest
  kind of institutional memory.
