# Routing policy crate (port sprint lane L40): attribution

The routing policy crate `crates/atlas-route-policy/` decides which backend
serves a request for an Atlas routing front: the eligible set before any
prediction, expected utility, prefer-local, the swap rule with hysteresis, and
decision timing. Two of its modules are translations from Go; the rest is
Atlas code built from published formulas. This file records what came from
where, under which license, and what Atlas changed. NOTICE carries the upstream
notice text; CITATIONS.md lists the sources. The feature is named "routing
policy" in Atlas; the upstream product names appear here only to identify the
sources.

## 1. rayline-core execution events and capacity facts (translation)

- Upstream: https://github.com/atlasfutures/semantic-router, branch
  `vsr-next`, commit `f6d0393`, module `rayline-core/`, published by Atlas
  Futures, Inc. License: Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`,
  identical to the module's `rayline-core/LICENSE`). The translated files carry
  no copyright line of their own and the repository has no NOTICE file.
- Files and lines translated (Go to Rust):

| Upstream | Lines | What | Atlas |
|---|---|---|---|
| `rayline-core/execution/events.go` | 12-26 | event kinds and bases | `src/events.rs` `Event`, `Basis` |
| same | 130-162 | `inferEvent`: fresh branch, last-message markers, tool-result wire groups | `src/events.rs` `infer_with_basis` |
| `rayline-core/execution/deployment_wire.go` | 11-23 | `hasExecutionHistory` | `src/events.rs` `has_execution_history` |
| `rayline-core/execution/capacity.go` | 14-68 | provenance values, the capacity fact and its fit rule (measured or bounded fits count, an estimated fit is unknown, any shortfall excludes) | `src/capacity.rs` `Provenance`, `CapacityFact`, `fits` |
| same | 161-206 | `relaxEstimatedShortfalls`, `yieldsToRuntime`, `everyTarget` | `src/capacity.rs` `settle`, `yields_to_runtime` |

- Changes by Atlas contributors:
  - the wire request is reduced to a per-message digest (role, tool-call and
    tool-result marks, wire group, a content fingerprint) that the front
    computes, so the policy crate parses no JSON and does no I/O;
  - the fresh-branch test reads the previous digest of the same branch instead
    of the server's branch state;
  - a transcript that no longer extends the previous one AND is shorter is
    inferred as a compaction (upstream infers compaction only from an explicit
    client event); an edit that keeps the length stays a user turn;
  - the capacity fact is split into a static window fact plus this request's
    prompt count, and `fits` returns a three-way verdict instead of a nullable
    fact; the runtime's own refusal is a provenance value;
  - the shortfall relaxation runs over the candidates left after the
    allow-list, health and locality filters, as a slice.

## 2. Raydock hybrid profile (naming only, no code)

- Upstream: https://github.com/rayline-ai/raydock, commit `20f02d5`,
  `crates/rayline-cli/src/hybrid_profile.rs` lines 597-644. Copyright 2026
  Atlas Futures, Inc. License: Apache License, Version 2.0. The Raydock NOTICE
  text is reproduced in `NOTICE`.
- Used: the name `min_p_pass` and the choice between a utility policy and a
  prefer-local policy. No code was copied. Atlas accepts a threshold in (0, 1]
  instead of (0, 1): 1.0 disables the local shortcut, so prefer-local then
  decides exactly as utility does (a property test checks this).

## 3. ARC technical report formulas (idea, no code)

- Source: "ARC: Learned Model Routing for Long-Horizon Agentic Workflows",
  Rayline Team, 2026 (https://static.rayline.ai/rayline/reports/arc/latest),
  section 2.2 and Appendix A; preset numbers from the Apache-2.0 ARC-1.0
  release's `presets.json` (Sage `value_usd` 4.5, Swift 0.1).
- Used, re-implemented from the formulas (the selection service is closed):
  utility `V * p - sum_k n_k * rho_k / 1e6` (`src/utility.rs`); prefer-local
  with threshold `min_p_pass`, default 0.85 (`src/prefer_local.rs`); the
  switching cost at user-turn boundaries
  `L_t * [max(rho_in, rho_write) - rho_read]+ / 1e6` (`src/utility.rs`); model
  changes only at task start, subagent start, compaction and user turns
  (`src/timing.rs`).
- Atlas additions with no upstream counterpart: the eligible set computed
  before any prediction with a hard local-only rule and a header that can only
  tighten (`src/eligibility.rs`); the PII signal that can only remove members;
  the swap rule `expected_gain * V * n_hard_queued > swap_seconds * lambda_wait`
  with its 10-minute dwell and hourly cap (`src/swap.rs`, `src/hysteresis.rs`);
  local prices from measured GPU-seconds (`utility::local_price`).

## 4. Cascade routing (idea, no code)

- Source: J. Dekoninck, M. Baader, M. Vechev, "A Unified Approach to Routing
  and Cascading for LLMs" (arXiv 2024, ICML 2025); repository
  eth-sri/cascade-routing (Apache-2.0, commit `9da6cf7`), not read for code.
- Used: the expected-utility framing of routing under a cost trade-off, which
  the utility module follows.

## License text

Apache License, Version 2.0: `LICENSES/Apache-2.0.txt`.
