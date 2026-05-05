# ADR-0010: Vendoring xgrammar-rs

**Status:** Accepted
**Date:** 2026-04-17

## Context

Tool calling and JSON-mode constrained decoding need a grammar engine
that can compile a JSON Schema (or regex / EBNF) into a per-token mask
fast enough to apply on every step. The two practical choices in Rust
in late-2025:

1. **`xgrammar-rs`** (MLC AI). Well-engineered, used by SGLang,
   minimal allocator pressure, supports the full pushdown automaton
   trick. C++ core with a Rust wrapper. Releases are sporadic and
   the upstream repo's build script has occasional breakage.
2. **Pure-Rust grammar libs** (e.g. `regex_automata`, `pest`).
   No CFG / pushdown support; would need a from-scratch
   implementation of the per-token-mask trick. Months of work.

Critically, we hit the upstream issue described in
`project_grammar_bytelevel_vocab` — xgrammar's `VocabType=RAW`
silently breaks ByteLevel-BPE tokenizers (Qwen, MiniMax, Mistral 4),
producing empty tool calls. Fixing this requires modifying xgrammar
itself, not just our wrapper.

## Decision

Atlas **vendors xgrammar-rs** under `vendor/xgrammar-rs/` with a
build-time fetch (no submodules, no live network at build). The
`Cargo.toml` references the vendored copy by path.

Modifications applied to the vendored fork:
- `VocabType` auto-detection via `TokenizerInfo` rather than the
  upstream RAW default that misbehaves on ByteLevel-BPE.
- Build-script fixes for the kernels we hit on `nvcc` 13 / Rust 1.93.
- Minor type-shim differences for our cudarc bindings.

Upstream-friendly changes are PR'd back; Atlas-specific glue stays
local. The vendored copy is **pinned to a specific upstream commit
hash** (called out in the build script) so we can compare against
upstream periodically.

## Consequences

**Better:**
- We can fix grammar bugs that block production use (the ByteLevel-BPE
  issue would otherwise have stalled tool-calling on three of our
  five model families).
- Build is hermetic: no live `git clone` step at build time, no
  network requirement past `cargo fetch`.
- Pinning to a known-good commit means upstream churn doesn't break
  our CI.

**Worse:**
- We carry maintenance for the fork. Every upstream bug-fix is a
  rebase; every Atlas-local change is a divergence we must justify.
- License compatibility check (xgrammar is Apache-2.0; AGPL-3.0 can
  consume Apache-2.0, so we are clear). Future xgrammar relicensing
  would force us to evaluate.
- Vendor directories are excluded from a few of our QA passes
  (`typos`, license-header check) on purpose. Anyone making changes
  inside `vendor/` is bypassing those checks.

**New problems we created:**
- Re-syncing with upstream is manual and risky. We must record every
  local change so the rebase doesn't drop fixes.
- `cargo deny` audits the vendored deps' transitive licenses; an
  upstream xgrammar dep change can silently introduce a forbidden
  license. We rely on the periodic `security.yml` workflow to catch
  this.
