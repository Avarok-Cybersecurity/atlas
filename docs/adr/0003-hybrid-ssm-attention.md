# ADR-0003: Hybrid SSM + attention as a first-class layer kind

**Status:** Accepted
**Date:** 2026-04-17

## Context

Several model families we care about — Qwen3.5 / Qwen3.6 / Qwen3-Next,
Nemotron-Nano / Super, MiniMax-M2 — interleave **gated-delta-net** (or
Mamba2) SSM blocks with full-attention blocks. The SSM blocks have a
*recurrent* state (per-token-update conv1d state + SSM hidden state) that
behaves nothing like a KV cache.

Naively wedging SSM into a KV-cache-shaped abstraction (treat SSM state as
a 1-token K/V) breaks chunked prefill, breaks speculative-decode rollback,
and breaks paged eviction semantics. We learned this the hard way during
Pass-9 / Pass-21 debugging.

The candidate approaches:

1. **One layer kind, hidden differences.** Pretend everything is "an
   attention layer" and let the layer impl decide. Simple at the layer
   call site, but the scheduler has to special-case rollback,
   verification, and snapshotting per layer anyway — the abstraction
   leaks.
2. **Two layer kinds.** Distinguish `FullAttention | LinearAttention` (and
   sub-kinds for SSM variants) at the type level, exposed via a `LayerType`
   enum in `atlas-core`. Scheduler and verify path branch explicitly on
   the kind.

## Decision

Atlas defines `LayerType` in `atlas-core` as a closed enum:

```rust
pub enum LayerType {
    FullAttention,
    LinearAttention, // SSM / GDN / Mamba2
    Mlp,
    Moe,
    Dense,
}
```

Per-architecture `TransformerLayer` impls live under
`crates/spark-model/src/layers/<family>/`, e.g.
`qwen3_attention/`, `qwen3_ssm/` (gated delta net),
`nemotron_mamba2/`. Each owns its own state-management primitives.

The scheduler's verify path (K=2 / K=3 / K=γ) explicitly handles SSM state
checkpoint + rollback: see `project_dflash_phase25k`-style state
snapshots before draft, restore on rejection. Chunked prefill is gated by
`Model::is_mla()` and a per-layer-kind check; SSM layers force
single-chunk prefill until per-chunk SSM state passing is wired up.

## Consequences

**Better:**
- Bugs of the form "SSM state was not rolled back after rejection" are
  now type-system-visible. A reviewer can ask "where do you snapshot the
  SSM state?" by reading the enum.
- New SSM variants slot in by adding a layer impl directory and wiring
  one variant arm; no scheduler-wide refactor.

**Worse:**
- Two code paths for prefill (paged-attn vs SSM-recurrent) and two for
  decode. We pay the duplication cost.
- Speculative decode is harder: K-token verify means K applications of
  the SSM recurrence, with K-step state checkpointing. Pure-attention
  models get K-token verify "for free" via paged attention.

**New problems we created:**
- Chunked prefill for SSM is genuinely hard (intermediate chunk states
  must be saved + restored across chunks). For MiniMax-M2 we punted by
  forcing single-chunk prefill; longer-context SSM models need this work
  before they scale.
- Cross-model regressions hide easily: a Qwen3-attention fix can break
  Qwen3-SSM if the change touches a shared scheduler code path. We rely
  on `tests/run_all_models.py` running every model variant on every PR.
