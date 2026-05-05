# Architecture Decision Records

This directory captures the major design decisions behind Atlas in the
[MADR](https://adr.github.io/madr/) lite format. Each record is a single
markdown file with **Status**, **Context**, **Decision**, **Consequences**.
ADRs are append-only — when a decision is revisited, write a new ADR that
*supersedes* the old one rather than editing history.

## When to write an ADR

Write one when you make a decision that:

- Constrains the shape of code other people will write (file-size cap, module
  idiom, error-handling style).
- Locks in a runtime contract that's hard to undo (license choice, API surface,
  on-disk format, kernel target tuples).
- Has a real "we considered X and rejected it" alternative that future
  contributors will rediscover unless we write it down.

Don't write one for routine refactors, dependency bumps, or single-PR
implementation choices — those belong in commit messages or design docs.

## Format

```markdown
# ADR-NNNN: Short title

**Status:** Accepted | Superseded by ADR-MMMM | Deprecated
**Date:** YYYY-MM-DD

## Context
What's the situation that forced a decision?

## Decision
What did we decide?

## Consequences
What's better, what's worse, what new problems did we create?
```

## Index

- [0001 — License: AGPL-3.0 with a CLA](0001-license-agpl-with-cla.md)
- [0002 — Pure-Rust runtime, no PyTorch](0002-pure-rust-no-pytorch.md)
- [0003 — Hybrid SSM + attention as a first-class layer kind](0003-hybrid-ssm-attention.md)
- [0004 — NVFP4 + FP8 as the primary quant formats](0004-nvfp4-fp8-quantization.md)
- [0005 — 500-LoC per-file cap, enforced in CI](0005-500-loc-file-cap.md)
- [0006 — Multi-file module idiom (`mod.rs` + sibling files)](0006-multi-file-module-idiom.md)
- [0007 — Composing tensor + expert parallelism](0007-tp-ep-composition.md)
- [0008 — NVMe-backed high-speed KV swap](0008-nvme-high-speed-swap.md)
- [0009 — `build.rs` PTX compilation per (hw, model, quant) tuple](0009-build-rs-ptx-tuples.md)
- [0010 — Vendoring xgrammar-rs](0010-vendor-xgrammar.md)
