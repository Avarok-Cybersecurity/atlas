# Copy drafter (lane L13): attribution

The merged copy drafter in `crates/spark-server/src/scheduler/copy_draft/` and
its round plan seam `crates/spark-server/src/scheduler/spec_round_plan.rs`
combine three sources into one design. This file records
what came from where, under which license, and what Atlas changed. NOTICE
carries the upstream NOTICE text; CITATIONS.md lists the sources.

## 1. Cinference / NInfer: prompt lookup and lookup rounds (translation)

- Upstream: https://github.com/satellitedown/cinference, commit `383e5db`,
  itself derived from NInfer by Neroued and contributors
  (https://github.com/Neroued/ninfer). License: Apache License, Version 2.0
  (`LICENSES/Apache-2.0.txt`). NOTICE text: see `NOTICE`.
- Files and lines translated (C++ to Rust):

| Upstream | Lines | What | Atlas |
|---|---|---|---|
| `src/models/qwen3_5/program/speculative/prompt_lookup.cpp` | 25-34 | `window_hash` | `copy_draft/index.rs` `window_hash` |
| same | 36-91 | open-addressed window tables, latest end per window, incremental `index_until` | `copy_draft/index.rs` `CopyIndex::{insert, find, index_until, sync}` |
| same | 93-125 | longest-window lookup, exact compare, match extension (`kMatchLimit` 128), period continuation | `copy_draft/index.rs` `longest`, `extend_match`, `continuation` |
| same | 9-23, 128-152 | estimator constants (decay 0.9, prior 1 hit / 3 misses, clamp 0.01-0.97, match floor 8), `observe`, `log_probability` | `copy_draft/estimator.rs` `WindowEstimates` |
| `prompt_lookup.h` | 13-35 | `propose` / `observe` / `confident` (0.96) contract: lookup rounds | `copy_draft/policy.rs` (`CopyState`, clause 5 of `select`) |

- Changes by Atlas contributors: a 16-token window in front of 8/4/2; a key
  collision never hides an occurrence (one entry per distinct content per key,
  exact compare on every hit; upstream skips to the next window on a
  collision); bounded memory (window ends older than the cap are dropped and
  the tables compacted); a committed stream that stops extending the index is
  indexed again rather than reset silently; the proposal is scored lazily
  against the tokens the target committed, at the next plan; lookup rounds go
  through Atlas's variable-width verify (`step_verify_dflash`), single sequence
  only, at quantized lengths (6, 8, 12, 15) bounded by the slot's verify
  capacity and the request's remaining budget.

## 2. TensorFold: copy index and chain fill (translation)

- Upstream: https://github.com/ashhart/TensorFold, commit `3885dce`.
  Copyright 2026 TensorFold contributors. License: Apache License, Version 2.0
  (`LICENSES/Apache-2.0.txt`); code written before TensorFold 0.6.0 is also
  available under the MIT License (`LICENSES/TensorFold-pre0.6-MIT.txt`).
  NOTICE text: see `NOTICE`.
- Files and lines translated (Zig to Rust): `zig/src/core/copy_index.zig:1-80`
  (`CopyIndex.extend`, `longest`, `chain`, and the plain-scan oracles
  `scanLongest` / `scanChain` with their every-prefix test).
- Atlas uses: the "latest earlier occurrence of the context's last tokens"
  contract and its every-prefix oracle (`copy_draft/index.rs`,
  `copy_draft/index_tests.rs`); filling the MTP verify chain from a copy
  instead of the head (`RoundPlan::CopyChain`, after the TensorFold report's
  Brief C); the legacy n-gram index (`copy_draft/legacy.rs`).
- Changes by Atlas contributors: positions are found through hashed windows
  instead of per-token position lists; the legacy index keys position lists
  by bigram and walks them EARLIEST first, so it returns exactly the token
  Atlas's previous `NgramProposer::propose` scan returned (its oracle is that
  scan, kept unchanged in `ngram.rs`); a shrinking history pops the stale tail
  entries instead of rebuilding.

## 3. KVMem: yield gate and source cursor (idea only)

- Source: the KVMem description (arXiv:2609.04852). The `kvmem/kvmem-llama.cpp` repository
  carries no license, so no file of it was opened and no code was copied.
- Ideas used: 16/8/4-token windows with backward match extension; a source
  cursor kept across fully accepted rounds and cleared on a rejection; the
  per-request yield gate (EWMA alpha 0.25 of the accepted fraction, a copy
  replaces the MTP chain only if `width * copy_avg > advantage * mtp_avg` with
  advantage 1.3, the first two rounds unconditional, one probe after eight
  refusals). Implemented independently in `copy_draft/estimator.rs`
  (`YieldGate`) and `copy_draft/policy.rs`.

## 4. Atlas-native parts

`copy_draft/{mod.rs, config.rs, metrics.rs}`, `spec_round_plan.rs`, the
`copy_draft` row resolver (`crates/spark-model/src/layers/ops/port_rows/copy_draft.rs`),
the hook lines in `scheduler/{mtp_step.rs, mtp_bootstrap_step.rs,
verify_k4_step.rs, spec_step.rs}` and `ngram.rs`, and every test module.
License: AGPL-3.0-only, like the rest of Atlas.

## Exactness note

Every plan is verified by the target through the same pre-sample pipeline
the MTP chain uses, so the drafter changes which tokens are offered, never
which tokens are emitted. With `copy_draft = off` (every target's default)
the per-sequence state is never allocated, the hooks return at their first
line and the legacy n-gram mode runs its original scan.
