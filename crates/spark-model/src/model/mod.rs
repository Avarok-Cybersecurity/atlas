// SPDX-License-Identifier: AGPL-3.0-only

//! Generic transformer model.
//!
//! The model loop (embed -> layers -> norm -> lm_head) is architecture-
//! agnostic. Layer-specific logic lives in `TransformerLayer` implementations.
//!
//! Wave 4b1 split: this module was originally a single 8,690 LoC `model.rs`.
//! Sub-modules now hold:
//!   - `types`             struct TransformerModel + PinnedMetaStaging
//!   - `ssm_pool`          SsmStatePool
//!   - `ssm_snapshot`      SsmSnapshotPool
//!   - `block_mgmt`        free-fn helpers (apply_evicted_blocks etc.)
//!   - `impl_a1/2/3`       first inherent `impl TransformerModel` block
//!   - `impl_b1/2/3`       second inherent `impl TransformerModel` block
//!   - `trait_impl`        single `impl Model for TransformerModel` block
//!     **FLAGGED ≤500 LoC cap** — Rust does NOT allow
//!     splitting one trait impl across files (E0119),
//!     and breaking it into inherent-helper delegation
//!     is a semantic refactor outside this wave's scope.
//!   - `drop`              `impl Drop for TransformerModel`
//!   - `tests`             extracted unit tests

#![allow(unused_imports, dead_code)]

pub(crate) mod block_mgmt;
pub(crate) mod drafter_context;
pub(crate) mod drop;
pub(crate) mod impl_a1;
pub(crate) mod impl_a1_init;
pub(crate) mod impl_a2;
pub(crate) mod impl_a3;
pub(crate) mod impl_b1;
pub(crate) mod impl_b2;
pub(crate) mod impl_b3;
pub(crate) mod impl_lora;
pub(crate) mod impl_lora_swap;
pub(crate) mod mtp_carry;
pub(crate) mod ssm_pool;
pub(crate) mod ssm_snapshot;
pub(crate) mod ssm_snapshot_spill;
pub(crate) mod ssm_tier;
pub(crate) mod trait_impl;
pub(crate) mod types;

// Served NLLB-200 / M2M-100 encoder-decoder model (CUDA/GB10 serving path).
#[cfg(feature = "cuda")]
pub mod nllb;

pub use types::TransformerModel;
