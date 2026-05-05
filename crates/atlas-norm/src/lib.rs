// SPDX-License-Identifier: AGPL-3.0-only

//! Normalization helpers (RMSNorm, LayerNorm).
//!
//! Scaffolding crate: defines the [`Normalize`] trait so other crates
//! can name a normalization without depending on a specific GPU
//! backend. Concrete RMSNorm kernels currently live in
//! `crates/spark-model/src/layers/ops/`; this crate will own the
//! host-side primitives once they are extracted.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod traits;

pub use traits::Normalize;
