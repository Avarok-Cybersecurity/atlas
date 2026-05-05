// SPDX-License-Identifier: AGPL-3.0-only

//! Reduction primitives (argmax, sum, max).
//!
//! Scaffolding crate: defines the [`Reduce`] trait so other crates can
//! name a reduction without depending on a specific GPU backend.
//! Concrete reductions today launch via
//! `crates/spark-model/src/layers/ops/` and the sampler in
//! `crates/spark-runtime/src/sampler.rs`.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod traits;

pub use traits::Reduce;
