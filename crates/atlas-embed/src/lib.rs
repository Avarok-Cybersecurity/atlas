// SPDX-License-Identifier: AGPL-3.0-only

//! Embedding + RoPE position-encoding helpers.
//!
//! Scaffolding crate covering the two embedding-shaped primitives:
//! [`token`] for embedding-table lookups, and [`rope`] for rotary
//! position encoding. Concrete kernels run via
//! `crates/spark-model/src/layers/ops/rope*` and the model-specific
//! prefill/decode paths; this crate hosts the host-side
//! type/config descriptors.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod rope;
pub mod token;
