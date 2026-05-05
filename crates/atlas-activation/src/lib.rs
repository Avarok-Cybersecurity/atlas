// SPDX-License-Identifier: AGPL-3.0-only

//! Activation-function helpers (SiLU, GeLU, sigmoid-mul).
//!
//! Scaffolding crate: defines the [`Activation`] trait so other crates
//! can name an activation without depending on a specific GPU backend.
//! Concrete kernel launches today live in
//! `crates/spark-model/src/layers/ops/`; this crate is the future home
//! for the host-side primitives once those are extracted.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod traits;

pub use traits::Activation;
