// SPDX-License-Identifier: AGPL-3.0-only

//! Quantization helpers (NVFP4, FP8, W4A16).
//!
//! Scaffolding crate that names the quant formats Atlas supports and
//! exposes type-level descriptors (e.g. group sizes, scale dtypes).
//! Detection of a model's quant format from its `config.json` lives in
//! `crates/atlas-core/src/config.rs`; per-format weight loading lives
//! under `crates/spark-model/src/weight_map/`.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod fp8;
pub mod nvfp4;
pub mod traits;
