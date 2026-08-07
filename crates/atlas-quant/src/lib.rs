// SPDX-License-Identifier: AGPL-3.0-only

//! Quantization helpers (NVFP4, FP8, W4A16).
//!
//! Scaffolding crate that names the quant formats Atlas supports and
//! exposes type-level descriptors (e.g. group sizes, scale dtypes).
//! Detection of a model's quant format from its `config.json` lives in
//! `crates/atlas-core/src/config.rs`; per-format weight loading lives
//! under `crates/spark-model/src/weight_map/`.
//!
//! # This crate is not wired to anything
//!
//! `atlas-quant` has ZERO dependents in the workspace — no crate lists it in
//! `Cargo.toml` and nothing `use`s it. It builds and its tests run (it is a
//! workspace member), but no code here executes in a serving binary. Do not
//! read a doc comment in this crate as a statement about production
//! behaviour, and do not "fix a bug" here expecting it to change how Atlas
//! runs: `fp8::FP8_E4M3_LUT` / `fp8::f32_to_bf16` are byte-identical
//! duplicates of the copies in `crates/spark-model/src/weight_map/fp8_lut.rs`,
//! and only that second copy is live. `fp8::Fp8Quantizer::quantize` is a
//! `todo!()`.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod fp8;
pub mod nvfp4;
pub mod traits;
