// SPDX-License-Identifier: AGPL-3.0-only
//! `MetalGpuBackend` integration / parity tests, sharded by kernel family.
//!
//! Each submodule below is `#[cfg(test)]`-gated transitively (this file is
//! only included via `#[cfg(test)] mod tests;` in `metal_backend.rs`); the
//! sharding is purely organisational so the file-size cap stays satisfied
//! and the parity coverage is easy to navigate by topic.

mod helpers;

mod parity_asym;
mod parity_attention;
mod parity_attention_full;
mod parity_basic;
mod parity_gdn;
mod parity_norms;
mod parity_quant;
mod parity_turbo;
mod parity_turbo23;
mod parity_turbo4;
mod parity_vision;

mod real_model_attention;
mod real_model_gemv;
mod real_model_misc;
mod real_model_vision;
