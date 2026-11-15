// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the root of the keep-packed GGUF 4-bit route (port lane L19, row `gguf_native_iq4`):
//! IQ4_XS, Q4_K and Q6_K weights served from the GGUF file's raw super-blocks on Ada (sm_89) through the
//! `W4Linear` contract.
//!
//! The kernels are thin entry points over the llama.cpp (MIT) quant engine Atlas already vendors in
//! `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/` (included, never edited): the q8_1 activation quantizers
//! (`kernels/ada/common/kquant_q8_1_ada.cu`), the decode GEMVs at 1..=8 rows (`kquant_mmvq_q4_ada.cu`), the
//! MMQ prefill GEMMs (`iq4_xs_mmq_ada.cu`) and the IQ4 / Q5_K dequant (`dequant_iq4xs_q5k_bf16_ada.cu`).
//! On every other target the modules are absent: [`launch::available`] is false, the constructors return an
//! error before any kernel lookup, and nothing changes. See `GGUF-IQ4-ATTRIBUTION.md`.
//!
//! Where the weights come from: spark-runtime's GGUF loader keeps the projections packed under the row
//! (`spark_runtime::weights::keep_packed_iq4`, tagging them `WeightDtype::{Iq4Xs, Q4K, Q6K}`); the W4 loader
//! routes wrap them with [`from_weight_tensor`] (or [`fuse_rows`] for a fused projection) and hand the
//! `W4Box` to the layer setters of lanes L21 and L22.
//!
//! | file | what |
//! |---|---|
//! | `launch.rs` | modules, entry names, handles, one launcher per kernel, buffer byte arithmetic |
//! | `w4linear.rs` | [`GgufQ4Linear`]: `impl W4Linear`, constructors, `decode_pair` partner registry |
//! | `reference.rs` | the CPU oracle: q8_1 and MMQ mirrors, float64 GEMV, Q4_K partial-sum term, bounds |
//! | `tests.rs` | mock-backend tests of every entry point and the oracle's planted controls |
//! | `reference_tests.rs` | tests of the prefill side of the oracle (MMQ mirror, Q4_K mins, bounds) |
//!
//! GPU oracle and timings: `examples/iq4xs_gemv_microtest.rs`.

pub mod launch;
pub mod reference;
pub mod w4linear;

pub use launch::{MmvqShape, Q4Type};
pub use w4linear::{
    GgufQ4Linear, GgufQ4Weight, from_device_blocks, from_gguf_tensor, from_weight_tensor,
    fuse_rows, gguf_of, q4_type_of, q4_type_of_dtype, release,
};

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;
