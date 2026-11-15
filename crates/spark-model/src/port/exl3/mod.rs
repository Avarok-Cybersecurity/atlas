// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the root of the EXL3 trellis weight route (port lane L20b, row `weights_exl3`): EXL3
//! matrices (ExLlamaV3's QTIP-style trellis codes, 1 to 8 bits a weight, half-bit steps with the mul1
//! codebook) served from their packed codes on Ada (sm_89) through the `W4Linear` contract,
//! after ExLlamaV3 (MIT) via TensorFold's CUDA translation (Apache-2.0). See EXL3-ATTRIBUTION.md.
//!
//! How a call runs: the input row is rotated into the Hadamard domain and scaled by `suh`
//! (`exl3_ada_rot_in_bf16`, fp16 out), the decode-row linear decodes each 16 x 16 tile straight into mma
//! B fragments and accumulates in fp32, and its epilogue rotates back and scales by `svh`
//! (`exl3_ada_linear_<K2>_<CB>_<WK>`). Decode serves 1..=16 rows, prefill 17..=128 rows on the same
//! kernel (the same bits), and a longer prefill declines to the BF16 dequant
//! (`exl3_ada_dequant_bf16_<K2>_<CB>`) and the caller's GEMM. On every other target the modules are
//! absent: [`kernels::available`] is false, the constructors fail before any kernel lookup, and nothing
//! changes. With the row off, [`from_exl3_tensors`] refuses on every target.
//!
//! | file | what |
//! |---|---|
//! | `format.rs` | [`Exl3Parts`] (one matrix as host bytes), codebooks, widths, admission checks, sign words |
//! | `plan.rs` | the shape plan (K splits, warps), strides, shared memory and scratch arithmetic |
//! | `kernels.rs` | module and entry names, [`Exl3Kernels`], one launcher per kernel |
//! | `w4linear.rs` | [`Exl3Linear`]: upload, `impl W4Linear`, [`release`] |
//! | `tensor.rs` | [`from_exl3_tensors`] (the reader's `Exl3Tensor`, behind the row), [`TensorBytes`], [`route_enabled`] |
//! | `reference.rs` | the CPU format decoder (scalar oracle) and the GPU lane-extraction mirror |
//! | `reference_math.rs` | bit-exact mirrors of the rotation and dequant kernels, FP64 references, the GEMV bound |
//! | `tests.rs`, `tests_math.rs` | CPU oracle tests (format, lane mirror, codebooks, Hadamard algebra) |
//! | `tests_mock.rs` | mock-backend tests of every entry point |
//! | `tests_fixture.rs` | TensorFold's numpy oracle and its public-checkpoint block, when `ATLAS_EXL3_ORACLE` names the file |
//!
//! GPU oracle and timings: `examples/exl3_microtest.rs`.

pub mod format;
pub mod kernels;
pub mod plan;
pub mod reference;
pub mod reference_math;
pub mod tensor;
pub mod w4linear;

pub use format::{Exl3Codebook, Exl3Parts, Exl3Scales};
pub use kernels::Exl3Kernels;
pub use tensor::{TensorBytes, from_exl3_tensors, route_enabled};
pub use w4linear::{Exl3Linear, Exl3Weight, exl3_of, release};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_fixture;
#[cfg(test)]
mod tests_math;
#[cfg(test)]
mod tests_mock;
