// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the affine 4-bit weight route's root (port lanes L17a, L17b, L17c): MLX-affine weights
//! quantized from a BF16 checkpoint at load and served on TensorFold's lane MMA GEMV on Ada (sm_89).
//!
//! Format (MLX's): per group of `group` inputs (32, 64 or 128) a bf16 scale `s` and bias `b`, value
//! `s * q + b`, code `q` in 0..15; 4.5 bits a weight at group 64. The kernels are Ada-owned files in
//! `kernels/ada/common/` (`affine4_lane_ada.cu`, `affine_pack_ada.cu`, `affine_frag_ada.cuh`); on every
//! other target the modules are absent, [`AffineKernels::available`] is false and no lookup is issued.
//!
//! Nothing here is reached unless a route row is on: the rows (`dense_affine4_from_bf16`,
//! `gdn_affine4_from_bf16`, `attn_affine4_from_bf16`, `affine4_prefill_int8`) and the loader routes that
//! call [`quantize_from_bf16`] are lane L23a's; the layer dispatch through `W4Linear` is L21's and L22's.
//!
//! | file | owner | what |
//! |---|---|---|
//! | `weight.rs` | L17a | [`AffineWeight`], layout, `split_k`, shared-memory and scratch arithmetic |
//! | `kernels.rs`, `params.rs` | L17a | handles, one launcher per entry point, the by-value descriptors |
//! | `quantize.rs` | L17a | [`quantize_from_bf16`] (BF16 -> tiles on the device) |
//! | `w4linear.rs` | L17a | [`AffineLinear`]: `impl W4Linear` (decode rows 1..=16, pair fusion) |
//! | `reference.rs`, `reference_gemv.rs` | L17a | bit-exact host mirrors, the float64 GEMV and its bound |
//! | `prefill.rs` | L17b | the prompt GEMMs (`prefill::run`, BF16 and W4A8 int8) |
//! | `prefill_launch.rs`, `prefill_oracle.rs`, `prefill_ffn.rs` | L17b | `#[path]` children of `prefill.rs` (launchers, CPU oracles, the fused dense FFN prompt path) |
//! | `widths.rs` | L17c | widths 2, 3, 5, 6 and 8 (`widths::quantize_from_bf16`) |

pub mod kernels;
pub mod params;
pub mod prefill;
pub mod quantize;
pub mod reference;
pub mod reference_gemv;
pub mod w4linear;
pub mod weight;
pub mod widths;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_layout;
#[cfg(test)]
mod tests_mock;

pub use kernels::{AffineKernels, group_sums};
pub use quantize::{quantize_from_bf16, quantize_weight};
pub use w4linear::{AffineLinear, affine_of, release};
pub use weight::{AffineWeight, Layout, split_k};
