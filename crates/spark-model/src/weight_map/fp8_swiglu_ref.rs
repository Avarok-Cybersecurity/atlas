// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU reference for the fused SwiGLU + per-token-group FP8
//! quantization (`kernels/ada/common/swiglu_quant_fp8_ada.cu`).
//!
//! The fused kernel is defined as the composition of two kernels the dense
//! FFN's W8A8 arm already runs, and this module is that composition on the
//! host:
//!
//! 1. `moe_silu_mul`: `r = bf16(g * (1 / (1 + __expf(-g))) * u)`, modelled
//!    instruction for instruction by [`swiglu_bf16_with`] as the PTX
//!    `--fmad=false` emits it (`mul.rn` by `-log2(e)`, `ex2.approx`,
//!    `add.rn`, `rcp.rn`, two `mul.rn`, `cvt.rn.bf16`);
//! 2. `per_token_group_quant_fp8` on `r`:
//!    [`super::fp8_blockscale_ref::per_token_group_quant`], which is exact.
//!
//! Step 2 is bit-exact. Step 1 is exact except for `ex2.approx.f32`, whose
//! last bits are the hardware's, so [`swiglu_bf16`] (which uses the host's
//! `exp2`) can differ from the device by a BF16 ulp on a few elements. The
//! GPU oracle (`examples/fp8_ada_swiglu_microtest.rs`) therefore gates the
//! fused kernel on [`quant_of_product`] of the DEVICE's unfused product, which
//! is exact, and reports the host SwiGLU only as a check on the model.

use super::fp8_blockscale_ref::{FP8_BLOCK, per_token_group_quant, round_bf16};

/// `-log2(e)` as the FP32 immediate the compiler folds `__expf(-g)` into
/// (`mul.rn.f32 t, g, 0fBFB8AA3B`).
pub const NEG_LOG2_E: f32 = f32::from_bits(0xBFB8_AA3B);

/// `moe_silu_mul`'s element, rounded to BF16 and widened back, with the
/// `ex2.approx.f32` step supplied by the caller.
///
/// Every other operation is a single IEEE round-to-nearest FP32 operation,
/// in the order the PTX issues them (Rust never contracts to FMA).
pub fn swiglu_bf16_with(g: f32, u: f32, exp2: impl Fn(f32) -> f32) -> f32 {
    let e = exp2(g * NEG_LOG2_E);
    let sigmoid_g = 1.0 / (1.0 + e);
    round_bf16(g * sigmoid_g * u)
}

/// [`swiglu_bf16_with`] using the host's `exp2` for `ex2.approx.f32`.
pub fn swiglu_bf16(g: f32, u: f32) -> f32 {
    swiglu_bf16_with(g, u, f32::exp2)
}

/// The BF16 SwiGLU product of `[m, k]` gate and up rows (values widened to
/// f32), element for element.
pub fn swiglu_product(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(
        gate.len(),
        up.len(),
        "swiglu_product: gate and up differ in length"
    );
    gate.iter()
        .zip(up)
        .map(|(g, u)| swiglu_bf16(*g, *u))
        .collect()
}

/// The quantizer half of the fused op on an already-rounded BF16 product:
/// `(bytes[m*k], scales[m * k/128])`, exactly as `per_token_group_quant_fp8`
/// computes them. The bit-exact contract of the fused kernel is
/// `fused(gate, up) == quant_of_product(moe_silu_mul(gate, up))`.
pub fn quant_of_product(product: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(
        product.iter().all(|x| *x == round_bf16(*x) || x.is_nan()),
        "quant_of_product: the product must already be rounded to BF16"
    );
    per_token_group_quant(product, m, k)
}

/// Host mirror of the whole fused op: [`swiglu_product`] then
/// [`quant_of_product`]. Exact up to the `ex2.approx` caveat in the module
/// doc.
pub fn swiglu_quant(gate: &[f32], up: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(
        gate.len(),
        m * k,
        "swiglu_quant: {} values for [{m},{k}]",
        gate.len()
    );
    assert!(
        k.is_multiple_of(FP8_BLOCK),
        "swiglu_quant: k={k} not a multiple of 128"
    );
    quant_of_product(&swiglu_product(gate, up), m, k)
}

#[cfg(test)]
#[path = "fp8_swiglu_ref_tests.rs"]
mod tests;
