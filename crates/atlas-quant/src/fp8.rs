// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 E4M3 quantization and dequantization utilities.
//!
//! Supports two FP8 checkpoint formats:
//!   1. **Per-tensor scaled**: `weight` (FP8) + `weight_scale` (f32 scalar).
//!   2. **Block-scaled**: `weight` (FP8) + `weight_scale_inv` (BF16 per-block).
//!
//! FP8 E4M3FN: sign(1) | exponent(4) | mantissa(3), bias=7, range [-448, 448].

use atlas_core::error::Result;
use atlas_core::tensor::TensorRef;

use crate::traits::Quantize;

mod dequant;
mod lut;
#[cfg(test)]
mod tests;

pub use dequant::{dequant_fp8_block_to_bf16, dequant_fp8_pertensor_to_bf16, is_fp8_tensor};
pub use lut::fp8_e4m3_to_f32;

// ── Format descriptors ──

/// Scale factor precision for block-scaled FP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDtype {
    Fp32,
    Bf16,
}

/// FP8 E4M3 block-scaled format descriptor.
///
/// Describes the layout of compressed-tensors FP8 checkpoints:
/// - Weight tensor: FP8 E4M3 bytes, shape [N, K].
/// - Scale tensor: per-block scales, shape [N/block_size, K/block_size].
#[derive(Debug, Clone)]
pub struct Fp8Format {
    /// Block size for block-scaled FP8 (e.g., 128 elements per scale in each dim).
    pub block_size: usize,
    /// Precision of the per-block scale factors.
    pub scale_dtype: ScaleDtype,
}

// ── GPU quantizer (stub for future 4B GEMM dispatch) ──

/// FP8 E4M3 quantization: FP32/BF16 -> FP8 with per-tensor or per-token scale.
pub struct Fp8Quantizer;

impl Quantize for Fp8Quantizer {
    fn quantize(
        &self,
        _input: &TensorRef,
        _output: &TensorRef,
        _scale: &TensorRef,
        _stream_ptr: u64,
    ) -> Result<()> {
        // TODO: Launch fp8_quant.cu kernel (Workstream 4B)
        todo!("FP8 quantization kernel launch")
    }
}
