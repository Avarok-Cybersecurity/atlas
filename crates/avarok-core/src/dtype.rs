// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

/// Supported quantization and precision types for Atlas kernels.
///
/// Each variant maps to a specific bit-width and numeric format used by
/// SM121 tensor cores or CUDA ALU paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 4-bit floating point (1 sign + 2 exponent + 1 mantissa)
    /// Used for NVFP4 weight storage. Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    E2M1,

    /// 8-bit floating point (1 sign + 4 exponent + 3 mantissa)
    /// Used for NVFP4 block scales and FP8 quantization
    FP8E4M3,

    /// 8-bit floating point (1 sign + 5 exponent + 2 mantissa)
    /// Used for some FP8 quantization schemes
    FP8E5M2,

    /// 16-bit brain floating point
    BF16,

    /// 16-bit IEEE floating point
    FP16,

    /// 32-bit IEEE floating point
    FP32,
}

impl DType {
    /// Size in bytes per element. E2M1 is sub-byte (4 bits) but we report
    /// the packed size (2 elements per byte).
    pub const fn element_size_bits(&self) -> usize {
        match self {
            DType::E2M1 => 4,
            DType::FP8E4M3 | DType::FP8E5M2 => 8,
            DType::BF16 | DType::FP16 => 16,
            DType::FP32 => 32,
        }
    }

    /// Number of elements that pack into a 32-bit word.
    pub const fn elements_per_u32(&self) -> usize {
        32 / self.element_size_bits()
    }
}

/// Quantization configuration for a weight tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Weight storage type (e.g., E2M1 for NVFP4)
    pub weight_type: DType,

    /// Scale factor type (e.g., FP8E4M3 for NVFP4 block scales)
    pub scale_type: DType,

    /// Number of weights per scale factor (block size)
    pub group_size: usize,

    /// Global scale factor type (FP32 for NVFP4)
    pub global_scale_type: DType,
}

impl QuantConfig {
    /// NVFP4: E2M1 weights with FP8 block scales (group_size=16)
    pub fn nvfp4() -> Self {
        Self {
            weight_type: DType::E2M1,
            scale_type: DType::FP8E4M3,
            group_size: 16,
            global_scale_type: DType::FP32,
        }
    }

    /// FP8 per-tensor quantization
    pub fn fp8() -> Self {
        Self {
            weight_type: DType::FP8E4M3,
            scale_type: DType::FP32,
            group_size: 0, // per-tensor
            global_scale_type: DType::FP32,
        }
    }
}
