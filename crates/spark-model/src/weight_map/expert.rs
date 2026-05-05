// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// FP8 expert weight: gate/up/down projections as FP8 block-scaled weights.
///
/// Used for native FP8 expert dispatch (no NVFP4 dequant overhead).
/// Each projection stores [N, K] FP8 E4M3 weights + block scales.
#[derive(Debug, Clone, Copy)]
pub struct Fp8ExpertWeight {
    pub gate_proj: Fp8Weight,
    pub up_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
}

/// Full attention layer weights (12 layers in Qwen3-Next).
#[derive(Debug, Clone, Copy)]
pub struct AttentionWeights {
    /// Q projection: [hidden_size, num_heads * head_dim] BF16.
    pub q_proj: DenseWeight,
    /// K projection: [hidden_size, num_kv_heads * head_dim] BF16.
    pub k_proj: DenseWeight,
    /// V projection: [hidden_size, num_kv_heads * head_dim] BF16.
    pub v_proj: DenseWeight,
    /// O projection: [num_heads * head_dim, hidden_size] NVFP4.
    pub o_proj: QuantizedWeight,
    /// Q RMS norm weight: `[head_dim]` BF16 (per-head Qwen3-family convention).
    pub q_norm: DenseWeight,
    /// K RMS norm weight: `[head_dim]` BF16 (per-head Qwen3-family convention).
    pub k_norm: DenseWeight,
    /// MiniMax-style full-hidden Q RMSNorm weight: [num_heads * head_dim] BF16.
    ///
    /// Set to `Some(..)` for models that apply RMSNorm over the concatenated
    /// Q projection output (MiniMax M2) before the view-into-heads and before
    /// RoPE. Mathematically different from the per-head `q_norm` above
    /// (MiniMax normalizes by the global hidden-dim RMS; Qwen3 normalizes
    /// per-head). Attention forward branches on `.is_some()` to pick which
    /// pre-RoPE norm to apply. Default `None` keeps all existing models on
    /// the per-head `q_norm` path — behavior-preserving for every non-
    /// MiniMax loader.
    pub q_norm_full: Option<DenseWeight>,
    /// MiniMax-style full-hidden K RMSNorm weight: [num_kv_heads * head_dim] BF16.
    pub k_norm_full: Option<DenseWeight>,
    /// K scale for FP8 KV cache.
    pub k_scale: f32,
    /// V scale for FP8 KV cache.
    pub v_scale: f32,
}

/// Linear attention (SSM / Gated Delta Net) layer weights (36 layers).
#[derive(Debug, Clone, Copy)]
pub struct SsmWeights {
    /// QKVZ projection: [hidden_size, qkvz_size] BF16.
    pub in_proj_qkvz: DenseWeight,
    /// Beta-Alpha projection: [hidden_size, ba_size] BF16.
    pub in_proj_ba: DenseWeight,
    /// Conv1d weight: [d_inner, 1, d_conv] BF16.
    pub conv1d: DenseWeight,
    /// A_log parameter: `[num_v_heads]` FP32.
    pub a_log: DenseWeight,
    /// dt_bias parameter: `[num_v_heads]` FP32.
    pub dt_bias: DenseWeight,
    /// Gate norm weight: `[hidden_size]` BF16.
    pub norm: DenseWeight,
    /// Output projection: [hidden_size, hidden_size] NVFP4.
    pub out_proj: QuantizedWeight,
}

/// MoE expert weights (shared across all 512 experts per layer).
#[derive(Debug, Clone, Copy)]
pub struct ExpertWeight {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

impl ExpertWeight {
    /// Null expert (all pointers NULL). Used for remote experts under EP.
    /// Kernels detect NULL pointers and write zero output for these experts.
    pub fn null() -> Self {
        Self {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
        }
    }
}

// ── Unified quantization types ──────────────────────────────────────
//
// These enums abstract over different quantization formats (NVFP4, FP8,
// BF16 dense) so that layer dispatch code uses a single type instead of
// cascading if/else chains checking multiple Optional fields.
//
// Adding a new quantization format requires:
//   1. Add a variant to QuantWeight
//   2. Add match arms in quant_gemv/quant_gemm (ops.rs)
//   3. Implement the weight loader for the new format

/// Quantized weight for any supported format.
///
/// Encapsulates all data a GEMV/GEMM kernel needs to dequantize and compute.
/// The forward path matches on this enum to select the correct kernel.
/// Enum branch compiles to ~1 cycle vs GPU kernel launch at ~5000 cycles.
#[derive(Debug, Clone, Copy)]
pub enum QuantWeight {
    /// NVFP4 E2M1: packed nibbles + FP8 group scales + f32 global scale.
    /// Kernel: w4a16_gemv (decode) / w4a16_gemm (prefill)
    Nvfp4(QuantizedWeight),

    /// FP8 E4M3: byte-packed weights + per-block BF16 scales.
    /// Kernel: w8a16_gemv (decode) / w8a16_gemm (prefill)
    Fp8(Fp8Weight),

    /// BF16 dense (unquantized). Kernel: dense_gemv / dense_gemm
    Dense(DenseWeight),
}

impl QuantWeight {
    /// Null weight (for remote experts under EP or unused projections).
    pub fn null() -> Self {
        Self::Nvfp4(QuantizedWeight::null())
    }

    /// Whether this weight points to NULL (placeholder).
    pub fn is_null(&self) -> bool {
        match self {
            Self::Nvfp4(w) => w.is_null(),
            Self::Fp8(w) => w.weight.is_null(),
            Self::Dense(w) => w.weight.is_null(),
        }
    }

    /// Extract as NVFP4, if this weight is that variant.
    pub fn as_nvfp4(&self) -> Option<&QuantizedWeight> {
        match self {
            Self::Nvfp4(w) => Some(w),
            _ => None,
        }
    }

    /// Extract as FP8, if this weight is that variant.
    pub fn as_fp8(&self) -> Option<&Fp8Weight> {
        match self {
            Self::Fp8(w) => Some(w),
            _ => None,
        }
    }

    /// Extract as Dense, if this weight is that variant.
    pub fn as_dense(&self) -> Option<&DenseWeight> {
        match self {
            Self::Dense(w) => Some(w),
            _ => None,
        }
    }
}

impl From<QuantizedWeight> for QuantWeight {
    fn from(w: QuantizedWeight) -> Self {
        Self::Nvfp4(w)
    }
}

impl From<Fp8Weight> for QuantWeight {
    fn from(w: Fp8Weight) -> Self {
        Self::Fp8(w)
    }
}

impl From<DenseWeight> for QuantWeight {
    fn from(w: DenseWeight) -> Self {
        Self::Dense(w)
    }
}

/// Per-expert weights in any supported quant format.
///
/// Replaces the separate `ExpertWeight` (NVFP4) and `Fp8ExpertWeight` (FP8)
/// types with a single unified type.
#[derive(Debug, Clone, Copy)]
pub struct QuantExpertWeight {
    pub gate_proj: QuantWeight,
    pub up_proj: QuantWeight,
    pub down_proj: QuantWeight,
}

impl QuantExpertWeight {
    pub fn null() -> Self {
        Self {
            gate_proj: QuantWeight::null(),
            up_proj: QuantWeight::null(),
            down_proj: QuantWeight::null(),
        }
    }
}

impl From<ExpertWeight> for QuantExpertWeight {
    fn from(w: ExpertWeight) -> Self {
        Self {
            gate_proj: QuantWeight::Nvfp4(w.gate_proj),
            up_proj: QuantWeight::Nvfp4(w.up_proj),
            down_proj: QuantWeight::Nvfp4(w.down_proj),
        }
    }
}

impl From<Fp8ExpertWeight> for QuantExpertWeight {
    fn from(w: Fp8ExpertWeight) -> Self {
        Self {
            gate_proj: QuantWeight::Fp8(w.gate_proj),
            up_proj: QuantWeight::Fp8(w.up_proj),
            down_proj: QuantWeight::Fp8(w.down_proj),
        }
    }
}
