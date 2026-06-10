// SPDX-License-Identifier: AGPL-3.0-only
//! Vendor-agnostic Qwen3.5 per-layer decoder forward.
//!
//! Extracted verbatim from the original Metal end-to-end driver
//! (`crates/spark-runtime/examples/metal_qwen35_inference/`). The two
//! exported functions — [`forward_full_attention`] and
//! [`forward_linear_attention`] — drive a single decoder layer end to
//! end (norm → projections → attention/GDN → residual+post-norm →
//! MLP → residual). Any end-to-end inference example calls these
//! through `&dyn GpuBackend` + a `QuantWeights` impl, regardless of
//! which hardware target the backend speaks.
//!
//! What the module does not do (intentional):
//! - Multi-token prefill / batched dispatch — single-token decode only
//!   (KV-append at `cache_pos`, attention at `seq_len_attn = cache_pos+1`).
//! - CUDA-graph capture, NCCL, paged-KV — the production decode path
//!   (`crate::model::trait_impl::decode_a`) handles those; this is the
//!   simpler shape an example or smoke driver wants.
//! - Tokenizer / sampler / weight loading — the caller owns these.
//!
//! Performance: the fused kernels (`gemv_silu_gate_resid`,
//! `gemv_gate_up_with`, `add_rms_norm`) all dispatch through trait
//! methods that backends override with their fused launches. Atlas's
//! Metal backend keeps decode at ~20 tok/s through this path
//! identically to the inlined version it replaces.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::quant_weights::QuantWeights;

mod full_attention;
mod linear_attention;

pub use full_attention::forward_full_attention;
pub use linear_attention::forward_linear_attention;

/// Compile-time-fixed dimensions for a Qwen3.5 checkpoint. Populate
/// from the model's `config.json` (`text_config`) at startup.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35ForwardConfig {
    // Top-level model dims.
    pub hidden: u32,
    pub intermediate: u32,
    pub num_layers: u32,
    pub vocab: u32,
    pub group_size: u32,
    pub rms_eps: f32,

    // Full-attention dims.
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    /// `head_dim * partial_rotary_factor` — Qwen3.5-VL rotates only
    /// the first `rotary_dim` of each head (=64 of 256 for the
    /// 4B checkpoint with `partial_rotary_factor = 0.25`).
    pub rotary_dim: u32,

    // Linear-attention (GDN) dims.
    pub num_k_heads_lin: u32,
    pub num_v_heads_lin: u32,
    pub k_head_dim_lin: u32,
    pub v_head_dim_lin: u32,
    pub conv_kernel_size: u32,
}

impl Qwen35ForwardConfig {
    /// Hardcoded constants for `mlx-community/Qwen3.5-4B-MLX-8bit`.
    /// Matches the Metal example's `dims.rs` exactly so the extracted
    /// forward path is byte-equivalent to the inlined version.
    pub const fn qwen3_5_4b_mlx_int8() -> Self {
        Self {
            hidden: 2560,
            intermediate: 9216,
            num_layers: 32,
            vocab: 248_320,
            group_size: 64,
            rms_eps: 1e-6,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            rotary_dim: 64, // = head_dim * partial_rotary_factor (0.25)
            num_k_heads_lin: 16,
            num_v_heads_lin: 32,
            k_head_dim_lin: 128,
            v_head_dim_lin: 128,
            conv_kernel_size: 4,
        }
    }

    /// `Q_TOTAL = num_heads * head_dim * 2` — Qwen3.5 packs the
    /// attention output gate into the same projection as Q, so the
    /// q_proj produces a `[num_heads, head_dim * 2]` interleaved
    /// tensor that needs a deinterleave step before normalisation.
    #[inline]
    pub const fn q_total(&self) -> u32 {
        self.num_heads * self.head_dim * 2
    }
    /// `Q_ONLY = num_heads * head_dim` — half of `Q_TOTAL`, the
    /// post-deinterleave Q size.
    #[inline]
    pub const fn q_only(&self) -> u32 {
        self.num_heads * self.head_dim
    }
    /// `KV_DIM = num_kv_heads * head_dim`.
    #[inline]
    pub const fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
    /// `Z_DIM_LIN = num_v_heads_lin * v_head_dim_lin`.
    #[inline]
    pub const fn z_dim_lin(&self) -> u32 {
        self.num_v_heads_lin * self.v_head_dim_lin
    }
    /// `QKV_TOTAL_LIN = (num_k_heads_lin + num_k_heads_lin) * k_head_dim_lin
    ///                + num_v_heads_lin * v_head_dim_lin`.
    #[inline]
    pub const fn qkv_total_lin(&self) -> u32 {
        2 * self.num_k_heads_lin * self.k_head_dim_lin + self.num_v_heads_lin * self.v_head_dim_lin
    }
    /// `NUM_STATE_HEADS = num_v_heads_lin` — the number of GDN heads
    /// the gate / beta / dt_bias / A_log vectors all run over.
    #[inline]
    pub const fn num_state_heads(&self) -> u32 {
        self.num_v_heads_lin
    }
}

/// Pre-resolved kernel handles. Resolve once at startup; pass `&` to
/// every per-layer call so name-lookup overhead doesn't appear in the
/// hot path.
pub struct Qwen35Kernels {
    pub rms: KernelHandle,
    pub rope: KernelHandle,
    pub kvap: KernelHandle,
    pub attn: KernelHandle,
    pub sg: KernelHandle,
    pub add_rms: KernelHandle,
    pub qkv_split: KernelHandle,
    pub conv1d: KernelHandle,
    pub gdn_gate: KernelHandle,
    pub sigmoid: KernelHandle,
    pub gdn_dec: KernelHandle,
    /// TurboQuant KV cache paths (Turbo8/4/3/2): quantizing appends,
    /// dequantizing decode attentions, and the WHT rotation bookends.
    /// Resolved unconditionally (the kernels live in the common set) so
    /// a turbo cache can never silently fall back to the bf16 kernels.
    pub kvap_turbo8: KernelHandle,
    pub attn_turbo8: KernelHandle,
    pub kvap_turbo4: KernelHandle,
    pub attn_turbo4: KernelHandle,
    pub kvap_turbo3: KernelHandle,
    pub attn_turbo3: KernelHandle,
    pub kvap_turbo2: KernelHandle,
    pub attn_turbo2: KernelHandle,
    pub kvap_bf16k_turbo4v: KernelHandle,
    pub attn_bf16k_turbo4v: KernelHandle,
    pub kvap_bf16k_turbo3v: KernelHandle,
    pub attn_bf16k_turbo3v: KernelHandle,
    pub kvap_bf16k_turbo2v: KernelHandle,
    pub attn_bf16k_turbo2v: KernelHandle,
    pub wht: KernelHandle,
    pub wht_inv: KernelHandle,
}

impl Qwen35Kernels {
    /// Look up every kernel the per-layer forward needs. Fails loudly
    /// if any are missing — better to surface that at startup than
    /// silently mid-decode.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms: gpu.kernel("rms_norm", "rms_norm")?,
            rope: gpu.kernel("rope_apply", "rope_apply")?,
            kvap: gpu.kernel("kv_cache_append", "kv_cache_append")?,
            attn: gpu.kernel("attention_decode", "attention_decode")?,
            sg: gpu.kernel("sigmoid_gate", "sigmoid_gate")?,
            add_rms: gpu.kernel("add_rms_norm", "add_rms_norm")?,
            qkv_split: gpu.kernel("qwen35_qkv_split", "qwen35_qkv_split")?,
            conv1d: gpu.kernel("causal_conv1d_update_l2norm", "causal_conv1d_update_l2norm")?,
            gdn_gate: gpu.kernel("gdn_helpers", "gdn_compute_gate")?,
            sigmoid: gpu.kernel("gdn_helpers", "sigmoid_bf16_to_f32")?,
            gdn_dec: gpu.kernel("gated_delta_rule_decode", "gated_delta_rule_decode")?,
            kvap_turbo8: gpu.kernel("kv_cache_append_turbo8", "kv_cache_append_turbo8")?,
            attn_turbo8: gpu.kernel("attention_decode_turbo8", "attention_decode_turbo8")?,
            kvap_turbo4: gpu.kernel("kv_cache_append_turbo4", "kv_cache_append_turbo4")?,
            attn_turbo4: gpu.kernel("attention_decode_turbo4", "attention_decode_turbo4")?,
            kvap_turbo3: gpu.kernel("kv_cache_append_turbo3", "kv_cache_append_turbo3")?,
            attn_turbo3: gpu.kernel("attention_decode_turbo3", "attention_decode_turbo3")?,
            kvap_turbo2: gpu.kernel("kv_cache_append_turbo2", "kv_cache_append_turbo2")?,
            attn_turbo2: gpu.kernel("attention_decode_turbo2", "attention_decode_turbo2")?,
            kvap_bf16k_turbo4v: gpu.kernel(
                "kv_cache_append_bf16k_turbov",
                "kv_cache_append_bf16k_turbo4v",
            )?,
            attn_bf16k_turbo4v: gpu.kernel(
                "attention_decode_bf16k_turbov",
                "attention_decode_bf16k_turbo4v",
            )?,
            kvap_bf16k_turbo3v: gpu.kernel(
                "kv_cache_append_bf16k_turbov",
                "kv_cache_append_bf16k_turbo3v",
            )?,
            attn_bf16k_turbo3v: gpu.kernel(
                "attention_decode_bf16k_turbov",
                "attention_decode_bf16k_turbo3v",
            )?,
            kvap_bf16k_turbo2v: gpu.kernel(
                "kv_cache_append_bf16k_turbov",
                "kv_cache_append_bf16k_turbo2v",
            )?,
            attn_bf16k_turbo2v: gpu.kernel(
                "attention_decode_bf16k_turbov",
                "attention_decode_bf16k_turbo2v",
            )?,
            wht: gpu.kernel("wht_bf16", "wht_bf16_inplace")?,
            wht_inv: gpu.kernel("wht_bf16", "wht_bf16_inplace_inv")?,
        })
    }
}

/// KV cache storage format for the Metal contiguous cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalKvDtype {
    /// Raw bfloat, 2 bytes/elem.
    Bf16,
    /// FP8 E4M3 data + bf16 group-of-16 scales, WHT-rotated basis.
    /// 2.13× smaller than bf16.
    Turbo8,
    /// 4-bit Lloyd-Max codebook indices + FP8 group-of-16 scales
    /// (matched-norm L2), WHT-rotated basis. 3.56× smaller than bf16.
    Turbo4,
    /// 3-bit Lloyd-Max (8 values → 3 bytes) + FP8 group scales.
    /// 4.57× smaller than bf16.
    Turbo3,
    /// 2-bit Lloyd-Max (4 elems/byte) + FP8 group scales.
    /// 6.4× smaller than bf16.
    Turbo2,
    /// Safer-asym: K raw bf16 (un-rotated), V Turbo4. Production-
    /// recommended frontier — K precision dominates retrieval quality.
    Bf16KTurbo4V,
    /// Safer-asym: K raw bf16, V Turbo3.
    Bf16KTurbo3V,
    /// Safer-asym: K raw bf16, V Turbo2.
    Bf16KTurbo2V,
}

impl MetalKvDtype {
    /// K side stored in the WHT-rotated basis (gates K rotation at
    /// append and the WHT(Q) decode bookend).
    pub fn k_is_rotated(self) -> bool {
        matches!(
            self,
            Self::Turbo8 | Self::Turbo4 | Self::Turbo3 | Self::Turbo2
        )
    }
    /// V side stored in the WHT-rotated basis (gates V rotation at
    /// append and the iWHT(out) decode bookend).
    pub fn v_is_rotated(self) -> bool {
        self != Self::Bf16
    }
}

impl std::str::FromStr for MetalKvDtype {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "turbo8" => Ok(Self::Turbo8),
            "turbo4" => Ok(Self::Turbo4),
            "turbo3" => Ok(Self::Turbo3),
            "turbo2" => Ok(Self::Turbo2),
            "bf16k_turbo4v" => Ok(Self::Bf16KTurbo4V),
            "bf16k_turbo3v" => Ok(Self::Bf16KTurbo3V),
            "bf16k_turbo2v" => Ok(Self::Bf16KTurbo2V),
            other => {
                anyhow::bail!(
                    "kv dtype {other:?} not supported on metal (bf16 | turbo8 | turbo4 | turbo3 | turbo2 | bf16k_turbo4v/3v/2v)"
                )
            }
        }
    }
}

/// Per-layer KV cache for a full-attention layer (single-batch).
///
/// `dtype` selects the storage format; for the turbo formats `k`/`v`
/// hold packed quantized data in the WHT-rotated basis and `scales`
/// holds the per-16-element group scales (bf16 for Turbo8, FP8 E4M3
/// bytes for Turbo4). The forward routes appends and attention through
/// the matching kernels with WHT(Q)/iWHT(out) bookends.
pub struct LayerKvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// Capacity in tokens — caller pre-allocates `max_seq_len * KV_DIM`.
    #[allow(dead_code)]
    pub capacity: u32,
    pub dtype: MetalKvDtype,
    /// Per-side group-scale buffers — `Some` only for quantized sides
    /// (both for symmetric turbo dtypes, V-only for the safer-asym
    /// Bf16K+TurboNV family, neither for Bf16).
    pub k_scales: Option<DevicePtr>,
    pub v_scales: Option<DevicePtr>,
}

impl LayerKvCache {
    /// Allocate a cache in the given storage format.
    pub fn alloc(
        gpu: &dyn GpuBackend,
        dtype: MetalKvDtype,
        max_seq: u32,
        kv_dim: u32,
    ) -> Result<Self> {
        assert!(
            dtype == MetalKvDtype::Bf16 || kv_dim.is_multiple_of(16),
            "turbo dtypes need KV_DIM divisible by 16"
        );
        let n = (max_seq * kv_dim) as usize;
        let scale_bytes_e4m3 = (max_seq * kv_dim / 16) as usize;
        // (k_bytes, v_bytes, k_scale_bytes, v_scale_bytes)
        let (kb, vb, ksb, vsb) = match dtype {
            MetalKvDtype::Bf16 => (n * 2, n * 2, 0, 0),
            // 1 byte/elem + bf16 scales (2 bytes per group of 16).
            MetalKvDtype::Turbo8 => (n, n, scale_bytes_e4m3 * 2, scale_bytes_e4m3 * 2),
            // 2 elems/byte + E4M3 scales.
            MetalKvDtype::Turbo4 => (n / 2, n / 2, scale_bytes_e4m3, scale_bytes_e4m3),
            // 8 values -> 3 bytes + E4M3 scales.
            MetalKvDtype::Turbo3 => (n * 3 / 8, n * 3 / 8, scale_bytes_e4m3, scale_bytes_e4m3),
            // 4 elems/byte + E4M3 scales.
            MetalKvDtype::Turbo2 => (n / 4, n / 4, scale_bytes_e4m3, scale_bytes_e4m3),
            // Safer-asym: K raw bf16, V packed + E4M3 scales.
            MetalKvDtype::Bf16KTurbo4V => (n * 2, n / 2, 0, scale_bytes_e4m3),
            MetalKvDtype::Bf16KTurbo3V => (n * 2, n * 3 / 8, 0, scale_bytes_e4m3),
            MetalKvDtype::Bf16KTurbo2V => (n * 2, n / 4, 0, scale_bytes_e4m3),
        };
        let alloc_opt = |bytes: usize| -> Result<Option<DevicePtr>> {
            Ok(if bytes > 0 {
                Some(gpu.alloc(bytes)?)
            } else {
                None
            })
        };
        Ok(Self {
            k: gpu.alloc(kb)?,
            v: gpu.alloc(vb)?,
            capacity: max_seq,
            dtype,
            k_scales: alloc_opt(ksb)?,
            v_scales: alloc_opt(vsb)?,
        })
    }
}

/// Full-attention layer weights, parameterised over the backend's
/// quantised weight type.
pub struct FullAttentionLayer<'a, Q: QuantWeights> {
    pub input_ln: DevicePtr,
    pub q_norm: DevicePtr,
    pub k_norm: DevicePtr,
    pub post_ln: DevicePtr,
    pub q_proj: &'a Q,
    pub k_proj: &'a Q,
    pub v_proj: &'a Q,
    pub o_proj: &'a Q,
    pub gate_proj: &'a Q,
    pub up_proj: &'a Q,
    pub down_proj: &'a Q,
}

/// Per-call scratch buffers for the full-attention forward.
pub struct FullAttentionScratch {
    pub x_norm: DevicePtr,
    pub q_full: DevicePtr,
    pub q_split: DevicePtr,
    pub gate_split: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub q_norm_out: DevicePtr,
    pub k_norm_out: DevicePtr,
    pub attn_out: DevicePtr,
    pub gated_attn: DevicePtr,
    pub o: DevicePtr,
    pub x_resid: DevicePtr,
    pub x_norm2: DevicePtr,
    pub gate_act: DevicePtr,
    pub up_act: DevicePtr,
    pub x_out: DevicePtr,
}

/// Linear-attention (GDN) layer weights.
pub struct LinearAttentionLayer<'a, Q: QuantWeights> {
    pub input_ln: DevicePtr,
    /// FP32 `[num_state_heads]`.
    pub a_log: DevicePtr,
    /// BF16 `[num_state_heads]`.
    pub dt_bias: DevicePtr,
    /// BF16 `[QKV_TOTAL_LIN, conv_kernel_size, 1]`.
    pub conv1d_weight: DevicePtr,
    pub in_proj_a: &'a Q,
    pub in_proj_b: &'a Q,
    pub in_proj_qkv: &'a Q,
    pub in_proj_z: &'a Q,
    /// BF16 `[v_head_dim_lin]`.
    pub norm_weight: DevicePtr,
    pub out_proj: &'a Q,
    pub post_ln: DevicePtr,
    pub gate_proj: &'a Q,
    pub up_proj: &'a Q,
    pub down_proj: &'a Q,
}

/// Per-layer SSM/conv state for a linear-attention layer. Persists
/// across tokens. Caller owns alloc + zero-init.
pub struct LinearAttentionState {
    /// FP32 `[QKV_TOTAL_LIN, conv_kernel_size]`.
    pub conv1d_state: DevicePtr,
    /// FP32 `[batch=1, num_v_heads_lin, k_head_dim_lin, v_head_dim_lin]`.
    pub gdn_state: DevicePtr,
}

/// Per-call scratch buffers for the linear-attention forward.
pub struct LinearAttentionScratch {
    pub x_norm: DevicePtr,
    pub dt_raw: DevicePtr,
    pub b_raw: DevicePtr,
    pub qkv: DevicePtr,
    pub qkv_smooth: DevicePtr,
    pub z: DevicePtr,
    /// FP32 `[num_state_heads]`.
    pub gate: DevicePtr,
    /// FP32 `[num_state_heads]`.
    pub beta: DevicePtr,
    pub y: DevicePtr,
    pub y_norm: DevicePtr,
    pub out: DevicePtr,
    pub x_resid: DevicePtr,
    pub x_norm2: DevicePtr,
    pub gate_act: DevicePtr,
    pub up_act: DevicePtr,
    pub x_final: DevicePtr,
}
