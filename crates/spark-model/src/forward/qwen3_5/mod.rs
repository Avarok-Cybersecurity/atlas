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
        })
    }
}

/// Per-layer KV cache for a full-attention layer (single-batch).
pub struct LayerKvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// Capacity in tokens — caller pre-allocates `max_seq_len * KV_DIM`.
    #[allow(dead_code)]
    pub capacity: u32,
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
