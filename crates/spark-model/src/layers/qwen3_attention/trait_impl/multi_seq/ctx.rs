// SPDX-License-Identifier: AGPL-3.0-only

//! Per-call context for multi-sequence batched decode. Bundles the
//! shared scalars and buffer pointers that every phase reads.

use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Shared scalars / buffer pointers for `super::decode_multi_seq_inner`.
/// Built once in the orchestrator, then handed to each phase by `&self`.
#[allow(dead_code)]
pub(super) struct MultiSeqCtx<'a> {
    /// Wrapping forward-pass context (kernels, buffers, gpu, config…).
    pub fwd: &'a ForwardContext<'a>,
    /// Per-token hidden state buffer (input).
    pub hidden: DevicePtr,
    /// Per-token residual buffer (FP32 or BF16 depending on config).
    pub residual: DevicePtr,
    /// Number of sequences in this batched decode.
    pub n: usize,
    /// CUDA stream.
    pub stream: u64,

    // Cached config / per-layer scalars.
    pub h: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub eps: f32,
    pub bs: u32,
    pub bf16: usize,
    pub q_dim: u32,
    pub q_proj_dim: u32,
    pub q_proj_bytes: usize,
    pub per_seq_qkv: usize,

    // Buffer pointers used by ≥2 phases.
    /// Buffer holding RMS-normed hidden (output of phase 1, input of QKV).
    pub normed: DevicePtr,
    /// Per-token concatenated QKV output [Q | K | V] strided by `per_seq_qkv`.
    pub qkv_buf: DevicePtr,
}

impl<'a> MultiSeqCtx<'a> {
    pub(super) fn new(
        layer: &Qwen3AttentionLayer,
        fwd: &'a ForwardContext<'a>,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        bs: u32,
        stream: u64,
    ) -> Self {
        let h = fwd.config.hidden_size;
        let nq = layer
            .num_q_heads_override
            .unwrap_or(fwd.config.num_attention_heads) as u32;
        let nkv = layer
            .num_kv_heads_override
            .unwrap_or(fwd.config.num_key_value_heads) as u32;
        let hd = layer.head_dim_override.unwrap_or(fwd.config.head_dim) as u32;
        let eps = fwd.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let q_dim = nq * hd;
        let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * bf16;
        let per_seq_qkv = q_proj_bytes + (nkv * hd) as usize * bf16 * 2;
        let normed = fwd.buffers.norm_output();
        let qkv_buf = fwd.buffers.qkv_output();
        Self {
            fwd,
            hidden,
            residual,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bs,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
        }
    }
}
