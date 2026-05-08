// SPDX-License-Identifier: AGPL-3.0-only
//! Full-attention layer forward (single-token decode).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};

use super::super::quant_weights::QuantWeights;
use super::{
    FullAttentionLayer, FullAttentionScratch, LayerKvCache, Qwen35ForwardConfig, Qwen35Kernels,
};

/// Single-token full-attention decoder forward. Returns the
/// `DevicePtr` containing the layer's output residual stream
/// (caller-owned `scratch.x_out`).
#[allow(clippy::too_many_arguments)]
pub fn forward_full_attention<Q: QuantWeights>(
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    k: &Qwen35Kernels,
    layer: &FullAttentionLayer<'_, Q>,
    scratch: &FullAttentionScratch,
    kv: &LayerKvCache,
    inv_freq_ptr: DevicePtr,
    positions_ptr: DevicePtr,
    x_in: DevicePtr,
    cache_pos: u32,
    seq_len_attn: u32,
    stream: u64,
) -> Result<DevicePtr> {
    // norm1
    gpu.launch_typed(
        k.rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.hidden.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(layer.input_ln),
            KernelArg::Buffer(scratch.x_norm),
        ],
    )?;
    layer
        .q_proj
        .gemv(gpu, scratch.x_norm, scratch.q_full, stream)?;
    // Fused k_proj + v_proj — both share x_norm and have identical
    // (N=KV_DIM, K=HIDDEN, group_size) shapes for Qwen3.5.
    layer.k_proj.gemv_gate_up_with(
        layer.v_proj,
        gpu,
        scratch.x_norm,
        scratch.k,
        scratch.v,
        stream,
    )?;

    // Qwen3.5 q_proj output is [num_heads, head_dim * 2] interleaved
    // per head as [Q_h | gate_h]. Deinterleave into separate buffers
    // before normalisation / RoPE / attention.
    gpu.launch_typed(
        k.qkv_split,
        [cfg.head_dim, cfg.num_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Buffer(scratch.q_full),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(scratch.gate_split),
        ],
    )?;
    let gate_view = scratch.gate_split;

    // per-head q/k norm (treat each head as a token)
    gpu.launch_typed(
        k.rms,
        [cfg.num_heads, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(layer.q_norm),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    gpu.launch_typed(
        k.rms,
        [cfg.num_kv_heads, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.k),
            KernelArg::Buffer(layer.k_norm),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // RoPE on the q_norm_out / k_norm_out buffers directly. Saves the
    // d2d copy that an in-place norm would have cost.
    let half_dim = cfg.rotary_dim / 2;
    let n_tokens = 1u32;
    gpu.launch_typed(
        k.rope,
        [half_dim, cfg.num_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    gpu.launch_typed(
        k.rope,
        [half_dim, cfg.num_kv_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // KV-cache append uses the post-RoPE k_norm_out.
    gpu.launch_typed(
        k.kvap,
        [cfg.head_dim, cfg.num_kv_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cache_pos.to_le_bytes()),
            KernelArg::Buffer(scratch.k_norm_out),
            KernelArg::Buffer(scratch.v),
            KernelArg::Buffer(kv.k),
            KernelArg::Buffer(kv.v),
        ],
    )?;

    // attention_decode with seq_len = seq_len_attn.
    let scale: f32 = 1.0 / (cfg.head_dim as f32).sqrt();
    gpu.launch_typed(
        k.attn,
        [cfg.num_heads, 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&seq_len_attn.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&scale.to_le_bytes()),
            KernelArg::Buffer(scratch.q_norm_out),
            KernelArg::Buffer(kv.k),
            KernelArg::Buffer(kv.v),
            KernelArg::Buffer(scratch.attn_out),
        ],
    )?;

    // sigmoid_gate(attn_gate, attn_out)
    let q_only = cfg.q_only();
    gpu.launch_typed(
        k.sg,
        [q_only.div_ceil(64), 1, 1],
        [64, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&q_only.to_le_bytes()),
            KernelArg::Buffer(gate_view),
            KernelArg::Buffer(scratch.attn_out),
            KernelArg::Buffer(scratch.gated_attn),
        ],
    )?;

    // o_proj
    layer
        .o_proj
        .gemv(gpu, scratch.gated_attn, scratch.o, stream)?;

    // Fused residual + post-attention RMSNorm.
    gpu.launch_typed(
        k.add_rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.hidden.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(scratch.o),
            KernelArg::Buffer(layer.post_ln),
            KernelArg::Buffer(scratch.x_resid),
            KernelArg::Buffer(scratch.x_norm2),
        ],
    )?;
    // Fused dual-output GEMV: shares x_norm2 across gate_proj and up_proj.
    layer.gate_proj.gemv_gate_up_with(
        layer.up_proj,
        gpu,
        scratch.x_norm2,
        scratch.gate_act,
        scratch.up_act,
        stream,
    )?;
    // Fused: x_out = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    layer.down_proj.gemv_silu_gate_resid(
        gpu,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_out,
        stream,
    )?;
    Ok(scratch.x_out)
}
