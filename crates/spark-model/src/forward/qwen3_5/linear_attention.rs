// SPDX-License-Identifier: AGPL-3.0-only
//! GDN (linear-attention) layer forward (single-token decode).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};

use super::super::quant_weights::QuantWeights;
use super::{
    LinearAttentionLayer, LinearAttentionScratch, LinearAttentionState, Qwen35ForwardConfig,
    Qwen35Kernels,
};

/// Single-token GDN (linear-attention) decoder forward. Returns the
/// `DevicePtr` containing the layer's output residual stream — that
/// pointer is `x_buf`, into which `scratch.x_final` was copied so the
/// caller's residual-stream buffer stays stable across layers.
#[allow(clippy::too_many_arguments)]
pub fn forward_linear_attention<Q: QuantWeights>(
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    k: &Qwen35Kernels,
    layer: &LinearAttentionLayer<'_, Q>,
    state: &LinearAttentionState,
    scratch: &LinearAttentionScratch,
    x_in: DevicePtr,
    x_buf: DevicePtr,
    stream: u64,
    intra_dump: Option<&dyn Fn(&str, DevicePtr, u32) -> Result<()>>,
) -> Result<DevicePtr> {
    // 1. norm
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
    // 2. projections — fused in_proj_a/in_proj_b share `x_norm`.
    layer.in_proj_a.gemv_gate_up_with(
        layer.in_proj_b,
        gpu,
        scratch.x_norm,
        scratch.dt_raw,
        scratch.b_raw,
        stream,
    )?;
    layer
        .in_proj_qkv
        .gemv(gpu, scratch.x_norm, scratch.qkv, stream)?;
    layer
        .in_proj_z
        .gemv(gpu, scratch.x_norm, scratch.z, stream)?;

    // 3. fused causal_conv1d_update_l2norm: conv + SiLU + per-head
    // L2-norm on Q+K, SiLU only on V.
    let batch_one: u32 = 1;
    let block_x: u32 = cfg.k_head_dim_lin;
    let qkv_total_lin = cfg.qkv_total_lin();
    let blocks_per_batch = qkv_total_lin.div_ceil(block_x);
    let qk_channels: u32 = 2 * cfg.num_k_heads_lin * cfg.k_head_dim_lin;
    let l2_eps: f32 = 1e-6;
    gpu.launch_typed(
        k.conv1d,
        [blocks_per_batch * batch_one, 1, 1],
        [block_x, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(state.conv1d_state),
            KernelArg::Buffer(scratch.qkv),
            KernelArg::Buffer(layer.conv1d_weight),
            KernelArg::Buffer(scratch.qkv_smooth),
            KernelArg::Bytes(&batch_one.to_le_bytes()),
            KernelArg::Bytes(&qkv_total_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.conv_kernel_size.to_le_bytes()),
            KernelArg::Bytes(&qk_channels.to_le_bytes()),
            KernelArg::Bytes(&cfg.k_head_dim_lin.to_le_bytes()),
            KernelArg::Bytes(&l2_eps.to_le_bytes()),
        ],
    )?;
    // Atlas applies the GDN `1/sqrt(d)` factor at the kernel output;
    // MLX applies its inv_scale at the rms_norm input. The two paths
    // are mathematically equivalent — see the rationale in
    // `/Users/.../atlas/memory/feedback_mlx_rms_norm_vs_l2_norm.md`.

    // 4. gate = exp(softplus(dt + dt_bias) * -exp(A_log))
    let num_state_heads = cfg.num_state_heads();
    gpu.launch_typed(
        k.gdn_gate,
        [num_state_heads.div_ceil(32), 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&num_state_heads.to_le_bytes()),
            KernelArg::Buffer(scratch.dt_raw),
            KernelArg::Buffer(layer.dt_bias),
            KernelArg::Buffer(layer.a_log),
            KernelArg::Buffer(scratch.gate),
        ],
    )?;
    // 5. beta = sigmoid(b_raw) → FP32
    gpu.launch_typed(
        k.sigmoid,
        [num_state_heads.div_ceil(32), 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&num_state_heads.to_le_bytes()),
            KernelArg::Buffer(scratch.b_raw),
            KernelArg::Buffer(scratch.beta),
        ],
    )?;

    // 6. Split qkv_smooth: Q | K | V (sequential).
    let k_offset = (cfg.num_k_heads_lin * cfg.k_head_dim_lin) as usize * 2;
    let v_offset = (2 * cfg.num_k_heads_lin * cfg.k_head_dim_lin) as usize * 2;
    let q_view = scratch.qkv_smooth;
    let k_view = scratch.qkv_smooth.offset(k_offset);
    let v_view = scratch.qkv_smooth.offset(v_offset);

    // 7. gated_delta_rule_decode
    let batch_size = 1u32;
    let total_groups = cfg.num_v_heads_lin * batch_size;
    gpu.launch_typed(
        k.gdn_dec,
        [total_groups, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(state.gdn_state),
            KernelArg::Buffer(q_view),
            KernelArg::Buffer(k_view),
            KernelArg::Buffer(v_view),
            KernelArg::Buffer(scratch.gate),
            KernelArg::Buffer(scratch.beta),
            KernelArg::Buffer(scratch.y),
            KernelArg::Bytes(&batch_size.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_k_heads_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_v_heads_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.k_head_dim_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.v_head_dim_lin.to_le_bytes()),
        ],
    )?;

    // 8. per-head rms_norm at head_dim=v_head_dim_lin over num_v_heads_lin tokens
    gpu.launch_typed(
        k.rms,
        [cfg.num_v_heads_lin, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.v_head_dim_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.y),
            KernelArg::Buffer(layer.norm_weight),
            KernelArg::Buffer(scratch.y_norm),
        ],
    )?;

    // 9+10+11. Fused: out = out_proj @ (silu(z) ⊙ y_norm).
    layer
        .out_proj
        .gemv_silu_gate(gpu, scratch.z, scratch.y_norm, scratch.out, stream)?;

    // 12+13. Fused residual + post-attention RMSNorm.
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
            KernelArg::Buffer(scratch.out),
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
    // Fused: x_final = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    layer.down_proj.gemv_silu_gate_resid(
        gpu,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_final,
        stream,
    )?;

    // Intra-layer dumps (debug-only; gated externally via Option).
    if let Some(dump) = intra_dump {
        gpu.synchronize(stream)?;
        let z_dim_lin = cfg.z_dim_lin();
        dump("gdn_x_norm", scratch.x_norm, cfg.hidden)?;
        dump("gdn_qkv_pre", scratch.qkv, qkv_total_lin)?;
        dump("gdn_qkv_smooth", scratch.qkv_smooth, qkv_total_lin)?;
        dump("gdn_y", scratch.y, z_dim_lin)?;
        dump("gdn_y_norm", scratch.y_norm, z_dim_lin)?;
        dump("gdn_out", scratch.out, cfg.hidden)?;
        dump("gdn_x_resid", scratch.x_resid, cfg.hidden)?;
        dump("gdn_x_final", scratch.x_final, cfg.hidden)?;
    }

    // Copy x_final (post-MLP-residual) to caller's stable buffer so the
    // next layer's input pointer stays the same across layers.
    gpu.copy_d2d_async(scratch.x_final, x_buf, cfg.hidden as usize * 2, stream)?;
    Ok(x_buf)
}
