// SPDX-License-Identifier: AGPL-3.0-only
//! Scratch / state buffer allocation for the inference driver.

use anyhow::Result;
use spark_model::forward::qwen3_5::{
    FullAttentionScratch, LinearAttentionScratch, LinearAttentionState,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::metal_backend::MetalGpuBackend;

use super::CFG;

pub fn alloc_full_attention_scratch(backend: &MetalGpuBackend) -> Result<FullAttentionScratch> {
    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    Ok(FullAttentionScratch {
        x_norm: alloc_bf16(CFG.hidden)?,
        q_full: alloc_bf16(CFG.q_total())?,
        q_split: alloc_bf16(CFG.q_only())?,
        gate_split: alloc_bf16(CFG.q_only())?,
        k: alloc_bf16(CFG.kv_dim())?,
        v: alloc_bf16(CFG.kv_dim())?,
        q_norm_out: alloc_bf16(CFG.q_only())?,
        k_norm_out: alloc_bf16(CFG.kv_dim())?,
        attn_out: alloc_bf16(CFG.q_only())?,
        gated_attn: alloc_bf16(CFG.q_only())?,
        o: alloc_bf16(CFG.hidden)?,
        x_resid: alloc_bf16(CFG.hidden)?,
        x_norm2: alloc_bf16(CFG.hidden)?,
        gate_act: alloc_bf16(CFG.intermediate)?,
        up_act: alloc_bf16(CFG.intermediate)?,
        x_out: alloc_bf16(CFG.hidden)?,
    })
}

pub fn alloc_linear_attention_scratch(backend: &MetalGpuBackend) -> Result<LinearAttentionScratch> {
    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    let alloc_f32 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 4)?) };
    Ok(LinearAttentionScratch {
        x_norm: alloc_bf16(CFG.hidden)?,
        dt_raw: alloc_bf16(CFG.num_state_heads())?,
        b_raw: alloc_bf16(CFG.num_state_heads())?,
        qkv: alloc_bf16(CFG.qkv_total_lin())?,
        qkv_smooth: alloc_bf16(CFG.qkv_total_lin())?,
        z: alloc_bf16(CFG.z_dim_lin())?,
        gate: alloc_f32(CFG.num_state_heads())?,
        beta: alloc_f32(CFG.num_state_heads())?,
        y: alloc_bf16(CFG.z_dim_lin())?,
        y_norm: alloc_bf16(CFG.z_dim_lin())?,
        out: alloc_bf16(CFG.hidden)?,
        x_resid: alloc_bf16(CFG.hidden)?,
        x_norm2: alloc_bf16(CFG.hidden)?,
        gate_act: alloc_bf16(CFG.intermediate)?,
        up_act: alloc_bf16(CFG.intermediate)?,
        x_final: alloc_bf16(CFG.hidden)?,
    })
}

pub fn alloc_linear_attention_state(backend: &MetalGpuBackend) -> Result<LinearAttentionState> {
    let conv_state_bytes = (CFG.qkv_total_lin() * CFG.conv_kernel_size) as usize * 4;
    let gdn_state_floats = (CFG.num_v_heads_lin * CFG.k_head_dim_lin * CFG.v_head_dim_lin) as usize;
    let conv1d_state = backend.alloc(conv_state_bytes)?;
    let gdn_state = backend.alloc(gdn_state_floats * 4)?;
    backend.memset(conv1d_state, 0, conv_state_bytes)?;
    backend.memset(gdn_state, 0, gdn_state_floats * 4)?;
    Ok(LinearAttentionState {
        conv1d_state,
        gdn_state,
    })
}
