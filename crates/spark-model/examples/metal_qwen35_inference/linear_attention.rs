// SPDX-License-Identifier: AGPL-3.0-only
//! GDN (linear-attention) decoder layer — weight loader + view bridge.
//!
//! Per-layer forward lives in
//! `spark_model::forward::qwen3_5::forward_linear_attention`; this
//! file owns the safetensors → GPU-buffer loader concrete to MLX 8-bit
//! and a `view()` shim that produces the borrowed
//! `LinearAttentionLayer<'_, MlxInt8Weight>` the shared forward
//! expects.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use spark_model::forward::qwen3_5;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::MlxInt8Weight;

pub(crate) struct LinearAttentionLayer {
    pub(crate) input_ln: DevicePtr,
    pub(crate) a_log: DevicePtr,         // F32 [num_state_heads]
    pub(crate) dt_bias: DevicePtr,       // BF16 [num_state_heads]
    pub(crate) conv1d_weight: DevicePtr, // BF16 [QKV_TOTAL_LIN, kernel_size]
    pub(crate) in_proj_a: MlxInt8Weight,
    pub(crate) in_proj_b: MlxInt8Weight,
    pub(crate) in_proj_qkv: MlxInt8Weight,
    pub(crate) in_proj_z: MlxInt8Weight,
    pub(crate) norm_weight: DevicePtr, // BF16 [v_head_dim_lin]
    pub(crate) out_proj: MlxInt8Weight,
    // Post-attention MLP — Qwen3.5 decoder layer applies it for both
    // GDN and full-attention layers.
    pub(crate) post_ln: DevicePtr,
    pub(crate) gate_proj: MlxInt8Weight,
    pub(crate) up_proj: MlxInt8Weight,
    pub(crate) down_proj: MlxInt8Weight,
}

impl LinearAttentionLayer {
    pub(crate) fn load(
        backend: &MetalGpuBackend,
        st: &SafeTensors,
        layer_idx: u32,
        group_size: u32,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_raw = |name: &str| -> Result<DevicePtr> {
            let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        let load_q = |suffix: &str| {
            MlxInt8Weight::load(backend, st, &format!("{prefix}.{suffix}"), group_size)
        };
        Ok(Self {
            input_ln: load_raw(&format!("{prefix}.input_layernorm.weight"))?,
            a_log: load_raw(&format!("{prefix}.linear_attn.A_log"))?,
            dt_bias: load_raw(&format!("{prefix}.linear_attn.dt_bias"))?,
            conv1d_weight: load_raw(&format!("{prefix}.linear_attn.conv1d.weight"))?,
            in_proj_a: load_q("linear_attn.in_proj_a")?,
            in_proj_b: load_q("linear_attn.in_proj_b")?,
            in_proj_qkv: load_q("linear_attn.in_proj_qkv")?,
            in_proj_z: load_q("linear_attn.in_proj_z")?,
            norm_weight: load_raw(&format!("{prefix}.linear_attn.norm.weight"))?,
            out_proj: load_q("linear_attn.out_proj")?,
            post_ln: load_raw(&format!("{prefix}.post_attention_layernorm.weight"))?,
            gate_proj: load_q("mlp.gate_proj")?,
            up_proj: load_q("mlp.up_proj")?,
            down_proj: load_q("mlp.down_proj")?,
        })
    }

    /// Borrow this layer as the shape the shared forward expects.
    pub(crate) fn view(&self) -> qwen3_5::LinearAttentionLayer<'_, MlxInt8Weight> {
        qwen3_5::LinearAttentionLayer {
            input_ln: self.input_ln,
            a_log: self.a_log,
            dt_bias: self.dt_bias,
            conv1d_weight: self.conv1d_weight,
            in_proj_a: &self.in_proj_a,
            in_proj_b: &self.in_proj_b,
            in_proj_qkv: &self.in_proj_qkv,
            in_proj_z: &self.in_proj_z,
            norm_weight: self.norm_weight,
            out_proj: &self.out_proj,
            post_ln: self.post_ln,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
        }
    }
}
