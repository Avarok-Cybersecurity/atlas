// SPDX-License-Identifier: AGPL-3.0-only
//! Full-attention decoder layer — weight loader + view bridge.
//!
//! The per-layer forward path lives in
//! `spark_model::forward::qwen3_5::forward_full_attention`. This file
//! is only the on-disk-weight → GPU-buffer loader (concrete to MLX
//! 8-bit) plus a `view()` shim that produces the borrowed
//! `FullAttentionLayer<'_, MlxInt8Weight>` the shared forward expects.
//!
//! Keeping the loader local keeps the example self-contained while
//! still routing every per-layer kernel launch through the shared
//! `spark-model` forward — a future CUDA-side example would write
//! its own loader against (e.g.) `Nvfp4Weight` and call the same
//! `forward_full_attention`.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use spark_model::forward::qwen3_5;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::MlxInt8Weight;

pub(crate) struct FullAttentionLayer {
    pub(crate) input_ln: DevicePtr,
    pub(crate) q_norm: DevicePtr,
    pub(crate) k_norm: DevicePtr,
    pub(crate) post_ln: DevicePtr,
    pub(crate) q_proj: MlxInt8Weight,
    pub(crate) k_proj: MlxInt8Weight,
    pub(crate) v_proj: MlxInt8Weight,
    pub(crate) o_proj: MlxInt8Weight,
    pub(crate) gate_proj: MlxInt8Weight,
    pub(crate) up_proj: MlxInt8Weight,
    pub(crate) down_proj: MlxInt8Weight,
}

impl FullAttentionLayer {
    pub(crate) fn load(
        backend: &MetalGpuBackend,
        st: &SafeTensors,
        layer_idx: u32,
        group_size: u32,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_bf16 = |name: &str| -> Result<DevicePtr> {
            let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        let load_q = |suffix: &str| {
            MlxInt8Weight::load(backend, st, &format!("{prefix}.{suffix}"), group_size)
        };
        Ok(Self {
            input_ln: load_bf16(&format!("{prefix}.input_layernorm.weight"))?,
            q_norm: load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?,
            post_ln: load_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: load_q("self_attn.q_proj")?,
            k_proj: load_q("self_attn.k_proj")?,
            v_proj: load_q("self_attn.v_proj")?,
            o_proj: load_q("self_attn.o_proj")?,
            gate_proj: load_q("mlp.gate_proj")?,
            up_proj: load_q("mlp.up_proj")?,
            down_proj: load_q("mlp.down_proj")?,
        })
    }

    /// Borrow this layer as the shape the shared forward expects.
    pub(crate) fn view(&self) -> qwen3_5::FullAttentionLayer<'_, MlxInt8Weight> {
        qwen3_5::FullAttentionLayer {
            input_ln: self.input_ln,
            q_norm: self.q_norm,
            k_norm: self.k_norm,
            post_ln: self.post_ln,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
        }
    }
}
