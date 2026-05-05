// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer context for Mistral MLA weight loading. Bundles the
//! immutable per-layer scalars + mutable shared state (yarn_inv_freq,
//! kernels) that the phase helpers need.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::weight_map::DenseWeight;

/// Per-layer arena for the MLA loading pipeline. Owned by
/// `load_one_layer`; phase fns mutate the optional fields as they fill in.
pub(super) struct MistralLayerCtx<'a> {
    // Immutable references / shared state
    pub store: &'a WeightStore,
    pub config: &'a ModelConfig,
    pub gpu: &'a dyn GpuBackend,
    pub absmax_k: spark_runtime::gpu::KernelHandle,
    pub quantize_k: spark_runtime::gpu::KernelHandle,
    pub stream: u64,
    pub layer_idx: usize,

    // Cached config scalars (avoid re-derefing config field by field).
    pub h: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    pub bf16: usize,

    // Filled in by phase A (LoRA QKV load + NVFP4 + TP).
    pub wq_a_dense: Option<DenseWeight>,
    pub wq_a_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub wq_b: Option<DenseWeight>,
    pub wq_b_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub q_a_norm: Option<DenseWeight>,
    pub wkv_a_dense: Option<DenseWeight>,
    pub wkv_a_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub wkv_a_rope_dense: Option<DenseWeight>,
    pub wkv_b: Option<DenseWeight>,
    pub kv_a_norm: Option<DenseWeight>,

    // Filled in by phase B (per-head transpose).
    pub w_uk_t: Option<DenseWeight>,
    pub w_uv: Option<DenseWeight>,
    pub wq_b_rope: Option<DenseWeight>,
    pub w_uk_host: Vec<u8>, // re-used by phase D for block-diag.

    // Filled in by phase C (W_QK absorbed CPU compute).
    pub w_qk_absorbed: Option<DenseWeight>,

    // Filled in by phase D (block-diagonal W_UK / W_UV).
    pub w_uk_block_diag: Option<DenseWeight>,
    pub w_uv_block_diag: Option<DenseWeight>,

    // Filled in by phase E (O projection).
    pub o_dense_bf16: Option<DenseWeight>,
    pub o_nvfp4: Option<crate::weight_map::QuantizedWeight>,
}

impl<'a> MistralLayerCtx<'a> {
    pub(super) fn new(
        store: &'a WeightStore,
        config: &'a ModelConfig,
        gpu: &'a dyn GpuBackend,
        absmax_k: spark_runtime::gpu::KernelHandle,
        quantize_k: spark_runtime::gpu::KernelHandle,
        stream: u64,
        layer_idx: usize,
    ) -> Self {
        Self {
            store,
            config,
            gpu,
            absmax_k,
            quantize_k,
            stream,
            layer_idx,
            h: config.hidden_size,
            n_heads: config.num_attention_heads,
            n_kv: config.num_key_value_heads,
            hd: config.head_dim,
            q_lora: config.q_lora_rank,
            kv_lora: config.kv_lora_rank,
            nope: config.qk_nope_head_dim,
            rope: config.qk_rope_head_dim,
            v_dim: config.v_head_dim,
            bf16: 2,
            wq_a_dense: None,
            wq_a_nvfp4: None,
            wq_b: None,
            wq_b_nvfp4: None,
            q_a_norm: None,
            wkv_a_dense: None,
            wkv_a_nvfp4: None,
            wkv_a_rope_dense: None,
            wkv_b: None,
            kv_a_norm: None,
            w_uk_t: None,
            w_uv: None,
            wq_b_rope: None,
            w_uk_host: Vec::new(),
            w_qk_absorbed: None,
            w_uk_block_diag: None,
            w_uv_block_diag: None,
            o_dense_bf16: None,
            o_nvfp4: None,
        }
    }

    pub(super) fn ap(&self) -> String {
        format!("layers.{}.attention", self.layer_idx)
    }
}

/// Compute or reuse the shared YaRN inv_freq table (computed once at
/// layer 0, returned by pointer for every subsequent layer).
pub(super) fn ensure_yarn_inv_freq(
    cached: &mut DevicePtr,
    config: &ModelConfig,
    rope: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    if !cached.is_null() {
        return Ok(*cached);
    }
    let ptr = super::yarn::compute_yarn_inv_freq(config, rope, gpu)?;
    *cached = ptr;
    Ok(ptr)
}
