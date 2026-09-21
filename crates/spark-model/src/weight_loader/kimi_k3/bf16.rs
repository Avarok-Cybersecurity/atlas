// SPDX-License-Identifier: AGPL-3.0-only

//! Bind 0.40B BF16 (or FP32) twin tensors. Packed experts land DSV4 E8M0
//! when `K3_ALLOW_MXFP4=1` (refused upstream otherwise).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use avarok_core::config::ModelConfig;
use avarok_core::kimi_k3::{K3Graph, kda_from, mla_from, moe_from};
use half::bf16;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::kimi_k3::bound::{K3BoundLayer, K3HostShared};
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, QuantizedWeight};

use super::mxfp4::{quantized_k3_mxfp4_e8m0, validate_packed_partition};

pub use avarok_core::kimi_k3::weights::{layer_keys, text_key};

pub fn load_embedding(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // Twin safetensors are F32 (HF tensor type). Engine embed/lm_head/norm
    // gather BF16 rows (`h * 2`). Leave FP32 as-is and the aviation prompt
    // becomes `自主性!!!!…` instead of C1 id 1459.
    dense_for_bf16_engine(store, &text_key(config, "model.embed_tokens.weight"), gpu)
}

pub fn load_final_norm(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    dense_for_bf16_engine(store, &text_key(config, "model.norm.weight"), gpu)
}

pub fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // Full `[vocab, hidden]` on every rank. Vocab-parallel GEMV + all-reduce
    // is `model::impl_a3::lmhead_vocab_shard` (same as minimax / glm5_next).
    // A load-time column shard would double-split with that offset.
    let _ = config.tp_world_size;
    let a = text_key(config, "lm_head.weight");
    if store.contains(&a) {
        return dense_for_bf16_engine(store, &a, gpu);
    }
    dense_for_bf16_engine(store, "lm_head.weight", gpu)
}

/// Engine embed/lm_head/final_norm are BF16 gathers. Host-convert F32 so we
/// do not issue `quantize_nvfp4::f32_to_bf16_trunc` (another unresolved
/// lookup on the kimi-k3 target).
fn dense_for_bf16_engine(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    anyhow::ensure!(
        avarok_core::kimi_k3::binding_memory::engine_bf16_weight(name),
        "K3 engine conversion missing from preflight contract: {name}"
    );
    let w = store.get(name)?;
    if w.dtype != WeightDtype::FP32 {
        return Ok(DenseWeight { weight: w.ptr });
    }
    let n = w.num_elements();
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut raw)?;
    let bf16_bytes = f32_le_to_bf16_bytes(&raw);
    let ptr = gpu.alloc(bf16_bytes.len())?;
    if let Err(error) = gpu.copy_h2d(&bf16_bytes, ptr) {
        let _ = gpu.free(ptr);
        return Err(error);
    }
    store
        .derived()
        .adopt("K3 FP32 engine projection to BF16", ptr, bf16_bytes.len());
    Ok(DenseWeight { weight: ptr })
}

pub(super) fn f32_le_to_bf16_bytes(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(4)
        .flat_map(|c| {
            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            bf16::from_f32(f).to_le_bytes()
        })
        .collect()
}

pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let marked = super::tp::is_prepartitioned(store, config)?;
    let packed = store.names().any(|n| n.ends_with(".weight_packed"));
    if config.tp_world_size > 1 && packed && !marked {
        bail!(
            "K3 TP does not slice packed MXFP4 after upload; use the rank-aware checkpoint loader"
        );
    }
    let mut moe = moe_from(config);
    if packed && config.tp_world_size > 1 {
        anyhow::ensure!(
            moe.expert_hidden.is_multiple_of(config.tp_world_size),
            "K3 packed expert width must divide TP world"
        );
        moe.expert_hidden /= config.tp_world_size;
    }
    let graph = K3Graph::from_config(config);
    if config.tp_world_size > 1 {
        tracing::info!(
            "kimi_k3: binding TP weights rank={}/{}",
            config.tp_rank,
            config.tp_world_size
        );
    }
    let out_proj_n = text_key(config, "model.output_attn_res_proj.weight");
    let out_norm_n = text_key(config, "model.output_attn_res_norm.weight");
    let out_proj_t = store.get(&out_proj_n)?;
    // GGUF ships one F32 score vector; atlas wants proj+norm. Alias when absent.
    let out_norm_t = match store.get(&out_norm_n) {
        Ok(t) => t,
        Err(_) => out_proj_t,
    };
    let shared = Arc::new(K3HostShared {
        config: config.clone(),
        graph: graph.clone(),
        kda: kda_from(config),
        mla: mla_from(config),
        moe,
        output_res_proj: DenseWeight {
            weight: out_proj_t.ptr,
        },
        output_res_norm: DenseWeight {
            weight: out_norm_t.ptr,
        },
        output_res_proj_meta: (out_proj_t.dtype, out_proj_t.num_elements()),
        output_res_norm_meta: (out_norm_t.dtype, out_norm_t.num_elements()),
        output_host: OnceLock::new(),
        kda_kernels: OnceLock::new(),
        mla_kernels: OnceLock::new(),
        moe_kernels: OnceLock::new(),
        attnres: Mutex::new(HashMap::new()),
    });
    if packed {
        // Resolve mandatory packed kernels before the boot audit seals lookup.
        let kernels = crate::kimi_k3::moe_cuda::K3MoeGemmKernels::resolve(gpu)?;
        let _ = shared.moe_kernels.set(kernels);
    }
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(graph.layers.len());
    for spec in &graph.layers {
        let keys = filter_keys_for_ep(
            layer_keys(config, spec.index, spec.mixer, spec.mlp, config.num_experts),
            config,
        );
        let mut weights = Vec::with_capacity(keys.len());
        let mut weight_meta = Vec::with_capacity(keys.len());
        let mut mxfp4_experts: Vec<(String, QuantizedWeight)> = Vec::new();
        for k in &keys {
            if let Some(prefix) = packed_expert_prefix(store, k) {
                validate_packed_partition(store, &prefix, config)?;
                mxfp4_experts.push((prefix.clone(), quantized_k3_mxfp4_e8m0(store, &prefix)?));
                continue;
            }
            let (dw, meta) = super::tp::load_sharded(store, k, spec.mixer, spec.mlp, config, gpu)?;
            weights.push(dw);
            weight_meta.push(meta);
        }
        layers.push(Box::new(K3BoundLayer {
            index: spec.index,
            spec: *spec,
            weights,
            weight_meta,
            mxfp4_experts,
            host: OnceLock::new(),
            shared: shared.clone(),
        }));
    }
    Ok(layers)
}

fn local_expert_range(config: &avarok_core::config::ModelConfig) -> (usize, usize) {
    let world = config.ep_world_size.max(1);
    if world <= 1 {
        return (0, config.num_experts);
    }
    let per = config.num_experts / world;
    let start = config.ep_rank * per;
    let end = if config.ep_rank + 1 == world {
        config.num_experts
    } else {
        start + per
    };
    (start, end)
}

fn filter_keys_for_ep(keys: Vec<String>, config: &avarok_core::config::ModelConfig) -> Vec<String> {
    let (lo, hi) = local_expert_range(config);
    if config.ep_world_size.max(1) <= 1 {
        return keys;
    }
    keys.into_iter()
        .filter(|k| {
            let Some(rest) = k.split("block_sparse_moe.experts.").nth(1) else {
                return true;
            };
            let Some(idx_str) = rest.split('.').next() else {
                return true;
            };
            let Ok(idx) = idx_str.parse::<usize>() else {
                return true;
            };
            idx >= lo && idx < hi
        })
        .collect()
}

fn packed_expert_prefix(store: &WeightStore, weight_key: &str) -> Option<String> {
    let prefix = weight_key.strip_suffix(".weight")?;
    if !prefix.contains("block_sparse_moe.experts.") {
        return None;
    }
    store
        .contains(&format!("{prefix}.weight_packed"))
        .then(|| prefix.to_string())
}

#[cfg(test)]
mod tests {
    use super::f32_le_to_bf16_bytes;
    use half::bf16;

    #[test]
    fn f32_embed_row_becomes_bf16() {
        let f = 1.5f32;
        let out = f32_le_to_bf16_bytes(&f.to_le_bytes());
        assert_eq!(out, bf16::from_f32(1.5).to_le_bytes());
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use avarok_core::scope::ModelResource;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightTensor;

    #[test]
    fn converted_engine_projection_is_owned_and_released() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(16).unwrap();
        gpu.copy_h2d(&[0u8; 16], ptr).unwrap();
        let mut store = WeightStore::from_map(HashMap::from([(
            "language_model.model.embed_tokens.weight".into(),
            WeightTensor {
                ptr,
                shape: vec![2, 2],
                dtype: WeightDtype::FP32,
            },
        )]));
        let weight =
            dense_for_bf16_engine(&store, "language_model.model.embed_tokens.weight", &gpu)
                .unwrap();
        assert_ne!(weight.weight, ptr);
        assert_eq!(store.derived().bytes(), 8);
        store.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}
