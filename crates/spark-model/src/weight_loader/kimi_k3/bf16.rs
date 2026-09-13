// SPDX-License-Identifier: AGPL-3.0-only

//! Bind 0.40B BF16 (or FP32) twin tensors. Packed MXFP4 is refused upstream.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{K3Graph, MixerKind, MlpKind, kda_from, mla_from, moe_from};
use half::bf16;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::kimi_k3::bound::{K3BoundLayer, K3HostShared, WeightMeta};
use crate::layer::TransformerLayer;
use crate::weight_map::DenseWeight;

pub fn text_key(config: &ModelConfig, rest: &str) -> String {
    let p = config.weight_prefix.trim_end_matches('.');
    if p.is_empty() {
        rest.to_string()
    } else {
        format!("{p}.{rest}")
    }
}

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
    let w = store.get(name)?;
    if w.dtype != WeightDtype::FP32 {
        return Ok(DenseWeight { weight: w.ptr });
    }
    let n = w.num_elements();
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut raw)?;
    let bf16_bytes = f32_le_to_bf16_bytes(&raw);
    let ptr = gpu.alloc(bf16_bytes.len())?;
    gpu.copy_h2d(&bf16_bytes, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

fn f32_le_to_bf16_bytes(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(4)
        .flat_map(|c| {
            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            bf16::from_f32(f).to_le_bytes()
        })
        .collect()
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

pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let graph = K3Graph::from_config(config);
    let out_proj_n = text_key(config, "model.output_attn_res_proj.weight");
    let out_norm_n = text_key(config, "model.output_attn_res_norm.weight");
    let out_proj_t = store.get(&out_proj_n)?;
    let out_norm_t = store.get(&out_norm_n)?;
    let shared = Arc::new(K3HostShared {
        config: config.clone(),
        graph: graph.clone(),
        kda: kda_from(config),
        mla: mla_from(config),
        moe: moe_from(config),
        output_res_proj: DenseWeight {
            weight: out_proj_t.ptr,
        },
        output_res_norm: DenseWeight {
            weight: out_norm_t.ptr,
        },
        output_res_proj_meta: (out_proj_t.dtype, out_proj_t.num_elements()),
        output_res_norm_meta: (out_norm_t.dtype, out_norm_t.num_elements()),
        output_host: OnceLock::new(),
        attnres: Mutex::new(HashMap::new()),
    });
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(graph.layers.len());
    for spec in &graph.layers {
        let keys = layer_keys(config, spec.index, spec.mixer, spec.mlp, config.num_experts);
        let mut weights = Vec::with_capacity(keys.len());
        let mut weight_meta = Vec::with_capacity(keys.len());
        for k in &keys {
            let t = store.get(k)?;
            weights.push(DenseWeight { weight: t.ptr });
            weight_meta.push(WeightMeta {
                name: k.clone(),
                dtype: t.dtype,
                numel: t.num_elements(),
            });
        }
        layers.push(Box::new(K3BoundLayer {
            index: spec.index,
            spec: *spec,
            weights,
            weight_meta,
            host: OnceLock::new(),
            shared: shared.clone(),
        }));
    }
    Ok(layers)
}

pub fn layer_keys(
    config: &ModelConfig,
    i: usize,
    mixer: MixerKind,
    mlp: MlpKind,
    n_experts: usize,
) -> Vec<String> {
    let lp = text_key(config, &format!("model.layers.{i}"));
    let mut k = vec![
        format!("{lp}.input_layernorm.weight"),
        format!("{lp}.post_attention_layernorm.weight"),
        format!("{lp}.mlp_res_norm.weight"),
        format!("{lp}.mlp_res_proj.weight"),
        format!("{lp}.self_attention_res_norm.weight"),
        format!("{lp}.self_attention_res_proj.weight"),
        format!("{lp}.self_attn.g_proj.weight"),
        format!("{lp}.self_attn.o_proj.weight"),
    ];
    match mixer {
        MixerKind::Kda => {
            k.extend([
                format!("{lp}.self_attn.A_log"),
                format!("{lp}.self_attn.b_proj.weight"),
                format!("{lp}.self_attn.dt_bias"),
                format!("{lp}.self_attn.f_a_proj.weight"),
                format!("{lp}.self_attn.f_b_proj.weight"),
                format!("{lp}.self_attn.k_conv1d.weight"),
                format!("{lp}.self_attn.k_proj.weight"),
                format!("{lp}.self_attn.o_norm.weight"),
                format!("{lp}.self_attn.q_conv1d.weight"),
                format!("{lp}.self_attn.q_proj.weight"),
                format!("{lp}.self_attn.v_conv1d.weight"),
                format!("{lp}.self_attn.v_proj.weight"),
            ]);
        }
        MixerKind::Mla => {
            k.extend([
                format!("{lp}.self_attn.kv_a_layernorm.weight"),
                format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
                format!("{lp}.self_attn.kv_b_proj.weight"),
                format!("{lp}.self_attn.q_a_layernorm.weight"),
                format!("{lp}.self_attn.q_a_proj.weight"),
                format!("{lp}.self_attn.q_b_proj.weight"),
            ]);
        }
    }
    match mlp {
        MlpKind::Dense => {
            k.extend([
                format!("{lp}.mlp.down_proj.weight"),
                format!("{lp}.mlp.gate_proj.weight"),
                format!("{lp}.mlp.up_proj.weight"),
            ]);
        }
        MlpKind::LatentMoe => {
            k.extend([
                format!("{lp}.block_sparse_moe.gate.e_score_correction_bias"),
                format!("{lp}.block_sparse_moe.gate.weight"),
                format!("{lp}.block_sparse_moe.routed_expert_down_proj.weight"),
                format!("{lp}.block_sparse_moe.routed_expert_norm.weight"),
                format!("{lp}.block_sparse_moe.routed_expert_up_proj.weight"),
                format!("{lp}.block_sparse_moe.shared_experts.down_proj.weight"),
                format!("{lp}.block_sparse_moe.shared_experts.gate_proj.weight"),
                format!("{lp}.block_sparse_moe.shared_experts.up_proj.weight"),
            ]);
            for e in 0..n_experts {
                k.extend([
                    format!("{lp}.block_sparse_moe.experts.{e}.w1.weight"),
                    format!("{lp}.block_sparse_moe.experts.{e}.w2.weight"),
                    format!("{lp}.block_sparse_moe.experts.{e}.w3.weight"),
                ]);
            }
        }
    }
    k
}
