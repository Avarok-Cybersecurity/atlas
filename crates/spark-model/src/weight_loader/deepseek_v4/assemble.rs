// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::MoeLayer;
use crate::layers::qwen3_attention::{MlaWeights, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense,
    quantize_to_nvfp4, quantized_v2,
};

#[allow(clippy::too_many_arguments)]
pub fn assemble_layer(
    layer_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    wq_a: DenseWeight,
    wq_a_nvfp4: Option<QuantizedWeight>,
    wq_b: DenseWeight,
    wq_b_nvfp4: Option<QuantizedWeight>,
    q_a_norm: DenseWeight,
    wkv_a: DenseWeight,
    wkv_a_nvfp4: Option<QuantizedWeight>,
    wkv_b: DenseWeight,
    kv_a_norm: DenseWeight,
    o_dense: DenseWeight,
    o_nvfp4: Option<QuantizedWeight>,
    w_uk_t: DenseWeight,
    w_uv: DenseWeight,
    wq_b_rope: DenseWeight,
    w_qk_absorbed: DenseWeight,
    w_uk_block_diag: DenseWeight,
    w_uv_block_diag: DenseWeight,
    yarn_inv_freq: DevicePtr,
    wo_a: DenseWeight,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Box<dyn TransformerLayer>> {
    // RedHatAI re-quant uses flattened naming: layers.N.* instead of model.layers.N.*
    let lp = format!("layers.{layer_idx}");
    let h = config.hidden_size;
    let kv_dtype = layer_kv_dtypes
        .get(layer_idx)
        .copied()
        .unwrap_or(KvCacheDtype::Bf16);

    // ── MoE FFN ──
    // RedHatAI re-quant uses ffn.gate.weight and ffn.experts.E.w1/w2/w3 naming.
    let p = &lp;
    let gate = dense(store, &format!("{p}.ffn.gate.weight"))?;
    let gate_nvfp4 = Some(quantize_to_nvfp4(
        &gate,
        config.num_experts,
        config.hidden_size,
        gpu,
        gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        gpu.default_stream(),
    )?);

    let mut experts = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.ffn.experts.{e}");
            let gate_proj = quantized_v2(store, &format!("{ep}.w1"), gpu)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w1"))?;
            let up_proj = quantized_v2(store, &format!("{ep}.w3"), gpu)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w3"))?;
            let down_proj = quantized_v2(store, &format!("{ep}.w2"), gpu)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w2"))?;
            experts.push(ExpertWeight {
                gate_proj,
                up_proj,
                down_proj,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }
    // Shared expert: DeepSeek-V4 has n_shared_experts=1, always-on and UNGATED
    // (reference MoE.forward does `y += shared_experts(x)` after the routed
    // all-reduce). It is NOT EP-sharded — every rank loads the full shared
    // expert and adds it once post-all-reduce (forward_prefill handles the
    // EP-once blend; moe_batched_blend treats a NULL gate as sigmoid=1.0).
    // The weights live under `ffn.shared_experts.{w1,w2,w3}` (NVFP4), same
    // packing as the routed experts. Leaving this null caused the MoE prefill
    // to dereference null gate/up/down pointers (CUDA illegal address).
    let sep = format!("{p}.ffn.shared_experts");
    let shared_expert = ExpertWeight {
        gate_proj: quantized_v2(store, &format!("{sep}.w1"), gpu)
            .with_context(|| "DeepSeek-V4 shared expert: w1")?,
        up_proj: quantized_v2(store, &format!("{sep}.w3"), gpu)
            .with_context(|| "DeepSeek-V4 shared expert: w3")?,
        down_proj: quantized_v2(store, &format!("{sep}.w2"), gpu)
            .with_context(|| "DeepSeek-V4 shared expert: w2")?,
    };

    // ── MoE routing correction bias (noaux_tc loss-free balancing) ──
    // DeepSeek-V4 uses sigmoid scoring + per-expert correction bias for
    // top-k selection (`topk(sigmoid(logits) + bias)`). Without the bias,
    // the MoE layer falls back to softmax routing (MoeLayer::new keys the
    // sigmoid path off `correction_bias`), selecting the wrong experts and
    // producing gibberish. The kernel consumes F32 bias, so convert BF16.
    let correction_bias = load_correction_bias(store, &lp, config.num_experts, gpu)?;

    let moe_weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias,
    };
    let moe = MoeLayer::new(moe_weights, config.num_experts, gate_nvfp4, gpu, config)?;

    // ── MLA weights ──
    // RedHatAI checkpoint: wkv_a may only contain kv_lora_rank rows (no rope).
    // Try loading a separate rope tensor; if absent, allocate a zero buffer.
    let wkv_a_rope = if let Ok(rope_w) = store.get(&format!("{lp}.attn.wkv_rope.weight")) {
        DenseWeight { weight: rope_w.ptr }
    } else {
        let rope_bytes = config.qk_rope_head_dim * h * 2;
        let rope_ptr = gpu.alloc(rope_bytes)?;
        gpu.memset(rope_ptr, 0, rope_bytes)?;
        DenseWeight { weight: rope_ptr }
    };

    let mla = MlaWeights {
        wq_a,
        wq_a_nvfp4,
        wq_b,
        wq_b_nvfp4,
        q_a_norm,
        wkv_a,
        wkv_a_nvfp4,
        wkv_b,
        kv_a_norm,
        wkv_a_rope,
        wkv_a_merged: DenseWeight {
            weight: wkv_a.weight,
        },
        wo: o_dense,
        wo_nvfp4: o_nvfp4,
        wo_a,
        wo_a_nvfp4: None,
        wo_b: o_dense,
        wo_b_nvfp4: None,
        w_uk_t,
        w_uv,
        wq_b_rope,
        w_qk_absorbed,
        w_uk_block_diag,
        w_uv_block_diag,
        yarn_inv_freq,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        o_lora_rank: config.o_lora_rank,
        nope: config.qk_nope_head_dim,
        rope: config.qk_rope_head_dim,
        v_dim: config.v_head_dim,
    };

    // ── Attention dummy + layer ──
    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        o_proj: QuantizedWeight::null(),
        q_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        FfnComponent::Moe(moe),
        layer_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        0,
        config,
    )?;
    layer.set_mla_weights(mla);
    Ok(Box::new(layer))
}

/// Load the DeepSeek-V4 MoE `e_score_correction_bias` (noaux_tc loss-free
/// balancing bias) as an F32 `[num_experts]` tensor. The RedHatAI re-quant
/// flattens layer keys to `layers.N.ffn.gate.*`. Returns `None` (softmax
/// routing) only if the checkpoint genuinely lacks the tensor.
fn load_correction_bias(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<DenseWeight>> {
    use spark_runtime::weights::WeightDtype;
    let candidates = [
        format!("{layer_prefix}.ffn.gate.e_score_correction_bias"),
        format!("{layer_prefix}.ffn.gate.correction_bias"),
        format!("{layer_prefix}.mlp.gate.e_score_correction_bias"),
    ];
    let Some(bias_t) = candidates.iter().find_map(|k| store.get(k).ok()) else {
        tracing::warn!(
            "DeepSeek-V4 {layer_prefix}: no e_score_correction_bias found — \
             MoE will fall back to softmax routing (output will be degraded)"
        );
        return Ok(None);
    };
    let n = bias_t.num_elements();
    anyhow::ensure!(
        n == num_experts,
        "DeepSeek-V4 correction_bias length {n} != num_experts {num_experts}"
    );
    // The moe_topk_sigmoid kernel consumes an F32 bias. F32 tensors pass
    // through unchanged; BF16 must be widened (zero-extend mantissa).
    if bias_t.dtype == WeightDtype::BF16 {
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(bias_t.ptr, &mut bf16_buf)?;
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            f32_buf[i * 4 + 2] = bf16_buf[i * 2];
            f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
        }
        let ptr = gpu.alloc(f32_buf.len())?;
        gpu.copy_h2d(&f32_buf, ptr)?;
        Ok(Some(DenseWeight { weight: ptr }))
    } else {
        Ok(Some(DenseWeight { weight: bias_t.ptr }))
    }
}
