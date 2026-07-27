// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::HcHeadWeights;
use crate::weight_map::{DenseWeight, dense_auto};

pub fn load_all_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let n = config.num_hidden_layers;
    tracing::info!(
        "DeepSeek-V4 load_layers: num_layers={}, hc_mult={}, hc_sinkhorn_iters={}, hc_eps={}",
        n,
        config.hc_mult,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    );
    tracing::info!(
        "DeepSeek-V4 architecture: hidden_size={}, num_experts={}, q_lora_rank={}, kv_lora_rank={}, o_lora_rank={}, head_dim={}",
        config.hidden_size,
        config.num_experts,
        config.q_lora_rank,
        config.kv_lora_rank,
        config.o_lora_rank,
        config.head_dim,
    );
    tracing::info!(
        "DeepSeek-V4 attention: q_heads={}, kv_heads={}, qk_rope_head_dim={}, qk_nope_head_dim={}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.qk_rope_head_dim,
        config.qk_nope_head_dim,
    );
    tracing::info!(
        "DeepSeek-V4 KV cache dtype: {:?}",
        layer_kv_dtypes
            .first()
            .copied()
            .unwrap_or(KvCacheDtype::Bf16),
    );

    let mut layers = Vec::with_capacity(n);
    let mut yarn_inv_freq = DevicePtr::NULL;

    // Load model-level HC head weights once (replicated to every layer).
    let hc_head = if config.hc_mult > 0 {
        let hc = config.hc_mult;
        let hc_dim = hc * config.hidden_size;
        let head_fn = super::assemble::load_hc_f32(
            store,
            &["hc_head_fn".to_string(), "model.hc_head.fn".to_string()],
            hc * hc_dim,
            gpu,
        )
        .ok();
        let head_base = super::assemble::load_hc_f32(
            store,
            &["hc_head_base".to_string(), "model.hc_head.base".to_string()],
            hc,
            gpu,
        )
        .ok();
        let head_scale = super::assemble::load_hc_f32(
            store,
            &[
                "hc_head_scale".to_string(),
                "model.hc_head.scale".to_string(),
            ],
            1,
            gpu,
        )
        .ok();
        match (head_fn, head_base, head_scale) {
            (Some(fn_ptr), Some(base_ptr), Some(scale_ptr)) => Some(HcHeadWeights {
                hc_fn: fn_ptr,
                hc_base: base_ptr,
                hc_scale: scale_ptr,
            }),
            (fn_ok, base_ok, scale_ok) => {
                anyhow::bail!(
                    "DeepSeek-V4: hc_head weights missing (fn={} base={} scale={}); \
                     tried hc_head_*, model.hc_head.*, head_hc.*",
                    if fn_ok.is_some() { "ok" } else { "MISSING" },
                    if base_ok.is_some() { "ok" } else { "MISSING" },
                    if scale_ok.is_some() { "ok" } else { "MISSING" },
                );
            }
        }
    } else {
        None
    };

    for i in 0..n {
        // RedHatAI re-quant uses flattened naming: layers.N.* instead of model.layers.N.*
        let lp = format!("layers.{i}");
        let ap = format!("{lp}.attn");

        let input_norm = dense_auto(store, &format!("{lp}.attn_norm.weight"), gpu)?;
        let post_attn_norm = dense_auto(store, &format!("{lp}.ffn_norm.weight"), gpu)?;

        // DeepSeek-V4-Flash attention weights are FP8 block-quantized in the
        // checkpoint (config quant group_0: float-quantized, block [128,128]);
        // the HF reference runs them through fp8_gemm, NOT NVFP4. Re-quantizing
        // them to NVFP4 here was both architecturally wrong and the source of an
        // out-of-bounds crash (wkv quantized as kv_lora+rope=576 rows and wo_b as
        // n_heads*head_dim=32768 cols, neither matching the real [512,4096] /
        // [4096,8192] buffers). Load as BF16 dense; the MLA decode/prefill paths
        // already fall back to dense_gemv/dense_gemm when the nvfp4 view is None.
        let wq_a = dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?;
        let wq_b = dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?;
        let q_a_norm = dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?;

        let wkv_a = dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?;
        let kv_a_norm = dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?;

        let is_v4_flash = config.o_lora_rank > 0;
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };

        // V4-Flash uses direct KV projection + grouped low-rank O projection (wo_a → wo_b).
        // V3-style absorption tensors (w_uk_t, w_uv, w_qk_absorbed, etc.) are not needed.
        let (
            wo_a,
            wo_b,
            wkv_b,
            w_uk_t,
            w_uv,
            wq_b_rope,
            w_qk_absorbed,
            w_uk_block_diag,
            w_uv_block_diag,
        ) = if is_v4_flash {
            let wo_a_w = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
            let wo_b_w = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;
            (wo_a_w, wo_b_w, null, null, null, null, null, null, null)
        } else {
            // V3 fallback: wo_a is kv_b_proj, wo_b is o_proj
            let wkv_b_w = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
            let wkv_b_shape = store.get(&format!("{ap}.wo_a.weight"))?.shape.clone();
            let o_dense = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;
            let wq_b_shape = store.get(&format!("{ap}.wq_b.weight"))?.shape.clone();
            let (w_uk_t, w_uv, wq_b_rope, w_uk_host) = super::compute::build_per_head_views(
                &wkv_b_w,
                &wkv_b_shape,
                &wq_b,
                &wq_b_shape,
                config,
                gpu,
            )?;
            let w_qk_absorbed =
                super::compute::build_w_qk_absorbed(&wq_b, &wq_b_shape, &w_uk_t, config, gpu)?;
            let (w_uk_block_diag, w_uv_block_diag) =
                super::compute::build_block_diagonals(&w_uk_host, &w_uv, config, gpu)?;
            (
                null,
                o_dense,
                wkv_b_w,
                w_uk_t,
                w_uv,
                wq_b_rope,
                w_qk_absorbed,
                w_uk_block_diag,
                w_uv_block_diag,
            )
        };
        yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

        let layer = super::assemble::assemble_layer(
            i,
            &lp,
            false, // force_all_experts — main layers use EP sharding
            input_norm,
            post_attn_norm,
            wq_a,
            None, // wq_a_nvfp4 — V4 attn is FP8/BF16, not NVFP4 (see above)
            wq_b,
            None, // wq_b_nvfp4
            q_a_norm,
            wkv_a,
            None, // wkv_a_nvfp4
            wkv_b,
            kv_a_norm,
            wo_b,
            None, // o_nvfp4
            w_uk_t,
            w_uv,
            wq_b_rope,
            w_qk_absorbed,
            w_uk_block_diag,
            w_uv_block_diag,
            yarn_inv_freq,
            wo_a,
            hc_head.clone(),
            store,
            config,
            gpu,
            layer_kv_dtypes,
        )?;
        layers.push(layer);
    }
    Ok(layers)
}
