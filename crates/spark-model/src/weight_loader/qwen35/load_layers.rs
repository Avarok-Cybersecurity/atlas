// SPDX-License-Identifier: AGPL-3.0-only

mod attention_arms;
mod linear_attn_arms;

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::super::{ModelWeightLoader, QuantFormat, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_fp8_block_scaled};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Nvfp4Variant, QuantizedWeight, dense, detect_nvfp4_variant,
    load_fp8_block_scaled_as_fp8weight, load_kv_scales, load_moe_qwen35,
    load_moe_qwen35_fp8_experts, quantize_to_nvfp4,
};

pub(super) fn load_layers(
    loader: &dyn ModelWeightLoader,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let layer_types = if config.layer_types.is_empty() {
        (0..config.num_hidden_layers)
            .map(|i| config.layer_type(i))
            .collect::<Vec<_>>()
    } else {
        config.layer_types.clone()
    };

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_idx = 0usize;

    // C.3 (2026-04-25): per-(layer, role) precision schedule. The
    // default trait impl returns the empty schedule — every lookup
    // yields `Dtype::Inherit`, preserving the existing per-checkpoint
    // detection logic byte-for-byte. When MODEL.toml ships a
    // `[precision]` block AND the loader's `precision_schedule`
    // method is overridden to honour it, the schedule directs
    // each tensor's dtype here. Below we plumb the schedule
    // through and log when overrides will engage; the actual
    // dispatch sites (router, attention QKV, expert weights,
    // LM head) check `schedule.dtype_for(...)` and select their
    // load path from it.
    let precision = loader.precision_schedule(config);
    if precision.has_any_override() {
        tracing::info!(
            "Precision schedule active: {:?} — overriding per-checkpoint dtype",
            precision,
        );
    }
    // Suppress unused warning when no dispatch site consumes it
    // yet (the schedule is wired but not all call sites have been
    // converted; remaining conversions track the structured-tag
    // grammar deployment in `project_xgrammar.md`).
    let _ = precision;

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let h = config.hidden_size;

    // Detect weight format and quantization strategy.
    let variant = detect_nvfp4_variant(store, config);
    let weight_format = WeightFormat::detect(store, config);

    // Resolve runtime quantization format from the detected on-disk
    // variant. This determines which kernels are used for
    // decode/prefill/verify.
    let quant_format = if variant == Nvfp4Variant::Fp8Dequanted {
        QuantFormat::Fp8
    } else {
        QuantFormat::Nvfp4
    };
    let native_fp8 = quant_format == QuantFormat::Fp8;
    tracing::info!(
        "Weight format: {:?}, NVFP4 variant: {:?}, quant_format: {:?}",
        weight_format,
        variant,
        quant_format,
    );

    // Estimate MoE transpose memory: 3 projections × num_experts × (packed + scale) per layer.
    // Skip transposition if GPU memory is insufficient — fallback grouped GEMM is used instead.
    let skip_moe_transpose = {
        let inter = config.moe_intermediate_size;
        let group_size = 16usize;
        // gate/up: [inter, h/2] packed + [inter, h/group] scale
        let gu_bytes = inter * h / 2 + inter * h / group_size;
        // down:    [h, inter/2] packed + [h, inter/group] scale
        let d_bytes = h * inter / 2 + h * inter / group_size;
        let per_layer = config.num_experts * (2 * gu_bytes + d_bytes);
        let total = per_layer * config.num_hidden_layers;
        let available = gpu.free_memory().unwrap_or(0);
        let headroom = 2 * 1024 * 1024 * 1024; // 2 GB for KV cache + buffers
        let skip = total > available.saturating_sub(headroom);
        if skip {
            tracing::warn!(
                "Skipping MoE weight transposition ({:.1} GB needed, {:.1} GB available). \
                 Prefill will use fallback grouped GEMM.",
                total as f64 / (1024.0 * 1024.0 * 1024.0),
                available as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        skip
    };

    for (i, lt) in layer_types.iter().enumerate() {
        let lp = config.layer_prefix(i);
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

        // When native_fp8, skip NVFP4 routed experts — FP8 fused batch1/2/3
        // kernels handle all MoE dispatch including MTP verify.
        // Saves ~33 GB on 122B EP=2, enabling FP8+MTP within memory budget.
        let skip_nvfp4_experts = native_fp8;
        if skip_nvfp4_experts {
            tracing::info!(
                "FP8: skipping NVFP4 routed experts (FP8 fused MoE batch1/2/3 handles all dispatch)"
            );
        }
        let moe_weights = load_moe_qwen35(
            store,
            &lp,
            config.num_experts,
            gpu,
            config,
            variant,
            absmax_k,
            quantize_k,
            stream,
            skip_nvfp4_experts,
        )?;
        let gate_nvfp4 = quantize_to_nvfp4(
            &moe_weights.gate,
            config.num_experts,
            h,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        let mut moe_layer = MoeLayer::new(
            moe_weights,
            config.num_experts,
            Some(gate_nvfp4),
            gpu,
            config,
        )?;
        // Phase 2.7 Tier C: flag DFlash capture layers so the MoE forward
        // can dispatch the Frankenstein kernel route (env-var-gated). The
        // capture-layer indices are already offset-adjusted in factory.rs
        // before being placed on `config.dflash_capture_layers`.
        moe_layer.is_dflash_capture_layer = config.dflash_capture_layers.contains(&i);
        // With native FP8, the FP8 fused MoE kernel handles both prefill and decode.
        // Skip transposition and predequant (saves ~30 GB + CPU time for 122B EP=2).
        if !native_fp8 && !skip_moe_transpose {
            moe_layer.transpose_for_prefill(gpu, config)?;
        }
        if !native_fp8 {
            moe_layer.predequant_for_prefill(gpu, config, stream)?;
        }

        // Native FP8 MoE: load FP8 expert weights for decode
        if native_fp8
            && let Ok(fp8_experts) =
                load_moe_qwen35_fp8_experts(store, &lp, config.num_experts, gpu, config)
        {
            let sp = format!("{lp}.mlp.shared_expert");
            use crate::weight_map::{Fp8ExpertWeight as FEW, Fp8Weight as FW};
            use spark_runtime::gpu::DevicePtr;
            let null_fw = FW {
                weight: DevicePtr::NULL,
                row_scale: DevicePtr::NULL,
                n: 0,
                k: 0,
            };
            let sh_gate =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.gate_proj"), gpu);
            let sh_up = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.up_proj"), gpu);
            let sh_down =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.down_proj"), gpu);
            if sh_gate.is_err() || sh_up.is_err() || sh_down.is_err() {
                tracing::warn!(
                    "Layer {i}: shared expert FP8 load failed (gate={}, up={}, down={})",
                    sh_gate.is_ok(),
                    sh_up.is_ok(),
                    sh_down.is_ok(),
                );
            }
            let shared_fp8 = FEW {
                gate_proj: sh_gate.unwrap_or(null_fw),
                up_proj: sh_up.unwrap_or(null_fw),
                down_proj: sh_down.unwrap_or(null_fw),
            };
            if let Err(e) = moe_layer.set_fp8_experts(&fp8_experts, shared_fp8, gpu) {
                tracing::error!("Layer {i}: failed to build FP8 expert pointer tables: {e:#}");
                tracing::warn!("Layer {i}: falling back to NVFP4-only decode for MoE experts");
            } else {
                tracing::info!("Layer {i}: MoE experts loaded as native FP8");
            }
        }

        let ffn = FfnComponent::Moe(moe_layer);

        match lt {
            LayerType::FullAttention if native_fp8 => {
                // ── Native FP8 path: FP8 for both decode AND prefill ──
                // NO NVFP4 dequant — saves ~30 GB peak memory on 122B EP=2.
                // Decode uses w8a16_gemv, prefill uses w8a16_gemm (both with
                // E4M3 LUT + BF16 2D block scales from checkpoint).
                let p = format!("{lp}.self_attn");
                tracing::info!("Layer {i}: loading attention FP8 native (zero-copy)");

                // FP8 block-scaled QKVO: column-parallel Q/K/V, row-parallel O.
                // Block size is 128 for Qwen3.5 native FP8 checkpoints.
                let tp_rank = config.tp_rank;
                let tp_size = config.tp_world_size.max(1);
                let block_size = 128usize;
                let load_fp8_proj = |name: &str,
                                     _full_n: usize,
                                     _full_k: usize,
                                     kind: TpShardKind|
                 -> Result<crate::weight_map::Fp8Weight> {
                    let src =
                        load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)?;
                    if tp_size == 1 {
                        return Ok(src);
                    }
                    let sharded =
                        shard_fp8_block_scaled(&src, kind, tp_rank, tp_size, block_size, gpu)?;
                    gpu.free(src.weight)?;
                    gpu.free(src.row_scale)?;
                    Ok(sharded)
                };
                let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
                tracing::info!(
                    "Layer {i}: FP8 Q/K/V/O loaded, {:.1} GB free",
                    gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0)
                );

                // O proj needs a QuantizedWeight placeholder for the AttentionWeights struct.
                // Use a dummy — the actual O proj uses o_fp8w via w8a16_gemv/gemm.
                let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                let dummy = DenseWeight {
                    weight: spark_runtime::gpu::DevicePtr::NULL,
                };
                let dummy_qw = QuantizedWeight::null();
                let attn = AttentionWeights {
                    q_proj: dummy,
                    k_proj: dummy,
                    v_proj: dummy,
                    o_proj: dummy_qw,
                    q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                    k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                    q_norm_full: None,
                    k_norm_full: None,
                    k_scale,
                    v_scale,
                };

                let layer_kv_dtype = layer_kv_dtypes[attn_idx];
                let mut layer = Qwen3AttentionLayer::new(
                    input_norm,
                    attn,
                    post_attn_norm,
                    ffn,
                    attn_idx,
                    None,
                    None,
                    None, // No NVFP4 — w8a16_gemm handles prefill
                    gpu,
                    layer_kv_dtype,
                    config.fp8_kv_calibration_tokens,
                    config,
                )?;

                // Set checkpoint FP8 weights for decode (w8a16_gemv) and prefill fallback (w8a16_gemm).
                layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));

                // Transpose FP8 weights for fast prefill (w8a16_gemm_t: coalesced reads).
                // This allocates N*K bytes per projection but gives ~14x prefill speedup.
                if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
                    tracing::warn!(
                        "Layer {i}: FP8 transpose failed, using non-transposed prefill: {e}"
                    );
                } else {
                    tracing::info!("Layer {i}: FP8 weights transposed for fast prefill");
                }

                layers.push(Box::new(layer));
                attn_idx += 1;
            }
            LayerType::FullAttention => {
                let layer = attention_arms::build_full_attention_nvfp4(
                    i,
                    store,
                    &lp,
                    gpu,
                    variant,
                    config,
                    h,
                    absmax_k,
                    quantize_k,
                    stream,
                    layer_kv_dtypes[attn_idx],
                    attn_idx,
                    input_norm,
                    post_attn_norm,
                    ffn,
                )?;
                layers.push(layer);
                attn_idx += 1;
            }
            // LinearAttention native-FP8 path was previously wired here as
            // `if native_fp8 && false => build_linear_attention_fp8(...)`
            // but the FP8 GDN kernels are still stabilizing and the arm
            // was permanently short-circuited. The NVFP4 fallback below
            // handles every LinearAttention layer regardless of `native_fp8`.
            // To re-enable: revive the call to
            // `linear_attn_arms::build_linear_attention_fp8` in this arm
            // and gate it behind a real predicate (env-var or feature
            // flag), not a literal `false`.
            LayerType::LinearAttention => {
                let layer = linear_attn_arms::build_linear_attention_nvfp4(
                    store,
                    &lp,
                    gpu,
                    variant,
                    config,
                    h,
                    absmax_k,
                    quantize_k,
                    stream,
                    input_norm,
                    post_attn_norm,
                    ffn,
                )?;
                layers.push(layer);
            }
            LayerType::Moe => unreachable!("Qwen3.5 has no standalone MoE layers"),
        }

        if (i + 1) % 10 == 0 || i < 5 {
            let free_gb = gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!("Loaded layers 0..{} — {free_gb:.1} GB free", i + 1);
        }
    }

    tracing::info!(
        "Qwen3.5 weight loader: {} layers ({} attention, {} linear_attn)",
        layers.len(),
        attn_idx,
        layers.len() - attn_idx,
    );

    Ok(layers)
}
