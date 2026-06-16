// SPDX-License-Identifier: AGPL-3.0-only

mod attention_arms;
mod fp8_attention_arms;
mod linear_attn_arms;
mod moe_arms;

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::super::{ModelWeightLoader, QuantFormat, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{
    Nvfp4Variant, dense, detect_nvfp4_variant, load_moe_qwen35, quantize_to_nvfp4,
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
        //
        // Diagnostic env: ATLAS_FORCE_NVFP4_MOE=1 forces the NVFP4 path even
        // for FP8 models — used to localize FP8 grouped-GEMM amplification
        // bug (L0 moe_out 3.3x too large vs HF). Keeps NVFP4 experts loaded
        // AND skips set_fp8_experts so forward dispatch falls through to the
        // NVFP4 path.
        let force_nvfp4_moe = std::env::var("ATLAS_FORCE_NVFP4_MOE").ok().as_deref() == Some("1");
        let skip_nvfp4_experts = native_fp8 && !force_nvfp4_moe;
        if skip_nvfp4_experts {
            tracing::info!(
                "FP8: skipping NVFP4 routed experts (FP8 fused MoE batch1/2/3 handles all dispatch)"
            );
        } else if native_fp8 && force_nvfp4_moe {
            tracing::warn!(
                "ATLAS_FORCE_NVFP4_MOE=1: routing MoE through NVFP4 path (diagnostic — slower)"
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
        // 2026-05-25 (final): gate stays in BF16 for `native_fp8` —
        // routes through `dense_gemm` BF16 fallback path.
        //
        // The MoE gate is a `[num_experts=512, h=2048]` BF16 matrix on
        // disk (explicitly `ignored_layers` in the FP8 release's
        // quantization_config). Runtime-quantizing it to NVFP4 (4-bit
        // E2M1) destroys the precision the router needs at late layers
        // where the top-8 weights cluster in `[0.105, 0.168]` — the
        // 4-bit ULP is wider than that range, so the router can't
        // distinguish them. The dense-code-output regression we see
        // on opencode multi-turn (`\n` collapsed to ` ` in tool-call
        // `content` args, `</br>` substituted for newlines, all on
        // first emission with the native FP8 SSM dispatch active)
        // is the visible symptom — the model wants to emit a
        // structure token but the post-MoE residual has drifted
        // toward a nearby-but-wrong attractor. Memory cost: 2 MB ×
        // 40 layers ≈ 80 MB. Non-FP8 variants keep the runtime
        // NVFP4 quantize (matched-shape self-compensation with
        // the on-disk NVFP4 experts).
        let gate_nvfp4 = if native_fp8 {
            None
        } else {
            Some(quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?)
        };
        let mut moe_layer =
            MoeLayer::new(moe_weights, config.num_experts, gate_nvfp4, gpu, config)?;
        // Phase 2.7 Tier C: flag DFlash capture layers so the MoE forward
        // can dispatch the Frankenstein kernel route (env-var-gated). The
        // capture-layer indices are already offset-adjusted in factory.rs
        // before being placed on `config.dflash_capture_layers`.
        moe_layer.is_dflash_capture_layer = config.dflash_capture_layers.contains(&i);
        // With native FP8, the FP8 fused MoE kernel handles both prefill and decode.
        // Skip transposition and predequant (saves ~30 GB + CPU time for 122B EP=2).
        // ATLAS_FORCE_NVFP4_MOE=1 inverts: do the prep so NVFP4 path is usable.
        if (!native_fp8 || force_nvfp4_moe) && !skip_moe_transpose {
            moe_layer.transpose_for_prefill(gpu, config)?;
        }
        if !native_fp8 || force_nvfp4_moe {
            moe_layer.predequant_for_prefill(gpu, config, stream)?;
        }

        // ATLAS_FP8_DEQUANT_MOE_TO_BF16: dequant FP8 experts to BF16 at load,
        // route MoE through the BF16 grouped GEMM + fused-decode kernels.
        // Eliminates the per-layer 0.989 FP8 cosine ceiling. Memory cost:
        // ~2× expert weights vs native FP8.
        // ATLAS_FP8_DEQUANT_LAYERS (PCND opt-in): restrict BF16 dequant to a
        // subset of absolute layer indices (e.g. "31-39" or "31,35,39"). Unset
        // → all layers (legacy behaviour). Selective late-layer BF16 targets
        // the worst-drift deep layers while keeping early layers FP8-fast,
        // cutting the ~2× MoE decode bandwidth that drives 360s harness
        // timeouts (the bit-perfect speed wall, task #231).
        let layer_sel = layer_dequant_selected(i);
        let dequant_moe_to_bf16 = native_fp8
            && std::env::var("ATLAS_FP8_DEQUANT_MOE_TO_BF16")
                .ok()
                .as_deref()
                == Some("1")
            && layer_sel;
        // Diagnostic: dequant attention Q/K/V/O FP8→BF16 at load and run them
        // through dense BF16 GEMM (isolates the FP8-attention contribution to
        // the Atlas↔vLLM cosine floor). TP=1 only.
        let dequant_attn_to_bf16 = native_fp8
            && std::env::var("ATLAS_FP8_DEQUANT_ATTN_TO_BF16")
                .ok()
                .as_deref()
                == Some("1")
            && layer_sel;

        if dequant_moe_to_bf16 {
            moe_arms::dequant_moe_experts_to_bf16(i, store, &lp, config, gpu, &mut moe_layer);
        }

        // Native FP8 MoE: load FP8 expert weights for decode
        if native_fp8 && !force_nvfp4_moe && !dequant_moe_to_bf16 {
            moe_arms::load_fp8_moe_experts(i, store, &lp, config, gpu, &mut moe_layer);
        }

        let ffn = FfnComponent::Moe(moe_layer);

        match lt {
            LayerType::FullAttention if native_fp8 && dequant_attn_to_bf16 => {
                let layer = fp8_attention_arms::build_full_attention_fp8_bf16_dequant(
                    i,
                    store,
                    &lp,
                    gpu,
                    config,
                    layer_kv_dtypes[attn_idx],
                    attn_idx,
                    input_norm,
                    post_attn_norm,
                    ffn,
                )?;
                layers.push(layer);
                attn_idx += 1;
            }
            LayerType::FullAttention if native_fp8 => {
                let layer = fp8_attention_arms::build_full_attention_fp8_native(
                    i,
                    store,
                    &lp,
                    gpu,
                    config,
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
            // LinearAttention dispatch.
            //
            // For `Fp8Dequanted` checkpoints (Qwen3.6-A3B-FP8), route
            // through the native-FP8 build that keeps decode in
            // block-scaled FP8 via `w8a16_gemv` (no 4-bit NVFP4 detour).
            // Prior to 2026-05-24 this branch was dead-coded because the
            // scale-concat in `build_linear_attention_fp8` did per-row
            // F32 byte math against a per-BLOCK BF16 buffer; that's now
            // fixed to copy block rows at the correct stride.
            // CAUSAL-PATHWAY-AUDIT Bug #1 closed.
            //
            // All other variants (NVFP4 native, BF16, etc.) keep the
            // existing NVFP4-quantized decode path.
            // LinearAttention dispatch.
            //
            // Native FP8 SSM path lit for `Fp8Dequanted` checkpoints
            // (Qwen3.6-35B-A3B-FP8). Decode runs `w8a16_gemv` with
            // block-scaled FP8 weights + `[N/BS,K/BS] BF16` scales
            // directly off disk — no BF16→NVFP4 detour. Prefill stays
            // on single-scale FP8 via `bf16_to_fp8` + `fp8_gemm_n128`.
            // See `linear_attn_arms::build_linear_attention_fp8` for
            // the byte-exact concat math (qkv + z along the N-block
            // axis at `(K/BS) * 2` bytes per scale row, BS=128). The
            // 2026-05-25 revert to the NVFP4 detour was a debugging
            // workaround — re-enabled now since downgrading hides the
            // real progress signal on the FP8 implementation.
            //
            // All non-FP8 variants (NVFP4 native, BF16, etc.) take the
            // existing NVFP4-quantized decode path.
            LayerType::LinearAttention => {
                let layer = match variant {
                    Nvfp4Variant::Fp8Dequanted => linear_attn_arms::build_linear_attention_fp8(
                        i,
                        store,
                        &lp,
                        gpu,
                        variant,
                        config,
                        h,
                        stream,
                        input_norm,
                        post_attn_norm,
                        ffn,
                    )?,
                    _ => linear_attn_arms::build_linear_attention_nvfp4(
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
                    )?,
                };
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

/// Whether absolute layer index `layer` is selected for BF16 dequant per
/// `ATLAS_FP8_DEQUANT_LAYERS` (PCND opt-in). The spec is a comma-separated
/// list of singletons and inclusive ranges, e.g. `"31-39"` or `"31,35,39"`.
/// Unset → every layer selected (legacy all-layers behaviour). Parsed once.
fn layer_dequant_selected(layer: usize) -> bool {
    use std::sync::OnceLock;
    // None  = env unset → all layers; Some(ranges) = explicit selection.
    static SPEC: OnceLock<Option<Vec<(usize, usize)>>> = OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        let s = std::env::var("ATLAS_FP8_DEQUANT_LAYERS").ok()?;
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    ranges.push((a.min(b), a.max(b)));
                }
            } else if let Ok(a) = part.parse::<usize>() {
                ranges.push((a, a));
            }
        }
        Some(ranges)
    });
    match spec {
        None => true,
        Some(ranges) => ranges.iter().any(|&(a, b)| layer >= a && layer <= b),
    }
}
