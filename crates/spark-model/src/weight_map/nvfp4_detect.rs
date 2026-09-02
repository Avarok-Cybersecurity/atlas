// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Detect the weight quantization variant from the weight store.
///
/// Dispatch order matches vLLM / TRT-LLM / SGLang:
///   1. **Config-declared scheme** (`config.quantization_config.quant_method`)
///      wins outright. This is the authoritative signal and the only one
///      that correctly handles checkpoints with an `ignore` list (e.g.
///      `lukealonso/MiniMax-M2.7-NVFP4`, whose MLP `gate_proj` is
///      intentionally unquantized and therefore has no `.weight_scale`
///      tensor — sniffing would mis-detect the whole checkpoint as
///      `Bf16Raw` and then read uint8-packed FP4 as BF16, which is the
///      4× byte overrun that surfaces as `CUDA_ERROR_ILLEGAL_ADDRESS`
///      ten seconds into load).
///   2. **Tensor-name sniffing** for the many checkpoints in the wild
///      that ship without a `quantization_config` block.
pub fn detect_nvfp4_variant(
    store: &WeightStore,
    config: &atlas_core::config::ModelConfig,
) -> Nvfp4Variant {
    // (1) Config-first dispatch. See module docs on `quant_format` for
    // the full rationale — this is the fix for the Discord 2026-04-17
    // `CUDA_ERROR_ILLEGAL_ADDRESS` bug.
    if let Some(qc) = &config.quantization_config {
        match qc.quant_method.as_str() {
            "modelopt" if qc.quant_algo.eq_ignore_ascii_case("NVFP4") => {
                return Nvfp4Variant::Standard;
            }
            "modelopt" if qc.quant_algo.eq_ignore_ascii_case("FP8") => {
                return Nvfp4Variant::Fp8Dequanted;
            }
            "compressed-tensors" => {
                // `format` is the sub-selector here. Block-scaled FP8 is tagged
                // either with a literal "fp8" OR with compressed-tensors'
                // `"float-quantized"` (8-bit float = FP8 E4M3, e.g.
                // Hcompany/Holo-3.1-*-FP8); the rest ("nvfp4-pack-quantized",
                // "pack-quantized") are NVFP4.
                let fmt = qc.format.to_ascii_lowercase();
                if fmt.contains("fp8") || fmt.contains("float-quant") {
                    return Nvfp4Variant::Fp8Dequanted;
                }
                return Nvfp4Variant::CompressedTensors;
            }
            "fp8" => {
                return Nvfp4Variant::Fp8Dequanted;
            }
            _ => {
                // Unknown method with non-empty ignore list — fall
                // through to heuristic detection. A warning was already
                // emitted by `quant_format::detect_quant_format`.
            }
        }
    }

    let lp = config.layer_prefix(0);

    // Check MoE expert key first (most models are MoE).
    let local_expert = config.local_expert_range().0;
    let moe_sehyo_key = format!("{lp}.mlp.experts.{local_expert}.gate_proj.weight_packed");
    if store.contains(&moe_sehyo_key) {
        return Nvfp4Variant::CompressedTensors;
    }

    // Check dense FFN key (non-MoE models like Qwen3.5-27B).
    let dense_sehyo_key = format!("{lp}.mlp.gate_proj.weight_packed");
    if store.contains(&dense_sehyo_key) {
        return Nvfp4Variant::CompressedTensors;
    }

    // Mistral uses "layers.{i}.experts.{e}.w1" naming (no "model." prefix, no ".mlp.").
    let mistral_key = format!("layers.0.experts.{local_expert}.w1.weight_packed");
    if store.contains(&mistral_key) {
        return Nvfp4Variant::CompressedTensors;
    }

    // Fallback: scan any tensor name for `.weight_packed` suffix.
    // Catches compressed-tensors checkpoints with unexpected naming conventions.
    if store.names().any(|k| k.ends_with(".weight_packed")) {
        return Nvfp4Variant::CompressedTensors;
    }

    // Check for FP8 block-scaled weights (e.g. Qwen/Qwen3.5-35B-A3B-FP8):
    // FP8 models have `weight_scale_inv` alongside FP8E4M3 weights.
    //
    // Two SPELLINGS of the same layer, not two layers. The second entry used to
    // be indexed by `local_expert_range().0` — an EXPERT index used as a LAYER
    // index. `local_expert_range` returns global expert ids (see its sibling
    // `is_local_expert`), so on a single node or EP rank 0 it is 0 and layer 0
    // is probed by accident; on rank >= 1 it is `ep_rank * num_experts /
    // ep_world_size` and each rank probes a DIFFERENT layer. With 256 experts
    // over 2 ranks, rank 1 probed layer 39 — a full-attention layer whose FP8
    // `q_proj` trips the attention sniff below — so two ranks loading one
    // checkpoint could disagree about its quantisation variant.
    //
    // Detection must not depend on EP rank: every rank sees the same file.
    const ALT_LAYER0_PREFIX: &str = "model.language_model.layers.0";
    let prefixes_to_check = [lp.clone(), ALT_LAYER0_PREFIX.to_string()];
    for pfx in &prefixes_to_check {
        let fp8_key = format!("{pfx}.mlp.experts.{local_expert}.gate_proj.weight_scale_inv");
        if store.contains(&fp8_key) {
            return Nvfp4Variant::Fp8Dequanted;
        }
        let fp8_dense_key = format!("{pfx}.mlp.gate_proj.weight_scale_inv");
        if store.contains(&fp8_dense_key) {
            return Nvfp4Variant::Fp8Dequanted;
        }
        let fp8_attn_key = format!("{pfx}.self_attn.q_proj.weight_scale_inv");
        if store.contains(&fp8_attn_key) {
            return Nvfp4Variant::Fp8Dequanted;
        }
        // compressed-tensors `float-quantized` FP8 (e.g. Hcompany/Holo-3.1-*-FP8)
        // ships block-FP8 as an FP8E4M3 `.weight` + 2D `.weight_scale` — NO
        // `.weight_packed` (that's NVFP4) and NO `.weight_scale_inv` (that's
        // DeepSeek/Qwen-native FP8). The `.weight_scale` name alias-collides
        // with compressed-tensors NVFP4, so the `.weight_scale` checks below
        // would misroute it to an NVFP4 variant. Disambiguate by the
        // unambiguous FP8E4M3 weight dtype: an FP8E4M3 projection weight is
        // always block-FP8 (Fp8Dequanted; the FP8→BF16→NVFP4 requant path in
        // `quantized_from_fp8` reads the 2D `.weight_scale`).
        for key in [
            format!("{pfx}.mlp.experts.{local_expert}.gate_proj.weight"),
            format!("{pfx}.mlp.gate_proj.weight"),
            format!("{pfx}.self_attn.q_proj.weight"),
        ] {
            if store
                .get(&key)
                .map(|w| w.dtype == WeightDtype::FP8E4M3)
                .unwrap_or(false)
            {
                return Nvfp4Variant::Fp8Dequanted;
            }
        }
    }
    // Fallback: scan any tensor name for `.weight_scale_inv` suffix.
    // Catches FP8 checkpoints where the layer prefix hasn't been resolved yet.
    if store.names().any(|k| k.ends_with(".weight_scale_inv")) {
        return Nvfp4Variant::Fp8Dequanted;
    }

    // BF16/FP16 fine-tune detection: no quantization markers at all.
    // If even `.weight_scale` is absent (i.e., not a Standard NVFP4 model
    // either), fall through to runtime quantization from raw BF16/FP16.
    // Catches third-party fine-tunes like samuelcardillo/Carnice-MoE-35B-A3B
    // that ship only `.weight` tensors with no per-channel scales.
    let any_standard_scale = store.names().any(|k| k.ends_with(".weight_scale"));
    if !any_standard_scale {
        tracing::warn!(
            "No NVFP4/FP8 quantization metadata found (no .weight_packed / .weight_scale_inv / .weight_scale). \
             Falling back to runtime BF16→NVFP4 quantization. Quality will be inferior to a calibrated NVFP4 release."
        );
        return Nvfp4Variant::Bf16Raw;
    }

    // Partial-NVFP4 guard: some upstream checkpoints (notably google/gemma-4-26B-A4B-it)
    // ship `.weight_scale` on KV-cache scale tensors but NOT on the MLP/MoE
    // projections Atlas actually consumes. If we claim Standard here the
    // loader will then fail with a cryptic `Weight '...mlp.gate_proj.weight_scale'
    // not found in store` half-way through load (logged against #bugs 2026-04-15
    // by kiiv6565). Sniff the canonical L0 MLP gate_proj — if its `.weight_scale`
    // is missing, the right answer is BF16 runtime quantization, not Standard.
    let has_mlp_scale = {
        let k_dense = format!("{lp}.mlp.gate_proj.weight_scale");
        let k_moe = format!("{lp}.mlp.experts.{local_expert}.gate_proj.weight_scale");
        store.contains(&k_dense) || store.contains(&k_moe)
    };
    if !has_mlp_scale {
        tracing::warn!(
            "Partial NVFP4 metadata: `.weight_scale` exists for some tensors (e.g. KV scales) \
             but not for MLP/MoE projections. Falling back to runtime BF16→NVFP4 quantization. \
             For best quality use a fully-quantized NVFP4 release (e.g. Sehyo/*-NVFP4)."
        );
        return Nvfp4Variant::Bf16Raw;
    }

    Nvfp4Variant::Standard
}

/// Load a quantized weight using the appropriate naming convention.
///
/// For `Fp8Dequanted`, requires `quant_ctx` (absmax_k, quantize_k, stream)
/// to runtime-quantize the dequanted BF16 to NVFP4.
pub(crate) fn quantized_auto(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<QuantizedWeight> {
    match variant {
        Nvfp4Variant::Standard => quantized(store, prefix, gpu),
        Nvfp4Variant::CompressedTensors => quantized_v2(store, prefix, gpu),
        Nvfp4Variant::Fp8Dequanted => {
            unreachable!("Fp8Dequanted must use quantized_auto_fp8 with quant context")
        }
        Nvfp4Variant::Bf16Raw => {
            unreachable!("Bf16Raw must use quantized_any with quant context")
        }
    }
}

/// Quantize context for FP8→BF16→NVFP4 runtime conversion.
#[derive(Clone, Copy)]
pub(crate) struct QuantizeCtx {
    pub absmax_k: spark_runtime::gpu::KernelHandle,
    pub quantize_k: spark_runtime::gpu::KernelHandle,
    pub stream: u64,
}

/// Per-key variant resolution for [`quantized_any`]: the checkpoint-wide
/// `variant` may not describe THIS key (mixed-precision exports, EXL3
/// materialization). Pure — reads only tensor names/dtypes from the store —
/// so the fallbacks are unit-testable without a GPU.
pub(crate) fn resolve_key_variant(
    store: &WeightStore,
    prefix: &str,
    variant: Nvfp4Variant,
    exl3_native_moe: bool,
) -> Nvfp4Variant {
    // Per-key fallback (B8 #bugs RedHatAI/Qwen3-Coder-Next-NVFP4): some
    // models that are CompressedTensors overall keep certain projections
    // (e.g. `linear_attn.out_proj`) as raw BF16 with no quantization
    // metadata. Detect that case here and runtime-quantize, instead of
    // failing the whole load with "weight_global_scale not found".
    let has_packed = store.contains(&format!("{prefix}.weight_packed"));
    let has_scale = store.contains(&format!("{prefix}.weight_scale"));
    let has_scale_inv = store.contains(&format!("{prefix}.weight_scale_inv"));
    let has_only_dense =
        !has_packed && !has_scale && !has_scale_inv && store.contains(&format!("{prefix}.weight"));

    // Per-key fallback #2 (unsloth/Qwen3.6-{27B,35B-A3B}-NVFP4, re-quantized
    // 2026-07-10): mixed-precision checkpoints that are NVFP4 for most of the
    // net but leave a tail of layers — and, in the MoE, the shared experts —
    // as FP8 E4M3 with a per-row `weight_scale` ([N,1] BF16). Those keys carry
    // no NVFP4 metadata at all (no `weight_packed`, no `weight_global_scale`,
    // no `weight_scale_2`), so the declared NVFP4 variant cannot load them and
    // the whole model dies on `weight_global_scale not found in store`.
    // Detect the FP8 layout per key and dequant→runtime-quantize instead.
    //
    // The three NVFP4 layouts are all excluded by construction, so this can
    // never steal a key that IS NVFP4:
    //   Standard (ModelOpt/nvidia) -> has `weight_scale_2`
    //   CompressedTensors (Sehyo)  -> has `weight_packed` + `weight_global_scale`
    //   this FP8 case              -> has neither, and `.weight` is FP8E4M3
    let has_fp8_dense = !has_packed
        && !store.contains(&format!("{prefix}.weight_global_scale"))
        && !store.contains(&format!("{prefix}.weight_scale_2"))
        && (has_scale || has_scale_inv)
        && store
            .get(&format!("{prefix}.weight"))
            .map(|w| w.dtype == WeightDtype::FP8E4M3)
            .unwrap_or(false);

    // Per-key fallback #3 (EXL3 native MoE, found by compute-sanitizer on the
    // qwen4_exp 2.05bpw single-node boot): the EXL3 materialize pass rewrites
    // the SHARED experts as ModelOpt-layout NVFP4 triplets (`.weight` UInt8
    // [n,k/2] + `.weight_scale` FP8 + `.weight_scale_2` FP32), but with the
    // routed experts kept packed the global detector sees no NVFP4 metadata on
    // MLP projections and declares the model Bf16Raw. The Bf16Raw arm then
    // reads the UInt8 packed buffer as BF16 [n,k] — 4x past the allocation
    // (CUDA 700 at a layout-dependent layer, or silent OOB garbage). Detect
    // the materialized triplet per key and load it as Standard. Scoped to the
    // native-MoE gate: it is the ONLY configuration that leaves triplet keys
    // inside a Bf16Raw-declared store, and gating keeps every other
    // checkpoint's detection byte-identical.
    let has_nvfp4_triplet = exl3_native_moe
        && !has_packed
        && has_scale
        && store.contains(&format!("{prefix}.weight_scale_2"))
        && store
            .get(&format!("{prefix}.weight"))
            .map(|w| w.dtype == WeightDtype::UInt8)
            .unwrap_or(false);

    if has_nvfp4_triplet && !matches!(variant, Nvfp4Variant::Standard) {
        tracing::debug!(
            "{prefix}: NVFP4 triplet key in a {variant:?} checkpoint (EXL3 \
             materialized shared expert); loading as Standard"
        );
        Nvfp4Variant::Standard
    } else if has_only_dense && !matches!(variant, Nvfp4Variant::Bf16Raw) {
        tracing::debug!("{prefix}: no quantization metadata; falling back to runtime BF16→NVFP4");
        Nvfp4Variant::Bf16Raw
    } else if has_fp8_dense
        && !matches!(variant, Nvfp4Variant::Fp8Dequanted | Nvfp4Variant::Bf16Raw)
    {
        tracing::debug!("{prefix}: FP8 key in an NVFP4 checkpoint; dequant FP8→BF16→NVFP4");
        Nvfp4Variant::Fp8Dequanted
    } else {
        variant
    }
}

/// Load a quantized weight, dispatching by variant. Handles all three on-disk formats
/// including FP8 block-scaled (requires dimensions for FP8→BF16→NVFP4 conversion).
pub(crate) fn quantized_any(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<QuantizedWeight> {
    let _t_detect = std::time::Instant::now();
    // Native-EXL3 probe FIRST (mirrors `dense_auto`): a kept-packed trellis
    // prefix has no `.weight`/`.weight_packed`, so every arm below would die
    // on a bare "not found". Name the real cause: the prefix is in the
    // native-serving set but its consumer still expects an NVFP4 weight.
    if super::exl3_native_enabled() && spark_runtime::weights::exl3::is_exl3_linear(store, prefix) {
        anyhow::bail!(
            "quantized_any: {prefix} is held as packed EXL3 trellis \
             (ATLAS_EXL3_NATIVE=1) but this consumer expects an NVFP4 weight — \
             the prefix is in the native-serving set without a routed dispatch \
             path. Remove it from `exl3_native_serves` or route the consumer \
             through ops::exl3_gemv/exl3_gemm."
        );
    }
    let effective_variant =
        resolve_key_variant(store, prefix, variant, super::exl3_native_moe_enabled());

    let _t_detect_ns = _t_detect.elapsed().as_nanos() as u64;
    match effective_variant {
        Nvfp4Variant::Standard => quantized(store, prefix, gpu),
        Nvfp4Variant::CompressedTensors => quantized_v2(store, prefix, gpu),
        Nvfp4Variant::Fp8Dequanted => quantized_from_fp8(
            store,
            prefix,
            n,
            k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        ),
        Nvfp4Variant::Bf16Raw => {
            use std::sync::atomic::{AtomicU64, Ordering};
            static T_DETECT: AtomicU64 = AtomicU64::new(0);
            static T_GET: AtomicU64 = AtomicU64::new(0);
            static T_QUANT: AtomicU64 = AtomicU64::new(0);
            static T_FREE: AtomicU64 = AtomicU64::new(0);
            static N: AtomicU64 = AtomicU64::new(0);
            T_DETECT.fetch_add(_t_detect_ns, Ordering::Relaxed);
            // Raw BF16/FP16 fine-tune: load the dense weight then runtime-quantize.
            let _t = std::time::Instant::now();
            let w = store.get(&format!("{prefix}.weight"))?;
            let bf16 = DenseWeight { weight: w.ptr };
            T_GET.fetch_add(_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let _t = std::time::Instant::now();
            let q = quantize_to_nvfp4(
                &bf16,
                n,
                k,
                gpu,
                qctx.absmax_k,
                qctx.quantize_k,
                qctx.stream,
            )?;
            T_QUANT.fetch_add(_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let _t = std::time::Instant::now();
            // Free the BF16 source: the NVFP4 buffer is a fresh allocation, so the
            // on-disk BF16 weight is now redundant. Without this a 35B BF16 MoE
            // (Bf16Raw, SEPARATE per-expert layout routed through here by #200's
            // `quantized_any`) holds BOTH the ~60GB BF16 experts AND the ~22GB
            // NVFP4 copies → ~109GB pre-KV, no room for KV. Safe + mirrors
            // `quantized_from_fp8` which frees its BF16 intermediate the same way.
            // Through the store so a pooled (arena-resident) source is skipped
            // rather than passed to cuMemFree as an interior pointer.
            store.release_ptr(gpu, w.ptr)?;
            T_FREE.fetch_add(_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let c = N.fetch_add(1, Ordering::Relaxed) + 1;
            if c.is_multiple_of(512) {
                let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1.0e6;
                tracing::info!(
                    "quantized_any(Bf16Raw) PROFILE after {c} calls (ms total): detect={:.1} \
                     store_get={:.1} quantize={:.1} free={:.1} | sum={:.1} per_call={:.3}ms",
                    ms(&T_DETECT),
                    ms(&T_GET),
                    ms(&T_QUANT),
                    ms(&T_FREE),
                    ms(&T_DETECT) + ms(&T_GET) + ms(&T_QUANT) + ms(&T_FREE),
                    (ms(&T_DETECT) + ms(&T_GET) + ms(&T_QUANT) + ms(&T_FREE)) / c as f64,
                );
            }
            Ok(q)
        }
    }
}

/// Load a quantized weight from FP8 block-scaled data: FP8→BF16→NVFP4.
///
/// `n` and `k` are the logical weight dimensions (e.g. [inter, hidden] for gate_proj).
pub(crate) fn quantized_from_fp8(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<QuantizedWeight> {
    let bf16 = dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)?;
    let result = quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)?;
    // Free the BF16 intermediate — only the NVFP4 result is needed.
    gpu.free(bf16.weight)?;
    Ok(result)
}

/// Load FP8 block-scaled weight as BF16 dense (no NVFP4 re-quantization).
///
/// Use this when the runtime NVFP4 quantization produces degenerate weights
/// (e.g., FP8 checkpoints where double-quantization degrades quality).
/// The weight stays in BF16 and uses `dense_gemv`/`dense_gemm` kernels.
#[allow(dead_code)]
pub(crate) fn dense_from_fp8(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)
}

/// Load full attention weights for Qwen3.5 (all Q/K/V/O are NVFP4 on disk).
#[allow(dead_code)]
pub(crate) fn load_attention_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<AttentionWeights> {
    let p = format!("{layer_prefix}.self_attn");
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    Ok(AttentionWeights {
        // Q/K/V are NVFP4 quantized — load packed, return as dense (the weight_packed data)
        // The weight_loader will handle creating QuantizedWeight from these
        q_proj: dense(store, &format!("{p}.q_proj.weight_packed"))?,
        k_proj: dense(store, &format!("{p}.k_proj.weight_packed"))?,
        v_proj: dense(store, &format!("{p}.v_proj.weight_packed"))?,
        o_proj: quantized_v2(store, &format!("{p}.o_proj"), gpu)?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    })
}

/// Load NVFP4 quantized projection for Qwen3.5 full attention layer.
#[allow(dead_code)]
pub(crate) fn load_quantized_proj_qwen35(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    quantized_v2(store, prefix, gpu)
}

#[cfg(test)]
#[path = "nvfp4_detect_tests.rs"]
mod tests;
