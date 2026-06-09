// SPDX-License-Identifier: AGPL-3.0-only

//! LM-head quantization setup — extracted from `build.rs` (≤500 LoC cap).
//!
//! Selects the LM-head representation (`--lm-head-dtype`): pre-packed or
//! runtime-quantized NVFP4, runtime FP8 (w8a16), or BF16 skip — plus the
//! draft-only NVFP4 head the MTP proposer needs when the main head stays
//! BF16. Returns `(lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4)`.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{Fp8DenseWeight, quantize_to_fp8, quantize_to_nvfp4};

#[allow(clippy::type_complexity)]
pub(super) fn setup_lm_heads(
    store: &WeightStore,
    lm_head: &crate::weight_map::DenseWeight,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    use_speculative: bool,
    have_mtp_weights: bool,
) -> Result<(
    Option<crate::weight_map::QuantizedWeight>,
    Option<Fp8DenseWeight>,
    Option<crate::weight_map::QuantizedWeight>,
)> {
    // ── Step 3: Quantize LM head to NVFP4 for fast decode ──
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    // nvidia/Qwen3.6-35B-A3B-NVFP4 (algo=MIXED_PRECISION) ships an already
    // NVFP4-packed lm_head (U8 `weight` + `weight_scale` + `weight_scale_2`).
    // `load_lm_head` dense-loads those packed bytes, and re-quantizing them as
    // if they were BF16 reads 2x the buffer and faults
    // (CUDA_ERROR_ILLEGAL_ADDRESS, issue #107). Detect the packed lm_head and
    // load it directly as NVFP4 instead of dequant->requantize.
    let lm_head_key = [
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ]
    .into_iter()
    .find(|k| store.contains(k));
    let lm_head_prepacked_nvfp4 = lm_head_key
        .and_then(|k| store.get(k).ok())
        .is_some_and(|w| w.dtype == spark_runtime::weights::WeightDtype::UInt8);

    // FP8 lm_head signal (`--lm-head-dtype fp8`): when we are NOT skipping
    // quantization, route the runtime LM-head quantization to FP8 (E4M3,
    // per-row scales, w8a16_gemv decode) instead of NVFP4. Additive: when
    // `config.lm_head_fp8` is false the NVFP4/BF16 paths below are unchanged.
    let mut lm_head_fp8: Option<Fp8DenseWeight> = None;
    let lm_head_nvfp4 = if config.skip_lm_head_quantization() {
        tracing::info!("LM head kept as BF16 (skip NVFP4 quantization per model config)");
        None
    } else if config.lm_head_fp8 {
        // Runtime FP8 head. `quantize_bf16_to_fp8` (module `gemv_fp8w`) writes
        // FP8 E4M3 bytes + per-row f32 scales, consumed by `w8a16_gemv` at
        // decode. The NVFP4 head stays `None` on this path.
        let quantize_fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
        let q = quantize_to_fp8(
            lm_head,
            config.vocab_size,
            config.hidden_size,
            gpu,
            quantize_fp8_k,
            stream,
        )?;
        tracing::info!(
            "LM head quantized to FP8 (w8a16, vocab={})",
            config.vocab_size
        );
        lm_head_fp8 = Some(q);
        None
    } else if lm_head_prepacked_nvfp4 {
        let prefix = lm_head_key.unwrap().strip_suffix(".weight").unwrap();
        let q = crate::weight_map::quantized(store, prefix, gpu)?;
        tracing::info!(
            "LM head loaded as pre-packed NVFP4 (vocab={}, skipped requantize)",
            config.vocab_size
        );
        Some(q)
    } else {
        let q = quantize_to_nvfp4(
            lm_head,
            config.vocab_size,
            config.hidden_size,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        tracing::info!("LM head quantized to NVFP4 (vocab={})", config.vocab_size);
        Some(q)
    };

    // ── Step 3a: Separate NVFP4 draft head (BF16-main + MTP decouple) ──
    //
    // When the main LM head is kept BF16 for argmax precision
    // (`skip_lm_head_quantization()`), the MTP draft proposer still needs an
    // NVFP4 vocab projection: `MtpHead::forward_one` hard-wires the final
    // hidden→vocab projection to `w4a16_gemv` over a `QuantizedWeight`. Build
    // a SEPARATE NVFP4 copy used ONLY for drafting. This is correctness-safe
    // because every draft is VERIFIED by the main BF16 `lm_head_batched`
    // (verify_*.rs) — an approximate draft head only affects acceptance rate,
    // never an emitted/accepted token. Only built when speculative decoding is
    // actually active and the checkpoint ships an MTP head; otherwise `None`.
    //
    // When the main head is NVFP4 (`lm_head_nvfp4.is_some()`), this stays
    // `None` and the proposer falls back to the main NVFP4 head — byte-for-byte
    // unchanged from the pre-decouple behavior.
    let mtp_lm_head_nvfp4 = if lm_head_nvfp4.is_none() && use_speculative && have_mtp_weights {
        let q = quantize_to_nvfp4(
            lm_head,
            config.vocab_size,
            config.hidden_size,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        tracing::info!(
            "Draft-only NVFP4 LM head built for MTP (main head stays BF16, vocab={})",
            config.vocab_size,
        );
        Some(q)
    } else {
        None
    };
    Ok((lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4))
}
