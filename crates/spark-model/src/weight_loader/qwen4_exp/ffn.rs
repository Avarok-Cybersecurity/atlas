// SPDX-License-Identifier: AGPL-3.0-only

//! The per-layer MoE block: 512 routed experts, top-10, plus a shared expert
//! and its sigmoid gate.
//!
//! This is `load_moe_qwen35` verbatim. Both the naming
//! (`mlp.gate`, `mlp.shared_expert.{gate,up,down}_proj`,
//! `mlp.shared_expert_gate`, `mlp.experts.{e}.{gate,up,down}_proj`) and the
//! on-disk quantization (standard ModelOpt NVFP4: packed E2M1 `weight`,
//! per-16 E4M3 `weight_scale`, per-tensor F32 `weight_scale_2`) are identical
//! to Qwen3.5/3.6 MoE. The expert COUNT and widths differ — 512 x 640 here
//! against 256 x 512 there — but those come from `config`, not from the
//! loader, so nothing needs re-deriving.
//!
//! The router (`mlp.gate`) is left BF16 and runtime-quantized to NVFP4 only
//! for the non-native-FP8 path, matching qwen35. Its precision matters more
//! than its size: at 512 experts the top-10 weights cluster tightly, and a
//! 4-bit ULP wider than that spread cannot tell them apart.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{Nvfp4Variant, load_moe_qwen35, quantize_to_nvfp4};

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let weights = load_moe_qwen35(
        store,
        lp,
        config.num_experts,
        gpu,
        config,
        variant,
        absmax_k,
        quantize_k,
        stream,
        false,
    )
    .with_context(|| format!("qwen4_exp: MoE block at {lp}"))?;

    let gate_nvfp4 = Some(quantize_to_nvfp4(
        &weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?);

    let mut moe = MoeLayer::new(weights, config.num_experts, gate_nvfp4, gpu, config)?;

    // CUTLASS grouped NVFP4 gate_up/down (ATLAS_HOLO_MOE_GROUPED_CUTLASS).
    // qwen4_exp serves from the checkpoint-native ORIGINAL [N,K/16] scales — it
    // never builds the transposed gate_ptrs_t/up_ptrs_t that qwen35 gates this
    // on — so it takes build_cutlass_grouped_sfb's n-major fallback, which
    // exists for exactly this layout. Without this call the CUTLASS flags are
    // INERT on this model: the dispatch site requires cutlass_grouped_host,
    // and nothing else populates it (measured: "CUTLASS SFB layers: 0" with the
    // full flag set).
    //
    // Opt-in because the SFB atoms are not free: ~100 KB per expert per
    // projection, x512 experts x3 projections x48 layers ~ 7 GB resident, which
    // comes straight out of the KV budget. Read the alloc ledger before
    // adopting it as a default.
    if std::env::var("ATLAS_HOLO_MOE_GROUPED_CUTLASS")
        .ok()
        .as_deref()
        == Some("1")
    {
        moe.build_cutlass_grouped_sfb(gpu, config, stream)?;
    }

    Ok(FfnComponent::Moe(moe))
}
