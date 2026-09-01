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

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::moe::{Exl3MoeState, build_exl3_ptr_table};
use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{Nvfp4Variant, load_moe_qwen4exp_exl3, load_moe_qwen35, quantize_to_nvfp4};

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    exl3: &mut super::exl3_dense::NativeExl3,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    // ── Native EXL3 routed experts (ATLAS_EXL3_NATIVE_MOE=1) ──
    // Re-derived PER LAYER from the store, not just the env gates: the
    // materialize pass keeps a layer's experts packed only when the whole
    // layer passed the K/cb-uniformity + envelope check (atomic per layer),
    // so "the first local expert still has its .trellis" is exactly "this
    // layer was kept". A fallen-back layer takes the NVFP4 path below with
    // zero special-casing.
    let (ep_local_start, _) = config.local_expert_range();
    let exl3_native_moe = crate::weight_map::exl3_native_enabled()
        && crate::weight_map::exl3_native_moe_enabled()
        && spark_runtime::weights::exl3::is_exl3_linear(
            store,
            &format!("{lp}.mlp.experts.{ep_local_start}.gate_proj"),
        );

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
        // Native EXL3 experts: routed experts become ExpertWeight::null()
        // (the FP8-native precedent) — the packed trellis serves them. The
        // shared expert and router still load/materialize exactly as before.
        exl3_native_moe,
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

    if exl3_native_moe {
        // The CUTLASS grouped path reads the NVFP4 expert tables, which are
        // null under native EXL3 — the combination would be silently inert
        // at best. Refuse loudly.
        ensure!(
            std::env::var("ATLAS_HOLO_MOE_GROUPED_CUTLASS").as_deref() != Ok("1"),
            "ATLAS_EXL3_NATIVE_MOE=1 is incompatible with \
             ATLAS_HOLO_MOE_GROUPED_CUTLASS=1 (no NVFP4 expert tables exist \
             to build SFB atoms from); unset one of the two"
        );
        let experts = load_moe_qwen4exp_exl3(store, lp, config.num_experts, gpu, config)
            .with_context(|| format!("qwen4_exp: native EXL3 experts at {lp}"))?;
        let gate_t = build_exl3_ptr_table(&experts.gate, gpu)?;
        let up_t = build_exl3_ptr_table(&experts.up, gpu)?;
        let down_t = build_exl3_ptr_table(&experts.down, gpu)?;
        // Fail at load, not mid-serve: probe the fp32-C mgemm instance for
        // each projection's (K, cb) — the shape-2 instance existing implies
        // the whole exl3_matmul module is compiled into this target (the
        // Exl3LmHead precedent).
        for t in [&gate_t, &up_t, &down_t] {
            gpu.kernel(
                "exl3_matmul",
                &format!("exl3_mgemm_k{}_cb{}_sh2_f32", t.k_bits, t.cb),
            )
            .context(
                "EXL3 native MoE needs the exl3_matmul kernel module (gb10 \
                 targets only) — unset ATLAS_EXL3_NATIVE_MOE on this target",
            )?;
        }
        // The prefill tier lives in a SEPARATE module (`exl3_moe`) — resolve
        // the exact instance the fused launch will select (same name rule as
        // `exl3_moe_fused`) so a missing module or its JIT compile is paid
        // here, not on the first prefill.
        {
            let ks = [gate_t.k_bits, up_t.k_bits, down_t.k_bits];
            let kname = if ks[0] == ks[1] && ks[1] == ks[2] {
                ks[0]
            } else {
                0
            };
            let inter = config.moe_intermediate_size;
            let n_tile = if h.is_multiple_of(256) && inter.is_multiple_of(256) {
                256
            } else {
                128
            };
            gpu.kernel(
                "exl3_moe",
                &format!("exl3_moe_k{kname}_n{n_tile}_cb{}", gate_t.cb),
            )
            .context(
                "EXL3 native MoE needs the exl3_moe (fused prefill) kernel \
                     module — unset ATLAS_EXL3_NATIVE_MOE on this target",
            )?;
        }
        // Over the MODEL-shared launch state (locks/fence/section) so the
        // MoE arm and the native dense arms serialize on ONE section.
        let state = Exl3MoeState::get_or_create_with_launch(
            &mut exl3.moe,
            &mut exl3.launch,
            gpu,
            h,
            config.moe_intermediate_size,
            config.num_experts_per_tok,
            config.num_experts,
        )?;
        tracing::info!(
            "{lp}: routed experts served natively from EXL3 trellis \
             ({} local experts; gate/up/down K={}/{}/{} cb={})",
            gate_t.num_local,
            gate_t.k_bits,
            up_t.k_bits,
            down_t.k_bits,
            gate_t.cb,
        );
        moe.set_exl3_experts([gate_t, up_t, down_t], state);
    }

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
