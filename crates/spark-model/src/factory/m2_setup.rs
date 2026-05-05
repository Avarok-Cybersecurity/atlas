// SPDX-License-Identifier: AGPL-3.0-only

//! MiniMax M2 layout-mode memory audit (extracted from `build_model`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::layer::TransformerLayer;

/// Pre-flight memory audit + MoE-transpose pass for MiniMax-M2 unified /
/// hybrid layout modes.
pub(super) fn maybe_run_minimax_m2_moe_transpose(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
) -> Result<()> {
    if config.model_type != "minimax_m2" {
        return Ok(());
    }
    let unified_layout = std::env::var("ATLAS_UNIFIED_MOE_LAYOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let hybrid_layout = std::env::var("ATLAS_HYBRID_MOE_LAYOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let local_experts: usize = (0..config.num_experts)
        .filter(|e| config.is_local_expert(*e))
        .count();
    // Per expert weight = packed (n*k/2 bytes) + scale (n*k/16 bytes) =
    // n*k * 9/16 bytes, where n*k = inter*hidden.
    let per_expert_one: usize = config.moe_intermediate_size * config.hidden_size * 9 / 16;
    let cost_full: usize = local_experts * 3 * per_expert_one * config.num_hidden_layers;
    let cost_gate_up: usize = local_experts * 2 * per_expert_one * config.num_hidden_layers;
    let safety: usize = 2 * 1024 * 1024 * 1024; // 2 GB
    let free = gpu.free_memory()?;
    let gb = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    // Hybrid mode pre-flight: Block C Path 2 needs ~2× the cost_full
    // budget — keeps originals AND builds transposed copies. Fall back
    // to unified if it doesn't fit (defends against KV-cache-heavy
    // configurations exceeding the 122 GB GB10 budget).
    let hybrid_fits = hybrid_layout && free >= 2 * cost_full + safety;
    if hybrid_layout && !hybrid_fits {
        tracing::warn!(
            "MoE transpose pass (hybrid layout): ATLAS_HYBRID_MOE_LAYOUT=1 \
             requested but doesn't fit (need {:.1} GB, free {:.1} GB) — \
             falling back to unified-layout (decode regression).",
            gb(2 * cost_full + safety),
            gb(free),
        );
    }
    if hybrid_fits {
        // Block C Path 2 hybrid-layout: keep originals + add transposed.
        // Decode + MTP verify route through originals (warp-reduction
        // 35 tok/s preserved); prefill (forward_batched) routes through
        // transposed (Phase 8a coalesced TTFT win retained).
        tracing::info!(
            "MoE transpose pass (hybrid layout): ATLAS_HYBRID_MOE_LAYOUT=1, \
             dual-layout (cost {:.1} GB), free pre-pass {:.1} GB → RUNNING",
            gb(2 * cost_full),
            gb(free),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_for_prefill_hybrid(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass (hybrid layout): done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if unified_layout {
        // Phase 8a unified-layout: phased transpose with originals
        // freed between phases. Fits in tight budgets that the
        // straight `transpose_moe_for_prefill` would reject.
        tracing::info!(
            "MoE transpose pass (unified layout): ATLAS_UNIFIED_MOE_LAYOUT=1, \
             phased transpose with frees, free pre-pass {:.1} GB → RUNNING",
            gb(free),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_for_prefill_unified(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass (unified layout): done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if free >= cost_full + safety {
        tracing::info!(
            "MoE transpose pass: cost {:.1} GB (full gate+up+down), \
             free {:.1} GB → RUNNING",
            gb(cost_full),
            gb(free),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_for_prefill(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass: done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if free >= cost_gate_up + safety {
        tracing::info!(
            "MoE transpose pass: full cost {:.1} GB > free {:.1} GB; \
             falling back to gate+up only (cost {:.1} GB) → RUNNING",
            gb(cost_full),
            gb(free),
            gb(cost_gate_up),
        );
        for layer in layers.iter_mut() {
            layer.transpose_moe_gate_up_for_prefill(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass: gate+up done, {:.1} GB free \
             (down: per-prefill scratch transpose)",
            gb(gpu.free_memory()?),
        );

        // ── Lazy down_proj transpose scratch ──
        //
        // The persistent transpose pass couldn't fit down (would have
        // needed another `cost_full - cost_gate_up` GB). Instead we
        // allocate a SINGLE shared scratch sized for ONE layer's down
        // weights, plus per-expert pointer tables that select into the
        // scratch. Every MoE layer reuses the same scratch — at the
        // start of each layer's prefill, a batched transpose kernel
        // populates the scratch from the layer's untransposed source
        // (`down_ptrs`). Decode unaffected — it still reads
        // `down_ptrs` directly.
        //
        // Memory: one layer's down packed = num_experts × N × K/2;
        // scale = num_experts × N × K/16. For MiniMax M2.7 NVFP4 EP=2
        // (128 local experts × 3072 × 1024/2 packed + same shape for
        // scale × 1/8 group): ~200 MB packed + ~24 MB scale ≈ 230 MB.
        // Fits comfortably in the 7 GB free post-gate+up.
        let n_per_expert_packed: usize = config.hidden_size * config.moe_intermediate_size / 2;
        let n_per_expert_scale: usize = config.hidden_size * config.moe_intermediate_size / 16;
        let local_experts_count: usize = (0..config.num_experts)
            .filter(|e| config.is_local_expert(*e))
            .count();
        let scratch_packed_bytes = local_experts_count * n_per_expert_packed;
        let scratch_scale_bytes = local_experts_count * n_per_expert_scale;
        let scratch_packed = gpu.alloc(scratch_packed_bytes)?;
        let scratch_scale = gpu.alloc(scratch_scale_bytes)?;

        // Build per-expert ptr tables pointing into the shared scratch.
        // Local experts get real offsets; remote experts get NULL so
        // the prefill GEMM kernel skips them (existing behaviour).
        let mut packed_ptrs_host = Vec::<u8>::with_capacity(config.num_experts * 8);
        let mut scale_ptrs_host = Vec::<u8>::with_capacity(config.num_experts * 8);
        let mut local_idx = 0usize;
        for e in 0..config.num_experts {
            let (p_ptr, s_ptr) = if config.is_local_expert(e) {
                let p = scratch_packed.0 + (local_idx * n_per_expert_packed) as u64;
                let s = scratch_scale.0 + (local_idx * n_per_expert_scale) as u64;
                local_idx += 1;
                (p, s)
            } else {
                (0u64, 0u64)
            };
            packed_ptrs_host.extend_from_slice(&p_ptr.to_le_bytes());
            scale_ptrs_host.extend_from_slice(&s_ptr.to_le_bytes());
        }
        let packed_ptrs_t = gpu.alloc(config.num_experts * 8)?;
        gpu.copy_h2d(&packed_ptrs_host, packed_ptrs_t)?;
        let scale_ptrs_t = gpu.alloc(config.num_experts * 8)?;
        gpu.copy_h2d(&scale_ptrs_host, scale_ptrs_t)?;
        for layer in layers.iter_mut() {
            layer.set_moe_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        tracing::info!(
            "MoE down scratch: {:.0} MB packed + {:.0} MB scale, shared across {} layers",
            scratch_packed_bytes as f64 / (1024.0 * 1024.0),
            scratch_scale_bytes as f64 / (1024.0 * 1024.0),
            config.num_hidden_layers,
        );
    } else {
        tracing::warn!(
            "MoE transpose pass: cost {:.1} GB (full) / {:.1} GB (gate+up), \
             free {:.1} GB → SKIP (insufficient memory; prefill uses \
             uncoalesced fallback, TTFT will be ~2× slower)",
            gb(cost_full),
            gb(cost_gate_up),
            gb(free),
        );
    }
    Ok(())
}
