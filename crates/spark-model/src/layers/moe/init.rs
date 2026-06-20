// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::new constructor.

use super::*;

impl MoeLayer {
    pub fn new(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_hash(weights, num_experts, gate_nvfp4, None, gpu, config)
    }

    /// Like [`MoeLayer::new`] but with an optional DeepSeek-V4 hash-routing
    /// `tid2eid` table ([vocab_size, top_k] i64). `Some` marks this as a
    /// hash-routed layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_hash(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        tid2eid_dev: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        // Sanity-check the routing config: top-k that exceeds the
        // expert count would index OOB in the topk kernel and produce
        // silent NaN routing. Catch the misconfiguration at load time.
        anyhow::ensure!(
            config.num_experts_per_tok <= num_experts && num_experts > 0,
            "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
            config.num_experts_per_tok,
            num_experts,
        );
        let gate_ptrs = build_ptr_table(&weights.experts, |e| &e.gate_proj, gpu)?;
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;

        // Extract the optional correction-bias device pointer before the
        // struct literal below moves `weights`. `.map(|dw| dw.weight)` turns
        // an `Option<DenseWeight>` into an `Option<DevicePtr>` for the
        // `moe_topk_sigmoid` kernel's bias arg.
        let weights_correction_bias: Option<DevicePtr> =
            weights.correction_bias.map(|dw| dw.weight);

        let _ = num_experts;
        let rms_norm_k = gpu.kernel("norm", "rms_norm")?;
        Ok(Self {
            weights,
            gate_nvfp4,
            pre_expert_norm: None,
            pre_expert_norm_k: rms_norm_k,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            // FP32 gate path (ATLAS_FP32_GATE) — optional; KernelHandle(0) if the
            // target's kernel set predates these symbols, dispatch then stays BF16.
            dense_gemm_f32out: super::super::try_kernel(gpu, "gemm", "dense_gemm_bf16_f32out"),
            dense_gemm_f32in: super::super::try_kernel(gpu, "gemm", "dense_gemm_f32in_f32out"),
            moe_topk_f32: super::super::try_kernel(gpu, "moe_topk", "moe_topk_softmax_f32"),
            moe_expert_gate_up_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")?,
            moe_expert_silu_down_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")?,
            moe_topk: gpu.kernel("moe_topk", "moe_topk_softmax")?,
            moe_weighted_sum_blend: gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            moe_topk_batched: gpu.kernel("moe_topk", "moe_topk_softmax_batched")?,
            moe_expert_gate_up_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_gate_up_shared_batch2")?,
            moe_expert_silu_down_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_silu_down_shared_batch2")?,
            moe_weighted_sum_blend_batch2: gpu
                .kernel("moe_fused_batch2", "moe_weighted_sum_blend_batch2")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            moe_expert_gate_up_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_gate_up_shared_batch3")?,
            moe_expert_silu_down_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_silu_down_shared_batch3")?,
            moe_weighted_sum_blend_batch3: gpu
                .kernel("moe_fused_batch3", "moe_weighted_sum_blend_batch3")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            moe_sort_by_expert: gpu.kernel("moe", "moe_sort_by_expert")?,
            moe_sorted_gate_up: gpu.kernel("moe_sorted", "moe_sorted_gate_up")?,
            moe_sorted_silu_down: gpu.kernel("moe_sorted", "moe_sorted_silu_down")?,
            moe_grouped_gemm: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?,
            moe_grouped_gemm_t: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t")?,
            moe_grouped_gemm_t_k64: gpu
                .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t_k64")?,
            moe_fused_gate_up_t: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t")?,
            moe_fused_gate_up_t_k64: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t_k64")?,
            // M=128 variant only present in models where Block D #3 has
            // been ported (currently minimax-m2-229b). Other models keep
            // KernelHandle(0) and dispatch falls through to M=64.
            moe_fused_gate_up_t_k64_m128: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_m128",
            ),
            moe_fp8_grouped_gemm_t: gpu.kernel("moe_w4a16", "moe_fp8_grouped_gemm_ptrtable_t")?,
            // THE routed-expert FP8 prefill kernel: grid-compaction (persistent
            // 96-CTA grid over a compacted work-list). Handle may be 0 on older
            // images that don't ship it.
            moe_fp8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_fp8_grouped_gemm",
                "moe_fp8_grouped_gemm",
            ),
            // Work-list builder (module "moe" = moe_permute.cu). Launched on the
            // SAME stream as the grouped GEMM (read-after-write of total_tiles).
            moe_build_tile_worklist_k: super::super::try_kernel(
                gpu,
                "moe",
                "moe_build_tile_worklist",
            ),
            moe_w8a8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_w8a8_grouped_gemm",
                "moe_w8a8_grouped_gemm",
            ),
            per_token_group_quant_fp8_k: super::super::try_kernel(
                gpu,
                "per_token_group_quant_fp8",
                "per_token_group_quant_fp8",
            ),
            fp8_gemm_t_blockscaled_k: super::super::try_kernel(
                gpu,
                "fp8_gemm_t_blockscaled",
                "fp8_gemm_t_blockscaled",
            ),
            moe_bf16_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_bf16_grouped_gemm",
                "moe_bf16_grouped_gemm",
            ),
            moe_expert_gate_up_shared_bf16_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16",
                "moe_expert_gate_up_shared_bf16",
            ),
            moe_expert_silu_down_shared_bf16_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16",
                "moe_expert_silu_down_shared_bf16",
            ),
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            moe_gate_topk_fused_k: super::super::try_kernel(
                gpu,
                "moe_gate_topk",
                "moe_gate_topk_fused",
            ),
            w4a16_gemm_t: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            bf16_to_fp8_k: gpu.kernel("w4a16", "bf16_to_fp8")?,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            moe_silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            moe_act_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?, // default: SiLU
            gelu_activation: false,
            moe_unpermute_reduce: gpu.kernel("moe", "moe_unpermute_reduce_indexed")?,
            moe_batched_blend: gpu.kernel("moe", "moe_batched_blend")?,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            gate_ptrs_t: None,
            up_ptrs_t: None,
            down_ptrs_t: None,
            down_t_scratch_packed: None,
            down_t_scratch_scale: None,
            moe_transpose_u8_batched_k: gpu
                .kernel("moe_transpose_batched", "moe_transpose_u8_batched")?,
            // ── Phase 8a transposed-layout decode kernels ──
            // Module name = file stem (default convention in atlas-kernels).
            moe_expert_gate_up_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_gate_up_shared_t")?,
            moe_expert_silu_down_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_silu_down_shared_t")?,
            // sqrtsoftplus kernels: lazy-loaded via try_kernel so models that
            // don't register them (all except DeepSeek-V4) start fine.
            moe_topk_sqrtsoftplus_k: super::super::try_kernel(
                gpu,
                "moe_topk_sqrt",
                "moe_topk_sqrtsoftplus",
            ),
            moe_topk_sqrtsoftplus_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sqrt",
                "moe_topk_sqrtsoftplus_batched",
            ),
            // Hash routing (DeepSeek-V4 hash_moe layers): lazy-loaded so other
            // models start fine. `tid2eid_dev` is the per-layer table (Some
            // only for hash layers).
            moe_hash_route_k: super::super::try_kernel(gpu, "moe_hash_route", "moe_hash_route"),
            moe_hash_route_batched_k: super::super::try_kernel(
                gpu,
                "moe_hash_route",
                "moe_hash_route_batched",
            ),
            tid2eid_dev,
            moe_expert_gate_up_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_gate_up_shared_batch2_t",
            )?,
            moe_expert_silu_down_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_silu_down_shared_batch2_t",
            )?,
            moe_expert_gate_up_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_gate_up_shared_batch3_t",
            )?,
            moe_expert_silu_down_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_silu_down_shared_batch3_t",
            )?,
            moe_expert_gate_up_shared_fp8_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_t",
                "moe_expert_gate_up_shared_fp8_t",
            )?,
            moe_expert_silu_down_shared_fp8_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_t",
                "moe_expert_silu_down_shared_fp8_t",
            )?,
            moe_expert_gate_up_shared_fp8_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2_t",
                "moe_expert_gate_up_shared_fp8_batch2_t",
            )?,
            moe_expert_silu_down_shared_fp8_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2_t",
                "moe_expert_silu_down_shared_fp8_batch2_t",
            )?,
            moe_expert_gate_up_shared_fp8_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3_t",
                "moe_expert_gate_up_shared_fp8_batch3_t",
            )?,
            moe_expert_silu_down_shared_fp8_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3_t",
                "moe_expert_silu_down_shared_fp8_batch3_t",
            )?,
            unified_layout: std::env::var("ATLAS_UNIFIED_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            hybrid_layout: std::env::var("ATLAS_HYBRID_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_gate_up_m128: std::env::var("ATLAS_NVFP4_GATE_UP_M128")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            shared_gate_t: None,
            shared_up_t: None,
            shared_down_t: None,
            gate_fp8: None,
            shared_gate_fp8: None,
            shared_up_fp8: None,
            shared_down_fp8: None,
            prefill_stream: gpu.create_stream()?,
            event_a: gpu.create_event()?,
            event_b: gpu.create_event()?,
            moe_expert_gate_up_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_gate_up_shared_fp8",
            )?,
            moe_expert_silu_down_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_silu_down_shared_fp8",
            )?,
            // FP8 batch2/3 kernels for MTP verify
            moe_expert_gate_up_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_gate_up_shared_fp8_batch2",
            )?,
            moe_expert_silu_down_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_silu_down_shared_fp8_batch2",
            )?,
            moe_weighted_sum_blend_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_weighted_sum_blend_fp8_batch2",
            )?,
            moe_expert_gate_up_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_gate_up_shared_fp8_batch3",
            )?,
            moe_expert_silu_down_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_silu_down_shared_fp8_batch3",
            )?,
            moe_weighted_sum_blend_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_weighted_sum_blend_fp8_batch3",
            )?,
            fp8_gate_weight_ptrs: None,
            fp8_up_weight_ptrs: None,
            fp8_down_weight_ptrs: None,
            bf16_gate_weight_ptrs: None,
            bf16_up_weight_ptrs: None,
            bf16_down_weight_ptrs: None,
            bf16_shared_gate: None,
            bf16_shared_up: None,
            bf16_shared_down: None,
            fp8_shared_expert: None,
            // Phase 2.7 Tier C — set by loader after construction (qwen35.rs).
            is_dflash_capture_layer: false,
            correction_bias_dev: weights_correction_bias,
            // `moe_topk_sig` is only registered for sigmoid-gated MoE models
            // (MiniMax-M2, Nemotron-Nano, Nemotron-Super). Softmax-gated MoEs
            // (Qwen3.5, Qwen3-Next, Gemma-4, Mistral) never hit the sigmoid
            // dispatch path, so a missing kernel is fine — fail at call time
            // via the KernelHandle(0) check in ops::moe_topk_sigmoid rather
            // than at MoeLayer::new(), which would otherwise block all
            // softmax-MoE model startup (observed on Qwen3.5-35B-A3B-FP8 in
            // alpha-2.43: "Module 'moe_topk_sig' not loaded" during model
            // build).
            moe_topk_sigmoid_k: super::super::try_kernel(gpu, "moe_topk_sig", "moe_topk_sigmoid"),
            moe_topk_sigmoid_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sig",
                "moe_topk_sigmoid_batched",
            ),
        })
    }
}
