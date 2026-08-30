// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3SsmLayer constructors + setters.

use super::*;

impl Qwen3SsmLayer {
    pub fn new(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let nk = config.linear_num_key_heads;
        let kd = config.linear_key_head_dim;
        let d_conv = config.linear_conv_kernel_dim;

        // conv_dim = Q_flat + K_flat + V_flat = 2*key_dim + value_dim = 8192
        let conv_dim = nk * kd * 2 + nv * vd;

        Ok(Self {
            // mHC is attached later by the loader, and only for models that
            // carry a hc_mult-wide residual highway. The handles are gated on
            // the same condition `ArchProbes` uses, so a plain GDN model
            // never issues the lookup.
            hc: None,
            ple: None,
            hc_pre_k: hc_kernel(config, gpu, "hc_pre"),
            hc_post_k: hc_kernel(config, gpu, "hc_post"),
            hc_expand_k: hc_kernel(config, gpu, "hc_expand"),
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            lora_out_proj: None,
            qkvz_nvfp4,
            qkvz_nvfp4_t: None,
            out_proj_nvfp4_t: None,
            out_proj_dense: None,
            qkvz_fp8w: None,
            out_proj_fp8w: None,
            qkvz_fp8w_rowwise: None,
            out_proj_fp8w_rowwise: None,
            qkvz_q2: None,
            q2_0_gemv_k: super::super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec"),
            dequant_q2_0_gn_k: super::super::try_kernel(
                gpu,
                "dequant_gguf_bf16",
                "dequant_q2_0_gn_to_bf16",
            ),
            // The keep-packed MMQ family ships only in targets that serve
            // GGUF Q2 checkpoints; probing here would fail the boot audit on
            // every other GDN target. `set_packed_q2_qkvz` resolves them —
            // the only path that installs weights their dispatch sites check.
            q2_0_mmq_nc_k: KernelHandle(0),
            q2_0_mmq_wc_k: KernelHandle(0),
            q4k_quant_act_k: KernelHandle(0),
            sequential_qkvz: false,
            // Resolved ONCE here from the driver, then carried on the layer:
            // the projection dispatch asks "does this grid still fill the
            // machine?" and that question has no portable answer.
            sm_count: gpu.sm_count()?,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            // `output_gate_type: "sigmoid"` (qwen4_exp) swaps the gated-norm
            // handles for the sigmoid twins ONCE, here, so no forward call
            // site branches on it. Every other model keeps the SiLU originals.
            gated_rms_norm_k: if config.gdn_norm_sigmoid {
                gpu.kernel("gated_norm_sigmoid", "gated_rms_norm_sigmoid")?
            } else {
                gpu.kernel("norm", "gated_rms_norm")?
            },
            gated_rms_norm_f32_k: if config.gdn_norm_sigmoid {
                super::super::try_kernel(
                    gpu,
                    "gated_norm_sigmoid",
                    "gated_rms_norm_f32_input_sigmoid",
                )
            } else {
                super::super::try_kernel(gpu, "norm", "gated_rms_norm_f32_input")
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batch2_k: gpu.kernel("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
            w4a16_gemv_qkvz_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qkvz")?,
            deinterleave_k: gpu.kernel("ssm_preprocess", "deinterleave_qkvz")?,
            conv1d_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            conv1d_l2norm_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            // FP32 conv1d output prevents BF16 truncation in the recurrent
            // path from compounding past ~8k tokens. The Metal backend
            // (kernels/metal/common/causal_conv1d_update_l2norm.metal) only
            // ships the BF16 variant; on those targets we fall back to the
            // BF16 kernel via the `.0 != 0` gate at the use site
            // (ssm_forward.rs). Warn instead of error: missing-on-Metal is
            // expected, and a startup `error!` would page on benign cases.
            // Strided twin of `conv1d_l2norm_f32_k` for the batched multi-seq
            // decode path. Optional: absent on older kernel sets, where the
            // multi-seq conv stays a per-sequence loop.
            conv1d_l2norm_f32_strided_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_f32_strided",
            ),
            conv1d_l2norm_f32_k: {
                let h = super::super::try_kernel(
                    gpu,
                    "causal_conv1d",
                    "causal_conv1d_update_l2norm_f32",
                );
                if h.0 == 0 {
                    tracing::warn!(
                        "FP32 conv1d kernel not loaded; SSM uses BF16 conv \
                         output. Expect long-context coherence drift past ~8k \
                         tokens on this backend."
                    );
                }
                h
            },
            gdn_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_decode")?,
            gdn_f32_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32",
            ),
            gdn_f32_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_norm",
            ),
            gdn_f32_conv_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_conv_norm",
            ),
            gdn_f32_strided_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided",
            ),
            gdn_f32_strided_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided_norm",
            ),
            gdn_f32_strided_norm_half_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided_norm_half",
            ),
            gdn_f32_strided_norm_smem_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided_norm_smem",
            ),
            gdn_f16_strided_norm_half_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f16_strided_norm_half",
            ),
            gdn_f16_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f16_norm",
            ),
            ssm_h_f16_to_f32_k: super::super::try_kernel(
                gpu,
                "ssm_h_dtype",
                "ssm_h_state_f16_to_f32",
            ),
            ssm_h_f32_to_f16_k: super::super::try_kernel(
                gpu,
                "ssm_h_dtype",
                "ssm_h_state_f32_to_f16",
            ),
            ba_gates_k: gpu.kernel("ssm_preprocess", "dense_gemv_ba_gates")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            l2_norm_k: gpu.kernel("norm", "l2_norm_bf16")?,
            residual_add_rms_norm_k: gpu.kernel("norm", "residual_add_rms_norm")?,
            residual_add_rms_norm_gatef32_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "residual_add_rms_norm_gatef32",
            ),
            gated_rms_norm_prefill_k: if config.gdn_norm_sigmoid {
                gpu.kernel("gated_norm_sigmoid", "gated_rms_norm_prefill_sigmoid")?
            } else {
                gpu.kernel("norm", "gated_rms_norm_prefill")?
            },
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: crate::layers::tgemm_kernel(gpu),
            w4a16_gemm_t_k64_k: crate::layers::k64_kernel(gpu)?,
            w4a16_gemm_t_k64_n64_k: crate::layers::k64_n64_kernel(gpu),
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            // 8-warp pipelined M128 (try_kernel: 0 when absent → falls back to m128/n128).
            w4a16_gemm_t_m128_v2_k: super::super::w4a16_v2_kernel(gpu),
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            // try_kernel: 0-handle if absent (gated at dispatch); the pipelined
            // BF16 GEMM lives in the same `gemm` module as dense_gemm_bf16.
            dense_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            gdn_prefill_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_prefill")?,
            gdn_prefill_split_k: gpu
                .kernel("gated_delta_rule", "gated_delta_rule_prefill_split")?,
            gdn_prefill_split4_k: gpu
                .kernel("gated_delta_rule", "gated_delta_rule_prefill_split4")?,
            gdn_prefill_persistent_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent",
            ),
            gdn_prefill_persistent_wy4_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_wy4",
            ),
            gdn_prefill_regresident_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_regresident",
                "gated_delta_rule_prefill_regresident",
            ),
            gdn_prefill_fla_recompute_wu_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_recompute_wu",
            ),
            gdn_prefill_fla_chunk_delta_h_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_ksplit",
            ),
            gdn_prefill_fla_chunk_delta_h_tc_vblock_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_tc_vblock",
            ),
            // ONE handle for the fused GDN state spine. DEFAULT is `..._vfused`
            // (SPLIT=2 / 256 threads): 2.01x over ksplit and 12/12 byte-identical on
            // the ssm-poisoning tripwire. `ATLAS_GDN_VTILE=1` swaps in the SPLIT=4 /
            // 512-thread build, which is 2.15x but scores 1/12 there and fails two
            // accuracy gates — kept reachable for whoever diagnoses it, never default.
            // The two are ABI-identical apart from block size, which the launcher
            // derives from the same env, so nothing else downstream changes.
            gdn_prefill_fla_chunk_delta_h_fused_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                // Logged, not silent: which spine ran is the single most
                // consequential fact about a GDN measurement, and a run record
                // that cannot say which one it used cannot be compared to
                // another. An A/B on this kernel is otherwise unfalsifiable —
                // both arms produce a number either way.
                {
                    let name = match (
                        std::env::var("ATLAS_GDN_PIPE").ok().as_deref(),
                        std::env::var("ATLAS_GDN_VTILE").ok().as_deref(),
                    ) {
                        (Some("1"), _) => "gated_delta_rule_chunk_delta_h_pipe",
                        (_, Some("1")) => "gated_delta_rule_chunk_delta_h_vtile",
                        _ => "gated_delta_rule_chunk_delta_h_vfused",
                    };
                    tracing::info!("GDN state spine: {name}");
                    name
                },
            ),
            gdn_prefill_fla_chunk_delta_h_tma_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_tma",
            ),
            gdn_prefill_fla_chunk_fwd_o_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_fwd_o",
            ),
            gdn_prefill_wy32_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy64_prefill",
                "gated_delta_rule_prefill_wy64",
            ),
            // ── Q12 Phase 2b: batched GDN kernel handles ──
            gdn_prefill_wy32_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy64_prefill",
                "gated_delta_rule_prefill_wy64_batched",
            ),
            gdn_prefill_persistent_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_batched",
            ),
            gdn_prefill_persistent_wy4_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_wy4_batched",
            ),
            gdn_prefill_split4_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_prefill_split4_batched",
            ),
            compute_gdn_gates_k: gpu.kernel("ssm_preprocess", "compute_gdn_gates")?,
            ba_gates_prefill_k: gpu.kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")?,
            conv1d_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv1d_prefill_tp_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp",
            ),
            gdn_chunk2_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk2")?,
            conv1d_chunk2_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_chunk2")?,
            gdn_chunk3_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            gdn_wy2_k: gpu.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
            // Register-resident wy2 twin (own gb10-common module so non-gb10
            // targets simply resolve 0 and keep the base wy2 — try_kernel
            // misses are a silent handle 0, so `wy2_kernel` logs the
            // resolution outcome once at first K=2 dispatch).
            gdn_wy2_resident_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy2_resident",
                "gated_delta_rule_wy2_resident",
            ),
            gdn_wy3_k: gpu.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
            // Register-resident wy3 twin (same pattern as wy2's above:
            // try_kernel so non-gb10 targets resolve 0 and keep base wy3;
            // `wy3_kernel` logs the resolution outcome at first K=3 dispatch).
            gdn_wy3_resident_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_resident",
                "gated_delta_rule_wy3_resident",
            ),
            gdn_wy4_k: gpu.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
            // ── ATLAS_SSM_H_FP16 stage 2: FP16 h-state twins of the MTP
            // verify WY kernels. try_kernel for the same reason as the
            // resident twins above — a miss is a silent handle 0, and the
            // selectors gate on `.0 != 0` before ever picking one. Without
            // these the flag and `--speculative` are mutually exclusive,
            // because every WY kernel above writes the state as FP32 and an
            // FP32 kernel over an FP16 pool produces fluent garbage, not an
            // error. Preflight refuses the combination unless the K values the
            // configured draft count can reach all have a twin here.
            gdn_wy2_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy_f16",
                "gated_delta_rule_wy2_f16",
            ),
            gdn_wy2_resident_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy2_resident_f16",
                "gated_delta_rule_wy2_resident_f16",
            ),
            gdn_wy3_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_f16",
                "gated_delta_rule_wy3_f16",
            ),
            gdn_wy3_resident_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_resident_f16",
                "gated_delta_rule_wy3_resident_f16",
            ),
            gdn_wy4_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy4_f16",
                "gated_delta_rule_wy4_f16",
            ),
            // STAGE 1 fused K=2 verify epilogue. Only present in the gb10
            // common PTX module set; NULL on targets lacking the .cu, in which
            // case the num_tokens==2 arm keeps the per-token path even when
            // ATLAS_GDN_FUSED_VERIFY is set.
            gdn_verify_fused_conv_k2_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_k2",
                "gdn_verify_fused_conv_k2",
            ),
            gdn_verify_fused_norm_k2_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_k2",
                "gdn_verify_fused_norm_k2",
            ),
            // Generic-K fused verify conv (K=17 DFlash arm). gb10 common
            // module; NULL on targets lacking the .cu, in which case the
            // K=17 arm keeps its per-token conv loop.
            gdn_verify_fused_conv_kn_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_conv_kn",
                "gdn_verify_fused_conv_kn",
            ),
            // Batched twin (gridDim.y = n_seq) for batched speculative decoding.
            gdn_verify_fused_conv_kn_batched_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_conv_kn",
                "gdn_verify_fused_conv_kn_batched",
            ),
            // Exact-verify `_snap` twins (#435): model-shadow staged
            // (qwen3.6-27b/nvfp4), 0 elsewhere — the exact arm then uses the
            // parent kernel + copy_d2d snapshots (same bits, more launches).
            // Every other GDN target declares these three lookups in its
            // MODEL.toml [expected_absent] (#438) — the boot gate fails closed.
            gdn_f32_norm_snap_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_snap",
                "gated_delta_rule_decode_f32_norm_snap",
            ),
            gdn_f32_strided_norm_snap_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_snap",
                "gated_delta_rule_decode_f32_strided_norm_snap",
            ),
            gdn_verify_fused_conv_kn_f32_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_conv_kn_f32",
                "gdn_verify_fused_conv_kn_f32",
            ),
            // wy17 ships only in qwen3.6-35b-a3b's and qwen3.6-27b's PTX sets;
            // NULL elsewhere (declared [expected_absent] in those MODEL.tomls).
            // decode_batched(K=17) checks for non-NULL before dispatching the fused path.
            gdn_wy17_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17",
                "gated_delta_rule_wy17",
            ),
            gdn_wyn_k: init_kernels::wyn_kernels(gpu),
            gdn_wyn_f16_k: init_kernels::wyn_f16_kernels(gpu),
            h_state_bytes: nv * vd * kd * 4, // FP32 [nv, kd, vd] transposed for coalescing
            conv_state_bytes: conv_dim * d_conv * 4, // FP32 [conv_dim, d_conv]
            qkvz_fp8: None,
            out_proj_fp8: None,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_gemm_t_m128")?,
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            w8a16_gemv_batch4_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch4",
            ),
            w8a16_gemv_batch16_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch16",
            ),
            // NVFP4 batched decode GEMV (all entries live in the w4a16_gemv module).
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            w4a16_gemv_batch16_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16"),
            w8a16_gemm_t_k: super::super::try_kernel(gpu, "w8a16_gemm_t", "w8a16_gemm_t"),
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
        })
    }

    // `new_sequential` moved to `init_sequential.rs` (≤500 LoC split).
}

#[path = "init_kernels.rs"]
mod init_kernels;
use init_kernels::hc_kernel;

#[path = "init_sequential.rs"]
mod init_sequential;
