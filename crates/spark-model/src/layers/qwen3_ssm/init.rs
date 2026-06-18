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
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qkvz_nvfp4,
            qkvz_nvfp4_t: None,
            out_proj_nvfp4_t: None,
            out_proj_dense: None,
            qkvz_fp8w: None,
            out_proj_fp8w: None,
            sequential_qkvz: false,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            gated_rms_norm_k: gpu.kernel("norm", "gated_rms_norm")?,
            gated_rms_norm_f32_k: super::super::try_kernel(gpu, "norm", "gated_rms_norm_f32_input"),
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
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
            ba_gates_k: gpu.kernel("ssm_preprocess", "dense_gemv_ba_gates")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            l2_norm_k: gpu.kernel("norm", "l2_norm_bf16")?,
            residual_add_rms_norm_k: gpu.kernel("norm", "residual_add_rms_norm")?,
            residual_add_rms_norm_gatef32_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "residual_add_rms_norm_gatef32",
            ),
            gated_rms_norm_prefill_k: gpu.kernel("norm", "gated_rms_norm_prefill")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            w4a16_gemm_t_k64_k: gpu.kernel("w4a16", "w4a16_gemm_t_k64")?,
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
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
            gdn_chunk2_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk2")?,
            conv1d_chunk2_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_chunk2")?,
            gdn_chunk3_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            gdn_wy2_k: gpu.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
            gdn_wy3_k: gpu.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
            gdn_wy4_k: gpu.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
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
            // wy17 only present in qwen3.6-35b-a3b's PTX module set; NULL on other targets.
            // decode_batched(K=17) checks for non-NULL before dispatching the fused path.
            gdn_wy17_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17",
                "gated_delta_rule_wy17",
            ),
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

    /// Construct an SSM layer where QKVZ projection output is already sequential.
    ///
    /// Used by Qwen3.5 where separate QKV and Z weights are concatenated at load
    /// time into `[Q|K|V|Z]` row order. The `deinterleave_qkvz` kernel is skipped
    /// and plain `w4a16_gemv` writes directly to the deinterleaved buffer.
    pub fn new_sequential(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        qkvz_nvfp4_t: Option<QuantizedWeight>,
        out_proj_nvfp4_t: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let mut layer = Self::new(
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qkvz_nvfp4,
            config,
            gpu,
        )?;
        layer.sequential_qkvz = true;
        layer.qkvz_nvfp4_t = qkvz_nvfp4_t;
        layer.out_proj_nvfp4_t = out_proj_nvfp4_t;
        Ok(layer)
    }

    /// Install native FP8 block-scaled weights for the decode GEMV path.
    ///
    /// Inputs MUST be tagged `WeightQuantFormat::Fp8BlockScaled` — that is
    /// the canonical input format for the `w8a16_gemv` kernel
    /// (`out[n] = sum_k A[k] * E4M3_LUT[B[n,k]] * block_scale[n/BS, k/BS]`,
    /// see `kernels/gb10/common/w8a16_gemv.cu`). The kernel reads the
    /// scale buffer at `[N/BS, K/BS]` BF16 — exactly the shape produced
    /// by `load_fp8_block_scaled_as_fp8weight`.
    ///
    /// This setter does NOT install the raw FP8 DevicePtr fields used by
    /// the prefill `fp8_gemm_n128` kernel — that kernel takes no scale
    /// argument and assumes single-scale FP8 (baked-in scale) produced
    /// by `bf16_to_fp8`. Block-scaled bytes would silently produce wrong
    /// outputs there. For prefill, call `set_fp8_prefill_only_weights`
    /// separately with single-scale FP8 derived from a BF16 dequant.
    pub fn set_fp8_decode_weights(&mut self, qkvz: Option<Fp8Weight>, out_proj: Option<Fp8Weight>) {
        if let Some(ref w) = qkvz {
            w.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "set_fp8_decode_weights::qkvz (w8a16_gemv expects [N/BS,K/BS] BF16 block scales)",
            );
        }
        if let Some(ref w) = out_proj {
            w.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "set_fp8_decode_weights::out_proj (w8a16_gemv expects [N/BS,K/BS] BF16 block scales)",
            );
        }
        self.qkvz_fp8w = qkvz;
        self.out_proj_fp8w = out_proj;
    }

    /// Set raw FP8 DevicePtrs for the prefill GEMM path ONLY (no decode GEMV
    /// scale fields). Used by the Qwen3.6-27B-FP8 native-FP8 SSM prefill path:
    /// the FP8 buffer here is a single-scale FP8 (BF16 → FP8 truncation; values
    /// already in FP8 range) suitable for `fp8_gemm_n128`. Decode falls back to
    /// the NVFP4/BF16 paths via the existing `qkvz_nvfp4*` fields. PCND:
    /// caller decides whether to install — never set implicitly.
    pub fn set_fp8_prefill_only_weights(
        &mut self,
        qkvz_fp8: Option<DevicePtr>,
        out_proj_fp8: Option<DevicePtr>,
    ) {
        if qkvz_fp8.is_some() {
            self.qkvz_fp8 = qkvz_fp8;
        }
        if out_proj_fp8.is_some() {
            self.out_proj_fp8 = out_proj_fp8;
        }
    }

    /// Pre-dequant NVFP4 → FP8 for QKVZ and out_proj transposed weights.
    /// Eliminates per-inference dequant overhead in prefill GEMMs.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let qkvz_size = config.ssm_qkvz_size();
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;

        // QKVZ FP8 predequant: tested at ISL=1019, FP8 is ~50% slower (1900µs vs 1228µs)
        // because weight matrix [12288, 2048] is bandwidth-dominated at M=1024 — the 2×
        // larger FP8 weights (25 MB vs 12.6 MB NVFP4) cost more than the dequant saves.
        let _ = qkvz_size; // suppress unused warning
        // Use NON-transposed out_proj (ssm.out_proj is [N, K/2] layout).
        // predequant_nvfp4_to_fp8 assumes [N, K/2] input layout.
        if self.out_proj_nvfp4_t.is_some() {
            self.out_proj_fp8 = Some(self.ssm.out_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                value_dim,
                stream,
            )?);
        }
        Ok(())
    }
}
