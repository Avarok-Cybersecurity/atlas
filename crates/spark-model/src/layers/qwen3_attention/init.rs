// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3AttentionLayer` constructors: `new`, `new_ungated`, and the
//! private `new_with_gating` (kernel-loading core).

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

use super::types::Qwen3AttentionLayer;
use crate::layers::FfnComponent;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantWeight, QuantizedWeight};

impl Qwen3AttentionLayer {
    pub fn new(
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        attn_layer_idx: usize,
        q_nvfp4: Option<QuantizedWeight>,
        k_nvfp4: Option<QuantizedWeight>,
        v_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
        fp8_calibration_tokens: usize,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_gating(
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            true,
            gpu,
            kv_dtype,
            fp8_calibration_tokens,
            config,
        )
    }

    pub fn new_ungated(
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        attn_layer_idx: usize,
        q_nvfp4: Option<QuantizedWeight>,
        k_nvfp4: Option<QuantizedWeight>,
        v_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
        fp8_calibration_tokens: usize,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_gating(
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            false,
            gpu,
            kv_dtype,
            fp8_calibration_tokens,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_gating(
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        attn_layer_idx: usize,
        q_nvfp4: Option<QuantizedWeight>,
        k_nvfp4: Option<QuantizedWeight>,
        v_nvfp4: Option<QuantizedWeight>,
        gated: bool,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
        fp8_calibration_tokens: usize,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let (reshape_mod, reshape_fn, decode_mod, decode_fn) = match kv_dtype {
            KvCacheDtype::Nvfp4 => (
                "reshape_and_cache",
                "reshape_and_cache_flash_nvfp4",
                "paged_decode_nvfp4",
                "paged_decode_attn_nvfp4",
            ),
            KvCacheDtype::Turbo4 => {
                let dm = if config.head_dim <= 128 {
                    "paged_decode_turbo4_128"
                } else {
                    "paged_decode_turbo4"
                };
                (
                    "reshape_and_cache_turbo",
                    "reshape_and_cache_flash_turbo4",
                    dm,
                    "paged_decode_attn_turbo4",
                )
            }
            KvCacheDtype::Turbo3 => {
                let dm = if config.head_dim <= 128 {
                    "paged_decode_turbo3_128"
                } else {
                    "paged_decode_turbo3"
                };
                (
                    "reshape_and_cache_turbo",
                    "reshape_and_cache_flash_turbo3",
                    dm,
                    "paged_decode_attn_turbo3",
                )
            }
            KvCacheDtype::Turbo8 => {
                let dm = if config.head_dim <= 128 {
                    "paged_decode_turbo8_128"
                } else {
                    "paged_decode_turbo8"
                };
                (
                    "reshape_and_cache_turbo",
                    "reshape_and_cache_flash_turbo8",
                    dm,
                    "paged_decode_attn_turbo8",
                )
            }
            KvCacheDtype::Bf16 => (
                "reshape_and_cache",
                "reshape_and_cache_flash",
                "paged_decode",
                "paged_decode_attn",
            ),
            _ => (
                "reshape_and_cache",
                "reshape_and_cache_flash_fp8",
                "paged_decode_fp8",
                "paged_decode_attn_fp8",
            ),
        };
        let mrope_interleaved = config.mrope_interleaved;
        Ok(Self {
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            gated,
            mrope_interleaved,
            kv_dtype,
            head_dim_override: None,
            num_q_heads_override: None,
            num_kv_heads_override: None,
            sliding_window: None,
            rope_theta_override: None,
            rotary_dim_override: None,
            rope_proportional: false,
            attn_scale_override: None,
            k_eq_v: false,
            v_norm_weight: None,
            post_attn_out_norm: None,
            post_ffn_out_norm: None,
            layer_scalar: None,
            moe_ffn: None,
            pre_moe_norm: None,
            post_moe_out_norm: None,
            post_dense_ffn_norm: None,
            sparse_v_threshold: 0.0,
            q_weight: q_nvfp4.map(QuantWeight::Nvfp4),
            k_weight: k_nvfp4.map(QuantWeight::Nvfp4),
            v_weight: v_nvfp4.map(QuantWeight::Nvfp4),
            o_weight: None,
            o_dense_bf16: None,
            mla: None,
            q_nvfp4_t: None,
            k_nvfp4_t: None,
            v_nvfp4_t: None,
            o_nvfp4_t: None,
            q_fp8w_t: None,
            k_fp8w_t: None,
            v_fp8w_t: None,
            o_fp8w_t: None,
            w8a16_gemm_t_k: super::super::try_kernel(gpu, "w8a16_gemm_t", "w8a16_gemm_t"),
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_residual_k: if config.use_fp32_residual() {
                if config.model_type == "gemma4" {
                    gpu.kernel("norm", "rms_norm_residual_f32_abs")
                        .or_else(|_| gpu.kernel("norm", "rms_norm_residual"))?
                } else {
                    gpu.kernel("norm", "rms_norm_residual_f32")
                        .or_else(|_| gpu.kernel("norm", "rms_norm_residual"))?
                }
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            rope_mrope_interleaved_k: super::super::try_kernel(
                gpu,
                "rope_mrope_interleaved",
                "rope_forward_mrope_interleaved",
            ),
            rope_yarn_k: super::super::try_kernel(gpu, "rope", "rope_forward_yarn"),
            rope_proportional_k: super::super::try_kernel(gpu, "rope", "rope_forward_proportional"),
            reshape_cache_k: gpu.kernel(reshape_mod, reshape_fn)?,
            wht_bf16_k: super::super::try_kernel(gpu, "wht_bf16", "wht_bf16_inplace"),
            paged_decode_k: gpu.kernel(decode_mod, decode_fn)?,
            paged_decode_512_k: match kv_dtype {
                KvCacheDtype::Bf16 => {
                    super::super::try_kernel(gpu, "paged_decode_attn_512", "paged_decode_attn")
                }
                KvCacheDtype::Turbo4 => super::super::try_kernel(
                    gpu,
                    "paged_decode_turbo4_512",
                    "paged_decode_attn_turbo4",
                ),
                KvCacheDtype::Turbo8 => super::super::try_kernel(
                    gpu,
                    "paged_decode_turbo8_512",
                    "paged_decode_attn_turbo8",
                ),
                KvCacheDtype::Turbo3 => super::super::try_kernel(
                    gpu,
                    "paged_decode_turbo4_512",
                    "paged_decode_attn_turbo4",
                ),
                _ => super::super::try_kernel(
                    gpu,
                    "paged_decode_attn_fp8_512",
                    "paged_decode_attn_fp8",
                ),
            },
            paged_decode_mla_k: super::super::try_kernel(
                gpu,
                "paged_decode_mla",
                "paged_decode_attn",
            ),
            mla_batched_gemv_k: super::super::try_kernel(gpu, "mla_absorbed", "mla_batched_gemv"),
            mla_q_rope_scatter_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_q_rope_scatter",
            ),
            mla_q_rope_writeback_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_q_rope_writeback",
            ),
            mla_cache_assemble_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_cache_assemble",
            ),
            mla_q_rope_extract_batched_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_q_rope_extract_batched",
            ),
            mla_q_rope_writeback_batched_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_q_rope_writeback_batched",
            ),
            mla_kv_assemble_batched_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_kv_assemble_batched",
            ),
            mla_cache_assemble_batched_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_cache_assemble_batched",
            ),
            prefill_attn_mla320_k: super::super::try_kernel(
                gpu,
                "mla_prefill_attn",
                "mla_prefill_attn_320",
            ),
            grouped_gemm_mla_k: super::super::try_kernel(
                gpu,
                "grouped_gemm_mla",
                "grouped_gemm_mla",
            ),
            mla_q_final_assemble_k: super::super::try_kernel(
                gpu,
                "mla_absorbed",
                "mla_q_final_assemble_batched",
            ),
            mla_fused_prefill_k: super::super::try_kernel(
                gpu,
                "mla_fused_prefill",
                "mla_fused_prefill",
            ),
            gemm_splitk_partial_k: super::super::try_kernel(
                gpu,
                "gemm_splitk",
                "dense_gemm_splitk_partial",
            ),
            gemm_splitk_reduce_k: super::super::try_kernel(
                gpu,
                "gemm_splitk",
                "dense_gemm_splitk_reduce",
            ),
            dense_gemm_tc_k: super::super::try_kernel(gpu, "gemm_tc", "dense_gemm_tc"),
            paged_decode_splitk_k: match kv_dtype {
                KvCacheDtype::Nvfp4 => {
                    Some(gpu.kernel("paged_decode_nvfp4", "paged_decode_attn_splitk_nvfp4")?)
                }
                KvCacheDtype::Turbo3 | KvCacheDtype::Turbo4 | KvCacheDtype::Turbo8 => None,
                _ => Some(gpu.kernel("paged_decode_fp8", "paged_decode_attn_splitk_fp8")?),
            },
            paged_decode_reduce_k: match kv_dtype {
                KvCacheDtype::Nvfp4 => {
                    Some(gpu.kernel("paged_decode_nvfp4", "paged_decode_attn_reduce_nvfp4")?)
                }
                KvCacheDtype::Turbo3 | KvCacheDtype::Turbo4 | KvCacheDtype::Turbo8 => None,
                _ => Some(gpu.kernel("paged_decode_fp8", "paged_decode_attn_reduce_fp8")?),
            },
            residual_add_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "f32_residual_add")
                    .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?
            } else {
                gpu.kernel("residual_add", "bf16_residual_add")?
            },
            // Gemma-4 rms-norm uses the absolute formula `out = x * rms * w`.
            rms_norm_f32_in_k: if config.use_fp32_residual() && config.model_type == "gemma4" {
                super::super::try_kernel(gpu, "norm", "rms_norm_f32_in_abs")
            } else {
                KernelHandle(0)
            },
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            deinterleave_qg_k: gpu.kernel("ssm_preprocess", "deinterleave_qg")?,
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            residual_add_rms_norm_k: if config.use_fp32_residual() {
                if config.model_type == "gemma4" {
                    gpu.kernel("norm", "residual_add_rms_norm_f32_abs")
                        .or_else(|_| gpu.kernel("norm", "residual_add_rms_norm"))?
                } else {
                    gpu.kernel("norm", "residual_add_rms_norm_f32")
                        .or_else(|_| gpu.kernel("norm", "residual_add_rms_norm"))?
                }
            } else {
                gpu.kernel("norm", "residual_add_rms_norm")?
            },
            w4a16_gemv_qg_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg_batch2")?,
            w4a16_gemv_dual_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_qg_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg_batch3")?,
            w4a16_gemv_dual_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            w4a16_gemm_t_k64_k: gpu.kernel("w4a16", "w4a16_gemm_t_k64")?,
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            w4a16_gemm_t_m128_v2_k: super::super::try_kernel(
                gpu,
                "w4a16_v2",
                "w4a16_gemm_t_m128_v2",
            ),
            w4a16_gemm_t_m128_v3_k: super::super::try_kernel(
                gpu,
                "w4a16_v3",
                "w4a16_gemm_t_m128_v3",
            ),
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            prefill_attn_k: gpu.kernel("inferspark_prefill", "inferspark_prefill")?,
            prefill_attn_512_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_512",
                "inferspark_prefill_512",
            ),
            prefill_attn_paged_512_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_512",
                "inferspark_prefill_paged_512",
            ),
            prefill_attn_64_k: gpu.kernel("inferspark_prefill", "inferspark_prefill_64")?,
            prefill_attn_paged_k: gpu.kernel("prefill_paged", "inferspark_prefill_paged")?,
            prefill_attn_paged_fp8_k: gpu
                .kernel("prefill_paged_fp8", "inferspark_prefill_paged_fp8")?,
            prefill_attn_paged_nvfp4_k: gpu
                .kernel("prefill_paged_nvfp4", "inferspark_prefill_paged_nvfp4")?,
            prefill_attn_paged_turbo4_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4",
                "inferspark_prefill_paged_turbo4",
            ),
            prefill_attn_paged_64_k: gpu.kernel("prefill_paged", "inferspark_prefill_paged_64")?,
            prefill_attn_paged_fp8_64_k: gpu
                .kernel("prefill_paged_fp8", "inferspark_prefill_paged_fp8_64")?,
            prefill_attn_paged_nvfp4_64_k: gpu
                .kernel("prefill_paged_nvfp4", "inferspark_prefill_paged_nvfp4_64")?,
            prefill_attn_paged_turbo4_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4",
                "inferspark_prefill_paged_turbo4_64",
            ),
            // ── Q12 Phase 3: batched paged-prefill kernel handles ──
            prefill_attn_paged_batched_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_batched",
                "inferspark_prefill_paged_batched",
            ),
            prefill_attn_paged_fp8_batched_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_fp8_batched",
                "inferspark_prefill_paged_fp8_batched",
            ),
            prefill_attn_paged_nvfp4_batched_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_nvfp4_batched",
                "inferspark_prefill_paged_nvfp4_batched",
            ),
            prefill_attn_paged_batched_64_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_batched",
                "inferspark_prefill_paged_batched_64",
            ),
            prefill_attn_paged_fp8_batched_64_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_fp8_batched",
                "inferspark_prefill_paged_fp8_batched_64",
            ),
            prefill_attn_paged_nvfp4_batched_64_k: super::super::try_kernel(
                gpu,
                "inferspark_prefill_paged_nvfp4_batched",
                "inferspark_prefill_paged_nvfp4_batched_64",
            ),
            deinterleave_qg_split_k: gpu.kernel("ssm_preprocess", "deinterleave_qg_split")?,
            deinterleave_qg_split_qnorm_k: gpu
                .kernel("ssm_preprocess", "deinterleave_qg_split_qnorm")?,
            sigmoid_gate_mul_batched_k: gpu.kernel("residual_add", "sigmoid_gate_mul_batched")?,
            q_fp8: None,
            k_fp8: None,
            v_fp8: None,
            o_fp8: None,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            bf16_to_fp8_k: gpu.kernel("w4a16", "bf16_to_fp8")?,
            fp8_fp8_gemm_k: gpu.kernel("w4a16", "fp8_fp8_gemm_t")?,
            fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_gemm_t_m128")?,
            fp8_fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_fp8_gemm_t_m128")?,
            fp8_calibration: if fp8_calibration_tokens > 0
                && !matches!(
                    kv_dtype,
                    KvCacheDtype::Nvfp4
                        | KvCacheDtype::Turbo4
                        | KvCacheDtype::Turbo3
                        | KvCacheDtype::Turbo8
                ) {
                Some(Fp8KvCalibration::new(fp8_calibration_tokens, gpu)?)
            } else {
                None
            },
        })
    }
}
