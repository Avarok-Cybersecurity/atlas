// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3AttentionLayer` constructors: `new`, `new_ungated`, and the
//! private `new_with_gating` (kernel-loading core).

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

// `gate` must be called through a real path, not through a `let`-bound
// function pointer: coercing a `#[track_caller]` fn to a pointer inserts a shim
// and the audit would name the shim instead of the dispatch site below.
use super::init_arch_gates::{ArchProbes, gated as gate};
use super::types::{HeadGateActivation, Qwen3AttentionLayer};
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
        let (reshape_mod, reshape_fn, decode_mod, decode_fn) =
            super::init_kernel_dispatch::kernel_modules_for_dtype(kv_dtype, config.head_dim);
        // Which cross-architecture kernel families this config says exist. A
        // family the model does not have is never LOOKED UP, so it leaves no
        // failed row in the boot audit. See `init_arch_gates`.
        let probes = ArchProbes::from_config(config);
        let mrope_interleaved = config.mrope_interleaved;
        Ok(Self {
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            lora: None,
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
            head_gate_weight: None,
            head_gate_activation: HeadGateActivation::Sigmoid,
            sigmoid_gate_head_broadcast_k: super::super::try_kernel(
                gpu,
                "residual_add",
                "sigmoid_gate_mul_head_broadcast",
            ),
            softplus_gate_head_broadcast_k: super::super::try_kernel(
                gpu,
                "residual_add",
                "softplus_gate_mul_head_broadcast",
            ),
            yarn_inv_freq: spark_runtime::gpu::DevicePtr::NULL,
            yarn_attention_factor: 1.0,
            post_attn_out_norm: None,
            post_ffn_out_norm: None,
            layer_scalar: None,
            moe_ffn: None,
            shortcut_carry_out: None,
            shortcut_carry_in: None,
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
            // ── DeepSeek-V4 Manifold-Constrained Hyper-Connections (mHC) ──
            // `hc` stays None for non-V4 models; the V4 loader attaches real
            // HcWeights after this constructor. Kernel handles are lazy (null
            // when the hyper_connection module is absent), so non-V4 models
            // still start cleanly.
            hc: None,
            qsa: None,
            hc_pre_k: gate(probes.hyper_connection, gpu, "hyper_connection", "hc_pre"),
            hc_post_k: gate(probes.hyper_connection, gpu, "hyper_connection", "hc_post"),
            hc_expand_k: gate(
                probes.hyper_connection,
                gpu,
                "hyper_connection",
                "hc_expand",
            ),
            hc_head_k: gate(probes.hyper_connection, gpu, "hyper_connection", "hc_head"),
            qkv_nvfp4_t: None,
            q_nvfp4_t: None,
            k_nvfp4_t: None,
            v_nvfp4_t: None,
            o_nvfp4_t: None,
            q_fp8w_t: None,
            k_fp8w_t: None,
            v_fp8w_t: None,
            o_fp8w_t: None,
            w8a16_gemm_t_k: super::super::try_kernel(gpu, "w8a16_gemm_t", "w8a16_gemm_t"),
            w8a16_gemm_t_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_t",
                "w8a16_gemm_t_pipelined",
            ),
            w8a16_gemm_t_m128_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_t_m128",
                "w8a16_gemm_t_m128",
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
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_w_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?
            } else {
                gpu.kernel("norm", "rms_norm")?
            },
            rms_norm_w_warp_row_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla_warp_row")
                    .unwrap_or(KernelHandle(0))
            } else {
                KernelHandle(0)
            },
            norm_vanilla: crate::ships_vanilla_norm_weights(config),
            rms_norm_residual_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("norm", "rms_norm_residual_vanilla")?
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dequant_q2_0_gn_k: super::super::try_kernel(
                gpu,
                "dequant_gguf_bf16",
                "dequant_q2_0_gn_to_bf16",
            ),
            // Resolved by `set_packed_q2_weights`, never here: q2_0_mmq /
            // the Q8_1 quantizer ship only in GGUF-serving targets, and an
            // unconditional probe fails the boot audit everywhere else.
            q2_0_mmq_nc_k: KernelHandle(0),
            q2_0_mmq_wc_k: KernelHandle(0),
            q4k_quant_act_k: KernelHandle(0),
            q2_0_gemv_k: super::super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec"),
            dense_gemv_batchm_k: gpu
                .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
            w8a16_gemv_batch4_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch4",
            ),
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            rope_strided_k: super::super::try_kernel(gpu, "rope", "rope_forward_strided"),
            rms_norm_strided_k: super::super::try_kernel(gpu, "norm", "rms_norm_strided"),
            rope_mrope_interleaved_k: super::super::try_kernel(
                gpu,
                "rope_mrope_interleaved",
                "rope_forward_mrope_interleaved",
            ),
            rope_mrope_interleaved_k_only_k: super::super::try_kernel(
                gpu,
                "rope_mrope_interleaved",
                "rope_forward_mrope_interleaved_k_only",
            ),
            rope_yarn_k: super::super::try_kernel(gpu, "rope", "rope_forward_yarn"),
            rope_yarn_scaled_k: super::super::try_kernel(gpu, "rope", "rope_forward_yarn_scaled"),
            // Interleaved (GPT-J / is_neox_style=False) YaRN RoPE — DeepSeek-V4 MLA.
            rope_yarn_interleaved_k: super::super::try_kernel(
                gpu,
                "rope",
                "rope_forward_yarn_interleaved",
            ),
            rope_yarn_interleaved_inv_k: super::super::try_kernel(
                gpu,
                "rope",
                "rope_forward_yarn_interleaved_inv",
            ),
            rope_proportional_k: super::super::try_kernel(gpu, "rope", "rope_forward_proportional"),
            reshape_cache_k: gpu.kernel(reshape_mod, reshape_fn)?,
            fused_k_norm_rope_cache_write_bf16_k: super::super::try_kernel(
                gpu,
                "fused_k_norm_rope_cache",
                "fused_k_norm_rope_cache_write_bf16",
            ),
            fused_k_norm_rope_mrope_cache_write_bf16_k: super::super::try_kernel(
                gpu,
                "fused_k_norm_rope_cache",
                "fused_k_norm_rope_mrope_cache_write_bf16",
            ),
            reshape_and_cache_flash_v_only_k: super::super::try_kernel(
                gpu,
                "reshape_and_cache",
                "reshape_and_cache_flash_v_only",
            ),
            wht_bf16_k: super::super::try_kernel(gpu, "wht_bf16", "wht_bf16_inplace"),
            wht_bf16_k_inv: super::super::try_kernel(gpu, "wht_bf16", "wht_bf16_inplace_inv"),
            innerq_apply_q_k: super::super::try_kernel(
                gpu,
                "tq_plus_innerq_apply",
                "tq_plus_innerq_apply_q",
            ),
            innerq_apply_k_k: super::super::try_kernel(
                gpu,
                "tq_plus_innerq_apply",
                "tq_plus_innerq_apply_k",
            ),
            paged_decode_k: gpu.kernel(decode_mod, decode_fn)?,
            // HDIM>256 decode arm. Every dispatch site gates on
            // `head_dim > 256 && paged_decode_512_k.0 != 0`, so on a head_dim
            // 128 model this whole family was a per-dtype probe that could
            // never be used. `probes.wide_head_dim` is derived from the same
            // `config.head_dim` those sites read.
            paged_decode_512_k: match kv_dtype {
                KvCacheDtype::Bf16 => gate(
                    probes.wide_head_dim,
                    gpu,
                    "paged_decode_attn_512",
                    "paged_decode_attn",
                ),
                KvCacheDtype::Turbo4 => gate(
                    probes.wide_head_dim,
                    gpu,
                    "paged_decode_turbo4_512",
                    "paged_decode_attn_turbo4",
                ),
                KvCacheDtype::Turbo8 => gate(
                    probes.wide_head_dim,
                    gpu,
                    "paged_decode_turbo8_512",
                    "paged_decode_attn_turbo8",
                ),
                KvCacheDtype::Turbo3 | KvCacheDtype::Turbo2 => gate(
                    probes.wide_head_dim,
                    gpu,
                    "paged_decode_turbo4_512",
                    "paged_decode_attn_turbo4",
                ),
                _ => gate(
                    probes.wide_head_dim,
                    gpu,
                    "paged_decode_attn_fp8_512",
                    "paged_decode_attn_fp8",
                ),
            },
            paged_decode_mla_k: gate(probes.mla, gpu, "paged_decode_mla", "paged_decode_attn"),
            // DeepSeek-V4-Flash MLA paged decode (compressed 576-dim KV cache).
            mla_paged_decode_k: gate(
                probes.mla,
                gpu,
                "mla_paged_decode",
                "mla_paged_decode_nvfp4",
            ),
            mla_paged_decode_fp8_k: gate(
                probes.mla,
                gpu,
                "mla_paged_decode_fp8",
                "mla_paged_decode_fp8",
            ),
            mla_batched_gemv_k: gate(probes.mla, gpu, "mla_absorbed", "mla_batched_gemv"),
            mla_q_rope_scatter_k: gate(probes.mla, gpu, "mla_absorbed", "mla_q_rope_scatter"),
            mla_q_rope_writeback_k: gate(probes.mla, gpu, "mla_absorbed", "mla_q_rope_writeback"),
            mla_cache_assemble_k: gate(probes.mla, gpu, "mla_absorbed", "mla_cache_assemble"),
            mla_q_rope_extract_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_q_rope_extract_batched",
            ),
            mla_q_rope_writeback_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_q_rope_writeback_batched",
            ),
            mla_kv_assemble_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_kv_assemble_batched",
            ),
            mla_cache_assemble_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_cache_assemble_batched",
            ),
            prefill_attn_mla320_k: gate(
                probes.mla,
                gpu,
                "mla_prefill_attn",
                "mla_prefill_attn_320",
            ),
            grouped_gemm_mla_k: gate(probes.mla, gpu, "grouped_gemm_mla", "grouped_gemm_mla"),
            mla_q_final_assemble_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_q_final_assemble_batched",
            ),
            mla_fused_prefill_k: gate(probes.mla, gpu, "mla_fused_prefill", "mla_fused_prefill"),
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
                KvCacheDtype::Turbo3
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo8
                | KvCacheDtype::Bf16KTurbo3V
                | KvCacheDtype::Bf16KTurbo4V
                | KvCacheDtype::Bf16KTurbo2V
                | KvCacheDtype::Fp8KTurbo3V
                | KvCacheDtype::Fp8KTurbo4V
                | KvCacheDtype::Fp8KTurbo2V
                | KvCacheDtype::Turbo4KTurbo3V
                | KvCacheDtype::Turbo4KTurbo8V
                | KvCacheDtype::Turbo3KTurbo8V => None,
                _ => Some(gpu.kernel("paged_decode_fp8", "paged_decode_attn_splitk_fp8")?),
            },
            paged_decode_reduce_k: match kv_dtype {
                KvCacheDtype::Nvfp4 => {
                    Some(gpu.kernel("paged_decode_nvfp4", "paged_decode_attn_reduce_nvfp4")?)
                }
                KvCacheDtype::Turbo3
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo8
                | KvCacheDtype::Bf16KTurbo3V
                | KvCacheDtype::Bf16KTurbo4V
                | KvCacheDtype::Bf16KTurbo2V
                | KvCacheDtype::Fp8KTurbo3V
                | KvCacheDtype::Fp8KTurbo4V
                | KvCacheDtype::Fp8KTurbo2V
                | KvCacheDtype::Turbo4KTurbo3V
                | KvCacheDtype::Turbo4KTurbo8V
                | KvCacheDtype::Turbo3KTurbo8V => None,
                _ => Some(gpu.kernel("paged_decode_fp8", "paged_decode_attn_reduce_fp8")?),
            },
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            // Gemma-4 rms-norm uses the absolute formula `out = x * rms * w`.
            rms_norm_f32_in_k: KernelHandle(0),
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            deinterleave_qg_k: gpu.kernel("ssm_preprocess", "deinterleave_qg")?,
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            residual_add_rms_norm_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("norm", "residual_add_rms_norm_vanilla")?
            } else {
                gpu.kernel("norm", "residual_add_rms_norm")?
            },
            residual_add_rms_norm_gatef32_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "residual_add_rms_norm_gatef32",
            ),
            w4a16_gemv_qg_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg_batch2")?,
            w4a16_gemv_dual_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_qg_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg_batch3")?,
            w4a16_gemv_dual_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: crate::layers::tgemm_kernel(gpu),
            w4a16_gemm_t_k64_k: crate::layers::k64_kernel(gpu)?,
            w4a16_gemm_t_k64_n64_k: crate::layers::k64_n64_kernel(gpu),
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            w4a16_gemm_t_m128_bf16_k: super::super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128_bf16",
            ),
            w4a16_gemm_t_m128_v2_k: super::super::w4a16_v2_kernel(gpu),
            w4a16_gemm_t_m128_v3_k: super::super::w4a16_v3_kernel(gpu),
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            prefill_attn_k: gpu.kernel("inferspark_prefill", "inferspark_prefill")?,
            // Name comes from the SSOT helper that also supplies the BR the
            // launcher builds its grid from — see `ops::wide_prefill_kernel`.
            // Module and entry share a name for both variants.
            // Resolved WITH FALLBACK — see `ops::wide_prefill_kernel`. A target
            // that ships only the scalar HDIM=512 kernel must still get it.
            prefill_attn_512_k: if probes.wide_head_dim {
                crate::layers::ops::wide_prefill_kernel(gpu).0
            } else {
                spark_runtime::gpu::KernelHandle(0)
            },
            // BR=32 is the tensor-core instantiation; BR=16 the scalar reference.
            prefill_attn_512_is_tc: probes.wide_head_dim
                && crate::layers::ops::wide_prefill_kernel(gpu).1 == 32,
            // DeepSeek-V4 sparse-attention compressor + compressed-KV prefill.
            csa_compress_k: gate(probes.compressed_attn, gpu, "csa_compress", "csa_compress"),
            prefill_attn_compressed_k: gate(
                probes.compressed_attn,
                gpu,
                "prefill_attn_compressed",
                "prefill_attn_compressed",
            ),
            v4_comp_pool_filled: std::sync::atomic::AtomicU32::new(0),
            v4_comp_prev_valid: std::sync::atomic::AtomicBool::new(false),
            v4_decode_started: std::sync::atomic::AtomicBool::new(false),
            v4_decode_first_pos: std::sync::atomic::AtomicU32::new(0),
            prefill_attn_paged_512_k: gate(
                probes.wide_head_dim,
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
            prefill_attn_paged_turbo2_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo2",
                "inferspark_prefill_paged_turbo2",
            ),
            prefill_attn_paged_turbo3_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo3",
                "inferspark_prefill_paged_turbo3_64",
            ),
            prefill_attn_paged_turbo4_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4",
                "inferspark_prefill_paged_turbo4_64",
            ),
            prefill_attn_paged_turbo8_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo8",
                "inferspark_prefill_paged_turbo8_64",
            ),
            // TurboQuant+ safer-asym Bf16K + Turbo3V BR=64 prefill kernel.
            // Compiled from inferspark_prefill_paged_bf16k_turbo3v.cu which
            // forks prefill_paged_compute_asym.cuh (LOAD_K_TILE = bf16,
            // LOAD_V_TILE = turbo3 3-bit dequant).
            prefill_attn_paged_bf16k_turbo3v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_bf16k_turbo3v",
                "inferspark_prefill_paged_bf16k_turbo3v_64",
            ),
            // Bf16K + Turbo4V BR=64 prefill (4-bit V dequant in LOAD_V_TILE).
            prefill_attn_paged_bf16k_turbo4v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_bf16k_turbo4v",
                "inferspark_prefill_paged_bf16k_turbo4v_64",
            ),
            // Bf16K + Turbo2V BR=64 prefill (2-bit V dequant in LOAD_V_TILE).
            prefill_attn_paged_bf16k_turbo2v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_bf16k_turbo2v",
                "inferspark_prefill_paged_bf16k_turbo2v_64",
            ),
            // Fp8K + TurboNV BR=64 prefill kernels — K loaded as FP8 (per-tensor
            // `k_scale` dequant in LOAD_K_TILE), V as 3/4/2-bit Lloyd-Max packed.
            prefill_attn_paged_fp8k_turbo3v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_fp8k_turbo3v",
                "inferspark_prefill_paged_fp8k_turbo3v_64",
            ),
            prefill_attn_paged_fp8k_turbo4v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_fp8k_turbo4v",
                "inferspark_prefill_paged_fp8k_turbo4v_64",
            ),
            prefill_attn_paged_fp8k_turbo2v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_fp8k_turbo2v",
                "inferspark_prefill_paged_fp8k_turbo2v_64",
            ),
            // Both-sides-quantized TurboQuant+ asym BR=64 prefill kernels.
            // K loaded via turbo* dequant in LOAD_K_TILE, V via the corresponding
            // turbo* dequant in LOAD_V_TILE — separate (block_stride, data_section)
            // pairs per side.
            prefill_attn_paged_turbo4k_turbo3v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4k_turbo3v",
                "inferspark_prefill_paged_turbo4k_turbo3v_64",
            ),
            prefill_attn_paged_turbo4k_turbo8v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4k_turbo8v",
                "inferspark_prefill_paged_turbo4k_turbo8v_64",
            ),
            prefill_attn_paged_turbo3k_turbo8v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo3k_turbo8v",
                "inferspark_prefill_paged_turbo3k_turbo8v_64",
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
            deinterleave_qg_split_qnorm_mrope_k: super::super::try_kernel(
                gpu,
                "ssm_preprocess",
                "deinterleave_qg_split_qnorm_mrope",
            ),
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
            w4a4_gemm_k: crate::layers::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_bf16_to_nvfp4",
            ),
            fp8_calibration: if fp8_calibration_tokens > 0
                && crate::layers::fp8_calibration::dtype_runs_online_fp8_kv_calibration(kv_dtype)
            {
                Some(Fp8KvCalibration::new(
                    fp8_calibration_tokens,
                    config.fp8_kv_headroom,
                    gpu,
                )?)
            } else {
                None
            },
        })
    }
}
