// SPDX-License-Identifier: AGPL-3.0-only
//
// The standard NVFP4-quantized LinearAttention arm of `load_layers`
// (`build_linear_attention_nvfp4`). Child module of `linear_attn_arms.rs`,
// split out for the ≤500 LoC cap; re-exported from there so callers keep
// the `linear_attn_arms::build_linear_attention_nvfp4` path.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3SsmLayer};
use crate::tp_shard::{
    TpGdnDims, shard_gdn_ba_rows, shard_gdn_conv_rows, shard_gdn_out_proj_row_parallel,
    shard_gdn_qkvz_rows, shard_gdn_value_vector,
};
use crate::weight_map::{
    DenseWeight, Exl3DenseFamily, Nvfp4Variant, SsmWeights, exl3_dense_family_kept,
    gpu_concat_rows, interleave_ba, load_ssm_qwen35, quantize_to_nvfp4,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_linear_attention_nvfp4(
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    // GDN HeadParallel: `config` holds per-rank-LOCAL linear head counts.
    // Slice each SSM projection to this rank's head range on the dense/BF16
    // intermediate BEFORE quantizing to NVFP4 (dequant→slice→requant is the
    // safe path — no NVFP4 packed-buffer surgery). For tp=1 the slicers return
    // the source pointer untouched → byte-identical fast path.
    let tp_size = config.tp_world_size.max(1);
    let dims = TpGdnDims::from_config(config);

    // The native-EXL3 GDN route lives on the BF16 arm only: a layer whose
    // family was kept packed has no `.weight` for this arm to requantize.
    ensure!(
        !exl3_dense_family_kept(store, lp, Exl3DenseFamily::Gdn),
        "{lp}: ATLAS_EXL3_NATIVE_DENSE=1 kept this layer's GDN family as packed EXL3 \
         trellis, but the NVFP4-requant GDN arm was selected (ATLAS_QWEN4EXP_BF16_GDN=0), \
         which has no native route — unset ATLAS_QWEN4EXP_BF16_GDN or ATLAS_EXL3_NATIVE_GDN=0"
    );
    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant)?;

    // Concat FULL [Q|K|V] || [Z] then SEGMENT-slice to local heads.
    let qkvz_full = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        dims.full_conv_dim(),
        &ssm35.in_proj_z,
        dims.full_value_dim(),
        h,
        gpu,
    )?;
    let (qkvz_ptr, _, _) = shard_gdn_qkvz_rows(qkvz_full.weight, &dims, gpu)?;
    if tp_size > 1 {
        let _ = gpu.free(qkvz_full.weight);
    }
    let qkvz_dense = DenseWeight { weight: qkvz_ptr };

    // BA: interleave FULL heads then slice to local (group-aligned).
    let ba_full = interleave_ba(
        &DenseWeight {
            weight: ssm35.in_proj_a.weight,
        },
        &DenseWeight {
            weight: ssm35.in_proj_b.weight,
        },
        dims.full_nv,
        dims.full_nk,
        h,
        gpu,
    )?;
    let (ba_ptr, _, _) = shard_gdn_ba_rows(ba_full.weight, &dims, gpu)?;
    if tp_size > 1 {
        let _ = gpu.free(ba_full.weight);
    }
    let ba_dense = DenseWeight { weight: ba_ptr };

    // conv1d / a_log / dt_bias / norm sliced to local heads (stay dense/FP32).
    let d_conv = config.linear_conv_kernel_dim;
    let (conv_ptr, _, _) = shard_gdn_conv_rows(ssm35.conv1d.weight, &dims, d_conv, gpu)?;
    let (a_log_ptr, _) = shard_gdn_value_vector(ssm35.a_log.weight, &dims, 1, 4, gpu)?;
    let (dt_bias_ptr, _) = shard_gdn_value_vector(ssm35.dt_bias.weight, &dims, 1, 4, gpu)?;
    // norm.weight is the gated-RMSNorm gain over the value HEAD-DIM ([vd]),
    // SHARED across all value heads — REPLICATE under HeadParallel. (a_log/dt_bias
    // above ARE per-head [nv] scalars, so they slice; slicing norm on the head
    // axis read past the [vd] buffer → cuMemcpyDtoDAsync INVALID_VALUE at load.)
    let norm_ptr = ssm35.norm.weight;
    let conv1d_local = DenseWeight { weight: conv_ptr };
    let a_log_local = DenseWeight { weight: a_log_ptr };
    let dt_bias_local = DenseWeight {
        weight: dt_bias_ptr,
    };
    let norm_local = DenseWeight { weight: norm_ptr };

    // out_proj is row-parallel: slice its input (value_dim) to local, then
    // quantize the LOCAL [h, local_value_dim] weight.
    let (out_proj_ptr, _, _) = shard_gdn_out_proj_row_parallel(ssm35.out_proj.weight, &dims, gpu)?;
    let out_proj_local = DenseWeight {
        weight: out_proj_ptr,
    };

    // All sizes below are LOCAL (config was TP-divided at load).
    let nv = config.linear_num_value_heads;
    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 =
        quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;

    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 = quantize_to_nvfp4(
        &out_proj_local,
        h,
        value_dim,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

    // Native FP8 SSM prefill GEMM (cross-port from qwen35_dense.rs,
    // 2026-05-20). Same conv-k SNR-collapse vulnerability as the dense
    // 27B: the MoE A3B's GDN config has identical asymmetric conv
    // weights (k-segment ~18× smaller than v-segment), so the triple-
    // quant FP8→BF16→NVFP4→BF16 chain attenuates direction in the
    // k-channel just as it did on dense. Bypass the NVFP4 intermediate
    // by installing a single-scale FP8 copy of `qkvz_dense` and
    // `ssm35.out_proj` and dispatching prefill through `fp8_gemm_n128`.
    // Unconditional for FP8-on-disk variants (mirrors dense).
    let (qkvz_fp8_prefill, out_proj_fp8_prefill) = if matches!(variant, Nvfp4Variant::Fp8Dequanted)
    {
        // Diagnostic: fires once per LinearAttention layer (~30
        // lines for 35B-A3B). Confirms the MoE Bug #1 cross-port
        // (commit 7d5e8fc) is active and the SSM prefill path
        // dispatches through fp8_gemm_n128, not w4a16_gemm.
        tracing::info!(
            "SSM[{lp}] in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128)"
        );
        let b2f_k = gpu.kernel("w4a16", "bf16_to_fp8")?;
        let qkvz_total = (qkvz_size * h) as u32;
        let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
        crate::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            qkvz_dense.weight,
            qkvz_fp8,
            qkvz_total,
            stream,
        )?;
        let out_total = (h * value_dim) as u32;
        let out_fp8 = gpu.alloc(h * value_dim)?;
        crate::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            out_proj_local.weight,
            out_fp8,
            out_total,
            stream,
        )?;
        gpu.synchronize(stream)?;
        (Some(qkvz_fp8), Some(out_fp8))
    } else {
        (None, None)
    };

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: conv1d_local,
        a_log: a_log_local,
        dt_bias: dt_bias_local,
        norm: norm_local,
        out_proj: out_proj_nvfp4,
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        Some(qkvz_nvfp4),
        Some(qkvz_nvfp4_t),
        Some(out_proj_nvfp4_t),
        config,
        gpu,
    )?;
    // Native-HIP (atlas_hip) lacks the FP8 *prefill* GEMM kernels
    // (fp8_gemm_n128 / fp8_gemm_t_blockscaled are inline-PTX, not yet
    // WMMA-ported). Skip the FP8→FP8 predequant AND the native-FP8 prefill
    // install so SSM qkvz/out_proj prefill falls to the NVFP4 w4a16 WMMA path
    // (qkvz_nvfp4* / out_proj_nvfp4_t fallbacks). SCALE/NVIDIA keep FP8 prefill.
    if !cfg!(atlas_hip) {
        layer.predequant_for_prefill(gpu, config, stream)?;
        // Install native FP8 prefill weights AFTER `predequant_for_prefill`
        // (which sets `out_proj_fp8` from NVFP4 + scale2). The FP8 path
        // overrides both pointers when active, routing prefill through
        // `fp8_gemm_n128` instead of `w4a16_gemm_t`. Decode batch paths
        // retain their NVFP4 fallback via the `qkvz_nvfp4*` fields above.
        if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
            layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
        }
    }
    // ATLAS_GDN_BF16_WEIGHTS=1 extension: also install BF16 out_proj so
    // the prefill dispatcher takes the dense_gemm BF16 path (highest
    // dispatch priority). Eliminates FP8/NVFP4 quant noise on out_proj
    // — the noise was previously amplified by post_attn_norm's RMSNorm
    // into wildly different gate inputs at the MoE block (cos=0.42 vs
    // HF). Test fix for long-context drift root cause (commit 1db7572
    // and onward investigation). ssm35.out_proj is the BF16 weight
    // (loaded via dense_auto with FP8→BF16 dequant).
    if matches!(
        std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref(),
        Some("1")
    ) {
        // out_proj_local weight is BF16 on GPU (from load_ssm_qwen35 →
        // dense_auto on Fp8Dequanted variant, sliced to this rank's value
        // heads). It's a separate buffer from out_proj_nvfp4 /
        // out_proj_fp8_prefill. Set as dense path.
        layer.out_proj_dense = Some(out_proj_local);
        tracing::info!(
            "SSM[{lp}] ATLAS_GDN_BF16_WEIGHTS: out_proj routed through BF16 dense_gemm (overrides FP8/NVFP4)"
        );
    }
    Ok(Box::new(layer))
}
