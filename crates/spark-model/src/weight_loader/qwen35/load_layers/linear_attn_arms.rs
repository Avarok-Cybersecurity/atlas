// SPDX-License-Identifier: AGPL-3.0-only
//
// Helper functions for the LinearAttention arms of `load_layers`. Two
// flavours: the native-FP8 path (block-scaled, w8a16 decode + prefill) and
// the standard NVFP4-quantized path.

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
    DenseWeight, Fp8Weight, Nvfp4Variant, QuantizedWeight, SsmWeights, WeightQuantFormat,
    dense_auto, dense_f32_safe, dense_keep_f32, gpu_concat_rows, interleave_ba,
    load_fp8_block_scaled_as_fp8weight, load_ssm_qwen35, quantize_to_nvfp4,
};

/// Native FP8 SSM build: keeps decode in block-scaled FP8 via `w8a16_gemv`,
/// and prefill in block-scaled FP8 via `w8a16_gemm`. No NVFP4 detour.
///
/// Disk format (Qwen3.5/3.6 FP8 release):
///   - `{p}.in_proj_qkv.weight`        : `[Nq, K]` FP8 E4M3
///   - `{p}.in_proj_qkv.weight_scale_inv`: `[Nq/BS, K/BS]` BF16, BS=128
///   - `{p}.in_proj_z.weight`          : `[Nz, K]` FP8 E4M3
///   - `{p}.in_proj_z.weight_scale_inv` : `[Nz/BS, K/BS]` BF16
///   - `{p}.out_proj.weight`           : `[H, V]` FP8 E4M3
///   - `{p}.out_proj.weight_scale_inv` : `[H/BS, V/BS]` BF16
///
/// Decode pipeline: concat `qkv` + `z` along the row (N) dim into a single
/// `[Nq+Nz, K]` FP8 buffer with a `[(Nq+Nz)/BS, K/BS]` BF16 scale buffer,
/// then `w8a16_gemv` consumes it directly. The scale concat copies
/// **block rows**, not raw F32 — that was the bug in the prior cut.
///
/// Load the SSM projection weights as block-scaled FP8 for the `w8a16_gemv`
/// (decode) / `w8a16_gemm` (batched decode) path: QKV and Z concatenated into
/// a single `[Nq+Nz, K]` FP8 buffer + matching `[(Nq+Nz)/BS, K/BS]` BF16 block
/// scales, plus the out_proj FP8 weight. Shared by the native-FP8 build and by
/// the decode-only FP8 overlay on the BF16 dense build (`ATLAS_HOLO_FP8_SSM_DECODE`).
fn load_ssm_fp8_decode_weights(
    layer_idx: usize,
    store: &WeightStore,
    p: &str,
    gpu: &dyn GpuBackend,
    h: usize,
) -> Result<(Fp8Weight, Fp8Weight)> {
    let qkv_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj_qkv"), gpu)?;
    let z_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj_z"), gpu)?;
    let out_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.out_proj"), gpu)?;

    qkv_fp8.scale_format.expect(
        WeightQuantFormat::Fp8BlockScaled,
        "load_ssm_fp8_decode_weights::qkv_fp8 from disk",
    );
    z_fp8.scale_format.expect(
        WeightQuantFormat::Fp8BlockScaled,
        "load_ssm_fp8_decode_weights::z_fp8 from disk",
    );
    out_fp8.scale_format.expect(
        WeightQuantFormat::Fp8BlockScaled,
        "load_ssm_fp8_decode_weights::out_fp8 from disk",
    );

    let qkv_rows = qkv_fp8.n as usize;
    let z_rows = z_fp8.n as usize;
    let qkvz_n = qkv_rows + z_rows;

    // Concat weight bytes along N: [Nq, K] || [Nz, K] → [Nq+Nz, K].
    let qkvz_weight_ptr = gpu.alloc(qkvz_n * h)?;
    gpu.copy_d2d(qkv_fp8.weight, qkvz_weight_ptr, qkv_rows * h)?;
    gpu.copy_d2d(
        z_fp8.weight,
        qkvz_weight_ptr.offset(qkv_rows * h),
        z_rows * h,
    )?;

    // Concat block scales along the N-block axis (BS=128, FP32).
    const BS: usize = 128;
    ensure!(
        qkv_rows.is_multiple_of(BS),
        "SSM L{layer_idx}: qkv_rows={qkv_rows} not divisible by BS={BS} (FP8 block size)",
    );
    ensure!(
        z_rows.is_multiple_of(BS),
        "SSM L{layer_idx}: z_rows={z_rows} not divisible by BS={BS} (FP8 block size)",
    );
    ensure!(
        h.is_multiple_of(BS),
        "SSM L{layer_idx}: hidden_size={h} not divisible by BS={BS}",
    );
    let scale_cols = h / BS;
    let scale_row_bytes = scale_cols * 4;
    let qkv_scale_rows = qkv_rows / BS;
    let z_scale_rows = z_rows / BS;
    let qkvz_scale_bytes = (qkv_scale_rows + z_scale_rows) * scale_row_bytes;
    let qkvz_scale_ptr = gpu.alloc(qkvz_scale_bytes)?;
    gpu.copy_d2d(
        qkv_fp8.row_scale,
        qkvz_scale_ptr,
        qkv_scale_rows * scale_row_bytes,
    )?;
    gpu.copy_d2d(
        z_fp8.row_scale,
        qkvz_scale_ptr.offset(qkv_scale_rows * scale_row_bytes),
        z_scale_rows * scale_row_bytes,
    )?;

    let qkvz_fp8 = Fp8Weight {
        weight: qkvz_weight_ptr,
        row_scale: qkvz_scale_ptr,
        n: qkvz_n as u32,
        k: h as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    Ok((qkvz_fp8, out_fp8))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_linear_attention_fp8(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    _variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    _stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    // GDN HeadParallel MVP: BF16 + NVFP4 only. Native block-scaled FP8 SSM
    // slicing (per-128-row block scales split on the head axis) is deferred —
    // slicing rows mid-block would corrupt the scale association. Fail loudly
    // rather than ship wrong FP8 scale slicing.
    ensure!(
        config.tp_world_size.max(1) == 1,
        "Native block-scaled FP8 SSM (linear_attn) supports TP=1 only (got tp={}); \
         GDN HeadParallel FP8 scale slicing is deferred. Use the NVFP4 decode path \
         (ATLAS_HOLO_FP4_PROJ_DECODE=1) or run --tp-size 1 for FP8.",
        config.tp_world_size,
    );

    let p = format!("{lp}.linear_attn");
    tracing::info!("Layer {layer_idx}: loading SSM FP8 native (block-scaled decode + prefill)");

    let (qkvz_fp8, out_fp8) = load_ssm_fp8_decode_weights(layer_idx, store, &p, gpu, h)?;
    tracing::info!(
        "Layer {layer_idx}: SSM QKVZ FP8 [{},{h}] block-scaled, out_proj FP8 [{},{}] block-scaled",
        qkvz_fp8.n,
        out_fp8.n,
        out_fp8.k
    );

    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let in_proj_a = dense_auto(store, &format!("{p}.in_proj_a.weight"), gpu)?;
    let in_proj_b = dense_auto(store, &format!("{p}.in_proj_b.weight"), gpu)?;
    let ba_dense = interleave_ba(
        &DenseWeight {
            weight: in_proj_a.weight,
        },
        &DenseWeight {
            weight: in_proj_b.weight,
        },
        nv,
        nk,
        h,
        gpu,
    )?;

    // ── 4. Wire into Qwen3SsmLayer.
    //       QKV/Z and out_proj stay in checkpoint FP8 form. The BF16 dense
    //       fields are dead fallback slots for this native path; keeping them
    //       null avoids materializing tens of GB of duplicate Holo weights.
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        },
        in_proj_ba: ba_dense,
        conv1d: dense_auto(store, &format!("{p}.conv1d.weight"), gpu)?,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?,
        out_proj: QuantizedWeight::null(),
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        None,
        None,
        None,
        config,
        gpu,
    )?;
    layer.set_fp8_decode_weights(Some(qkvz_fp8), Some(out_fp8));
    tracing::info!("Layer {layer_idx}: SSM native FP8 — w8a16 decode + prefill");
    Ok(Box::new(layer))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_linear_attention_dense_bf16(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    // GDN HeadParallel: `config` already holds per-rank-LOCAL linear head
    // counts (topology.rs divided them by tp_size). `TpGdnDims::from_config`
    // multiplies back up to the full pre-shard sizes the on-disk weights use;
    // the slicers below cut each rank's contiguous head range. For tp=1 every
    // slicer returns the source pointer untouched → byte-identical fast path.
    let tp_size = config.tp_world_size.max(1);
    let dims = TpGdnDims::from_config(config);
    tracing::info!(
        "Layer {layer_idx}: loading SSM FP8 projections as BF16 dense \
         (tp={tp_size}, local_nk={}, local_nv={})",
        dims.local_nk,
        dims.local_nv,
    );

    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant)?;

    // Concat FULL [Q|K|V] || [Z] (on-disk sizes) then SEGMENT-slice to this
    // rank's heads (Q/K/V/Z sliced independently, re-packed local — a naive
    // "first half of QKVZ" split is WRONG).
    let qkvz_full = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        dims.full_conv_dim(),
        &ssm35.in_proj_z,
        dims.full_value_dim(),
        h,
        gpu,
    )?;
    // `gpu_concat_rows` allocates an independent combined buffer (alloc +
    // copy_d2d), so the per-projection BF16 expansions of in_proj_qkv /
    // in_proj_z are dead after this point. They are freshly-allocated
    // FP8→BF16 dequant outputs (not WeightStore aliases), ~50 MB/layer ×
    // ~30 GDN layers ≈ 1.5 GB. Free them here — identical numerics.
    let _ = gpu.free(ssm35.in_proj_qkv.weight);
    let _ = gpu.free(ssm35.in_proj_z.weight);
    let (qkvz_ptr, _, _) = shard_gdn_qkvz_rows(qkvz_full.weight, &dims, gpu)?;
    if tp_size > 1 {
        let _ = gpu.free(qkvz_full.weight);
    }
    let qkvz_dense = DenseWeight { weight: qkvz_ptr };

    // BA: interleave FULL heads (per-group β/α) then slice to local heads —
    // the rank boundary always lands on a key-head group boundary.
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

    // conv1d (per-QKV-channel filter), a_log/dt_bias (per value head, FP32),
    // norm (per value_dim, BF16), out_proj (row-parallel on value_dim).
    let d_conv = config.linear_conv_kernel_dim;
    let (conv_ptr, _, _) = shard_gdn_conv_rows(ssm35.conv1d.weight, &dims, d_conv, gpu)?;
    let (a_log_ptr, _) = shard_gdn_value_vector(ssm35.a_log.weight, &dims, 1, 4, gpu)?;
    let (dt_bias_ptr, _) = shard_gdn_value_vector(ssm35.dt_bias.weight, &dims, 1, 4, gpu)?;
    // norm.weight is the gated-RMSNorm gain over the value HEAD-DIM ([vd]),
    // SHARED across all value heads — REPLICATE under HeadParallel. (a_log/dt_bias
    // above ARE per-head [nv] scalars, so they slice; slicing norm on the head
    // axis read past the [vd] buffer → cuMemcpyDtoDAsync INVALID_VALUE at load.)
    let norm_ptr = ssm35.norm.weight;
    let (out_proj_ptr, _, _) = shard_gdn_out_proj_row_parallel(ssm35.out_proj.weight, &dims, gpu)?;

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: DenseWeight { weight: conv_ptr },
        a_log: DenseWeight { weight: a_log_ptr },
        dt_bias: DenseWeight {
            weight: dt_bias_ptr,
        },
        norm: DenseWeight { weight: norm_ptr },
        out_proj: QuantizedWeight::null(),
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        None,
        None,
        None,
        config,
        gpu,
    )?;
    layer.out_proj_dense = Some(DenseWeight {
        weight: out_proj_ptr,
    });
    // Decode-only FP8 SSM overlay (ATLAS_HOLO_FP8_SSM_DECODE=1): install the
    // on-disk block-scaled FP8 QKVZ/out_proj so DECODE runs through
    // w8a16_gemv / w8a16_gemm (half the BF16 weight bandwidth — SSM weights
    // are the bulk of the per-step fixed decode cost), while PREFILL keeps the
    // stable BF16 dense path (sidesteps the native-FP8 FLA-prefill crash at
    // layer 36). Costs ~25 MB/GDN layer extra (BF16 kept for prefill).
    // The FP8 decode overlay loads FULL (unsliced) block-scaled FP8 weights;
    // its per-128-row scale slicing is deferred (same reason as the native-FP8
    // path). Skip under TP>1 so the sharded BF16 path stays correct.
    if tp_size == 1 && std::env::var("ATLAS_HOLO_FP8_SSM_DECODE").ok().as_deref() == Some("1") {
        let p = format!("{lp}.linear_attn");
        let (qkvz_fp8, out_fp8) = load_ssm_fp8_decode_weights(layer_idx, store, &p, gpu, h)?;
        layer.set_fp8_decode_weights(Some(qkvz_fp8), Some(out_fp8));
        tracing::info!("Layer {layer_idx}: SSM FP8 decode overlay installed (BF16 prefill kept)");
    }
    Ok(Box::new(layer))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_linear_attention_nvfp4(
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
