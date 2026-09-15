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
    DenseWeight, Exl3DenseFamily, Fp8Weight, Nvfp4Variant, QuantizedWeight, SsmWeights,
    WeightQuantFormat, dense_auto, dense_f32_safe, dense_keep_f32, exl3_dense_family_kept,
    gpu_concat_rows, interleave_ba, load_fp8_block_scaled_as_fp8weight, load_ssm_qwen35_parts,
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
pub(crate) fn build_linear_attention_dense_bf16(
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
    exl3_stage: Option<&std::sync::Arc<crate::layers::ops::Exl3DenseStage>>,
) -> Result<Box<dyn TransformerLayer>> {
    // GDN HeadParallel: `config` already holds per-rank-LOCAL linear head
    // counts (topology.rs divided them by tp_size). `TpGdnDims::from_config`
    // multiplies back up to the full pre-shard sizes the on-disk weights use;
    // the slicers below cut each rank's contiguous head range. For tp=1 every
    // slicer returns the source pointer untouched → byte-identical fast path.
    let tp_size = config.tp_world_size.max(1);
    let dims = TpGdnDims::from_config(config);

    // Native EXL3 GDN (ATLAS_EXL3_NATIVE_DENSE=1): re-derived PER LAYER from
    // the store, not just the env gates — the materialize pass keeps the
    // routed GDN leaves (`Exl3DenseFamily::Gdn.leaves()`: in_proj_qkv,
    // in_proj_z, out_proj) packed only as an atomic set, so "the .trellis
    // tensors are still here" is exactly "this layer was kept". A fallen-back
    // layer takes the BF16 arm below with zero special-casing. On the native
    // arm the in_proj pair is NOT loaded or concatenated (packed trellis
    // weights cannot be fused; the layer serves them as a shared-A pair into
    // the same fused [Q|K|V|Z] arena row) and out_proj is left NULL.
    let native_gdn = exl3_dense_family_kept(store, lp, Exl3DenseFamily::Gdn);
    if native_gdn {
        ensure!(
            tp_size == 1,
            "Layer {layer_idx}: EXL3 native GDN serves the unsharded packed trellis; \
             TP={tp_size} is not supported (qwen4_exp does not load under TP)"
        );
        ensure!(
            std::env::var("ATLAS_HOLO_FP8_SSM_DECODE").as_deref() != Ok("1"),
            "ATLAS_EXL3_NATIVE_DENSE=1 is incompatible with ATLAS_HOLO_FP8_SSM_DECODE=1 \
             (no block-scaled FP8 qkvz/out_proj exists in an EXL3 checkpoint); unset one"
        );
        tracing::info!(
            "Layer {layer_idx}: SSM in_proj_qkv/in_proj_z/out_proj kept as packed EXL3 \
             trellis (native dense arm, no fused QKVZ concat); BA/conv/gates BF16 as before"
        );
    } else {
        tracing::info!(
            "Layer {layer_idx}: loading SSM FP8 projections as BF16 dense \
             (tp={tp_size}, local_nk={}, local_nv={})",
            dims.local_nk,
            dims.local_nv,
        );
    }

    let ssm35 = load_ssm_qwen35_parts(store, lp, gpu, variant, !native_gdn, !native_gdn)?;

    let qkvz_dense = if native_gdn {
        // Packed pair served by the layer (`Exl3GdnWeights::in_proj_linear`);
        // the fused BF16 slot stays NULL — the FP8-native precedent.
        DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        }
    } else {
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
        DenseWeight { weight: qkvz_ptr }
    };

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
    let out_proj_ptr = if native_gdn {
        spark_runtime::gpu::DevicePtr::NULL
    } else {
        shard_gdn_out_proj_row_parallel(ssm35.out_proj.weight, &dims, gpu)?.0
    };

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
    if native_gdn {
        super::exl3_dense_arms::install_native_gdn(
            &mut layer, gpu, store, lp, layer_idx, h, &dims, exl3_stage,
        )?;
        return Ok(Box::new(layer));
    }
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

// The NVFP4-quantized arm (`build_linear_attention_nvfp4`) lives in the child
// module (≤500 LoC split); the re-export keeps callers' path unchanged.
#[path = "linear_attn_nvfp4_arm.rs"]
mod nvfp4_arm;
pub(crate) use nvfp4_arm::build_linear_attention_nvfp4;
