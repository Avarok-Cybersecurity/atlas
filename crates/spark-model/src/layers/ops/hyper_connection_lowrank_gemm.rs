// SPDX-License-Identifier: AGPL-3.0-only

//! The GEMM-formulated low-rank collapse — split from
//! `hyper_connection_lowrank.rs` (500-LoC cap).
//!
//! One body serves two regimes through the `use_cublas` switch: prefill
//! (large T, tensor-core `dense_gemm_bf16_pipelined`) and decode-shaped T
//! (cuBLASLt, where the tile GEMM wastes the machine at M<=64).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::qwen3_attention::HcLowRank;
use spark_runtime::kernel_args::KernelLaunch;

/// LARGE T (prefill): the down/up projections are GEMM-shaped and the fused
/// kernel ran them as hand-rolled FP32 warp loops at ~4% of the machine —
/// measured 45 ms/call, 47% of the whole prefill. Stage `normed` in BF16 and
/// hand both projections (and the tiny injection one) to the tensor-core
/// `dense_gemm_bf16_pipelined`, keeping only the elementwise seams custom.
/// Slabbed at <= 2048 tokens to bound the scratch region.
///
/// `ATLAS_QWEN4EXP_NO_HC_GEMM=1` falls back to the fused kernel (kill switch,
/// same convention as ATLAS_NO_GDN_FLA).
#[allow(clippy::too_many_arguments)]
/// `ATLAS_HC_FUSE_UP_MIX=1`: do the mix in the up-GEMM epilogue so `up_pre`
/// is never materialised. See `hc_pre_up_mix` in the model's
/// `hyper_connection.cu` for the stream-interleaved N-tile that makes one
/// output dim's four streams meet in a single thread's registers.
///
/// 161 MB written + 161 MB read per call at T=7841, ~30.8 GB a chunk across
/// the 96 prefill sites, to hand one kernel's output to the next.
fn hc_fuse_up_mix() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_HC_FUSE_UP_MIX").as_deref() == Ok("1"))
}

/// `ATLAS_HC_FUSE_DOWN_INJ=1`: append the injection projection to the down
/// GEMM as extra output columns instead of launching it separately. See
/// `hc_down_inj` in the model's `hyper_connection.cu`.
///
/// The injection GEMM is N=4 on a 128-wide N-tile: 62 CTAs each streaming a
/// 2.6 MB A-tile with 96.9% of the MMA masked, ~15.4 GB a chunk across the 96
/// prefill sites for four numbers per token. Folding it into the down GEMM's
/// half-empty third column tile is free — ceil(324/128) == ceil(320/128).
fn hc_fuse_down_inj() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_HC_FUSE_DOWN_INJ").as_deref() == Ok("1"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_gemm(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    use_cublas: bool,
    row_exact: bool,
    stream: u64,
) -> Result<()> {
    hc_pre_gemm_fused(
        gpu,
        streams,
        w,
        y_out,
        inj_out,
        scratch,
        num_tokens,
        hidden_size,
        hc_mult,
        norm_eps,
        inject,
        use_cublas,
        row_exact,
        hc_fuse_up_mix(),
        hc_fuse_down_inj(),
        stream,
    )
}

/// `hc_pre_gemm` with the fusion arm passed EXPLICITLY rather than read from
/// the environment. The env reader is a `OnceLock`, so a single test process
/// can only ever observe one arm through it — and the whole case for this
/// change is that the two arms are bit-identical, which takes both in one
/// process to assert. Production goes through the wrapper above.
pub(crate) fn hc_pre_gemm_fused(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    use_cublas: bool,
    row_exact: bool,
    fuse_up_mix: bool,
    fuse_down_inj: bool,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !inject || !w.inject_w.is_null(),
        "HC injection requires block_inject_weight"
    );
    anyhow::ensure!(
        !row_exact || use_cublas,
        "row-exact HC requires decode cuBLAS projections"
    );
    // SSOT with `hc_lowrank_scratch` sizing — a mismatch writes past the arena.
    // Also the launch multiplier: an 8192 chunk at slab 2048 is four passes of
    // every slabbed kernel here. ATLAS_HC_GEMM_SLAB overrides both sides.
    let slab: u32 = spark_runtime::buffers::hc_gemm_slab() as u32;
    #[allow(non_snake_case)]
    let SLAB: u32 = slab;
    let hc_dim = (hc_mult * hidden_size) as usize;
    let rank = w.rank as u32;
    // Scratch layout (BF16): normed [L, hc_dim], up_pre [L, hc_dim],
    // low [L, rank], inj_pre [L, hc], where L = min(T, 2048). sizes.rs sizes
    // the region with m.min(2048) and T <= m always, so L-based offsets fit
    // even when the arena was sized for fewer than 2048 tokens.
    let lay = num_tokens.min(SLAB) as usize;
    let normed = scratch;
    let up_pre = scratch.offset(lay * hc_dim * 2);
    let low = scratch.offset(2 * lay * hc_dim * 2);
    let inj_pre = scratch.offset(2 * lay * hc_dim * 2 + lay * w.rank * 2);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage_bf16")?;
    let k_silu = gpu.kernel("hyper_connection", "hc_silu_scale")?;
    let k_mix = gpu.kernel("hyper_connection", "hc_pre_mix")?;
    let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    // try_kernel, not kernel: a target without the fused pair degrades to the
    // stock arm instead of refusing to serve.
    let k_up_mix = crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_up_mix");
    let k_inj_gate = crate::layers::try_kernel(gpu, "hyper_connection", "hc_inj_gate");
    // Every conjunct is load-bearing:
    //   !use_cublas  - excludes BOTH decode entries (T<=64 and the row-exact
    //                  batched verify) and ATLAS_HC_PREFILL_CUBLAS.
    //   inject       - excludes `hc_head_lowrank`, which is the model's FINAL
    //                  NORM (no `model.norm.weight` in the checkpoint). Worth
    //                  1/97th of the win and removes a wrong-logits failure
    //                  mode from the first commit.
    //   hc_mult == 4 - pins the 4x32=128 N-tile identity the epilogue relies on.
    //   hidden % 32  - grid.x is hidden/32, and a partial 32-dim group would
    //                  read up_w rows belonging to the next stream.
    let fuse = fuse_up_mix
        && !use_cublas
        && inject
        && hc_mult == 4
        && hidden_size.is_multiple_of(32)
        && k_up_mix.0 != 0
        && k_inj_gate.0 != 0;
    if fuse {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(
                hidden_size,
                hc_mult,
                rank,
                "hc_pre arm: FUSED up-GEMM+mix (up_pre not materialised)"
            )
        });
    }
    // Same fail-soft shape as the pair above: a target without the fused
    // down+inject kernel degrades to two launches instead of refusing to serve.
    let k_down_inj = crate::layers::try_kernel(gpu, "hyper_connection", "hc_down_inj");
    //   !use_cublas - excludes BOTH decode entries, as above.
    //   inject      - the whole point is folding the injection rows in; with
    //                 no injection there is nothing to fold and `inject_w` is
    //                 NULL (`hc_head_lowrank`).
    // No hc_mult/hidden constraint: unlike `hc_pre_up_mix` this kernel keeps
    // the stock B-row map, so it is shape-generic in N0 and NI.
    let fuse_di = fuse_down_inj && !use_cublas && inject && k_down_inj.0 != 0;
    if fuse_di {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(
                rank,
                hc_mult,
                n_total = rank + hc_mult,
                "hc_pre arm: FUSED down+inject GEMM (separate inject launch removed)"
            )
        });
    }
    let inv_hc = 1.0f32 / hc_mult as f32;

    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = SLAB.min(num_tokens - t0);
        let streams_s = streams.offset(t0 as usize * hc_dim * 4);

        KernelLaunch::new(gpu, k_stage)
            .grid([ts, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams_s)
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;

        // low_pre = normed x down_w^T   [ts, rank]
        // Under `fuse_di` this same launch also writes `inj_pre`, carried as
        // output columns [rank, rank+hc) of one N=324 GEMM.
        if fuse_di {
            KernelLaunch::new(gpu, k_down_inj)
                .grid([(rank + hc_mult).div_ceil(128), ts.div_ceil(128), 1])
                .block([256, 1, 1])
                .arg_ptr(normed)
                .arg_ptr(w.down_w)
                .arg_ptr(w.inject_w)
                .arg_ptr(low)
                .arg_ptr(inj_pre)
                .arg_u32(ts)
                .arg_u32(rank)
                .arg_u32(hc_mult)
                .arg_u32(hc_dim as u32)
                .launch(stream)?;
        } else if use_cublas {
            project_rows(
                gpu,
                normed,
                w.down_w,
                low,
                ts,
                rank,
                hc_dim as u32,
                row_exact,
                stream,
            )?;
        } else {
            gemm_raw(
                gpu,
                k_gemm,
                normed,
                w.down_w,
                low,
                ts,
                rank,
                hc_dim as u32,
                stream,
            )?;
        }
        let n_low = ts * rank;
        KernelLaunch::new(gpu, k_silu)
            .grid([n_low.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(low)
            .arg_u32(n_low)
            .arg_f32(inv_hc)
            .launch(stream)?;

        // up_pre = low x up_w^T   [ts, hc_dim]  — SKIPPED under `fuse`, which
        // folds this GEMM and the mix below into one launch.
        if fuse {
            KernelLaunch::new(gpu, k_up_mix)
                .grid([hidden_size / 32, ts.div_ceil(128), 1])
                .block([256, 1, 1])
                .arg_ptr(low)
                .arg_ptr(w.up_w)
                .arg_ptr(normed)
                .arg_ptr(y_out.offset(t0 as usize * hidden_size as usize * 2))
                .arg_u32(ts)
                .arg_u32(hidden_size)
                .arg_u32(rank)
                .arg_f32(inv_hc)
                .launch(stream)?;
        } else if use_cublas {
            project_rows(
                gpu,
                low,
                w.up_w,
                up_pre,
                ts,
                hc_dim as u32,
                rank,
                row_exact,
                stream,
            )?;
        } else {
            gemm_raw(
                gpu,
                k_gemm,
                low,
                w.up_w,
                up_pre,
                ts,
                hc_dim as u32,
                rank,
                stream,
            )?;
        }
        if inject && !fuse_di {
            // inj_pre = normed x inject_w^T   [ts, hc]  (already done above
            // when `fuse_di`, as the tail columns of the down GEMM).
            if use_cublas {
                project_rows(
                    gpu,
                    normed,
                    w.inject_w,
                    inj_pre,
                    ts,
                    hc_mult,
                    hc_dim as u32,
                    row_exact,
                    stream,
                )?;
            } else {
                gemm_raw(
                    gpu,
                    k_gemm,
                    normed,
                    w.inject_w,
                    inj_pre,
                    ts,
                    hc_mult,
                    hc_dim as u32,
                    stream,
                )?;
            }
        }

        if fuse {
            // The mix already happened in the epilogue; only the injection
            // tail of `hc_pre_mix` is left, carried verbatim by `hc_inj_gate`.
            KernelLaunch::new(gpu, k_inj_gate)
                .grid([ts, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(inj_pre)
                .arg_ptr(inj_out.offset(t0 as usize * hc_mult as usize * 4))
                .arg_u32(hc_mult)
                .arg_f32(inv_hc)
                .launch(stream)?;
            t0 += ts;
            continue;
        }

        KernelLaunch::new(gpu, k_mix)
            .grid([ts, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(up_pre)
            .arg_ptr(if inject { inj_pre } else { DevicePtr::NULL })
            .arg_ptr(y_out.offset(t0 as usize * hidden_size as usize * 2))
            .arg_ptr(inj_out.offset(t0 as usize * hc_mult as usize * 4))
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(inv_hc)
            .launch(stream)?;

        t0 += ts;
    }
    Ok(())
}

/// `ATLAS_HC_DENSE_GEMV=1` (presence) routes decode-shaped rows through the
/// batched dense GEMV instead of cuBLASLt. OPT-IN, default OFF — measured
/// 2026-09-05 on qwen3.8-flash-next EXL3 (GB10, 2 drafts, prefix cache on,
/// fresh server per arm): the GEMV arm was faster per kernel but draft
/// acceptance fell 1.47 → 1.37 per step and decode 30.36 → 29.05 tok/s;
/// serial 23.48 → 23.70 (noise). Serial, verify and the MTP draft module
/// all took the same row-invariant kernel, so this is not a serial/verify
/// mismatch — the dense GEMV's numerics themselves cost draft agreement.
/// Kept as an A/B arm; records in .research/exl3_decode_perf/ab_hc_*.
fn hc_dense_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_HC_DENSE_GEMV").is_some())
}

/// Opt-in arm (see [`hc_dense_gemv_enabled`]): decode-shaped rows
/// (`m <= 8`) through `dense_gemv_bf16_batchm` — one pass over the `[n, k]`
/// BF16 weight for all rows, bit-identical per row to the M=1 kernel (fixed
/// K order, `--fmad=false`), one launch and no reduce kernel. Default: the
/// cuBLASLt path (a kernel per M with split-K + reduce; under the row-exact
/// contract one M=1 call per row).
#[allow(clippy::too_many_arguments)]
fn project_rows(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    row_exact: bool,
    stream: u64,
) -> Result<()> {
    if m <= 8 && k.is_multiple_of(8) && n.is_multiple_of(4) && hc_dense_gemv_enabled() {
        let kernel = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
        let dw = crate::weight_map::DenseWeight { weight: w };
        return crate::layers::ops::dense_gemv_batchm(gpu, kernel, a, &dw, out, m, n, k, n, stream);
    }
    let batch = if row_exact { 1 } else { m };
    for row in (0..m).step_by(batch as usize) {
        crate::layers::ops::cublas_bf16_proj_dense(
            a.offset(row as usize * k as usize * 2),
            w,
            out.offset(row as usize * n as usize * 2),
            batch,
            n,
            k,
            stream,
        )?;
    }
    Ok(())
}

fn gemm_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
