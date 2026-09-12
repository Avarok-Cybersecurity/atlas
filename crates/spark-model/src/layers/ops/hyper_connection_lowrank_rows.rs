// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Qwen3.8-Flash-Next low-rank mHC: the decode-rows arm (T <= 8), split out of
//! `hyper_connection_lowrank.rs` under the 500-line cap. Kernels: `hc_dec_up` /
//! `hc_dec_down` / `hc_pre_stage` / `hc_post` in `kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank::hc_variant_down;
use crate::layers::qwen3_attention::HcLowRank;

/// Decode-shaped (T <= 8) collapse that reads every low-rank weight row ONCE
/// per site with 16-byte lane loads and applies it to all T tokens
/// (`hc_dec_down` / `hc_dec_up`, see the kernel file). ON by default;
/// `ATLAS_HC_DECODE_ROWS=0` restores the cuBLASLt arm (the A/B and rollback
/// switch). The shape contract below falls back to the existing arms for
/// anything it does not cover.
/// `ATLAS_HC_PRE_CHUNK=1` — OPT-IN: chunk T > [`HC_DEC_MAX_T`] onto the
/// decode-rows arm instead of the cuBLASLt GEMM decomposition.
///
/// DEFAULT OFF on a SPLIT result. Same binary, NVFP4 EP=2, 5 reps per arm:
///
///     C=1   51.88 chunked vs 49.21   +5.4%  (ranges do not overlap)
///     C=2   62.13         vs 62.30   -0.3%
///     C=4   64.62         vs 68.69   -5.9%
///
/// It helps C=1 and hurts C=4, and NEITHER theory for it survives the data.
/// "Re-reading the ~13 MB low-rank weights per chunk costs more than the GEMM
/// wastes" predicts damage scaling with chunk COUNT — but the verify's 11 rows
/// (2 chunks) is where it hurts, while the 26-row path (4 chunks) is the only
/// place it can be helping C=1, since a C=1 verify is 3 rows and never chunks
/// at all. C=2 should then look like C=1 and does not.
///
/// A width threshold fitted to two contradictory points is a guess wearing a
/// rule's clothes, so this ships off with the measurement recorded instead.
/// What would actually settle it: identify the 26-row path (unaccounted work
/// the hc engagement log named, `num_tokens=26`, present even at C=1), and
/// time hc_pre per width directly rather than inferring from end-to-end.
pub(crate) fn hc_pre_chunk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_HC_PRE_CHUNK").as_deref() == Ok("1"))
}

pub(crate) fn hc_decode_rows_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_HC_DECODE_ROWS").as_deref() != Ok("0"))
}

/// Maximum row count the decode-rows arm handles in one launch pair
/// (`QHC_DEC_MAX_T` in the kernel file).
pub(crate) const HC_DEC_MAX_T: u32 = 8;
/// `hc_dec_down`: warps per weight row (kernel `QHC_DOWN_SPLIT`).
const HC_DOWN_SPLIT: u32 = 4;
/// `hc_dec_up`: hidden columns per block (kernel `QHC_UP_D_PER_BLOCK`).
const HC_UP_D_PER_BLOCK: u32 = 8;

/// The decode-rows arm's shape contract (mirrors the kernel-file comment):
/// `hc*H % 256 == 0` (256 elements per warp step), `H % 16 == 0` (16 outputs
/// per block), `rank % 32 == 0` (four 16-byte-aligned quarter rows),
/// `hc <= 8` (block = hc * 64 <= 512 threads), `1 <= T <= 8`.
pub(crate) fn hc_decode_rows_shape_ok(
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    rank: u32,
) -> bool {
    let hc_dim = hc_mult * hidden_size;
    (1..=HC_DEC_MAX_T).contains(&num_tokens)
        && hc_dim.is_multiple_of(HC_DOWN_SPLIT * 256)
        && hidden_size.is_multiple_of(HC_UP_D_PER_BLOCK)
        && rank.is_multiple_of(64)
        && rank <= 512
        && (1..=8).contains(&hc_mult)
}

/// The decode-rows collapse: `hc_pre_stage` (existing) + `hc_dec_down` +
/// `hc_dec_up`. Same math and the same FP32 `normed` as the split arm; the
/// parity probe holds it to the split arm's tight bound.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_rows(
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
    stream: u64,
) -> Result<()> {
    let rank = w.rank as u32;
    anyhow::ensure!(
        hc_decode_rows_shape_ok(num_tokens, hidden_size, hc_mult, rank),
        "hc_pre_rows: shape outside the decode-rows contract (T={num_tokens} H={hidden_size} hc={hc_mult} rank={rank})"
    );
    let hc_dim = hc_mult * hidden_size;
    // Scratch: normed F32 [T, hc_dim] at offset 0, then low F32 [T, rank]
    // IMMEDIATELY after the T rows actually staged. NOT the split arm's fixed
    // `64 * hc_dim * 4` offset: `sizes.rs` sizes this region with
    // `t = m.min(64)`, so an arena whose token capacity `m` is below 64 (the
    // MTP draft module's private arena is sized for a few draft rows) is
    // SMALLER than that offset, and a fixed offset writes `low` past the end
    // of the region into the next live buffer. T <= 8 rows at T*hc_dim*4
    // bytes stay inside the region for every arena with capacity >= T.
    let normed = scratch;
    let low = scratch.offset(num_tokens as usize * hc_dim as usize * 4);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage")?;
    let k_down = gpu.kernel("hyper_connection", "hc_dec_down")?;
    let k_up = gpu.kernel("hyper_connection", "hc_dec_up")?;

    KernelLaunch::new(gpu, k_stage)
        .grid([num_tokens, hc_mult, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(normed)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)?;

    // HC_DOWN_SPLIT warps per weight row (a contiguous slice each, partials
    // summed in a fixed order through shared memory): rank rows plus (when
    // injecting) hc inject rows, 8 warps per block.
    let rows = rank + if inject { hc_mult } else { 0 };
    // ATLAS_HC_DOWN_KERNEL / ATLAS_HC_UP_KERNEL (2026-09-09): variant names
    // under measurement; unset = the kernels above.
    // hc_dec_down_v5 (two rows per warp, byte-identical to hc_dec_down, ~6 us
    // faster per site DRAM-streaming) is the default; ATLAS_HC_DOWN_KERNEL=
    // hc_dec_down restores the one-row form for A/B.
    let (k_down, down_grid) = match hc_variant_down() {
        "hc_dec_down" => (k_down, rows.div_ceil(8 / HC_DOWN_SPLIT)),
        _ => (
            gpu.kernel("hyper_connection", "hc_dec_down_v5")?,
            rows.div_ceil(4),
        ),
    };
    KernelLaunch::new(gpu, k_down)
        .grid([down_grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down_w)
        .arg_ptr(if inject { w.inject_w } else { DevicePtr::NULL })
        .arg_ptr(low)
        .arg_ptr(inj_out)
        .arg_u32(num_tokens)
        .arg_u32(hc_dim)
        .arg_u32(hc_mult)
        .arg_u32(rank)
        .launch(stream)?;

    // Eight lanes per (stream, d) row (one contiguous 128-byte segment per
    // load instruction), four rows per warp, HC_UP_D_PER_BLOCK outputs per
    // block (block = hc*64); chunk partials reduce by shuffle, the stream
    // mean in smem.
    let (up_grid, up_block, smem) = (
        hidden_size / HC_UP_D_PER_BLOCK,
        hc_mult * 64,
        (HC_DEC_MAX_T * rank + hc_mult * HC_UP_D_PER_BLOCK * HC_DEC_MAX_T) * 4,
    );
    KernelLaunch::new(gpu, k_up)
        .grid([up_grid, 1, 1])
        .block([up_block, 1, 1])
        .shared_mem(smem)
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up_w)
        .arg_ptr(y_out)
        .arg_u32(num_tokens)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(rank)
        .launch(stream)
}
