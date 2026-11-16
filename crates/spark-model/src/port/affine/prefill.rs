// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/w4a8.zig:1-55 (the quantize-then-prompt order, the scratch layout and the row-band formula).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust over `AffineWeight` and `W4Scratch`; the exceptional verdict is
// a device-side gate (no host readback per projection, graph-safe) with a fallback counter read at the caller's
// own sync points, instead of upstream's synchronous flag read; shared-input grids; the fused down projection.

//! Responsibility: the affine 4-bit prompt GEMMs (M > `max_decode_rows` rows) for [`AffineWeight`] on Ada,
//! with BF16 activations or W4A8 int8 activations, and the dense FFN's fused SwiGLU + group sums + int8
//! down-projection input (lane L17b; kernels `kernels/ada/common/affine_prefill_ada.cu` and
//! `affine_swiglu_sums_ada.cu`, launchers in [`launch`], CPU oracles in [`oracle`]).
//!
//! [`run`] is what `AffineLinear::prefill` (L17a, `W4Linear::prefill`) delegates to, with
//! `AffineWeight::int8_prefill` (the `affine4_prefill_int8` row, L23a, through `W4Opts`). `Ok(false)` means
//! DECLINED and nothing was launched: the weight is not a 4-bit group-64 matrix, its shape is not servable, or
//! the Ada modules are absent; the caller then dequantizes to BF16 and runs cuBLASLt as for any W4 format. An
//! exceptional activation group (a non-finite input, an overflowing group sum) is NOT a decline: the quantizer
//! raises a device flag, the int8 GEMM skips and the BF16 prompt GEMM runs on the same input, all on the device,
//! so a call never syncs and is safe inside a captured graph; a NaN input still reaches the output as it would on
//! the BF16 route. [`take_fallbacks`] reads and clears the device count of such fallbacks where the caller syncs
//! anyway and records each one as a declined prefill (`port::w4::note_prefill_decline`, the source of L23b's
//! `atlas_affine4_prefill_int8_fallbacks_total`); [`int8_fallbacks_total`] is this file's own process total.
//! [`harvest_fallbacks`] is the serve path's call: one line after the prefill chunk's metadata sync
//! (`model/trait_impl/prefill_b.rs`), a no-op until an int8 prompt GEMM has been prepared in this process.
//!
//! [`ffn::ffn_int8`] serves the dense FFN's whole prompt path on the int8 route (gate + up in one `_many`
//! grid over one quantization of x, then the fused SwiGLU + sums + quantizer and the down projection), byte for
//! byte what the per-projection path gives.
//!
//! Scratch (`W4Scratch`, L21's `port::w4::scratch`; affine decode uses only `ws`): `xq` holds at least
//! [`xq_bytes`]`(m, k)` and `xs` at least [`xs_bytes`]`(m, k)` bytes, which L21's sizes cover (`m * k` and one
//! `f32` per 16 inputs against the one `f32` per 32 used here). Both regions are shared by every W4 consumer of
//! the backend, so nothing that must outlive one call lives in them: the two flag words (word 0: the current
//! call's exceptional mark, word 1: the fallback counter) are this file's own backend-scoped region
//! ([`FLAGS_TAG`] in the op cache, zeroed when created). [`prepare`] resolves the kernels and creates it at load
//! (L23a's route, request R1), because a prompt GEMM may run inside a captured graph (a batched verify wider than
//! the decode window) and creating the region allocates; without `prepare` the first prompt GEMM creates it. A
//! null `xq` or `xs` turns an int8 request into the BF16 GEMM (logged once per backend).
//!
//! Every path here is reached only through an affine weight, which exists only when a route row is on, and only
//! on Ada (the modules exist only in `kernels/ada`), so no other target and no row-off Ada serve changes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::weight::AffineWeight;
use crate::port::w4::W4Scratch;

/// The dense FFN's whole prompt path on the int8 route (gate + up, fused SwiGLU, down).
#[path = "prefill_ffn.rs"]
pub mod ffn;
/// Kernel handles and one launcher per entry point.
#[path = "prefill_launch.rs"]
pub mod launch;
/// CPU oracles: bit-exact models of the int8 path, float64 references and bounds for both GEMMs.
#[path = "prefill_oracle.rs"]
pub mod oracle;

#[cfg(test)]
#[path = "prefill_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "prefill_tests_mock.rs"]
mod tests_mock;

pub use ffn::{FfnIo, ffn_int8};
pub use launch::{
    ENTRIES, Entry, gemm_bf16, gemm_bf16_many, gemm_int8, gemm_int8_many, kernels_present,
    quantize_int8_rows, swiglu_quantize,
};

/// Inputs a group: the tiled layout's only group size on these kernels.
pub const GROUP: usize = 64;
/// CTA tile (both GEMMs): 128 rows x 128 columns on 2 x 4 warps.
pub const TILE_M: usize = 128;
pub const TILE_N: usize = 128;
pub const THREADS: u32 = 256;
/// Activation kernels: four warps a CTA, one 64-input group a warp.
pub const ACT_THREADS: u32 = 128;
pub const ACT_GROUPS_PER_CTA: usize = 4;
/// `Tile<128, 128, 2, 4, 3>::SMEM`: 3 stages of 128 x 64 BF16 inputs, 128 x 32 weight bytes, 2 x 128 bf16.
pub const BF16_SMEM_BYTES: u32 =
    (3 * (TILE_M * GROUP * 2 + TILE_N * GROUP / 2 + 2 * TILE_N * 2)) as u32;
/// `Tile8<128, 128, 2, 4, 4>::SMEM`: 4 stages of 128 x 64 int8 inputs, their FP32 factors, the weights.
pub const INT8_SMEM_BYTES: u32 =
    (4 * (TILE_M * GROUP + TILE_M * 8 + TILE_N * GROUP / 2 + 2 * TILE_N * 2)) as u32;
/// The op-cache tag of the flags region: word 0 the current call's exceptional mark, word 1 the fallback counter.
pub const FLAGS_TAG: &str = "port.affine.prefill.flags";
const FLAGS_ZEROED: &str = "port.affine.prefill.flags.zeroed";
/// Bytes of the flags region (two words used, one cache line reserved).
pub const FLAG_BYTES: usize = 256;
/// Projections a `_many` grid.
pub const MAX_PARTS: usize = 4;
/// `sizeof(Part)` and `sizeof(Parts)` in affine_prefill_ada.cu (static_asserts there).
pub const PART_BYTES: usize = 48;
pub const PARTS_BYTES: usize = 200;
/// The weight tiles one row band keeps hot in L2: 12 MiB (w4a8.zig, `band`).
pub const BAND_BYTES: usize = 12 << 20;

static INT8_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// Set when a flags region is first created in this process: until then [`harvest_fallbacks`] touches nothing.
static ARMED: AtomicBool = AtomicBool::new(false);

/// Projections whose int8 prompt GEMM fell back to BF16, as read by [`take_fallbacks`] so far (process total).
pub fn int8_fallbacks_total() -> u64 {
    INT8_FALLBACKS.load(Ordering::Relaxed)
}

/// Device view of a packed 4-bit group-64 affine matrix (the layout of `affine_pack_ada.cu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packed {
    pub words: DevicePtr,
    pub scales: DevicePtr,
    pub biases: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub npad: usize,
}

impl Packed {
    /// The view of `w`, or `None` where these kernels do not serve it (bits other than 4, group other than 64,
    /// a shape [`Packed::servable`] refuses).
    pub fn of(w: &AffineWeight) -> Option<Self> {
        let p = Self {
            words: w.words,
            scales: w.scales,
            biases: w.biases,
            n: w.n,
            k: w.k,
            npad: w.npad,
        };
        (w.bits == 4 && usize::from(w.group) == GROUP && p.servable()).then_some(p)
    }

    /// 1 <= N, K < 2^31, K % 64 == 0, npad == N rounded up to 128, 16-byte aligned buffers.
    pub fn servable(&self) -> bool {
        let fits = |v: usize| v > 0 && v <= i32::MAX as usize;
        fits(self.n)
            && fits(self.k)
            && self.k.is_multiple_of(GROUP)
            && self.npad == self.n.div_ceil(TILE_N) * TILE_N
            && fits(self.npad)
            && [self.words, self.scales, self.biases]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(16))
    }
}

pub(crate) fn i32_of(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).with_context(|| format!("affine prefill: {what} = {v} does not fit an i32"))
}

/// Bytes of `W4Scratch::xq` for `m` rows of `k` inputs: one int8 code an input.
pub fn xq_bytes(m: usize, k: usize) -> usize {
    m * k
}

/// Bytes of `W4Scratch::xs` for `m` rows of `k` inputs: FP32 scales, then FP32 sums, one each a group.
pub fn xs_bytes(m: usize, k: usize) -> usize {
    2 * m * (k / GROUP) * 4
}

/// This backend's flags region, created (and zeroed on `stream`) on first use. Creation allocates, so it must
/// not first happen inside a graph capture: [`prepare`] creates it at load.
pub fn flags(gpu: &dyn GpuBackend, stream: u64) -> Result<DevicePtr> {
    let p = gpu.op_cache().scratch(gpu, FLAGS_TAG, FLAG_BYTES)?;
    if gpu.op_cache().once(FLAGS_ZEROED) {
        gpu.memset_async(p, 0, FLAG_BYTES, stream)?;
        ARMED.store(true, Ordering::Relaxed);
    }
    Ok(p)
}

/// Load-time preparation (L23a's affine route calls it when `affine4_prefill_int8` is on, before the KV cache
/// is sized and before any graph capture): resolve every entry, so the lookups sit on the eager build path the
/// boot kernel audit sees, and create the flags region, so no prompt GEMM (a wide verify included) allocates
/// later. False where the Ada modules are absent (nothing is allocated then).
pub fn prepare(gpu: &dyn GpuBackend, stream: u64) -> Result<bool> {
    if !kernels_present(gpu) {
        return Ok(false);
    }
    flags(gpu, stream)?;
    Ok(true)
}

/// The int8 activation buffers carved from a scratch for `m` rows of `k` inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8Act {
    /// `[m, k]` int8 codes in mma slot order.
    pub xq: DevicePtr,
    /// `[m, k / 64]` FP32 group scales.
    pub xa: DevicePtr,
    /// `[m, k / 64]` FP32 group sums.
    pub xs: DevicePtr,
    /// The two flag words ([`flags`]).
    pub flags: DevicePtr,
}

impl Int8Act {
    /// `None` when the scratch has no int8 activation buffers or a buffer is not 16-byte aligned.
    pub fn carve(s: &W4Scratch, m: usize, k: usize, flags: DevicePtr) -> Option<Self> {
        let ok = |p: DevicePtr| !p.is_null() && p.0.is_multiple_of(16);
        if !ok(s.xq) || !ok(s.xs) || flags.is_null() {
            return None;
        }
        Some(Self {
            xq: s.xq,
            xa: s.xs,
            xs: s.xs.offset(m * (k / GROUP) * 4),
            flags,
        })
    }
}

/// Row tiles a band: as many as keep a band's weight tiles within [`BAND_BYTES`], at least 1, at most all.
pub fn row_band(m: usize, k: usize) -> usize {
    let rows_t = m.div_ceil(TILE_M).max(1);
    (BAND_BYTES / (TILE_M * k.max(1) * 2)).min(rows_t).max(1)
}

/// CTAs of one projection's grid.
pub fn grid_blocks(m: usize, n: usize) -> usize {
    m.div_ceil(TILE_M) * n.div_ceil(TILE_N)
}

/// The `Parts` struct of a `_many` grid (the C layout of affine_prefill_ada.cu) and the grid's CTA count.
pub fn encode_parts(
    members: &[(Packed, DevicePtr)],
    m: usize,
) -> Result<([u8; PARTS_BYTES], usize)> {
    ensure!(
        (1..=MAX_PARTS).contains(&members.len()),
        "affine prompt GEMM: {} projections a grid (1..={MAX_PARTS})",
        members.len()
    );
    let k = members[0].0.k;
    let mut b = [0u8; PARTS_BYTES];
    let mut first = 0usize;
    for (i, (w, out)) in members.iter().enumerate() {
        ensure!(
            w.servable() && w.k == k && out.0.is_multiple_of(4),
            "affine prompt GEMM part {i}: {w:?} (k must be {k}, out 4-byte aligned)"
        );
        let at = i * PART_BYTES;
        for (j, p) in [w.words, w.scales, w.biases, *out].into_iter().enumerate() {
            b[at + 8 * j..at + 8 * j + 8].copy_from_slice(&p.0.to_le_bytes());
        }
        for (j, v) in [w.n, w.npad, first].into_iter().enumerate() {
            b[at + 32 + 4 * j..at + 36 + 4 * j]
                .copy_from_slice(&i32_of(v, "part field")?.to_le_bytes());
        }
        first += grid_blocks(m, w.n);
    }
    let count = i32_of(members.len(), "count")?;
    b[MAX_PARTS * PART_BYTES..MAX_PARTS * PART_BYTES + 4].copy_from_slice(&count.to_le_bytes());
    Ok((b, first))
}

/// The prompt GEMM `out[m, n] = x[m, k] . W^T` of one affine weight (`W4Linear::prefill`). `Ok(false)`:
/// declined, nothing launched. `int8`: the W4A8 route with its device-side BF16 fallback.
#[allow(clippy::too_many_arguments)]
pub fn run(
    weight: &AffineWeight,
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    m: usize,
    out: DevicePtr,
    s: &W4Scratch,
    int8: bool,
    stream: u64,
) -> Result<bool> {
    match Packed::of(weight) {
        Some(w) => run_packed(&[(w, out)], gpu, x, m, s, int8, stream),
        None => Ok(false),
    }
}

/// Up to four affine projections that share `x` (gate + up, q + k + v) with one quantization and one grid. The
/// bits of every output equal [`run`]'s for that projection alone. Declines unless every member is servable
/// with the same K.
pub fn run_shared(
    members: &[(&AffineWeight, DevicePtr)],
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    m: usize,
    s: &W4Scratch,
    int8: bool,
    stream: u64,
) -> Result<bool> {
    let packed: Option<Vec<(Packed, DevicePtr)>> = members
        .iter()
        .map(|(w, o)| Packed::of(w).map(|p| (p, *o)))
        .collect();
    match packed {
        Some(p)
            if !p.is_empty() && p.len() <= MAX_PARTS && p.iter().all(|(w, _)| w.k == p[0].0.k) =>
        {
            run_packed(&p, gpu, x, m, s, int8, stream)
        }
        _ => Ok(false),
    }
}

/// [`run_shared`] over packed views (the microtest calls it directly). `x` is `[m, k]` with row stride k.
pub fn run_packed(
    members: &[(Packed, DevicePtr)],
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    m: usize,
    s: &W4Scratch,
    int8: bool,
    stream: u64,
) -> Result<bool> {
    use launch::{ACT_QUANT, BF16_GEMM, BF16_GEMM_MANY, INT8_GEMM, INT8_GEMM_MANY, handle};
    let Some((first, out0)) = members.first().copied() else {
        return Ok(false);
    };
    let many = members.len() > 1;
    let bf16 = if many { BF16_GEMM_MANY } else { BF16_GEMM };
    if !members.iter().all(|(w, _)| w.servable()) || handle(gpu, bf16).is_none() {
        return Ok(false);
    }
    if m == 0 {
        return Ok(true);
    }
    let k = first.k;
    let int8_ready = int8
        && handle(gpu, ACT_QUANT).is_some()
        && handle(gpu, if many { INT8_GEMM_MANY } else { INT8_GEMM }).is_some();
    let act = if int8_ready {
        let a = Int8Act::carve(s, m, k, flags(gpu, stream)?);
        if a.is_none() && gpu.op_cache().once("affine_prefill:int8_without_scratch") {
            tracing::warn!(
                "affine prefill: int8 requested but W4Scratch has no xq/xs; running the BF16 prompt GEMM"
            );
        }
        a
    } else {
        None
    };
    let bf16_gemm = |run_if: DevicePtr| {
        if many {
            gemm_bf16_many(gpu, members, x, k, m, run_if, stream)
        } else {
            gemm_bf16(gpu, &first, x, k, m, out0, run_if, stream)
        }
    };
    match act {
        Some(a) => {
            if gpu.op_cache().once("affine_prefill:int8_route") {
                tracing::info!(
                    "[avarok] affine prefill: W4A8 int8 prompt GEMM (affine4_w4a8_prefill_ada, first at {m} rows, \
                     {} projection(s) a grid), with the device-side BF16 fallback on an exceptional group",
                    members.len()
                );
            }
            quantize_int8_rows(gpu, x, k, m, k, &a, stream)?;
            if many {
                gemm_int8_many(gpu, members, &a, m, stream)?;
            } else {
                gemm_int8(gpu, &first, &a, m, out0, stream)?;
            }
            bf16_gemm(a.flags)?;
        }
        None => bf16_gemm(DevicePtr::NULL)?,
    }
    Ok(true)
}

/// The dense FFN's down projection on the int8 route straight from the gate and up GEMM outputs (`[m, k]` BF16,
/// row stride `ld_in`): the fused SwiGLU + sums + quantizer writes h (BF16, row stride `ld_h`, may alias `gate`
/// with `ld_h == ld_in`) and h's int8 input, then the int8 GEMM, with the BF16 GEMM over h as the device-side
/// fallback. `Ok(false)`: declined (not servable, a kernel or the int8 scratch absent, a null `h`); the caller
/// then runs its SwiGLU and [`run`].
#[allow(clippy::too_many_arguments)]
pub fn run_down_fused(
    down: &AffineWeight,
    gpu: &dyn GpuBackend,
    gate: DevicePtr,
    up: DevicePtr,
    ld_in: usize,
    h: DevicePtr,
    ld_h: usize,
    m: usize,
    out: DevicePtr,
    s: &W4Scratch,
    stream: u64,
) -> Result<bool> {
    match Packed::of(down) {
        Some(w) => run_down_packed(&w, gpu, gate, up, ld_in, h, ld_h, m, out, s, stream),
        None => Ok(false),
    }
}

/// [`run_down_fused`] over a packed view.
#[allow(clippy::too_many_arguments)]
pub fn run_down_packed(
    w: &Packed,
    gpu: &dyn GpuBackend,
    gate: DevicePtr,
    up: DevicePtr,
    ld_in: usize,
    h: DevicePtr,
    ld_h: usize,
    m: usize,
    out: DevicePtr,
    s: &W4Scratch,
    stream: u64,
) -> Result<bool> {
    use launch::{BF16_GEMM, INT8_GEMM, SWIGLU_ACT_QUANT, handle};
    if !w.servable() {
        return Ok(false);
    }
    let present = [SWIGLU_ACT_QUANT, INT8_GEMM, BF16_GEMM]
        .iter()
        .all(|e| handle(gpu, *e).is_some());
    let usable = present && !h.is_null() && ld_h.is_multiple_of(8) && h.0.is_multiple_of(16);
    if !usable {
        return Ok(false);
    }
    let Some(act) = Int8Act::carve(s, m, w.k, flags(gpu, stream)?) else {
        return Ok(false);
    };
    if m == 0 {
        return Ok(true);
    }
    swiglu_quantize(gpu, gate, up, ld_in, h, ld_h, m, w.k, &act, stream)?;
    gemm_int8(gpu, w, &act, m, out, stream)?;
    gemm_bf16(gpu, w, h, ld_h, m, out, act.flags, stream)?;
    Ok(true)
}

/// Read and clear the device count of int8 GEMMs that fell back to BF16 since the last read, add it to
/// [`int8_fallbacks_total`] and record each one as a declined prefill (`port::w4::note_prefill_decline`, which
/// `atlas_affine4_prefill_int8_fallbacks_total` renders). Syncs `stream`: call it where the caller syncs anyway.
/// Returns 0 without touching the device on a capturing stream or where the Ada modules are absent (no affine
/// prompt GEMM can have run).
pub fn take_fallbacks(gpu: &dyn GpuBackend, stream: u64) -> Result<u32> {
    if gpu.stream_is_capturing(stream) || launch::handle(gpu, launch::BF16_GEMM).is_none() {
        return Ok(0);
    }
    let counter = flags(gpu, stream)?.offset(4);
    let mut b = [0u8; 4];
    gpu.copy_d2h_on_stream(counter, &mut b, stream)?;
    let n = u32::from_le_bytes(b);
    if n != 0 {
        gpu.memset_async(counter, 0, 4, stream)?;
        INT8_FALLBACKS.fetch_add(u64::from(n), Ordering::Relaxed);
        for _ in 0..n {
            crate::port::w4::note_prefill_decline();
        }
    }
    Ok(n)
}

/// The serve path's harvest of [`take_fallbacks`]: one call right after the single-sequence prefill chunk's
/// metadata sync (`model/trait_impl/prefill_b.rs`), where the stream is idle, so the read costs one 4-byte copy.
/// A fallback is counted at the next chunk that passes there (batched prefill traffic included, since the
/// device count is sticky). Until an int8 prompt GEMM's flags region exists in this process it returns at
/// once and touches nothing: every non-Ada target, and Ada with `affine4_prefill_int8` off, is unchanged. An
/// error is logged once and never fails the prefill.
pub fn harvest_fallbacks(gpu: &dyn GpuBackend, stream: u64) {
    if !ARMED.load(Ordering::Relaxed) {
        return;
    }
    if let Err(e) = take_fallbacks(gpu, stream)
        && gpu.op_cache().once("affine_prefill:harvest_failed")
    {
        tracing::warn!(
            "affine prefill: reading the int8 fallback count failed ({e:#}); the count may lag"
        );
    }
}
