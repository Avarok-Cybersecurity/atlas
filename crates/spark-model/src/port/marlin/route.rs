// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the loader-facing half of the `marlin_w4a16` row: whether one BF16 projection
//! takes the Marlin RTN route ([`eligibility`]), the quantize call a loader route makes in place of
//! its own quantizer ([`quantize_projection`]), and the ledger helpers such a route needs
//! ([`buffers_of`], [`marlin_bytes`], [`disown`]).
//!
//! # What the row means
//!
//! `marlin_w4a16` (off on every target) swaps the QUANTIZER and the KERNEL of a 4-bit BF16 route
//! that is already on (`dense_affine4_from_bf16`, `gdn_affine4_from_bf16`, `attn_affine4_from_bf16`,
//! lane L23a): every projection Marlin serves is quantized by round-to-nearest int4 at group 128
//! (`kU4B8`, symmetric, `--w4-clip-search` shrink candidates per group) instead of the MLX affine
//! codes, and decoded by the vLLM Marlin GEMM (rows 1..=64 in one launch, prompts natively). A
//! projection Marlin does not serve (shape, group, module absent) keeps the affine quantizer, with
//! one log line per reason. With the row off, [`quantize_projection`] returns `None` at once and
//! issues no kernel lookup, so the route is byte-identical to today. GPTQ and AWQ checkpoints do not
//! need the row's numerics argument (their codes are the checkpoint's): they go through
//! `port::marlin::from_gptq` / `from_awq` at the packed install sites (lane L23b).

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::{self, Problem, ThreadConfig};
use super::kernels::{self, KernelKey};
use super::{DEFAULT_GROUP, MarlinLinear, MarlinType};
use crate::port::w4::{W4Box, W4Linear, W4Opts};

/// The group of the RTN route (the only group the Ada build instantiates).
pub const RTN_GROUP: u16 = 128;

/// The load-time options of the RTN route: 4 bits, group 128, the operator's clip search, no int8
/// prompt path (Marlin has none).
pub fn rtn_opts(clip_search: u8) -> W4Opts {
    W4Opts {
        bits: super::BITS,
        group: RTN_GROUP,
        clip_search: clip_search.min(super::quantize::MAX_CLIP_SEARCH),
        int8_prefill: false,
    }
}

/// Why a projection does not take the Marlin route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `marlin_w4a16` is off (the default on every target).
    RowOff,
    /// The build does not carry `marlin_ada` (every non-Ada target).
    ModuleAbsent,
    /// The shape has no instance (N not a multiple of 64, K not a multiple of 128, one group, ...).
    Shape(String),
}

impl Decline {
    /// A stable label for the log line.
    pub fn label(&self) -> &'static str {
        match self {
            Self::RowOff => "row off",
            Self::ModuleAbsent => "module absent",
            Self::Shape(_) => "shape",
        }
    }
}

/// Whether the instance table carries `(type, g128, thread_m_blocks, m_block_size_8, cfg)`, without a
/// kernel lookup (the table equals `marlin_ada.cu`'s export list, pinned by the tests).
pub fn in_table(
    btype: MarlinType,
    thread_m_blocks: usize,
    m_block_size_8: bool,
    cfg: &ThreadConfig,
) -> bool {
    let key = KernelKey::of(
        btype,
        Some(DEFAULT_GROUP / 16),
        thread_m_blocks,
        m_block_size_8,
        cfg,
    );
    kernels::instances().contains(&key)
}

/// The shape half of [`eligibility`]: the Marlin layout, an instantiated group, and a decode
/// configuration valid at every decode tile. Pure; the decision the GPU build would make.
pub fn shape_ok(n: usize, k: usize, btype: MarlinType) -> std::result::Result<(), String> {
    config::supports_shape(n, k, DEFAULT_GROUP)?;
    if !kernels::group_supported(DEFAULT_GROUP, k) {
        return Err(format!(
            "K={k} is a single group of {DEFAULT_GROUP} (no channelwise instance)"
        ));
    }
    let p = Problem::of(n, k, DEFAULT_GROUP, btype.has_zp());
    let avail = |t: usize, m8: bool, c: &ThreadConfig| in_table(btype, t, m8, c);
    match config::invariant_decode_config(
        &p,
        &kernels::DECODE_TMBS,
        config::ADA_MAX_SHARED_MEM_OPTIN,
        &avail,
    ) {
        Some(_) => Ok(()),
        None => Err(format!(
            "[{n}x{k}] has no 128-thread configuration at every decode tile"
        )),
    }
}

/// Whether a BF16 `[n, k]` projection takes the RTN route, given the resolved row. The row is checked
/// first, so a row-off call touches nothing else.
pub fn eligibility(
    gpu: &dyn GpuBackend,
    row_on: bool,
    n: usize,
    k: usize,
) -> std::result::Result<(), Decline> {
    if !row_on {
        return Err(Decline::RowOff);
    }
    if !super::available(gpu) {
        return Err(Decline::ModuleAbsent);
    }
    shape_ok(n, k, MarlinType::U4B8).map_err(Decline::Shape)
}

static LOGGED_ON: AtomicBool = AtomicBool::new(false);
static LOGGED_ABSENT: AtomicBool = AtomicBool::new(false);
static LOGGED_SHAPE: AtomicBool = AtomicBool::new(false);

/// One log line per decline class and process (a model has hundreds of projections).
fn log_decline(d: &Decline, name: &str) {
    let flag = match d {
        Decline::RowOff => return,
        Decline::ModuleAbsent => &LOGGED_ABSENT,
        Decline::Shape(_) => &LOGGED_SHAPE,
    };
    if flag.swap(true, Ordering::Relaxed) {
        return;
    }
    match d {
        Decline::Shape(why) => tracing::info!(
            "marlin_w4a16: {name} keeps the affine quantizer ({why}); further shape declines are not logged"
        ),
        _ => tracing::info!(
            "marlin_w4a16: {} is not compiled for this target; the 4-bit routes keep the affine quantizer",
            kernels::MODULE
        ),
    }
}

/// The RTN route for one BF16 projection: `Some(weight)` when the row is on and Marlin serves the
/// shape, `None` when the caller should use its own quantizer (nothing allocated, nothing looked up
/// when the row is off). `src` is not freed; the caller frees it after this returns (the quantize
/// synchronizes `stream` before returning, so nothing still reads it). `name` is for the log.
pub fn quantize_projection(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    n: usize,
    k: usize,
    clip_search: u8,
    stream: u64,
    name: &str,
) -> Result<Option<W4Box>> {
    quantize_projection_with(
        gpu,
        super::row_enabled(),
        src,
        n,
        k,
        clip_search,
        stream,
        name,
    )
}

/// [`quantize_projection`] with the row given (the tests and the microtest).
#[allow(clippy::too_many_arguments)]
pub fn quantize_projection_with(
    gpu: &dyn GpuBackend,
    row_on: bool,
    src: DevicePtr,
    n: usize,
    k: usize,
    clip_search: u8,
    stream: u64,
    name: &str,
) -> Result<Option<W4Box>> {
    if let Err(d) = eligibility(gpu, row_on, n, k) {
        log_decline(&d, name);
        return Ok(None);
    }
    let opts = rtn_opts(clip_search);
    let w = super::quantize_rtn_from_bf16(gpu, src, n, k, &opts, stream)?;
    if !LOGGED_ON.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "marlin_w4a16: {name} [{n}x{k}] quantized to Marlin RTN int4 g{} (clip search {}), {} device bytes; \
             further projections are not logged",
            opts.group,
            opts.clip_search,
            w.device_bytes()
        );
    }
    Ok(Some(w))
}

/// The concrete Marlin weight behind a `W4Linear`, if it is one.
pub fn marlin_of(w: &dyn W4Linear) -> Option<&MarlinLinear> {
    w.as_any()?.downcast_ref::<MarlinLinear>()
}

/// The device buffers of a Marlin weight with their sizes (codes, scales, zero points when the type
/// has them, locks), or `None` for any other `W4Linear`. Null buffers are left out.
pub fn buffers_of(w: &dyn W4Linear) -> Option<Vec<(DevicePtr, usize)>> {
    let m = marlin_of(w)?.weight();
    let groups = m.groups();
    let mut v = vec![(m.words, m.k * m.n / 2), (m.scales, groups * m.n * 2)];
    if !m.zeros.is_null() {
        v.push((m.zeros, groups * m.n / 2));
    }
    v.push((m.locks, config::locks_bytes(m.sms)));
    Some(v)
}

/// Device bytes one Marlin `[n, k]` weight keeps at group 128 (codes, scales, zero points when
/// `has_zp`, locks for `sms` SMs): the residency arithmetic of the route.
pub fn marlin_bytes(n: usize, k: usize, has_zp: bool, sms: usize) -> usize {
    let groups = k / DEFAULT_GROUP;
    let zeros = if has_zp { groups * n / 2 } else { 0 };
    k * n / 2 + groups * n * 2 + zeros + config::locks_bytes(sms)
}

/// Hand a route-installed weight's buffers to the caller's ledger: `port::marlin::release` of it
/// becomes a no-op, so the ledger is the one owner that frees them. False when `w` is not a live
/// Marlin weight.
pub fn disown(w: &dyn W4Linear) -> bool {
    marlin_of(w).is_some() && super::w4linear::forget(w)
}

#[cfg(test)]
#[path = "route_tests.rs"]
mod tests;
