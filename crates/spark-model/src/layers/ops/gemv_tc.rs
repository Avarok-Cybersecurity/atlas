// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-core routing for the narrow NVFP4 batched GEMV
//! (`w4a16_gemv_tc.cu`, module `w4a16_gemv_tc`).
//!
//! # Why
//!
//! Every `ops::w4a16_gemv_batchm` launch (decode, MTP verify and lm_head at
//! 1..=16 rows) used to run a CUDA-core template whose per-weight work grows
//! with M: at M=4 it streams weights at ~220 GB/s but draws 77-82 W on the
//! GB10 GPU rail, against ~51 W for `w4a16_gemv_tc8` at the same time per
//! launch (tcbench, cold weights, real 27B shapes; -32..-35% mJ per launch at
//! M=4, -43% at M=8, and 2.4x faster than `w4a16_gemv_batch16` at M=16). The
//! whole C<=2 J/token gap against vLLM sat in that kernel family.
//!
//! # Contract
//!
//! Same arguments as the CUDA-core tiers, different launch geometry. Numerics
//! differ only in FP32 summation order (tensor-core reduction), the same
//! class of difference as the tile GEMMs above 8 rows; the dequant is exact.
//! Routing happens in ONE place — [`w4a16_gemv_batchm`](super::w4a16_gemv_batchm)
//! — so all fourteen call sites inherit it and none re-derives it.
//!
//! # Kill switch
//!
//! `AVAROK_NO_W4A16_TC=1` (any non-empty value, read once) restores the
//! CUDA-core tiers bit-for-bit — the A/B lever for the energy campaign. Gate
//! records disclose it in `perf_env`.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::weight_map::QuantizedWeight;

/// Rows the `w4a16_gemv_tc8` entry covers (A-fragment rows 0..7).
pub const TC8_MAX_M: u32 = 8;
/// Rows the `w4a16_gemv_tc16` entry covers (both A-fragment halves).
pub const TC16_MAX_M: u32 = 16;
/// Columns per CTA: tc8 runs NT=1 tile of 8, tc16 NT=2 tiles of 8.
pub const TC8_COLS_PER_CTA: u32 = 8;
pub const TC16_COLS_PER_CTA: u32 = 16;
/// 8 warps split K inside a CTA (`TC_WARPS` in the .cu).
pub const TC_BLOCK: u32 = 256;

/// Which tensor-core entry serves a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcKind {
    M8,
    M16,
}

impl TcKind {
    pub fn cols_per_cta(self) -> u32 {
        match self {
            TcKind::M8 => TC8_COLS_PER_CTA,
            TcKind::M16 => TC16_COLS_PER_CTA,
        }
    }
}

/// PURE routing decision. `None` keeps the caller's CUDA-core tier.
///
/// The kernel reads each quad's 128 contiguous k per step (so `K % 128`);
/// anything else declines rather than guessing at a K tail. Any N routes: a
/// partial last column tile (the 248077-row lm_head) is guarded in-kernel.
pub fn tc_route(
    m: u32,
    n: u32,
    k: u32,
    enabled: bool,
    have8: bool,
    have16: bool,
) -> Option<TcKind> {
    if !enabled || m == 0 || n == 0 || k == 0 || !k.is_multiple_of(128) {
        return None;
    }
    if m <= TC8_MAX_M && have8 {
        Some(TcKind::M8)
    } else if m <= TC16_MAX_M && have16 {
        Some(TcKind::M16)
    } else {
        None
    }
}

/// `AVAROK_NO_W4A16_TC` unset (or exported empty)? Any non-empty value,
/// `0` included, turns the tensor-core path off, and gate records disclose it
/// as `unset` otherwise (`avarok-plugin` `PERF_CONTROLS`). Read once: the
/// predicate sits on the decode path and a graph-captured launch must see a
/// stable choice.
pub fn tc_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AVAROK_NO_W4A16_TC").is_none_or(|v| v.is_empty()))
}

/// Widest row count the dispatch sites may send down the narrow-GEMV arms.
/// 8 is the CUDA-core family's measured edge; with the tensor-core path on,
/// `tc16` streams 9..=16 rows at ~190 GB/s (2.4x `w4a16_gemv_batch16`), so
/// the verify steps of C=4/8 (9..16 rows) stop falling onto the tile GEMMs /
/// W4A4 MMQ. The same switch also routes the FIXED-M launchers
/// (`w4a16_gemv_batch2/3`, `w4a16_gemv_dual_batch2/3`: the C=2/C=3 decode
/// and K=1/K=2 verify arms) to the tensor-core kernel.
///
/// ★ OPT-IN (`AVAROK_W4A16_TC_WIDE=1`, any non-empty value), because it is a
/// SPEED lever that costs ENERGY. Measured on dgx3 (gate throughput config,
/// same binary, 2 reps each): C=4 went 73.5/74.3 -> 77.9/77.9 tok/s (+5.5%),
/// but 49.1/48.9 -> 61.8/61.8 W, i.e. 0.668/0.658 -> 0.794/0.793 J/token
/// (+19%). The tensor-core MMA with 9..16 LIVE rows draws far more than the
/// W4A4 MMQ / tile path it replaces. C=2 and C=8 did not move. With the base
/// switch `AVAROK_NO_W4A16_TC` set it is off regardless.
pub const NARROW_MAX_ROWS: u32 = 8;
pub const WIDE_MAX_ROWS: u32 = TC16_MAX_M;

pub fn wide_rows_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        tc_enabled() && std::env::var_os("AVAROK_W4A16_TC_WIDE").is_some_and(|v| !v.is_empty())
    })
}

/// Row edge of the narrow-GEMV dispatch arms (see [`wide_rows_enabled`]).
pub fn narrow_gemv_max_rows() -> u32 {
    if wide_rows_enabled() {
        WIDE_MAX_ROWS
    } else {
        NARROW_MAX_ROWS
    }
}

/// Resolved handles, cached per backend. A `KernelHandle` is a function in
/// ONE backend's loaded module, so the cache is keyed by the backend object's
/// address; a process serves from one backend for its lifetime, so this holds
/// one entry in production.
#[derive(Clone, Copy)]
struct TcHandles {
    tc8: KernelHandle,
    tc16: KernelHandle,
}

fn tc_handles(gpu: &dyn GpuBackend) -> TcHandles {
    static CACHE: OnceLock<Mutex<Vec<(usize, TcHandles)>>> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let h = TcHandles {
        tc8: crate::layers::try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc8"),
        tc16: crate::layers::try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc16"),
    };
    guard.push((key, h));
    h
}

/// The tensor-core kernel and grid-x for this launch, or `None` to keep the
/// CUDA-core tier.
pub fn tc_kernel(gpu: &dyn GpuBackend, m: u32, n: u32, k: u32) -> Option<(KernelHandle, u32)> {
    if !tc_enabled() {
        return None;
    }
    let h = tc_handles(gpu);
    let kind = tc_route(m, n, k, true, h.tc8.0 != 0, h.tc16.0 != 0)?;
    let handle = match kind {
        TcKind::M8 => h.tc8,
        TcKind::M16 => h.tc16,
    };
    Some((handle, n.div_ceil(kind.cols_per_cta())))
}

/// The fixed-M launchers' tensor-core route (`gemv_tc::wide_rows_enabled`):
/// `Ok(true)` when it launched, `Ok(false)` to keep the CUDA-core kernel.
#[allow(clippy::too_many_arguments)]
pub fn tc_fixed_m(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if !wide_rows_enabled() {
        return Ok(false);
    }
    let Some((tc, grid_x)) = tc_kernel(gpu, m, n, k) else {
        return Ok(false);
    };
    KernelLaunch::new(gpu, tc)
        .grid([grid_x, 1, 1])
        .block([TC_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "gemv_tc_tests.rs"]
mod gemv_tc_tests;
