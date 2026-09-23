// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-core routing for the MTP drafter's BF16 small-M GEMV
//! (`dense_gemv_bf16_tc.cu`, module `dense_gemv_bf16_tc`).
//!
//! # Why
//!
//! The BF16 drafter (`--mtp-quantization bf16`) streams ~849 MB of weights per
//! draft position through `dense_gemv_bf16` (M=1, the C=1 propose) or
//! `dense_gemv_bf16_batchm` (M=2..8, the batched propose). Both are CUDA-core
//! kernels that do FMUL+FADD per weight per row (the kernel dir builds
//! `--fmad=false`), so their issue work grows with M while the bytes do not.
//! The drafter is hot at every concurrency, which puts it on the GB10
//! J/token critical path against vLLM, whose drafter runs on tensor cores.
//! The tensor-core entries run the same contract with one `mma.sync.m16n8k16`
//! per 16x16 weight block and no per-weight ALU work.
//!
//! # Contract
//!
//! `dense_gemv_bf16_batchm`'s arguments, argument for argument (A `[M,K]`
//! contiguous, W `[N,K]`, output rows `out_stride` elements apart). Numerics
//! differ only in FP32 summation order, the same class of difference as
//! `dense_gemm_bf16_pipelined` (mma.sync), which the drafter already runs
//! above 8 rows. The reduction order is fixed, so the output is
//! deterministic.
//!
//! # Switch
//!
//! `AVAROK_MTP_TC=1` opts in (exactly `1`; unset, empty or any other value
//! is OFF, and the drafter keeps its CUDA-core kernels bit-for-bit). Read
//! once. Gate records disclose it in `perf_env` (`avarok-plugin`
//! `PERF_CONTROLS`, default `0`).

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// Token rows each entry covers (8 per B-fragment token tile).
pub const TC8_MAX_M: u32 = 8;
pub const TC16_MAX_M: u32 = 16;
pub const TC32_MAX_M: u32 = 32;
/// Weight rows per CTA (one 16-row A tile per CTA; `NT = 1` in the .cu).
pub const ROWS_PER_CTA: u32 = 16;
/// 8 warps split K inside a CTA (`DTC_WARPS` in the .cu).
pub const BLOCK: u32 = 256;
/// K per warp step (`DTC_KB` in the .cu): each quad reads one 64-k block.
pub const K_STEP: u32 = 64;

/// Which tensor-core entry serves a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtcKind {
    M8,
    M16,
    M32,
}

impl DtcKind {
    pub fn entry(self) -> &'static str {
        match self {
            DtcKind::M8 => "dense_gemv_bf16_tc8",
            DtcKind::M16 => "dense_gemv_bf16_tc16",
            DtcKind::M32 => "dense_gemv_bf16_tc32",
        }
    }
}

/// PURE routing decision. `None` keeps the caller's CUDA-core kernel.
///
/// `have` is `[tc8, tc16, tc32]` resolved. The narrowest resolved entry that
/// covers `m` wins: a wider entry serves narrow M correctly (it skips token
/// tiles past M) but issues more reduction work. `K % 64` is required (each
/// quad reads 64 contiguous k per step); anything else declines rather than
/// guessing at a K tail. Any N routes: a partial last weight tile is guarded
/// in-kernel.
pub fn route(m: u32, n: u32, k: u32, enabled: bool, have: [bool; 3]) -> Option<DtcKind> {
    if !enabled || m == 0 || n == 0 || k == 0 || !k.is_multiple_of(K_STEP) {
        return None;
    }
    [
        (TC8_MAX_M, DtcKind::M8),
        (TC16_MAX_M, DtcKind::M16),
        (TC32_MAX_M, DtcKind::M32),
    ]
    .into_iter()
    .zip(have)
    .find(|&((cap, _), ok)| ok && m <= cap)
    .map(|((_, kind), _)| kind)
}

/// The `AVAROK_MTP_TC` rule over a looked-up value: ON only for exactly `1`.
pub fn mtp_tc_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `AVAROK_MTP_TC=1`? Read once: the predicate sits on the per-draft-position
/// path and every launch in a process must see one choice.
pub fn mtp_tc_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mtp_tc_from(std::env::var("AVAROK_MTP_TC").ok().as_deref()))
}

/// Resolved handles `[tc8, tc16, tc32]`, cached per backend (a
/// `KernelHandle` is a function in ONE backend's loaded module).
fn handles(gpu: &dyn GpuBackend) -> [KernelHandle; 3] {
    static CACHE: OnceLock<Mutex<Vec<(usize, [KernelHandle; 3])>>> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let h = [DtcKind::M8, DtcKind::M16, DtcKind::M32]
        .map(|kind| crate::layers::try_kernel(gpu, "dense_gemv_bf16_tc", kind.entry()));
    guard.push((key, h));
    h
}

/// The tensor-core kernel and grid-x for this launch when `AVAROK_MTP_TC=1`
/// and an entry resolves, else `None`.
pub fn kernel_for(gpu: &dyn GpuBackend, m: u32, n: u32, k: u32) -> Option<(KernelHandle, u32)> {
    if !mtp_tc_enabled() {
        return None;
    }
    let h = handles(gpu);
    let kind = route(m, n, k, true, h.map(|x| x.0 != 0))?;
    let handle = match kind {
        DtcKind::M8 => h[0],
        DtcKind::M16 => h[1],
        DtcKind::M32 => h[2],
    };
    Some((handle, div_ceil(n, ROWS_PER_CTA)))
}

/// Launch one tensor-core entry explicitly (the oracle and microbench use
/// this to pin a kernel; production goes through [`try_dense_gemv_tc`]).
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    grid_x: u32,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Run `C[t] = A[t] @ W^T` on the tensor-core entry if `AVAROK_MTP_TC=1` and
/// it routes. `Ok(false)` means nothing was launched and the caller must run
/// its own kernel.
#[allow(clippy::too_many_arguments)]
pub fn try_dense_gemv_tc(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<bool> {
    let Some((kernel, grid_x)) = kernel_for(gpu, m, n, k) else {
        return Ok(false);
    };
    launch(
        gpu, kernel, grid_x, input, weight, output, m, n, k, out_stride, stream,
    )?;
    Ok(true)
}

#[cfg(test)]
#[path = "dense_gemv_tc_tests.rs"]
mod dense_gemv_tc_tests;
