// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-core DENSE BF16 decode GEMM with a 16-row M tile — the BF16 LM-head
//! arm (#927/#928).
//!
//! WHY. nsys on 1xH100 (round 7, 2026-09-11, Qwen/Qwen3.8-27B-FP8 with
//! `--lm-head-dtype bf16` and `ATLAS_LM_HEAD_BATCHM_MAX=16`, decode batch 16)
//! puts `dense_gemv_bf16_batchm` — the LM head — at **3,571 µs in ONE launch,
//! 8.19% of the 43.6 ms step**, for a single pass over the 2.54 GB
//! `[248077, 5120]` BF16 vocab weight. That is **~710 GB/s** against ~3,350
//! GB/s of HBM3. The SAME kernel at C=1 costs 798 µs — 3.2 TB/s-class, at the
//! roofline — so 16 rows cost 4.47x one row for an identical weight read: the
//! batched tier is FP32-FMA-bound, not bandwidth-bound.
//!
//! `dense_gemm_m16_bf16` replaces the per-row scalar FFMA with one
//! `mma.sync.m16n8k16` lane slot. The M tile IS 16 rows, so nothing is padded —
//! the difference from `dense_gemm_tc`'s 16Mx64N tile (which pads the same way
//! but still reads the weight through a scalar inner loop) and from the tile
//! GEMMs that pad M to 128. Target: **<= 1.3 ms at M=16 = >= 1,950 GB/s**.
//!
//! NUMERICS — REASSOCIATED ON PURPOSE. `dense_gemv_bf16` / `_batchm` reduce
//! each output in ONE FP32 accumulator in strict K order, and the batched tier
//! is bit-identical to M serial scalar GEMVs. An MMA reduces 16 K-products in
//! the tensor core's own order first, so this is NOT. The contract is the
//! predicate `w8a16_gemm_m16` already answers to —
//! [`crate::layers::dense_ffn::m16_tc::within_m16_tc_budget`]: <= 2 ordinal
//! BF16 ULP, or an absolute error under the accumulation floor. Oracle:
//! `examples/native_bf16_lm_head_m16_microtest.rs`.
//!
//! At the LM head a near-tie argmax flip changes the emitted token, which is
//! why the head arm is behind `ATLAS_LM_HEAD_M16_TC` and defaults OFF
//! (`model/trait_impl/lm_head_batched.rs`).
//!
//! Unlike `w8a16_gemm_m16` there is no block scale and no dequant: B is already
//! BF16, so a staged weight word IS a B fragment register, the accumulator is
//! ONE level, and the K constraint is the 64-wide pipeline step rather than the
//! 128-wide scale block.
//!
//! Kernels: `dense_gemm_m16_bf16` / `dense_gemm_m16_bf16_n64` (module
//! `dense_gemm_m16_bf16`). Grid: (ceil(N/N_TILE), 1, 1)  Block: (128, 1, 1).

use crate::weight_map::DenseWeight;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// N columns one CTA owns on the DEFAULT instantiation. SSOT for the launch
/// geometry AND for the dispatch rule's CTA-count reasoning, so the two cannot
/// drift: the kernel's `DGM16_N_TILE` must equal this.
pub const DENSE_GEMM_M16_BF16_N_TILE: u32 = 32;

/// The wide instantiation's N tile (`dense_gemm_m16_bf16_n64`, kernel
/// `DGM16_N_TILE_WIDE`) — opt-in via `ATLAS_LM_HEAD_M16_TC_NTILE=64`. Halves
/// the CTA count for a given N and halves the L2 traffic the re-read A tile
/// costs. WHY it exists: `dense_gemm_m16_bf16.cu`.
pub const DENSE_GEMM_M16_BF16_N_TILE_WIDE: u32 = 64;

/// The kernel's M tile. Rows past it are simply not computed, so the wrapper
/// REFUSES rather than writing part of the block and leaving the rest stale.
pub const DENSE_GEMM_M16_BF16_MAX_M: u32 = 16;

/// K granularity: the 4-stage cp.async pipeline advances 64 elements per step,
/// and 64 BF16 is also what keeps every 16-byte weight-row chunk aligned.
pub const DENSE_GEMM_M16_BF16_K_STEP: u32 = 64;

/// The shared shape of both instantiations, so a caller that picks between them
/// (and between them and `dense_gemv_batchm`) can hold ONE function pointer and
/// the tile stays a dispatch choice rather than a code path.
pub type DenseM16Bf16Gemm = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    &DenseWeight,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

/// The default 32-wide CTA. `input` is `[m, a_row_stride]` BF16 with `k` used,
/// `weight.weight` is the raw `[n, k]` BF16 checkpoint tensor (NOT copied, NOT
/// quantized — the same pointer `dense_gemv_batchm` reads), and `output` is
/// `[m, c_row_stride]` BF16 with `n` used.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_m16_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    launch(
        gpu,
        kernel,
        DENSE_GEMM_M16_BF16_N_TILE,
        "dense_gemm_m16_bf16",
        input,
        weight,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// `N_TILE=64` twin of [`dense_gemm_m16_bf16`] — identical arguments and
/// identical per-output arithmetic, `ceil(n/64)` CTAs instead of `ceil(n/32)`.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_m16_bf16_n64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    launch(
        gpu,
        kernel,
        DENSE_GEMM_M16_BF16_N_TILE_WIDE,
        "dense_gemm_m16_bf16_n64",
        input,
        weight,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// The guards and the launch both instantiations share — only the CTA width
/// differs, and it is the ONE thing a reader has to check to tell them apart.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    n_tile: u32,
    who: &str,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DENSE_GEMM_M16_BF16_MAX_M).contains(&m),
        "{who}: m={m} outside 1..={DENSE_GEMM_M16_BF16_MAX_M} (kernel M tile; \
         rows past it are never computed, not a launch failure)"
    );
    ensure!(
        k.is_multiple_of(DENSE_GEMM_M16_BF16_K_STEP),
        "{who}: K={k} not a multiple of {DENSE_GEMM_M16_BF16_K_STEP} \
         (cp.async pipeline step, and what keeps each [n, k] weight row 16B-aligned)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "{who}: row pitches (a={a_row_stride}, c={c_row_stride}) must cover the \
         used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "{who}: a_row_stride={a_row_stride} must keep rows 16B-aligned \
         (cp.async stages A in 16-byte chunks)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, n_tile), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_row_stride)
        .arg_u32(c_row_stride)
        .launch(stream)
}

/// Host simulation of the staging / fragment / store index math against a
/// reference GEMM in the scalar `dense_gemv_bf16`'s reduction order — the test
/// that says WHERE a mismatch is, without an H100.
#[cfg(test)]
#[path = "dense_gemm_m16_bf16_tests.rs"]
mod tests;

/// Host simulation of the ACCUMULATION FLOOR — the round-9 LM-head red cell,
/// why round 6's fixed `2^-20 * rms` constant could not carry this tier, and
/// the margins the K-aware floor keeps. Split from `tests` for the 500-line
/// cap.
#[cfg(test)]
#[path = "dense_gemm_m16_bf16_floor_tests.rs"]
mod floor_tests;
