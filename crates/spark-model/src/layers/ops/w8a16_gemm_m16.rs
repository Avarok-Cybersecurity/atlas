// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-core W8A16 decode GEMM with a 16-row M tile (#927).
//!
//! WHY. `w8a16_gemv_batch16` streams the FP8 weight once for up to 16 rows and
//! is bit-exact, but at M=16 it is FP32-FMA-bound, not bandwidth-bound: on
//! 1xH100 with Qwen/Qwen3.8-27B-FP8 it measured 0.260 ms / **342 GB/s** for
//! gate/up (N=17408, K=5120) and 0.330 ms / **270 GB/s** for down (N=5120,
//! K=17408), against ~3,000 GB/s of HBM3 — an 89 MB weight matrix should
//! stream in ~30 us. Its inner loop spends ~37 ALU ops per weight BYTE (16
//! scalar FFMA, 16 BF16->FP32 converts, a LUT lookup, a scale multiply);
//! `w8a16_gemm_m16` turns those 16 FFMA into one `mma.sync.m16n8k16` lane-slot
//! and the dequant into ~2 instructions per byte, so the shape becomes
//! weight-bandwidth bound.
//!
//! NUMERICS — REASSOCIATED ON PURPOSE. The MMA reduces 16 K-products in the
//! tensor core's own order before they reach the FP32 accumulator, so this is
//! NOT bit-identical to the scalar `w8a16_gemv` the way the batched GEMVs are.
//! The contract is <= 2 BF16 ULP per element (oracle:
//! `examples/native_fp8_ffn_m16_tc_microtest.rs`). This is not a new seam: the
//! arm the FFN used at these widths BEFORE #927 (`w8a16_gemm_n128_m128` /
//! `w8a16_gemm_pipelined`) reassociates identically. It is why every call site
//! sits behind `ATLAS_FFN_M16_TC`, default OFF.
//!
//! The 128-K block scale is folded ONCE per block onto an FP32 outer
//! accumulator (two-level fold, preserved exactly from `w8a16_gemm_pipelined`),
//! never per element and never into BF16.
//!
//! Kernels: `w8a16_gemm_m16` / `w8a16_gemm_m16_strided` (module
//! `w8a16_gemm_m16`). Grid: (ceil(N/32), 1, 1)  Block: (128, 1, 1).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// N columns one CTA owns on the DEFAULT instantiation. SSOT for the launch
/// geometry AND for the dispatch rule's "does this shape have enough CTAs"
/// reasoning, so the two cannot drift: the kernel's `M16_N_TILE` must equal
/// this.
pub const W8A16_GEMM_M16_N_TILE: u32 = 32;

/// The wide instantiation's N tile (`w8a16_gemm_m16_n64`, kernel
/// `M16_N_TILE_WIDE`) — opt-in via `ATLAS_FFN_M16_TC_NTILE=64`. Halves the CTA
/// count for a given N and doubles the reuse of each staged A fragment. WHY it
/// exists and what it is meant to settle: `dense_ffn_m16_tc.rs`.
pub const W8A16_GEMM_M16_N_TILE_WIDE: u32 = 64;

/// The shared shape of [`w8a16_gemm_m16`], so a caller that picks between it
/// and `w8a16_gemv_batch16` can hold one function pointer.
pub type ContiguousM16Gemm = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

/// Contiguous `input` `[m, k]` BF16 and `output` `[m, n]` BF16; `weight` /
/// `block_scale` are the raw `w8a16_gemv` pointers (`[N, K]` FP8 E4M3 and
/// `[N/128, K/128]` FP32).
///
/// REFUSES m > 16: the kernel's M tile IS the MMA's 16 rows, and rows past it
/// are simply not computed (stale output, not a launch failure). Callers with
/// 17..=32 rows run it twice on contiguous row halves, the way
/// `dense_ffn_m16_tc.rs` does.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemm_m16: m={m} outside 1..=16 (kernel M tile)"
    );
    ensure!(
        k.is_multiple_of(128),
        "w8a16_gemm_m16: K={k} not a multiple of 128 (block-scale granularity)"
    );
    launch_contiguous(
        gpu,
        kernel,
        W8A16_GEMM_M16_N_TILE,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        stream,
    )
}

/// `N_TILE=64` twin of [`w8a16_gemm_m16`] — identical arguments and identical
/// per-output arithmetic, `ceil(N/64)` CTAs instead of `ceil(N/32)`. Same
/// signature, so a caller holds ONE [`ContiguousM16Gemm`] pointer and the tile
/// is a dispatch choice, not a code path.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_m16_n64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemm_m16_n64: m={m} outside 1..=16 (kernel M tile)"
    );
    ensure!(
        k.is_multiple_of(128),
        "w8a16_gemm_m16_n64: K={k} not a multiple of 128 (block-scale granularity)"
    );
    launch_contiguous(
        gpu,
        kernel,
        W8A16_GEMM_M16_N_TILE_WIDE,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        stream,
    )
}

/// The launch both contiguous instantiations share — only the CTA width
/// differs, and it is the ONE thing a reader has to check to tell them apart.
#[allow(clippy::too_many_arguments)]
fn launch_contiguous(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    n_tile: u32,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, n_tile), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Strided sibling of [`w8a16_gemm_m16`]: `a_row_stride` / `c_row_stride` are
/// the A and C row pitches in ELEMENTS, for callers whose rows are not
/// contiguous. The multi-seq decode QKV buffer is `[n, per_seq_qkv]` with Q at
/// offset 0, K after Q and V after K inside every row, so one launch per
/// projection writes all `m` rows straight into their slots — the same reason
/// `w8a16_gemv_batch16_strided` exists, and the same argument order.
///
/// `a_row_stride` must keep each activation row 16-byte aligned (a multiple of
/// 8 BF16): the kernel stages A with 16-byte `cp.async` chunks.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_m16_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemm_m16_strided: m={m} outside 1..=16 (kernel M tile)"
    );
    ensure!(
        k.is_multiple_of(128),
        "w8a16_gemm_m16_strided: K={k} not a multiple of 128 (block-scale granularity)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemm_m16_strided: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemm_m16_strided: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (cp.async stages A in 16-byte chunks)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, W8A16_GEMM_M16_N_TILE), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_row_stride)
        .arg_u32(c_row_stride)
        .launch(stream)
}
