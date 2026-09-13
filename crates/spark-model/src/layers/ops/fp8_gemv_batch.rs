// SPDX-License-Identifier: AGPL-3.0-only

//! FP8-weight dual-GEMV (batch=2) dispatch.
//!
//! `dense_gemv_fp8w_batch2` computes two output rows from one pass over the
//! FP8 weight matrix — the batch=2 sibling of `dense_gemv_fp8w`. It halves
//! FP8 weight bandwidth vs two M=1 GEMV launches and is bit-identical to
//! running `dense_gemv_fp8w` twice (per-token reduction order unchanged).
//! Used by the K=2 MTP verify path where the two verify positions share
//! weights but have distinct activations (lm_head, attention Q/K/V/O, SSM
//! out_proj).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8DenseWeight;

/// Register-tiled batched row-scaled FP8 GEMV (M<=8, T=2 outputs/thread) —
/// the FP8 twin of `w4a16_gemv_batch8_rt2`, for the DFlash drafter PROPOSE
/// path. `input` `[M, K]` BF16, `output` `[M, N]` BF16; per-row f32 scale
/// applied at write-out inside the kernel. Replaces the prefill-class tile
/// GEMMs (`fp8_gemm_t_row_scaled` M64-tile / `_m16`) that pad 87%/50% of
/// their M-tile at M=8 (~100 GB/s measured vs 180+ for the rt family).
/// Drafter-side numerics: correctness-free under strict-argmax accept.
/// Kernel: `fp8_gemv_rowscale_batch8_rt2` (module `fp8_gemv_rt`).
/// Grid: (ceil(N/8), 1, 1)  Block: (256, 1, 1). Requires K % 16 == 0.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemv_rowscale_batch8_rt2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=8).contains(&m),
        "fp8_gemv_rowscale_batch8_rt2: m={m} outside 1..=8 (kernel MAX_M)"
    );
    ensure!(
        k.is_multiple_of(16),
        "fp8_gemv_rowscale_batch8_rt2: K={k} not a multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// MAX_M=16 sibling of [`fp8_gemv_rowscale_batch8_rt2`] for the γ>8 DFlash
/// propose window (flags 9..17). Same template, same launch geometry; added
/// 2026-08-29 after STEP_TIMING measured propose 18.2ms (flag 8, rt2) vs
/// 38.0ms (flag 9, tile fallback) — the whole γ>8 step tax.
/// Kernel: `fp8_gemv_rowscale_batch16_rt2` (module `fp8_gemv_rt`).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemv_rowscale_batch16_rt2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "fp8_gemv_rowscale_batch16_rt2: m={m} outside 1..=16 (kernel MAX_M)"
    );
    ensure!(
        k.is_multiple_of(16),
        "fp8_gemv_rowscale_batch16_rt2: K={k} not a multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// FP8-weight dual-GEMV. `input` is `[2, K]` BF16, `output` is `[2, N]` BF16.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv_fp8w_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// The shared shape of `w8a16_gemv_batch4` / `w8a16_gemv_batch16` (contiguous
/// A and C), so a caller that picks its MAX_M tier by row count can hold the
/// wrapper and the handle as one pair instead of duplicating the call site.
/// The `_strided` pair's sibling alias lives with its own callers.
pub type ContiguousBatchGemv = fn(
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

/// Block-scaled FP8 batched GEMV (M<=4). `input` is `[M, K]` BF16, `output` is
/// `[M, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers (2D
/// block-scaled FP8). One pass over the FP8 weight serves all M rows — the M=4
/// sibling of `w8a16_gemv`, replacing `w8a16_gemm_pipelined` for n<=4 batched
/// decode (which pads M to a 128-row MMA tile). Bit-identical per-row to
/// `w8a16_gemv`. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
///
/// REFUSES m>4. The kernel is `w8a16_gemv_batchm_impl<4>`: at M=5 it computes
/// rows 0..3 and never writes rows 4.. — stale memory, not a launch failure.
/// Callers with 5..=16 rows want [`w8a16_gemv_batch16`], which takes the same
/// arguments and the same launch geometry (issue #927).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4(
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
        (1..=4).contains(&m),
        "w8a16_gemv_batch4: m={m} outside 1..=4 (kernel MAX_M; use w8a16_gemv_batch16)"
    );
    contiguous_batch_launch(
        gpu,
        kernel,
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

/// MAX_M=16 sibling of [`w8a16_gemv_batch4`], for decode concurrency 5..=16.
///
/// WHY (#927). On 1xH100 with Qwen/Qwen3.8-27B-FP8 the decode step measured
/// 44 ms at 4 active rows and 224 ms at 16 — C=16 aggregate FELL from 76 to
/// 62 tok/s when the batch cap went 4 -> 16, because every native-FP8 site
/// stopped at the M<=4 GEMV and handed 5..16 rows to the transposed /
/// pipelined tile GEMMs (5-12 TFLOP/s class, M padded to a 128-row MMA tile).
/// This kernel streams the weight ONCE for up to 16 rows instead.
///
/// Same template body, same K-iteration order and the same per-row reduction
/// tree as `w8a16_gemv_batch4`, so each row is bit-identical to the scalar
/// `w8a16_gemv` (H100 receipt on #932: M=8/16 `unequal_bf16=0`). The wider
/// register array is the only difference.
///
/// Kernel: `w8a16_gemv_batch16` (module `w8a16_gemv_batch4`).
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16(
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
        "w8a16_gemv_batch16: m={m} outside 1..=16 (kernel MAX_M)"
    );
    contiguous_batch_launch(
        gpu,
        kernel,
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

/// Shared launch body for the two contiguous entry points. Identical argument
/// order and geometry — the only thing that differs above is the MAX_M bound
/// the caller must respect, exactly as for the `_strided` pair below.
#[allow(clippy::too_many_arguments)]
fn contiguous_batch_launch(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Block-scaled FP8 dual-GEMV (batch=2). `input` is `[2, K]` BF16, `output` is
/// `[2, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Strided sibling of [`w8a16_gemv_batch4`] (M<=4).
///
/// WHY: the multi-sequence decode Q/K/V buffer is `[n, per_seq_qkv]` with Q at
/// offset 0, K after Q and V after K inside every row, so the contiguous
/// `[M, N]` writer cannot address one projection across rows. Without a
/// strided writer the native-FP8 attention projections fell back to three
/// scalar `w8a16_gemv` launches PER ROW at decode concurrency 2..=8 — the
/// third-largest bucket in the C=4 decode profile (issue #927). This writes one
/// projection for all M rows in ONE launch.
///
/// LAYOUT: `input` `[M, a_row_stride]` BF16, only the first `k` elements of
/// each row read; `weight`/`block_scale` are the raw `w8a16_gemv` pointers
/// (`[N, K]` FP8 E4M3 and `[N/128, K/128]` FP32); `output`
/// `[M, c_row_stride]` BF16, only the first `n` elements of each row written.
/// Both strides are in ELEMENTS. `a_row_stride` must keep each activation row
/// 16-byte aligned (multiple of 8) — the kernel's activation loads are `uint4`.
///
/// Bit-identical per row to `w8a16_gemv`: same template body, same K-iteration
/// order and same reduction tree as [`w8a16_gemv_batch4`]; only the row pitches
/// change. Verified by `examples/native_fp8_qkv_batch_microtest`.
///
/// Kernel: `w8a16_gemv_batch4_strided` (module `w8a16_gemv_batch4`).
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4_strided(
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
        (1..=4).contains(&m),
        "w8a16_gemv_batch4_strided: m={m} outside 1..=4 (kernel MAX_M)"
    );
    strided_batch_launch(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// MAX_M=16 sibling of [`w8a16_gemv_batch4_strided`], for decode concurrency
/// 5..=16. Same template, same launch geometry, same per-row accumulation
/// order; the wider register array is the only difference.
///
/// Kernel: `w8a16_gemv_batch16_strided` (module `w8a16_gemv_batch4`).
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16_strided(
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
        "w8a16_gemv_batch16_strided: m={m} outside 1..=16 (kernel MAX_M)"
    );
    strided_batch_launch(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// Shared launch body for the two `_strided` entry points — identical argument
/// order and geometry, so the only thing that differs above is the MAX_M bound
/// the caller must respect.
#[allow(clippy::too_many_arguments)]
fn strided_batch_launch(
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
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemv batch strided: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemv batch strided: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (uint4 activation loads)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
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
