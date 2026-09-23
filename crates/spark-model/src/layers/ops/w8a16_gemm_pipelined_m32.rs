// SPDX-License-Identifier: AGPL-3.0-only

//! `w8a16_gemm_pipelined_m32` — the 32-row M-tile twin of
//! [`super::w8a16_gemm_pipelined`], and THE rule for when the twin runs.
//!
//! WHY (G18, MoE Qwen3.6-35B-A3B-FP8 at C=16 / R=32 verify rows, nsys
//! 2026-09-22/23): the block-scaled W8A16 projections on the batched verify
//! path at 17..=32 rows all pay the 128-row MMA tile. The GDN
//! `in_proj_qkvz` (N=12288, K=2048) measured 296 us for 25.2 MB of weight and
//! `out_proj` (N=2048, K=4096, 64 CTAs on 48 SMs) 166 us for 8.4 MB — 3.6x
//! their bandwidth floor — while the attention Q/K/V, whose batched tier
//! stopped at 16 rows, fell to a 32-row scalar loop (128 launches and 32
//! weight passes per layer). One kernel serves both: a 32x32 tile with 128-K
//! steps and caller-supplied A/C row pitches (kernel header:
//! `kernels/gb10/common/w8a16_gemm_pipelined_m32.cu`).
//!
//! NUMERICS: bit-identical to `w8a16_gemm_pipelined` (same sub-MMA windows,
//! same fold points; oracle `examples/native_fp8_gdn_proj_m32_microtest`).
//! Against the scalar GEMV family it is the tensor-core reassociation every
//! `*_m16` tier carries (`layers::dense_ffn::m16_tc::oracle`).
//!
//! The twin lives in its own module (`w8a16_gemm_pipelined_m32`, one entry
//! point). Targets that do not carry the module resolve it through
//! `try_target_kernel`, so a zero handle keeps every caller on the kernel it
//! ran before this file existed.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Rows one M tile covers (`PM32_M_TILE`). Above it the kernel still runs —
/// `grid.y = ceil(M / 32)`, one full weight pass per tile — but a caller with
/// a contiguous alternative prefers the 128-tile kernel past this bound; see
/// [`w8a16_pipelined_prefers_m32`].
pub const W8A16_M32_TILE_ROWS: u32 = 32;

/// N tile (`PM32_N_TILE`) — grid.x granularity.
const W8A16_M32_TILE_COLS: u32 = 32;

/// K granularity: one FP8 scale block per resident K-step (`PM32_K_STEP ==
/// PM32_FP8_BLOCK`), which is what makes the fold points line up with the
/// 128-tile kernel's and is why `K % 128 == 0` is REQUIRED, not padded.
pub const W8A16_M32_K_BLOCK: u32 = 128;

/// THE selection rule between the 128-tile `w8a16_gemm_pipelined` and its
/// 32-tile twin at a contiguous call site: the twin takes `1..=32` rows when
/// it is linked and `K` is whole scale blocks. Above 32 rows the 128 tile's
/// single pass wins back its padding, and the twin's `grid.y > 1` re-streams
/// the weight per tile; a `K % 128 != 0` shape (none of the GDN or attention
/// callers today) keeps the 128 tile, which tolerates a partial block.
///
/// One function, consulted by [`w8a16_gemm_pipelined_by_m`] and pinned by
/// the tests, so the GDN in_proj/out_proj arms and any future caller cannot
/// disagree on the boundary.
pub fn w8a16_pipelined_prefers_m32(m: u32, k: u32, m32_kernel: KernelHandle) -> bool {
    (1..=W8A16_M32_TILE_ROWS).contains(&m)
        && k >= W8A16_M32_K_BLOCK
        && k.is_multiple_of(W8A16_M32_K_BLOCK)
        && m32_kernel.0 != 0
}

/// Strided launch of `w8a16_gemm_pipelined_m32`.
///
/// `input` `[M, a_row_stride]` BF16 (first `k` of each row read), `weight`
/// `[N, K]` FP8 E4M3 with `block_scale` `[N/128, K/128]` FP32, `output`
/// `[M, c_row_stride]` BF16 (first `n` of each row written). Strides in
/// ELEMENTS. The A rows feed 16-byte `cp.async` chunks, so `a_row_stride`
/// must keep them 16 B-aligned (multiple of 8).
///
/// Same argument order as `w8a16_gemv_batch{4,16}_strided`, so the multi-seq
/// Q/K/V tier holds it as one more `StridedBatchGemv` arm.
///
/// Grid: (ceil(N/32), ceil(M/32), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined_m32_strided(
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
        m >= 1,
        "w8a16_gemm_pipelined_m32: m=0 has no rows to compute"
    );
    ensure!(
        k >= W8A16_M32_K_BLOCK && k.is_multiple_of(W8A16_M32_K_BLOCK),
        "w8a16_gemm_pipelined_m32: K={k} must be a positive multiple of \
         {W8A16_M32_K_BLOCK} (one FP8 scale block per K-step)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemm_pipelined_m32: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemm_pipelined_m32: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (cp.async A tile)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, W8A16_M32_TILE_COLS),
            div_ceil(m, W8A16_M32_TILE_ROWS),
            1,
        ])
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

/// Contiguous `[M, K]` -> `[M, N]` launch of the same kernel: the pitches are
/// `k` and `n`. Same signature as [`super::w8a16_gemm_pipelined`] so a caller
/// can hold either as one `fn` pointer.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined_m32(
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
    w8a16_gemm_pipelined_m32_strided(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        k,
        n,
        stream,
    )
}

/// Contiguous block-scaled W8A16 GEMM, tile chosen by `m`
/// ([`w8a16_pipelined_prefers_m32`]): the 32-tile twin at 1..=32 rows when
/// `m32_kernel` is linked, else the 128-tile `w8a16_gemm_pipelined` on
/// `full_kernel`. The GDN `in_proj_qkvz` / `out_proj` batched-verify arms
/// call this instead of the 128-tile wrapper directly.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined_by_m(
    gpu: &dyn GpuBackend,
    full_kernel: KernelHandle,
    m32_kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if w8a16_pipelined_prefers_m32(m, k, m32_kernel) {
        w8a16_gemm_pipelined_m32(
            gpu,
            m32_kernel,
            input,
            weight,
            block_scale,
            output,
            m,
            n,
            k,
            stream,
        )
    } else {
        super::w8a16_gemm_pipelined(
            gpu,
            full_kernel,
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
}

#[cfg(test)]
#[path = "w8a16_gemm_pipelined_m32_tests.rs"]
mod tests;
