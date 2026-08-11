// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for CANDIDATE B of the native keep-packed ternary Q2_0
//! decode GEMV (`kernels/gb10/common/q2_0_gemv_vec.cu`, module stem
//! `q2_0_gemv_vec`).
//!
//! Same call surface as [`super::gemv_q2`] — the Q2_0 weight carries its fp16
//! scale INLINE in each `block_q2_0`, so there is no separate scale-pointer
//! argument. The only launch-geometry difference is the CANDIDATE-B thread map:
//! ONE warp per output row, EIGHT rows per 256-thread CTA, so the grid is
//! `(ceil(N/8),1,1)` (vs the 2-warp/4-row `ceil(N/4)` of the baseline kernel).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// Q2_0 GEMV (M=1 decode), CANDIDATE B: `C[1,N] = A[1,K] @ dequant(B)`.
///
/// Vectorized code loads (one `uint32` = 16 ternary codes per lane) + shared-
/// memory activation staging. `A` BF16 `[1,K]`, `B` raw `block_q2_0`, `C` BF16
/// `[1,N]`. Dequant `(code-1)*d` happens inside the dot-product.
///
/// Kernel: `q2_0_gemv_vec(A, B, C, N, K, group)`  Grid: (ceil(N/8),1,1) Block: (256,1,1)
pub fn q2_0_gemv_vec(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .launch(stream)
}

/// Shared-memory row cap of `q2_0_gemv_vec_batchm` (`MAX_M` in the .cu). The
/// kernel stages exactly `M` activation rows in `s_A[MAX_M * TILE_K]`; passing
/// `M > MAX_M` overflows that tile (OOB smem write) AND drops output rows >= 8
/// (compute/write loops iterate `m < MAX_M`). Callers with more rows MUST chunk
/// — done transparently by [`q2_0_gemv_vec_batchm`].
pub const Q2_BATCHM_MAX_M: u32 = 8;

/// Row-group boundaries for driving the batchm kernel at arbitrary `m`: yields
/// `(r0, m_chunk)` with `1 <= m_chunk <= Q2_BATCHM_MAX_M`, contiguous from 0,
/// summing to `m` (empty when `m == 0`). Each output row is independent and the
/// kernel is bit-consistent with M=1, so splitting by rows is numerically inert.
fn batchm_row_chunks(m: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..m)
        .step_by(Q2_BATCHM_MAX_M as usize)
        .map(move |r0| (r0, (m - r0).min(Q2_BATCHM_MAX_M)))
}

/// Q2_0 batched GEMV (M>=1 decode), CANDIDATE B: `C[M,N] = A[M,K] @ dequant(B)`.
///
/// Reads each weight word once and MAC's it into all `m` accumulators (all `m`
/// activation rows staged in smem). `A` BF16 `[M,K]` row-major, `C` BF16
/// `[M,N]` row-major. Bit-consistent with running the M=1 kernel `M` times.
///
/// The kernel itself is capped at `Q2_BATCHM_MAX_M` rows/launch, so `m` beyond
/// that is served by CHUNKING: successive <=8-row launches with the `[M,K]`
/// input and `[M,N]` output base pointers advanced by whole rows (BF16 = 2 B).
/// Chunking a caller with `m <= 8` costs one launch (identical to the direct
/// call); it exists so a wide concurrent-decode step (max-num-seqs up to 16)
/// can never drive the kernel into its OOB path.
///
/// Kernel: `q2_0_gemv_vec_batchm(A, B, C, N, K, group, M)`.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_gemv_vec_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    // BF16 row strides: input is [M,K], output is [M,N].
    let k_row_bytes = weight.k as usize * 2;
    let n_row_bytes = weight.n as usize * 2;
    for (r0, m_chunk) in batchm_row_chunks(m) {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(weight.n, 8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input.offset(r0 as usize * k_row_bytes))
            .arg_ptr(weight.weight)
            .arg_ptr(output.offset(r0 as usize * n_row_bytes))
            .arg_u32(weight.n)
            .arg_u32(weight.k)
            .arg_u32(weight.group as u32)
            .arg_u32(m_chunk)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::batchm_row_chunks;

    use half::{bf16, f16};

    // Weight scale `d` is fp16-precise (stored inline fp16 in each block_q2_0);
    // round-trip it exactly as the kernel's inline scale read does.
    fn f16_rt(x: f32) -> f32 {
        f16::from_f32(x).to_f32()
    }

    /// One output row of the Q2_0 GEMV — `out[n] = sum_k a[k] * (code(n,k)-1) *
    /// d(n, k/128)` in fp32, then BF16 store. This is BOTH the M=1 kernel and any
    /// single `m`-row of `q2_0_gemv_vec_batchm` (the batchm kernel keeps M
    /// independent fp32 accumulators over the SAME per-row arithmetic), so a
    /// batched output row must equal this bit-for-bit. Returns BF16 bit patterns.
    fn gemv_row(a: &[f32], codes: &[u8], scales: &[f32], n: usize, k: usize) -> Vec<u16> {
        let bpr = k / 128; // group-128 blocks per weight row
        let mut out = vec![0u16; n];
        for (col, o) in out.iter_mut().enumerate() {
            let mut acc = 0f32;
            for ki in 0..k {
                let d = f16_rt(scales[col * bpr + ki / 128]);
                let w = ((codes[col * k + ki] & 0x3) as i32 - 1) as f32 * d;
                acc += a[ki] * w;
            }
            *o = bf16::from_f32(acc).to_bits();
        }
        out
    }

    fn gen_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let mut act = vec![0f32; m * k];
        for (i, a) in act.iter_mut().enumerate() {
            *a = (i as f32 * 0.09131).sin() * 1.3;
        }
        let mut codes = vec![0u8; n * k];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = (((i * 2654435761usize) >> 6) % 3) as u8; // ternary {0,1,2}
        }
        let mut scales = vec![0f32; n * (k / 128)];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0.015 + 0.01 * ((i % 5) as f32);
        }
        (act, codes, scales)
    }

    #[test]
    fn batchm_row_chunks_cover_all_rows_exactly() {
        for &m in &[0u32, 1, 2, 3, 7, 8, 9, 15, 16, 17] {
            let chunks: Vec<(u32, u32)> = batchm_row_chunks(m).collect();
            let mut next_r0 = 0u32;
            let mut total = 0u32;
            for &(r0, mc) in &chunks {
                assert_eq!(r0, next_r0, "M={m}: chunk starts must be contiguous");
                assert!(
                    (1..=super::Q2_BATCHM_MAX_M).contains(&mc),
                    "M={m}: bad chunk {mc}"
                );
                next_r0 += mc;
                total += mc;
            }
            assert_eq!(total, m, "M={m}: chunks must cover every row once");
            assert_eq!(chunks.is_empty(), m == 0);
        }
    }

    /// The invariant the wiring relies on: driving the batchm kernel via the
    /// chunking loop (row-group launches with `r0*K` / `r0*N` pointer offsets)
    /// reproduces the per-row M=1 GEMV for EVERY row, bit-for-bit — including the
    /// M>8 chunk boundaries. Models the launch loop in `q2_0_gemv_vec_batchm`.
    #[test]
    fn chunked_batchm_equals_per_row_gemv() {
        let (n, k) = (5usize, 256usize);
        for &m in &[1usize, 2, 3, 8, 9, 16] {
            let (act, codes, scales) = gen_inputs(m, n, k);

            // Reference: each row through the standalone M=1 GEMV.
            let mut reference = vec![0u16; m * n];
            for row in 0..m {
                let r = gemv_row(&act[row * k..(row + 1) * k], &codes, &scales, n, k);
                reference[row * n..(row + 1) * n].copy_from_slice(&r);
            }

            // Simulate the wrapper: iterate chunk boundaries, offset the [M,K]
            // input by r0*K and write [M,N] output at r0*N per chunk row.
            let mut got = vec![0u16; m * n];
            for (r0, mc) in batchm_row_chunks(m as u32) {
                for i in 0..mc as usize {
                    let row = r0 as usize + i;
                    let r = gemv_row(&act[row * k..(row + 1) * k], &codes, &scales, n, k);
                    got[row * n..(row + 1) * n].copy_from_slice(&r);
                }
            }
            assert_eq!(
                got, reference,
                "M={m}: chunked batchm must equal per-row M=1 GEMV bit-for-bit"
            );
        }
    }
}
