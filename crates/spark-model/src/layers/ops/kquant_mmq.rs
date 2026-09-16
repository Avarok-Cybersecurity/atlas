// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Host launches for the DeepSeek-V4.1 K-quant expert kernels (`kquant_moe` module,
//! `kernels/gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu`): raw Q2_K / Q3_K expert
//! blocks consumed in place. Decode = `kquant_q8_1_rows` + `kquant_mmvq`; prefill =
//! the MMQ activation quantiser in the type's q8_1 layout (launched through
//! [`super::quantize_act_q8_1`] with the right kernel handle) + [`kquant_mmq_gemm`].

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::q4k_mmq::QK_K;

pub const KQUANT_MODULE: &str = "kquant_moe";
/// Bytes per 256-value super-block on disk.
pub const Q2K_BLOCK_BYTES: usize = 84;
pub const Q3K_BLOCK_BYTES: usize = 110;

/// A resident `[N, K]` row-major projection as the GGUF path left it on the
/// device: expanded bf16, or the raw `Q2_K` / `Q3_K` blocks
/// (`WeightDtype::Q2K` / `Q3K`) for the K-quant GEMV (m <= 8, q8_1 rows) and
/// MMQ (tensor cores, D2S6 / D4 tiles) kernels.
#[derive(Clone, Copy, Debug)]
pub enum ResidentMat {
    Bf16(DevicePtr),
    Q2K(DevicePtr),
    Q3K(DevicePtr),
}

impl ResidentMat {
    /// The sub-matrix starting `rows` rows in (each row `k` weights long).
    pub fn at_rows(self, rows: usize, k: usize) -> ResidentMat {
        let at = |p: DevicePtr, off: usize| DevicePtr(p.0 + off as u64);
        match self {
            ResidentMat::Bf16(p) => ResidentMat::Bf16(at(p, rows * k * 2)),
            ResidentMat::Q2K(p) => ResidentMat::Q2K(at(p, rows * (k / 256) * Q2K_BLOCK_BYTES)),
            ResidentMat::Q3K(p) => ResidentMat::Q3K(at(p, rows * (k / 256) * Q3K_BLOCK_BYTES)),
        }
    }
}
/// Dynamic shared memory for the 128x128 MMQ tile on the MMA path (`mmq_get_nbytes_shared`
/// with MMQ_TILE_NE_K = 32): ids 512 B + tile_x (128 rows x tile_x_k ints) + tile_y (128 x 144 B).
/// Q2_K tile_x_k = 2*32 + 32 + 4 = 100; Q3_K = 2*32 + 16 + 4 = 84.
pub const Q2K_MMQ_SMEM: u32 = 512 + 128 * 100 * 4 + 128 * 144;
pub const Q3K_MMQ_SMEM: u32 = 512 + 128 * 84 * 4 + 128 * 144;
/// `block_q8_1`: 32 int8 + half2 (d, sum).
pub const Q8_1_BLOCK_BYTES: usize = 36;

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Raw bytes of an `[n, k]` K-quant weight.
pub fn kquant_weight_bytes(n: u32, k: u32, block_bytes: usize) -> usize {
    (n as usize) * (k as usize / QK_K as usize) * block_bytes
}

/// Bytes of the plain `block_q8_1` buffer for `[m, k]` activations.
pub fn kquant_q8_1_rows_bytes(m: u32, k: u32) -> usize {
    (m as usize) * (k as usize / 32) * Q8_1_BLOCK_BYTES
}

/// Bytes of the MMQ-layout q8_1 buffer for `[m, k]` activations (what
/// `quantize_act_q8_1` writes: 144 B per 128 values, rows padded to the 128 tile,
/// k padded to 256).
pub fn kquant_mmq_act_bytes(m: u32, k: u32) -> usize {
    let mpad = div_ceil(m, 128) * 128;
    let kpad = div_ceil(k, QK_K) * QK_K;
    (mpad as usize) * (kpad as usize / 128) * 144
}

/// bf16 `[m, k]` -> `block_q8_1 [m, k/32]` (`kquant_q8_1_rows_bf16`). `k % 32 == 0`.
pub fn kquant_q8_1_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x_bf16: DevicePtr,
    out_q8: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        k % 32 == 0,
        "kquant_q8_1_rows: k={k} is not a multiple of 32"
    );
    let warps = m * (k / 32);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(warps * 32, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(x_bf16)
        .arg_ptr(out_q8)
        .arg_u32(k)
        .arg_u32(m)
        .launch(stream)
}

/// Decode GEMV on raw blocks: `out[m][n] = sum_k xq8[m][k] * W[n][k]`, `m <= 8`
/// (`kquant_mmvq_q2_k` / `kquant_mmvq_q3_k`). `w_blocks` is `[n][k/256]` raw
/// super-blocks; `y_q8` is the `block_q8_1 [m][k/32]` buffer; `out_bf16` is `[m][n]`.
pub fn kquant_mmvq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_blocks: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(m >= 1 && m <= 8, "kquant_mmvq: m={m} outside 1..=8");
    anyhow::ensure!(
        k % QK_K == 0,
        "kquant_mmvq: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([32, 4, 1])
        .arg_ptr(w_blocks)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .launch(stream)
}

/// Prefill GEMM on raw blocks through the vendored MMQ engine:
/// `out[m][n]` (bf16) = `A_q8[m][k]` x `W[n][k]`. `a_q8` must be in the type's MMQ
/// q8_1 layout (D2S6 for Q2_K, D4 for Q3_K); `smem` is the matching `*_MMQ_SMEM`.
#[allow(clippy::too_many_arguments)]
/// `kquant_mmvq_*_experts`: one launch over `n_experts` experts whose block
/// pointers sit in the device table `vxs` (`n_experts` pointers). The
/// activation for expert `e` is `y_q8 + e * y_stride_bytes` (0 = shared), its
/// output `out_bf16 + e * m * n`. Same per-row math as [`kquant_mmvq`].
pub fn kquant_mmvq_experts(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    vxs: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    n_experts: u32,
    y_stride_bytes: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(m >= 1 && m <= 8, "kquant_mmvq_experts: m={m} outside 1..=8");
    anyhow::ensure!(
        k % QK_K == 0,
        "kquant_mmvq_experts: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([n, n_experts, 1])
        .block([32, 4, 1])
        .arg_ptr(vxs)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(y_stride_bytes)
        .launch(stream)
}

/// `kquant_mmvq_*_w`: the warp-per-row GEMV (four rows a block, no shared
/// memory), same arguments as [`kquant_mmvq`].
pub fn kquant_mmvq_w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_blocks: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(m >= 1 && m <= 8, "kquant_mmvq_w: m={m} outside 1..=8");
    anyhow::ensure!(
        k % QK_K == 0,
        "kquant_mmvq_w: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([32, 4, 1])
        .arg_ptr(w_blocks)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .launch(stream)
}

/// `kquant_mmvq_*_experts_w`: the warp-per-row expert batch, same arguments
/// as [`kquant_mmvq_experts`].
pub fn kquant_mmvq_experts_w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    vxs: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    n_experts: u32,
    y_stride_bytes: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m >= 1 && m <= 8,
        "kquant_mmvq_experts_w: m={m} outside 1..=8"
    );
    anyhow::ensure!(
        k % QK_K == 0,
        "kquant_mmvq_experts_w: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), n_experts, 1])
        .block([32, 4, 1])
        .arg_ptr(vxs)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(y_stride_bytes)
        .launch(stream)
}

pub fn kquant_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_q8: DevicePtr,
    w_blocks: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        k % QK_K == 0,
        "kquant_mmq_gemm: k={k} is not a multiple of {QK_K}"
    );
    let kernel = if n.is_multiple_of(128) {
        kernel_nc
    } else {
        kernel_wc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([32, 8, 1])
        .shared_mem(smem)
        .arg_ptr(w_blocks) // x = weights
        .arg_ptr(a_q8) // y = q8_1 activations
        .arg_ptr(out_bf16) // dst
        .arg_u32(n) // nrows_x
        .arg_u32(m) // ncols_dst
        .arg_u32(k) // ncols_x
        .arg_u32(k / QK_K) // stride_row_x = K/256
        .arg_u32(m) // ncols_y
        .arg_u32(n) // stride_col_dst
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_block_geometry() {
        // one V4.1 expert gate slice: 2304 rows x 5120, Q2_K
        assert_eq!(kquant_weight_bytes(2304, 5120, Q2K_BLOCK_BYTES), 3_870_720);
        // its down slice: 5120 rows x 2304, Q3_K
        assert_eq!(kquant_weight_bytes(5120, 2304, Q3K_BLOCK_BYTES), 5_068_800);
        assert_eq!(kquant_q8_1_rows_bytes(1, 5120), 160 * 36);
        assert_eq!(kquant_mmq_act_bytes(200, 5120), 256 * 40 * 144);
        assert_eq!((Q2K_MMQ_SMEM, Q3K_MMQ_SMEM), (70_144, 61_952));
    }
}
