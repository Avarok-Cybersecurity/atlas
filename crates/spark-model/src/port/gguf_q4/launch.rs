// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-cuda/mmvq.cu and mmq.cuh (launch geometry: one block a row for mmvq, 128 x 128
// tiles of 8 warps for MMQ, the q8_1 activation layouts and their byte sizes), ggml/src/ggml-cuda/quantize.cu
// (the MMQ quantizer grid). Changes: translated from the C++ host launchers to Rust launch helpers over
// Atlas's KernelLaunch, by Atlas contributors.

//! Responsibility: the kernel handles and launch helpers of the keep-packed GGUF 4-bit route (port lane
//! L19): one function per kernel family in `kernels/ada/common/` (`kquant_q8_1_ada.cu`,
//! `kquant_mmvq_q4_ada.cu`, `iq4_xs_mmq_ada.cu`, `dequant_iq4xs_q5k_bf16_ada.cu`) plus the gb10-common
//! `dequant_gguf_bf16` the Ada tree links, and the byte arithmetic of every buffer they touch.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Weights a super-block.
pub const QK_K: usize = 256;
/// Values a `block_q8_1` (and bytes of one: 32 int8 plus a half2 of scale and sum).
pub const QK8_1: usize = 32;
pub const Q8_1_BLOCK_BYTES: usize = 36;
/// Rows one decode launch takes (`ATLAS_Q4_MAX_M` in `kquant_mmvq_q4_ada.cu`).
pub const MAX_DECODE_ROWS: usize = 8;
/// Dynamic shared memory of every 128 x 128 MMQ entry (`ATLAS_MMQ128_SMEM`, asserted in the `.cu`).
pub const GGUF_Q4_MMQ_SMEM: u32 = 57_856;
/// Threads a block of the MMQ activation quantizer (`CUDA_QUANTIZE_BLOCK_SIZE_MMQ`).
const QUANTIZE_BLOCK_MMQ: u32 = 128;

/// Modules (file stems) of the route.
pub const Q8_MODULE: &str = "kquant_q8_1_ada";
pub const MMVQ_MODULE: &str = "kquant_mmvq_q4_ada";
pub const MMQ_MODULE: &str = "iq4_xs_mmq_ada";
pub const DEQUANT_ADA_MODULE: &str = "dequant_iq4xs_q5k_bf16_ada";
/// gb10's GGUF dequant (Q4_K, Q6_K), linked into the Ada tree through `kernels/ada/common`.
pub const DEQUANT_GB10_MODULE: &str = "dequant_gguf_bf16";

/// The three packed GGUF types this route serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Q4Type {
    /// ggml id 23: 136 bytes a super-block, 4.25 bits a weight.
    Iq4Xs,
    /// ggml id 12: 144 bytes, 4.5 bits.
    Q4K,
    /// ggml id 14: 210 bytes, 6.5625 bits.
    Q6K,
}

/// The two decode GEMV shapes of `kquant_mmvq_q4_ada.cu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmvqShape {
    /// One block (4 warps) per output row: the shape the route serves.
    Block,
    /// One warp per output row, four rows a block: the microtest's A/B arm.
    Warp,
}

impl Q4Type {
    pub fn block_bytes(self) -> usize {
        match self {
            Q4Type::Iq4Xs => 136,
            Q4Type::Q4K => 144,
            Q4Type::Q6K => 210,
        }
    }

    /// The ggml type id.
    pub fn ggml_id(self) -> u32 {
        match self {
            Q4Type::Iq4Xs => 23,
            Q4Type::Q4K => 12,
            Q4Type::Q6K => 14,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Q4Type::Iq4Xs => "iq4_xs",
            Q4Type::Q4K => "q4_k",
            Q4Type::Q6K => "q6_k",
        }
    }

    /// The decode GEMV entry of `shape`.
    pub fn mmvq_entry(self, shape: MmvqShape) -> String {
        match shape {
            MmvqShape::Block => format!("atlas_mmvq_{}_ada", self.suffix()),
            MmvqShape::Warp => format!("atlas_mmvq_{}_w_ada", self.suffix()),
        }
    }

    /// The MMQ entries `(_nc, _wc)`.
    pub fn mmq_entries(self) -> (String, String) {
        let s = self.suffix();
        (
            format!("atlas_{s}_mmq128_nc"),
            format!("atlas_{s}_mmq128_wc"),
        )
    }

    /// The MMQ activation quantizer: D4 for IQ4_XS and Q6_K, DS4 for Q4_K (`mmq_get_q8_1_ds_layout`).
    pub fn mmq_act_entry(self) -> &'static str {
        match self {
            Q4Type::Q4K => "atlas_q8_1_mmq_ds4_bf16_ada",
            Q4Type::Iq4Xs | Q4Type::Q6K => "atlas_q8_1_mmq_d4_bf16_ada",
        }
    }

    /// `(module, entry)` of the BF16 dequant.
    pub fn dequant_entry(self) -> (&'static str, &'static str) {
        match self {
            Q4Type::Iq4Xs => (DEQUANT_ADA_MODULE, "dequant_iq4_xs_to_bf16_ada"),
            Q4Type::Q4K => (DEQUANT_GB10_MODULE, "dequant_q4_k_to_bf16"),
            Q4Type::Q6K => (DEQUANT_GB10_MODULE, "dequant_q6_k_to_bf16"),
        }
    }
}

/// Bytes of an `[n, k]` packed weight.
pub fn weight_bytes(ty: Q4Type, n: usize, k: usize) -> usize {
    n * (k / QK_K) * ty.block_bytes()
}

/// Bytes of the plain `block_q8_1 [m][k/32]` decode activations.
pub fn q8_rows_bytes(m: usize, k: usize) -> usize {
    m * (k / QK8_1) * Q8_1_BLOCK_BYTES
}

/// Bytes of the MMQ-layout activations for `[m, k]`: 144 bytes per 128 values, rows padded to the
/// 128-row tile (the tile loader reads whole tiles), `k` a multiple of 256.
pub fn mmq_act_bytes(m: usize, k: usize) -> usize {
    m.div_ceil(128) * 128 * (k / 128) * 144
}

/// Workspace for up to `max_rows` activation rows: the decode rows buffer, and for `max_rows` beyond the
/// decode tier the MMQ buffer, rounded up to 256 bytes.
pub fn scratch_bytes(k: usize, max_rows: usize) -> usize {
    let decode = q8_rows_bytes(max_rows.clamp(1, MAX_DECODE_ROWS), k);
    let prefill = if max_rows > MAX_DECODE_ROWS {
        mmq_act_bytes(max_rows, k)
    } else {
        0
    };
    decode.max(prefill).div_ceil(256) * 256
}

/// Every handle one packed weight launches.
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    pub q8_rows: KernelHandle,
    pub mmvq_block: KernelHandle,
    pub mmvq_warp: KernelHandle,
    pub mmq_act: KernelHandle,
    pub mmq_nc: KernelHandle,
    pub mmq_wc: KernelHandle,
    pub dequant: KernelHandle,
}

/// True when this build carries the route's modules (the Ada targets). No kernel lookup is issued, so a
/// target without them (GB10 and every other) records nothing in the boot audit.
pub fn available(gpu: &dyn GpuBackend, ty: Q4Type) -> bool {
    [Q8_MODULE, MMVQ_MODULE, MMQ_MODULE, ty.dequant_entry().0]
        .iter()
        .all(|m| gpu.has_module(m))
}

impl Kernels {
    /// Resolve every handle for `ty`. Errors when the build lacks a module (checked first, without a
    /// lookup) or an entry.
    pub fn resolve(gpu: &dyn GpuBackend, ty: Q4Type) -> Result<Self> {
        if !available(gpu, ty) {
            bail!(
                "gguf_q4: the keep-packed {ty:?} kernels are not in this build (modules {Q8_MODULE}, \
                 {MMVQ_MODULE}, {MMQ_MODULE}, {}); build the ada target",
                ty.dequant_entry().0
            );
        }
        let k = |m: &str, f: &str| {
            gpu.kernel(m, f)
                .with_context(|| format!("gguf_q4: {m}::{f}"))
        };
        let (nc, wc) = ty.mmq_entries();
        let (dm, df) = ty.dequant_entry();
        Ok(Self {
            q8_rows: k(Q8_MODULE, "atlas_q8_1_rows_bf16_ada")?,
            mmvq_block: k(MMVQ_MODULE, &ty.mmvq_entry(MmvqShape::Block))?,
            mmvq_warp: k(MMVQ_MODULE, &ty.mmvq_entry(MmvqShape::Warp))?,
            mmq_act: k(Q8_MODULE, ty.mmq_act_entry())?,
            mmq_nc: k(MMQ_MODULE, &nc)?,
            mmq_wc: k(MMQ_MODULE, &wc)?,
            dequant: k(dm, df)?,
        })
    }
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).with_context(|| format!("gguf_q4: {what} = {v} does not fit u32"))
}

/// bf16 `[m, k]` -> `block_q8_1 [m][k/32]` (`atlas_q8_1_rows_bf16_ada`). `k % 32 == 0`.
pub fn q8_rows(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    x: DevicePtr,
    out_q8: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(QK8_1),
        "q8_rows: k={k} is not a multiple of 32"
    );
    let threads = to_u32(m * (k / QK8_1) * 32, "q8_rows threads")?;
    KernelLaunch::new(gpu, kern.q8_rows)
        .grid([threads.div_ceil(128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(x)
        .arg_ptr(out_q8)
        .arg_u32(to_u32(k, "k")?)
        .arg_u32(to_u32(m, "m")?)
        .launch(stream)
}

/// Decode GEMV on the raw blocks: `out[m][n] = sum_k q8[m][k] * W[n][k]`, `1 <= m <= 8`, bf16 out.
#[allow(clippy::too_many_arguments)]
pub fn mmvq(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    shape: MmvqShape,
    w: DevicePtr,
    y_q8: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    m: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=MAX_DECODE_ROWS).contains(&m),
        "mmvq: m={m} outside 1..={MAX_DECODE_ROWS}"
    );
    ensure!(
        k.is_multiple_of(QK_K),
        "mmvq: k={k} is not a multiple of {QK_K}"
    );
    let (kernel, grid_x) = match shape {
        MmvqShape::Block => (kern.mmvq_block, to_u32(n, "n")?),
        MmvqShape::Warp => (kern.mmvq_warp, to_u32(n.div_ceil(4), "n / 4")?),
    };
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([32, 4, 1])
        .arg_ptr(w)
        .arg_ptr(y_q8)
        .arg_ptr(out)
        .arg_u32(to_u32(k, "k")?)
        .arg_u32(to_u32(n, "n")?)
        .arg_u32(to_u32(m, "m")?)
        .launch(stream)
}

/// bf16 `[m, k]` -> the MMQ q8_1 layout of the weight type (`atlas_q8_1_mmq_{d4,ds4}_bf16_ada`).
pub fn mmq_act(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    x: DevicePtr,
    out_q8: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(QK_K),
        "mmq_act: k={k} is not a multiple of {QK_K}"
    );
    let grid_y = to_u32(k.div_ceil(4 * QUANTIZE_BLOCK_MMQ as usize), "k / 512")?;
    KernelLaunch::new(gpu, kern.mmq_act)
        .grid([to_u32(m, "m")?, grid_y, 1])
        .block([QUANTIZE_BLOCK_MMQ, 1, 1])
        .arg_ptr(x)
        .arg_ptr(out_q8)
        .arg_u64(k as u64) // ne00
        .arg_u64(k as u64) // s01: contiguous rows
        .arg_u64(k as u64) // ne0: k is already a multiple of 256
        .arg_u32(to_u32(m, "m")?) // ne1
        .launch(stream)
}

/// Prefill GEMM through the vendored MMQ engine: `out[m][n]` (bf16) = `A_q8[m][k] x W[n][k]`.
#[allow(clippy::too_many_arguments)]
pub fn mmq_gemm(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    a_q8: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(QK_K),
        "mmq_gemm: k={k} is not a multiple of {QK_K}"
    );
    let (m32, n32, k32) = (to_u32(m, "m")?, to_u32(n, "n")?, to_u32(k, "k")?);
    let kernel = if n.is_multiple_of(128) {
        kern.mmq_nc
    } else {
        kern.mmq_wc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([n32.div_ceil(128), m32.div_ceil(128), 1])
        .block([32, 8, 1])
        .shared_mem(GGUF_Q4_MMQ_SMEM)
        .arg_ptr(w) // x = weights
        .arg_ptr(a_q8) // y = q8_1 activations
        .arg_ptr(out) // dst
        .arg_u32(n32) // nrows_x
        .arg_u32(m32) // ncols_dst
        .arg_u32(k32) // ncols_x
        .arg_u32(k32 / QK_K as u32) // stride_row_x
        .arg_u32(m32) // ncols_y
        .arg_u32(n32) // stride_col_dst
        .launch(stream)
}

/// The weight expanded to bf16 `[n, k]` (one CUDA block per super-block).
#[allow(clippy::too_many_arguments)]
pub fn dequant(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    ty: Q4Type,
    w: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let n_blocks = to_u32(n * (k / QK_K), "super-blocks")?;
    KernelLaunch::new(gpu, kern.dequant)
        .grid([n_blocks, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(n_blocks)
        .arg_u32(to_u32(ty.block_bytes(), "block bytes")?)
        .launch(stream)
}
