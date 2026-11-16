// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:358-452 (AffineNKernels: resolve, packAt, packTable, decode). Copyright 2026
// TensorFold contributors. Licensed under the Apache License, Version 2.0; code written before
// TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md. Changes for
// Atlas: Zig to Rust over Atlas's `GpuBackend` and `KernelLaunch`; lazy, target-gated lookups
// (`try_target_kernel`); scalar kernel arguments instead of a by-value Part; the split form's work and
// tickets come from the caller's `W4Scratch`; quantizer and dequant launchers are Atlas additions.
//! Responsibility: resolve the `affine_n_ada` kernels for one width and launch them: input group
//! sums, the decode GEMV (unsplit or split-K), the BF16 quantizer, the two tile packers, the table
//! transpose and the tile dequant. Arguments are plain device pointers and validated sizes; the
//! weight object and the `W4Linear` contract live in the parent module.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::layout;

/// Kernel module: the stem of `kernels/ada/common/affine_n_ada.cu` (built on Ada only).
pub const MODULE: &str = "affine_n_ada";
/// 64-group input sums.
pub const SUMS: &str = "affine_n_sums_ada";
/// BF16 to MLX quantizer.
pub const QUANTIZE: &str = "affine_n_quantize_ada";
/// MLX words to the narrow (2/3/5/6-bit) tiles.
pub const PACK_NARROW: &str = "affine_n_pack_ada";
/// MLX words to the 8-bit tiles.
pub const PACK_WIDE: &str = "affine8_pack_ada";
/// `(n, kg)` tables to `(kg, npad)`.
pub const TRANSPOSE: &str = "affine_n_transpose_pad16_ada";
/// Tiles to BF16 rows.
pub const DEQUANT: &str = "affine_n_dequant_bf16_ada";
/// Threads per block of the one-dimensional helper kernels.
const HELPER_THREADS: u32 = 256;

/// The gemv entry of one width.
pub fn gemv_name(bits: u8) -> String {
    format!("affine{bits}_gemv_ada")
}

/// The split-K entry of one width.
pub fn split_name(bits: u8) -> String {
    format!("affine{bits}_split_ada")
}

/// True when this target built the module (Ada only); no lookup is issued on any other target.
pub fn available(gpu: &dyn GpuBackend) -> bool {
    gpu.has_module(MODULE)
}

/// Every handle one width needs, resolved once per weight.
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    pub gemv: KernelHandle,
    pub split: KernelHandle,
    pub sums: KernelHandle,
    pub quantize: KernelHandle,
    pub pack: KernelHandle,
    pub transpose: KernelHandle,
    pub dequant: KernelHandle,
}

impl Kernels {
    /// Resolve the handles of `bits`; an error (never a panic) when the module is not built here or
    /// an entry is missing, so the caller keeps its current route.
    pub fn resolve(gpu: &dyn GpuBackend, bits: u8) -> Result<Self> {
        if layout::index(bits).is_none() {
            bail!("affine widths: no kernels for {bits}-bit codes");
        }
        if !available(gpu) {
            bail!("affine widths: kernel module {MODULE} is not built for this target (Ada only)");
        }
        let get = |name: &str| -> Result<KernelHandle> {
            let h = crate::layers::try_target_kernel(gpu, MODULE, name);
            if h.0 == 0 {
                bail!("affine widths: {MODULE}::{name} did not resolve");
            }
            Ok(h)
        };
        Ok(Self {
            gemv: get(&gemv_name(bits))?,
            split: get(&split_name(bits))?,
            sums: get(SUMS)?,
            quantize: get(QUANTIZE)?,
            pack: get(if layout::is_wide(bits) {
                PACK_WIDE
            } else {
                PACK_NARROW
            })?,
            transpose: get(TRANSPOSE)?,
            dequant: get(DEQUANT)?,
        })
    }
}

fn i32_of(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).with_context(|| format!("affine widths: {what} = {v} exceeds i32"))
}

fn blocks_for(total: usize, per: u32) -> Result<u32> {
    u32::try_from(total.div_ceil(per as usize)).context("affine widths: grid exceeds u32")
}

/// One decode launch: `out[rows][n] = x[rows][k] . W^T` for the packed weight.
#[derive(Clone, Copy, Debug)]
pub struct Decode {
    pub x: DevicePtr,
    pub xs: DevicePtr,
    pub w: DevicePtr,
    pub scales_t: DevicePtr,
    pub biases_t: DevicePtr,
    pub out: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub bits: u8,
    pub rows: usize,
    /// `(work, tickets)` for the split form; `None` runs the unsplit gemv (bit-identical output).
    pub split: Option<(DevicePtr, DevicePtr)>,
    /// The unsplit gemv's grid (`layout::gemv_grid`).
    pub grid: u32,
}

/// `xs[r][g] = sum of x[r][64 g .. 64 g + 64]` for `rows` rows of `k` bf16 inputs.
pub fn launch_sums(
    gpu: &dyn GpuBackend,
    h: &Kernels,
    x: DevicePtr,
    xs: DevicePtr,
    rows: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let gy = blocks_for(layout::groups(k), 64)?;
    KernelLaunch::new(gpu, h.sums)
        .grid([
            u32::try_from(rows).context("affine widths: rows exceed u32")?,
            gy,
            1,
        ])
        .block([64, 1, 1])
        .arg_ptr(x)
        .arg_ptr(xs)
        .arg_i32(i32_of(k, "k")?)
        .launch(stream)
}

/// The decode GEMV (group sums already in `d.xs`).
pub fn launch_decode(gpu: &dyn GpuBackend, h: &Kernels, d: &Decode, stream: u64) -> Result<()> {
    let sk = layout::split_k(d.n, d.k);
    let tiles = layout::tiles(d.n);
    let (handle, grid, smem) = match d.split {
        Some(_) => (
            h.split,
            blocks_for(tiles * sk, 1)?,
            layout::split_smem(d.bits),
        ),
        None => (h.gemv, d.grid, layout::gemv_smem(d.bits)),
    };
    let launch = KernelLaunch::new(gpu, handle)
        .grid([grid, 1, 1])
        .block([layout::GEMV_THREADS, 1, 1])
        .shared_mem(smem)
        .arg_ptr(d.x)
        .arg_ptr(d.xs)
        .arg_ptr(d.w)
        .arg_ptr(d.scales_t)
        .arg_ptr(d.biases_t)
        .arg_ptr(d.out)
        .arg_i32(i32_of(d.n, "n")?)
        .arg_i32(i32_of(layout::npad(d.n), "npad")?)
        .arg_i32(i32_of(sk, "sk")?)
        .arg_i32(i32_of(tiles, "tiles")?)
        .arg_i32(i32_of(d.rows, "rows")?)
        .arg_i32(i32_of(d.k, "k")?)
        .arg_i32(i32_of(d.k, "ldx")?);
    match d.split {
        Some((work, tickets)) => launch.arg_ptr(work).arg_ptr(tickets).launch(stream),
        None => launch.launch(stream),
    }
}

/// One quantizer launch: bf16 rows `src` `[n][k]` to zeroed MLX `words`, `scales`, `biases`; the
/// first failing group's index lands in `bad` (seeded `0xFFFFFFFF`).
#[derive(Clone, Copy, Debug)]
pub struct Quantize {
    pub src: DevicePtr,
    pub n: usize,
    pub k: usize,
    pub bits: u8,
    pub clip: u8,
    pub words: DevicePtr,
    pub scales: DevicePtr,
    pub biases: DevicePtr,
    pub bad: DevicePtr,
}

/// The quantizer: one warp a 64-group, four groups a block.
pub fn launch_quantize(gpu: &dyn GpuBackend, h: &Kernels, q: &Quantize, stream: u64) -> Result<()> {
    let groups = q.n * layout::groups(q.k);
    KernelLaunch::new(gpu, h.quantize)
        .grid([blocks_for(groups, 4)?, 1, 1])
        .block([layout::GEMV_THREADS, 1, 1])
        .arg_ptr(q.src)
        .arg_i32(i32_of(q.n, "n")?)
        .arg_i32(i32_of(q.k, "k")?)
        .arg_i32(i32::from(q.bits))
        .arg_i32(i32::from(q.clip))
        .arg_ptr(q.words)
        .arg_ptr(q.scales)
        .arg_ptr(q.biases)
        .arg_ptr(q.bad)
        .launch(stream)
}

/// MLX words `[n][k * bits / 32]` to the tiles of `bits` at `out` (`layout::tile_words` words).
pub fn launch_pack(
    gpu: &dyn GpuBackend,
    h: &Kernels,
    words: DevicePtr,
    n: usize,
    k: usize,
    bits: u8,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let total = layout::tile_words(n, k, bits);
    KernelLaunch::new(gpu, h.pack)
        .grid([blocks_for(total, HELPER_THREADS)?, 1, 1])
        .block([HELPER_THREADS, 1, 1])
        .arg_ptr(words)
        .arg_i32(i32::from(bits))
        .arg_i32(i32_of(n, "n")?)
        .arg_i32(i32_of(layout::mlx_row_words(k, bits), "row words")?)
        .arg_i32(i32_of(layout::groups(k), "groups")?)
        .arg_ptr(out)
        .arg_u64(total as u64)
        .launch(stream)
}

/// `(n, k / 64)` bf16 table `src` to `(k / 64, npad)` at `out`, padded rows zero.
pub fn launch_transpose(
    gpu: &dyn GpuBackend,
    h: &Kernels,
    src: DevicePtr,
    n: usize,
    k: usize,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let total = layout::table_elems(n, k);
    KernelLaunch::new(gpu, h.transpose)
        .grid([blocks_for(total, HELPER_THREADS)?, 1, 1])
        .block([HELPER_THREADS, 1, 1])
        .arg_ptr(src)
        .arg_i32(i32_of(n, "n")?)
        .arg_i32(i32_of(layout::groups(k), "groups")?)
        .arg_i32(i32_of(layout::npad(n), "npad")?)
        .arg_ptr(out)
        .launch(stream)
}

/// The packed weight to bf16 rows `[n][k]` at `out`.
pub fn launch_dequant(gpu: &dyn GpuBackend, h: &Kernels, d: &Decode, stream: u64) -> Result<()> {
    let total = d.n * d.k;
    KernelLaunch::new(gpu, h.dequant)
        .grid([blocks_for(total, HELPER_THREADS)?, 1, 1])
        .block([HELPER_THREADS, 1, 1])
        .arg_ptr(d.w)
        .arg_ptr(d.scales_t)
        .arg_ptr(d.biases_t)
        .arg_i32(i32_of(d.n, "n")?)
        .arg_i32(i32_of(d.k, "k")?)
        .arg_i32(i32_of(layout::npad(d.n), "npad")?)
        .arg_i32(i32::from(d.bits))
        .arg_ptr(d.out)
        .arg_u64(total as u64)
        .launch(stream)
}
