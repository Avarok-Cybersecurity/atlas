// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/affine4_kernels.zig:107-133, 222-250 and 291-318 (`pack`, `decode`, `gemvSeg`,
// `gemvSplit`, `gemv`) and zig/src/cuda/qlinear.zig:250-268 (the by-value launch descriptors).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust over `GpuBackend::launch_typed`; handles resolve through
// `try_target_kernel`, so a target without the Ada modules issues no lookup; the sk == 1 decode takes the
// lane `_gemv` body instead of the cluster kernel (qmm_group, not ported); a Seg carries its weight's own
// ticket pointer; group sizes 32 and 128 are added.

//! Responsibility: the affine route's kernel handles ([`AffineKernels`]) and one launcher per entry point
//! of `kernels/ada/common/affine4_lane_ada.cu` and `affine_pack_ada.cu`, with the by-value descriptors
//! those entry points take, serialized byte for byte.
//!
//! Every launcher is capture-safe: no allocation, no host synchronization, no host read. The only
//! state a launch leaves behind is in its outputs (tickets return to zero).

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

pub use super::params::{Member, ParamBytes, many_bytes, part_bytes, segs_bytes};
use super::weight::{
    AffineWeight, MAX_DECODE_ROWS, MAX_SEGS, blocks_per_sm, lane_smem, lane_stages,
};

/// The lane GEMV module (`kernels/ada/common/affine4_lane_ada.cu`).
pub const LANE_MODULE: &str = "affine4_lane_ada";
/// The quantize, pack, sums and dequant module (`kernels/ada/common/affine_pack_ada.cu`).
pub const PACK_MODULE: &str = "affine_pack_ada";

/// The four lane entry points and the quantizer of one group size.
#[derive(Clone, Copy, Debug)]
pub struct GroupKernels {
    pub gemv: KernelHandle,
    pub split: KernelHandle,
    pub seg: KernelHandle,
    pub many: KernelHandle,
    pub quantize: KernelHandle,
}

/// Every handle the affine 4-bit route launches, plus the SM count the persistent grid is sized by.
#[derive(Clone, Copy, Debug)]
pub struct AffineKernels {
    pub g32: GroupKernels,
    pub g64: GroupKernels,
    pub g128: GroupKernels,
    pub pack_tiles: KernelHandle,
    pub transpose: KernelHandle,
    pub sums: KernelHandle,
    pub dequant: KernelHandle,
    pub sms: u32,
}

fn need(gpu: &dyn GpuBackend, module: &str, entry: &str) -> Result<KernelHandle> {
    let h = crate::layers::try_target_kernel(gpu, module, entry);
    if h.0 == 0 {
        bail!(
            "affine 4-bit kernel {module}::{entry} is not compiled for this target \
             (the affine route needs the Ada build, AVAROK_TARGET_HW=ada)"
        );
    }
    Ok(h)
}

fn group_kernels(gpu: &dyn GpuBackend, gs: u16) -> Result<GroupKernels> {
    Ok(GroupKernels {
        gemv: need(gpu, LANE_MODULE, &format!("affine4_lane_gemv_g{gs}"))?,
        split: need(gpu, LANE_MODULE, &format!("affine4_lane_split_g{gs}"))?,
        seg: need(gpu, LANE_MODULE, &format!("affine4_lane_seg_g{gs}"))?,
        many: need(gpu, LANE_MODULE, &format!("affine4_lane_many_g{gs}"))?,
        quantize: need(gpu, PACK_MODULE, &format!("affine4_quantize_bf16_g{gs}"))?,
    })
}

impl AffineKernels {
    /// True when both modules are compiled into this build. Issues no kernel lookup, so a route's
    /// decision table can ask it on any target.
    pub fn available(gpu: &dyn GpuBackend) -> bool {
        gpu.has_module(LANE_MODULE) && gpu.has_module(PACK_MODULE)
    }

    /// Resolve every handle; an error names the first missing entry point.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        ensure!(
            Self::available(gpu),
            "affine 4-bit modules {LANE_MODULE} and {PACK_MODULE} are not compiled for this target \
             (the affine route needs the Ada build, AVAROK_TARGET_HW=ada)"
        );
        Ok(Self {
            g32: group_kernels(gpu, 32)?,
            g64: group_kernels(gpu, 64)?,
            g128: group_kernels(gpu, 128)?,
            pack_tiles: need(gpu, PACK_MODULE, "affine4_pack_tiles")?,
            transpose: need(gpu, PACK_MODULE, "affine_transpose_pad16")?,
            sums: need(gpu, PACK_MODULE, "affine_group_sums")?,
            dequant: need(gpu, PACK_MODULE, "affine4_dequant_bf16")?,
            sms: gpu.sm_count().unwrap_or(1).max(1),
        })
    }

    /// The entry points of `group`.
    pub fn group(&self, group: u16) -> Result<&GroupKernels> {
        match group {
            32 => Ok(&self.g32),
            64 => Ok(&self.g64),
            128 => Ok(&self.g128),
            g => bail!("affine: no kernels for group {g}"),
        }
    }
}

fn launch(
    gpu: &dyn GpuBackend,
    f: KernelHandle,
    grid: [u32; 3],
    block: u32,
    smem: u32,
    stream: u64,
    args: &[KernelArg<'_>],
) -> Result<()> {
    gpu.launch_typed(f, grid, [block, 1, 1], smem, stream, args)?;
    if gpu.debug_sync_kernels() {
        gpu.synchronize(stream)?;
    }
    Ok(())
}

fn grid_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v.max(1)).map_err(|_| anyhow::anyhow!("affine: {what} grid {v} exceeds u32"))
}

fn i32b(v: usize) -> [u8; 4] {
    (v as i32).to_le_bytes()
}

/// BF16 `[n, k]` at `src` -> MLX words `[n, k/8]`, scales and biases `[n, k/group]`; `bad` (one u32,
/// pre-set to 0xFFFFFFFF) receives the first group that is not representable.
pub fn quantize(
    gpu: &dyn GpuBackend,
    k: &AffineKernels,
    src: DevicePtr,
    n: usize,
    kdim: usize,
    group: u16,
    clip: u8,
    words: DevicePtr,
    scales: DevicePtr,
    biases: DevicePtr,
    bad: DevicePtr,
    stream: u64,
) -> Result<()> {
    let groups = n * (kdim / group as usize);
    let (n4, k4, c4) = (i32b(n), i32b(kdim), i32b(clip as usize));
    let args = [
        KernelArg::Buffer(src),
        KernelArg::Bytes(&n4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&c4),
        KernelArg::Buffer(words),
        KernelArg::Buffer(scales),
        KernelArg::Buffer(biases),
        KernelArg::Buffer(bad),
    ];
    let grid = grid_u32(groups.div_ceil(4), "quantize")?;
    launch(
        gpu,
        k.group(group)?.quantize,
        [grid, 1, 1],
        128,
        0,
        stream,
        &args,
    )
}

/// MLX words `[n, k/8]` -> the tiled codes (`npad * k / 8` words).
pub fn pack_tiles(
    gpu: &dyn GpuBackend,
    k: &AffineKernels,
    words: DevicePtr,
    n: usize,
    kdim: usize,
    group: u16,
    npad: usize,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let total = npad * kdim / 8;
    let (n4, k8, kg, gs) = (
        i32b(n),
        i32b(kdim / 8),
        i32b(kdim / group as usize),
        i32b(group as usize),
    );
    let t8 = (total as i64).to_le_bytes();
    let args = [
        KernelArg::Buffer(words),
        KernelArg::Bytes(&n4),
        KernelArg::Bytes(&k8),
        KernelArg::Bytes(&kg),
        KernelArg::Bytes(&gs),
        KernelArg::Buffer(out),
        KernelArg::Bytes(&t8),
    ];
    let grid = grid_u32(total.div_ceil(256), "pack")?;
    launch(gpu, k.pack_tiles, [grid, 1, 1], 256, 0, stream, &args)
}

/// `(n, kg)` 16-bit -> `(kg, npad)` with zero columns past n.
pub fn transpose_pad16(
    gpu: &dyn GpuBackend,
    k: &AffineKernels,
    src: DevicePtr,
    n: usize,
    kg: usize,
    npad: usize,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (n4, kg4, np4) = (i32b(n), i32b(kg), i32b(npad));
    let args = [
        KernelArg::Buffer(src),
        KernelArg::Bytes(&n4),
        KernelArg::Bytes(&kg4),
        KernelArg::Bytes(&np4),
        KernelArg::Buffer(out),
    ];
    let grid = grid_u32((kg * npad).div_ceil(256), "transpose")?;
    launch(gpu, k.transpose, [grid, 1, 1], 256, 0, stream, &args)
}

/// `port::affine::group_sums`: `xs[r * (k/group) + g]` = the in-order fp32 sum of the bf16 inputs
/// `x[r, g*group ..][..group]`, for `rows` contiguous rows `k` wide (`x` 16-byte aligned).
pub fn group_sums(
    gpu: &dyn GpuBackend,
    k: &AffineKernels,
    x: DevicePtr,
    rows: usize,
    kdim: usize,
    group: u16,
    xs: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(rows > 0, "affine: group sums of zero rows");
    let kg = kdim / group as usize;
    let (r4, k4, g4) = (i32b(rows), i32b(kdim), i32b(group as usize));
    let args = [
        KernelArg::Buffer(x),
        KernelArg::Buffer(xs),
        KernelArg::Bytes(&r4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&g4),
    ];
    let grid = [
        grid_u32(rows, "sums")?,
        grid_u32(kg.div_ceil(64), "sums")?,
        1,
    ];
    launch(gpu, k.sums, grid, 64, 0, stream, &args)
}

fn check_rows(rows: usize) -> Result<()> {
    ensure!(
        (1..=MAX_DECODE_ROWS).contains(&rows),
        "affine: decode takes 1..={MAX_DECODE_ROWS} rows, got {rows}"
    );
    Ok(())
}

fn rows_args(rows: usize, kdim: usize) -> [[u8; 4]; 3] {
    [i32b(rows), i32b(kdim), i32b(kdim)]
}

/// The persistent lane GEMV: every K slice of a tile in one CTA, `min(tiles, resident CTAs)` CTAs.
/// Valid for any sk; the same bits as [`gemv_split`].
pub fn gemv_unsplit(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    xs: DevicePtr,
    w: &AffineWeight,
    out: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    check_rows(rows)?;
    let g = w.group as usize;
    let smem = lane_smem(g, lane_stages(g).0);
    let resident = blocks_per_sm(smem) as usize * w.kernels.sms as usize;
    let mut p = ParamBytes::default();
    part_bytes(&mut p, w, out);
    let [r4, k4, l4] = rows_args(rows, w.k);
    let args = [
        KernelArg::Buffer(x),
        KernelArg::Buffer(xs),
        KernelArg::Bytes(&p.0),
        KernelArg::Bytes(&r4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&l4),
    ];
    let grid = grid_u32(w.tiles().min(resident), "gemv")?;
    launch(
        gpu,
        w.kernels.group(w.group)?.gemv,
        [grid, 1, 1],
        128,
        smem,
        stream,
        &args,
    )
}

/// One CTA a (tile, K slice); the tile's last CTA sums the slices in order. Needs sk > 1, `work` of
/// [`AffineWeight::work_bytes`] and the weight's zeroed tickets.
pub fn gemv_split(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    xs: DevicePtr,
    w: &AffineWeight,
    out: DevicePtr,
    rows: usize,
    work: DevicePtr,
    stream: u64,
) -> Result<()> {
    check_rows(rows)?;
    let sk = w.sk();
    ensure!(sk > 1, "affine: the split GEMV needs sk > 1 (sk is {sk})");
    let g = w.group as usize;
    let smem = lane_smem(g, lane_stages(g).1);
    let mut p = ParamBytes::default();
    part_bytes(&mut p, w, out);
    let [r4, k4, l4] = rows_args(rows, w.k);
    let args = [
        KernelArg::Buffer(x),
        KernelArg::Buffer(xs),
        KernelArg::Bytes(&p.0),
        KernelArg::Bytes(&r4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&l4),
        KernelArg::Buffer(work),
        KernelArg::Buffer(w.tickets),
    ];
    let grid = grid_u32(w.tiles() * sk, "split")?;
    launch(
        gpu,
        w.kernels.group(w.group)?.split,
        [grid, 1, 1],
        128,
        smem,
        stream,
        &args,
    )
}

/// The decode GEMV of one weight (`Affine4Kernels.decode` off GB10): the split form when the shape
/// splits K, else the lane body.
pub fn gemv(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    xs: DevicePtr,
    w: &AffineWeight,
    out: DevicePtr,
    rows: usize,
    work: DevicePtr,
    stream: u64,
) -> Result<()> {
    if w.sk() > 1 {
        gemv_split(gpu, x, xs, w, out, rows, work, stream)
    } else {
        gemv_unsplit(gpu, x, xs, w, out, rows, stream)
    }
}

fn same_input(members: &[Member<'_>]) -> Result<()> {
    ensure!(
        !members.is_empty() && members.len() <= MAX_SEGS,
        "affine: a multi-projection launch takes 1..={MAX_SEGS} members, got {}",
        members.len()
    );
    let (k0, g0) = (members[0].0.k, members[0].0.group);
    for (i, (w, out)) in members.iter().enumerate() {
        ensure!(
            w.k == k0 && w.group == g0,
            "affine: members must share K and the group (member {i}: [{}x{}] g{} vs K {k0} g{g0})",
            w.n,
            w.k,
            w.group
        );
        ensure!(
            members[..i].iter().all(|(_, o)| o != out),
            "affine: member {i} repeats an output"
        );
    }
    Ok(())
}

/// Up to four projections of one input in one launch, each with its own K split, partial slots and
/// tickets (`gemvSeg`): the same bits per member as [`gemv`]. `work` must hold the members' partials
/// together (the sum of their [`AffineWeight::work_bytes`]).
pub fn gemv_seg(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    xs: DevicePtr,
    members: &[Member<'_>],
    rows: usize,
    work: DevicePtr,
    stream: u64,
) -> Result<()> {
    check_rows(rows)?;
    same_input(members)?;
    let w0 = members[0].0;
    let g = w0.group as usize;
    let smem = lane_smem(g, lane_stages(g).1);
    let (segs, blocks, _) = segs_bytes(members);
    let [r4, k4, l4] = rows_args(rows, w0.k);
    let args = [
        KernelArg::Buffer(x),
        KernelArg::Buffer(xs),
        KernelArg::Bytes(&segs),
        KernelArg::Bytes(&r4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&l4),
        KernelArg::Buffer(work),
    ];
    let grid = grid_u32(blocks, "seg")?;
    launch(
        gpu,
        w0.kernels.group(w0.group)?.seg,
        [grid, 1, 1],
        128,
        smem,
        stream,
        &args,
    )
}

/// Up to four projections of one input in one launch, each as the persistent lane body on its own CTA
/// range (`tf_lane_gemv_many`): the same bits per member as [`gemv`], no scratch.
pub fn gemv_many(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    xs: DevicePtr,
    members: &[Member<'_>],
    rows: usize,
    stream: u64,
) -> Result<()> {
    check_rows(rows)?;
    same_input(members)?;
    let w0 = members[0].0;
    let g = w0.group as usize;
    let smem = lane_smem(g, lane_stages(g).0);
    let each = blocks_per_sm(smem) as usize * w0.kernels.sms as usize / members.len();
    let (parts, blocks) = many_bytes(members, each.max(1));
    let [r4, k4, l4] = rows_args(rows, w0.k);
    let args = [
        KernelArg::Buffer(x),
        KernelArg::Buffer(xs),
        KernelArg::Bytes(&parts),
        KernelArg::Bytes(&r4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&l4),
    ];
    let grid = grid_u32(blocks, "many")?;
    launch(
        gpu,
        w0.kernels.group(w0.group)?.many,
        [grid, 1, 1],
        128,
        smem,
        stream,
        &args,
    )
}

/// BF16 `[n, k]` at `out` = bf16(s * q + b), from the tiled layout.
pub fn dequant_bf16(
    gpu: &dyn GpuBackend,
    w: &AffineWeight,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (n4, k4, np4, g4) = (i32b(w.n), i32b(w.k), i32b(w.npad), i32b(w.group as usize));
    let args = [
        KernelArg::Buffer(w.words),
        KernelArg::Buffer(w.scales),
        KernelArg::Buffer(w.biases),
        KernelArg::Bytes(&n4),
        KernelArg::Bytes(&k4),
        KernelArg::Bytes(&np4),
        KernelArg::Bytes(&g4),
        KernelArg::Buffer(out),
    ];
    let grid = grid_u32((w.n * w.k).div_ceil(256), "dequant")?;
    launch(gpu, w.kernels.dequant, [grid, 1, 1], 256, 0, stream, &args)
}
