// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:28-32 (`Affine4.layout`), 271-275 (the lane shared-memory sizes) and 308-314
// (`splitK`). Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code
// written before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust; the group size is a parameter (32, 64 or 128; at 64
// every number is upstream's); the scratch arithmetic for Atlas's `W4Scratch` and the per-weight tickets
// are new.

//! Responsibility: the affine 4-bit weight descriptor ([`AffineWeight`]) and the pure arithmetic of its
//! layout: padded sizes, the K split fixed by the weight's shape, the lane GEMV's shared memory, grid
//! and scratch sizes.
//!
//! The descriptor is `Copy` and owns nothing by itself: the device buffers it names are released with
//! [`AffineWeight::release`] by whoever owns the weight (the `W4Linear` wrapper in `w4linear.rs` does it
//! through [`super::release`]). Nothing here touches the GPU except `release`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::kernels::AffineKernels;

/// Rows one decode or verify launch takes (one m16 MMA row tile).
pub const MAX_DECODE_ROWS: usize = 16;
/// Output columns one CTA computes.
pub const TILE_N: usize = 64;
/// Columns are padded to this for the tiled layout (`qlinear.zig:29`).
pub const NPAD_ALIGN: usize = 128;
/// Group sizes the kernels are instantiated for.
pub const GROUPS: [u16; 3] = [32, 64, 128];
/// Projections one `_seg` or `_many` launch takes.
pub const MAX_SEGS: usize = 4;
/// Largest clip-search candidate count (shrink ratios 0.95 down to 0.50).
pub const MAX_CLIP_SEARCH: u8 = 10;
/// Upper bound of `split_k` (`qlinear.zig:311`).
pub const MAX_SPLIT: usize = 8;
/// Scratch sub-buffers start on this boundary.
pub const SCRATCH_ALIGN: usize = 256;

/// The padded sizes of an `n x k` affine 4-bit weight at `group` (`Affine4.layout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// `n` rounded up to [`NPAD_ALIGN`].
    pub npad: usize,
    /// Bytes of the tiled codes (`npad * k / 2`).
    pub words: usize,
    /// Bytes of the tiled scales, and again of the biases (`k / group * npad * 2`).
    pub scales: usize,
}

impl Layout {
    pub fn of(n: usize, k: usize, group: usize) -> Self {
        let npad = n.div_ceil(NPAD_ALIGN) * NPAD_ALIGN;
        Self {
            npad,
            words: npad * k / 2,
            scales: k / group * npad * 2,
        }
    }

    /// Device bytes of the codes, scales and biases together.
    pub fn total(&self) -> usize {
        self.words + 2 * self.scales
    }
}

/// K slices of an `n x k` weight at `group`: fixed by the weight's shape, never by the row count, so a
/// row's bits do not depend on how many rows a launch carries (`qmm.split_k`, `qlinear.zig:308-314`).
/// At `group` 64 this is upstream's function exactly.
pub fn split_k(n: usize, k: usize, group: usize) -> usize {
    let tiles = n.div_ceil(TILE_N);
    let groups = k / group;
    let mut sk = 1;
    while sk < MAX_SPLIT
        && tiles * sk < 192
        && groups.is_multiple_of(sk * 2)
        && groups / (sk * 2) >= 8
    {
        sk *= 2;
    }
    sk
}

/// Pipeline stages of the lane kernels at `group`: `(gemv/many, split/seg)`. Must match the
/// `AFFINE4_LANE_ENTRIES` instantiations in `kernels/ada/common/affine4_lane_ada.cu`.
pub const fn lane_stages(group: usize) -> (usize, usize) {
    match group {
        128 => (4, 3),
        _ => (8, 4),
    }
}

/// `LaneTile<group, 16, 64, 1, 4, stages>::SMEM`: the dynamic shared bytes of a lane launch
/// (`affine_frag_ada.cuh`). At group 64 this gives upstream's 35,328 (8 stages) and 17,664 (4 stages),
/// `qlinear.zig:271,275`.
pub fn lane_smem(group: usize, stages: usize) -> u32 {
    let stage = 16 * group * 2 + TILE_N * group / 2 + 2 * TILE_N * 2 + 16 * 4;
    let partials = 2 * 4 * 128 * 4;
    (stages * stage).max(partials) as u32
}

/// Resident lane CTAs an SM (sm_89: 100 KiB of shared memory an SM, 1 KiB reserved a CTA, at most 12
/// CTAs of 128 threads). Only the persistent `_gemv` grid uses it, and a grid size never changes an
/// output bit (every tile's chain is the same whichever CTA runs it).
pub fn blocks_per_sm(smem: u32) -> u32 {
    (102_400 / (smem + 1024)).clamp(1, 12)
}

/// Round `x` up to [`SCRATCH_ALIGN`].
pub const fn align_scratch(x: usize) -> usize {
    x.div_ceil(SCRATCH_ALIGN) * SCRATCH_ALIGN
}

/// One affine 4-bit weight on the device, in the lane GEMV's tiled layout.
///
/// `words`: `[npad/64][k/group][8][32][group/32]` u32 codes; `scales`, `biases`: `[k/group][npad]` bf16
/// with zero columns past `n`; `tickets`: `ceil(n/64)` u32 split-K tickets, zero at rest (the split
/// kernel returns each to zero), owned by this weight so concurrent layers never share one. A weight is
/// decoded by one stream at a time (its tickets are its own; the scheduler's decode is one stream).
#[derive(Clone, Copy, Debug)]
pub struct AffineWeight {
    pub words: DevicePtr,
    pub scales: DevicePtr,
    pub biases: DevicePtr,
    pub tickets: DevicePtr,
    /// Outputs (rows of the checkpoint matrix).
    pub n: usize,
    /// Inputs.
    pub k: usize,
    pub npad: usize,
    pub group: u16,
    /// Always 4 for a weight built here; widths 2/3/5/6/8 are `super::widths`'.
    pub bits: u8,
    /// `prefill` uses the int8 (W4A8) prompt GEMM when true, the BF16 one otherwise.
    pub int8_prefill: bool,
    /// The resolved kernel handles the launches use.
    pub kernels: AffineKernels,
}

impl AffineWeight {
    /// Groups along K.
    pub fn kg(&self) -> usize {
        self.k / self.group as usize
    }

    /// 64-column tiles.
    pub fn tiles(&self) -> usize {
        self.n.div_ceil(TILE_N)
    }

    /// K slices ([`split_k`]).
    pub fn sk(&self) -> usize {
        split_k(self.n, self.k, self.group as usize)
    }

    pub fn layout(&self) -> Layout {
        Layout::of(self.n, self.k, self.group as usize)
    }

    /// Device bytes the weight occupies (codes, scales, biases and tickets).
    pub fn device_bytes(&self) -> usize {
        self.layout().total() + self.tickets_bytes()
    }

    /// Bytes of the per-weight ticket array.
    pub fn tickets_bytes(&self) -> usize {
        4 * self.tiles().max(1)
    }

    /// Bytes of the split kernel's fp32 slice partials (`tiles * sk * 16 * 64 * 4`), 0 when sk is 1.
    pub fn work_bytes(&self) -> usize {
        let sk = self.sk();
        if sk > 1 {
            self.tiles() * sk * MAX_DECODE_ROWS * TILE_N * 4
        } else {
            0
        }
    }

    /// Bytes of the group sums of `rows` input rows, rounded to [`SCRATCH_ALIGN`].
    pub fn sums_bytes(&self, rows: usize) -> usize {
        align_scratch(rows * self.kg() * 4)
    }

    /// `W4Linear::scratch_bytes`: the sums of `max_rows` rows (at most [`MAX_DECODE_ROWS`]) plus room for
    /// this weight's slice partials TWICE, so a gate/up pair of the same shape fits one fused `_seg`
    /// launch. `decode` needs `sums_bytes(rows) + work_bytes()`; a smaller pair scratch falls back to two
    /// launches with the same bits.
    pub fn scratch_bytes(&self, max_rows: usize) -> usize {
        self.sums_bytes(max_rows.clamp(1, MAX_DECODE_ROWS)) + 2 * align_scratch(self.work_bytes())
    }

    /// Check the shape rules every launcher relies on.
    pub fn validate(&self) -> Result<()> {
        let g = self.group as usize;
        ensure!(
            GROUPS.contains(&self.group),
            "affine: group {g} is not one of {GROUPS:?}"
        );
        ensure!(
            self.bits == 4,
            "affine: AffineWeight serves 4-bit codes, got {}",
            self.bits
        );
        ensure!(
            self.n > 0 && self.k > 0,
            "affine: empty weight [{}x{}]",
            self.n,
            self.k
        );
        ensure!(
            self.k.is_multiple_of(g),
            "affine: K {} is not a multiple of the group {g}",
            self.k
        );
        ensure!(
            self.npad == Layout::of(self.n, self.k, g).npad,
            "affine: npad {} does not match n {}",
            self.npad,
            self.n
        );
        ensure!(
            self.n <= i32::MAX as usize && self.k <= i32::MAX as usize,
            "affine: [{}x{}] exceeds the kernels' int range",
            self.n,
            self.k
        );
        Ok(())
    }

    /// Free the four device buffers. The caller guarantees no launch still reads them and that the
    /// weight is released once.
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.words, self.scales, self.biases, self.tickets] {
            if !p.is_null() {
                gpu.free(p)?;
            }
        }
        Ok(())
    }
}
