// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:35-47 (AffineN.layout), 283-288 (Split.tiles_max, work bytes), 308-314
// (splitK) and 326-344 (gemvSmem, splitSmem, tile_widths). Copyright 2026 TensorFold contributors.
// Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is also
// available under the MIT License. See TENSORFOLD-ATTRIBUTION.md. Changes for Atlas: Zig to Rust;
// the split work and ticket sizes are derived per weight instead of one fixed 384-slot buffer; a
// resident-CTA estimate replaces the driver occupancy query (it only sizes the gemv grid, never the
// bits); shape validation returns `anyhow` errors.
//! Responsibility: the pure layout arithmetic of the 2/3/5/6/8-bit affine tiles (lane L17c): which
//! widths are served, padded rows, tile and table sizes, shared-memory bytes per kernel, the shape-only
//! split-K choice and the scratch each decode needs. No device access; every number here is pinned by
//! `widths_tests.rs` and by the `static_assert`s in `kernels/ada/common/affine_n_ada.cu`.

use anyhow::{Result, bail};

/// The widths this module serves, in the order of its kernel tables. Width 4 is the 4-bit lane path
/// (lane L17a), never this module.
pub const BITS: [u8; 5] = [2, 3, 5, 6, 8];
/// The only group size the narrow and 8-bit tiles are built for.
pub const GROUP: usize = 64;
/// Rows one decode call accepts: one 16-row MMA tile, rows above M zero-filled.
pub const MAX_DECODE_ROWS: usize = 16;
/// Output columns one CTA tile covers.
pub const TILE_COLS: usize = 64;
/// Output rows are padded to this many (the 4-bit tiles' padding, so tables share one shape).
pub const ROW_PAD: usize = 128;
/// Threads per GEMV CTA (`__launch_bounds__(128)`).
pub const GEMV_THREADS: u32 = 128;
/// The split form runs only below this many column tiles (`split_k` never splits a wider weight).
pub const SPLIT_TILES_MAX: usize = 192;
/// Shared memory per SM the resident-CTA estimate assumes (sm_89: 100 KiB).
const SMEM_PER_SM: u32 = 100 * 1024;
/// Per-CTA shared-memory reservation the estimate adds (the driver's 1 KiB per block).
const SMEM_RESERVED: u32 = 1024;
/// Resident GEMV CTAs per SM the grid assumes at most (register-bound at 128 threads).
const MAX_RESIDENT: u32 = 4;

/// Position of `bits` in [`BITS`], or `None` for a width this module does not serve.
pub fn index(bits: u8) -> Option<usize> {
    BITS.iter().position(|&b| b == bits)
}

/// True when this module serves `bits` at `group` (group 64 only).
pub fn supports(bits: u8, group: u16) -> bool {
    index(bits).is_some() && usize::from(group) == GROUP
}

/// True for the 8-bit tiles' layout; false for the narrow (bit-sliced) one.
pub fn is_wide(bits: u8) -> bool {
    bits == 8
}

/// Output rows padded to [`ROW_PAD`].
pub fn npad(n: usize) -> usize {
    n.div_ceil(ROW_PAD) * ROW_PAD
}

/// Column tiles of a weight with `n` outputs (the GEMV's `tiles`).
pub fn tiles(n: usize) -> usize {
    n.div_ceil(TILE_COLS)
}

/// 64-groups along K.
pub fn groups(k: usize) -> usize {
    k / GROUP
}

/// u32 words of one MLX row of `k` codes at `bits`.
pub fn mlx_row_words(k: usize, bits: u8) -> usize {
    k * usize::from(bits) / 32
}

/// u32 words of the tiles: `npad * k * bits / 32` (for both layouts).
pub fn tile_words(n: usize, k: usize, bits: u8) -> usize {
    npad(n) * k * usize::from(bits) / 32
}

/// u32 words one (tile, group) block holds: 128 * bits narrow, 1024 on the 8-bit tiles.
pub fn block_words(bits: u8) -> usize {
    if is_wide(bits) {
        1024
    } else {
        128 * usize::from(bits)
    }
}

/// bf16 entries of one transposed scale (or bias) table: `[k / 64][npad]`.
pub fn table_elems(n: usize, k: usize) -> usize {
    groups(k) * npad(n)
}

/// Device bytes of a packed weight: tiles plus the two bf16 tables.
pub fn device_bytes(n: usize, k: usize, bits: u8) -> usize {
    tile_words(n, k, bits) * 4 + 2 * table_elems(n, k) * 2
}

/// Dynamic shared bytes of the gemv kernel (`Tile<64, 64, 6, bits>::SMEM`).
pub fn gemv_smem(bits: u8) -> u32 {
    if is_wide(bits) {
        38_784
    } else {
        14_208 + 3_072 * u32::from(bits)
    }
}

/// Dynamic shared bytes of the split kernel (`Tile<64, 64, 4, bits>::SMEM`).
pub fn split_smem(bits: u8) -> u32 {
    if is_wide(bits) {
        25_856
    } else {
        9_472 + 2_048 * u32::from(bits)
    }
}

/// K slices, fixed by the weight's shape and never by the row count: double while the grid stays
/// under 192 CTAs, the groups split evenly and every slice keeps at least 8 groups (at most 8).
pub fn split_k(n: usize, k: usize) -> usize {
    let t = tiles(n);
    let g = groups(k);
    let mut sk = 1usize;
    while sk < 8 && t * sk < SPLIT_TILES_MAX && g.is_multiple_of(sk * 2) && g / (sk * 2) >= 8 {
        sk *= 2;
    }
    sk
}

/// fp32 bytes of the split kernel's slice partials (`tiles * sk * 16 * 64 * 4`), 0 when the shape
/// does not split.
pub fn split_work_bytes(n: usize, k: usize) -> usize {
    let sk = split_k(n, k);
    if sk > 1 {
        tiles(n) * sk * MAX_DECODE_ROWS * TILE_COLS * 4
    } else {
        0
    }
}

/// Bytes of the split kernel's per-tile tickets (`tiles` u32; they must start zero, the kernel
/// leaves them zero).
pub fn tickets_bytes(n: usize) -> usize {
    tiles(n) * 4
}

/// fp32 bytes of the group sums `rows` input rows need (`[rows][k / 64]`).
pub fn sums_bytes(rows: usize, k: usize) -> usize {
    rows * groups(k) * 4
}

/// Resident gemv CTAs per SM, estimated from shared memory and capped at [`MAX_RESIDENT`]; only the
/// grid size depends on it, never a row's bits.
pub fn resident_per_sm(bits: u8) -> u32 {
    (SMEM_PER_SM / (gemv_smem(bits) + SMEM_RESERVED)).clamp(1, MAX_RESIDENT)
}

/// The gemv grid: one CTA a column tile, at most what the card holds at once.
pub fn gemv_grid(n: usize, bits: u8, sm_count: u32) -> u32 {
    let cap = sm_count.max(1).saturating_mul(resident_per_sm(bits));
    u32::try_from(tiles(n)).unwrap_or(u32::MAX).min(cap).max(1)
}

/// Refuse a shape or width the tiles cannot hold.
pub fn check_shape(n: usize, k: usize, bits: u8) -> Result<()> {
    if index(bits).is_none() {
        bail!(
            "affine widths: {bits}-bit codes are not served here (served: {BITS:?}; 4 is the lane path)"
        );
    }
    if n == 0 || k == 0 || !k.is_multiple_of(GROUP) {
        bail!("affine widths: shape n={n} k={k} needs n > 0 and k a positive multiple of {GROUP}");
    }
    if i32::try_from(npad(n)).is_err() || i32::try_from(k).is_err() {
        bail!("affine widths: shape n={n} k={k} does not fit the kernels' 32-bit dimensions");
    }
    Ok(())
}
