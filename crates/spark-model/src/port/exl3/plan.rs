// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/exl3.zig:73-112 and 202-217 (`wide_blocks`, `plan`, `strides`, `sharedBytes`, `zBytes`,
// `counterBytes`) and :466-474 (the plan's test table), after ExLlamaV3 (turboderp-org/exllamav3, MIT,
// Copyright (c) 2025 Turboderp). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust; the block target is 4 x the device's SM count instead
// of GB10's fixed 192; the per-call scratch is laid out inside `W4Scratch::ws` ([`ScratchLayout`]).

//! Responsibility: what one EXL3 matrix launches, from its shape alone ([`plan`]: the K splits and the
//! warps a block), and the byte arithmetic of every buffer a call touches (strides, shared memory, the
//! rotated input, the split partials, the per-weight arrival counters).
//!
//! The plan never looks at the row count: that is the row-invariance rule of the `W4Linear` contract
//! (row r's bits must not depend on how many rows the call carries).

use super::format::HADAMARD_DIM;

/// Largest `rows` [`super::Exl3Linear`]'s decode serves (the contract's `max_decode_rows`).
pub const MAX_DECODE_ROWS: usize = 16;
/// Largest `m` one launch of the linear kernel takes (8 passes of 16 rows); a prefill above it declines.
pub const MAX_KERNEL_ROWS: usize = 128;
/// From this many 128-column blocks a shape takes one K split and 8 warps (`wide_blocks`).
pub const WIDE_BLOCKS: usize = 64;
/// The fewest k tiles a warp walks before the plan trades warps or splits away (`min_tiles`).
pub const MIN_TILES: usize = 8;
/// Blocks the plan aims to fill per SM.
pub const BLOCKS_PER_SM: usize = 4;
/// Passes of 16 rows the per-weight counters cover (128 rows).
pub const COUNTER_PASSES: usize = MAX_KERNEL_ROWS / 16;
/// Every scratch region inside `ws` starts on this boundary.
pub const SCRATCH_ALIGN: usize = 256;
/// Dynamic shared memory of the dequant kernel: 128 x 129 fp32.
pub const DEQUANT_SMEM: u32 = 128 * 129 * 4;
/// Threads a block of the dequant kernel.
pub const DEQUANT_THREADS: u32 = 256;
/// Threads a block of the input rotation (four 128-value blocks of one row).
pub const ROT_THREADS: u32 = 128;

/// The launch shape of one matrix: `sk` K splits, `wk` warps a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub sk: u32,
    pub wk: u32,
}

/// `exl3.zig`'s `plan`: aim for `blocks` blocks with narrow outputs by splitting K (up to 64 ways), use 8
/// warps for wide outputs, then back off (8 -> 4 warps, then halve the splits) until every warp walks a
/// whole number of at least `min_tiles` k tiles. `k % 128 == 0` makes the 4-warp, 1-split fallback always
/// divide.
pub fn plan(k: usize, n: usize, blocks: usize, min_tiles: usize) -> Plan {
    let kt = k / 16;
    let nb = n / HADAMARD_DIM;
    let mut sk: usize = 1;
    let mut wk: usize = if nb >= WIDE_BLOCKS { 8 } else { 4 };
    if nb < WIDE_BLOCKS {
        while nb * sk < blocks && sk < 64 {
            sk *= 2;
        }
    }
    while (wk > 4 || sk > 1) && (!kt.is_multiple_of(sk * wk) || kt / (sk * wk) < min_tiles) {
        if wk > 4 {
            wk = 4;
        } else if sk > 1 {
            sk /= 2;
        } else {
            break;
        }
    }
    Plan {
        sk: sk as u32,
        wk: wk as u32,
    }
}

/// The plan for a device with `sms` streaming multiprocessors.
pub fn plan_for_device(k: usize, n: usize, sms: u32) -> Plan {
    plan(k, n, BLOCKS_PER_SM * sms.max(1) as usize, MIN_TILES)
}

/// `u32` words between consecutive k tiles and between 128-column blocks of the stored
/// `[k/16, n/16, words]` layout: `(stride_k, stride_nb)`.
pub fn strides(n: usize, halves: u16) -> (u64, u64) {
    let words = 4 * halves as u64;
    ((n as u64 / 16) * words, 8 * words)
}

fn align(bytes: usize) -> usize {
    bytes.div_ceil(SCRATCH_ALIGN) * SCRATCH_ALIGN
}

/// Dynamic shared memory of one linear launch: `wk * min(rows, 8) * 128` fp32 warp sums.
pub fn linear_smem(p: Plan, rows: usize) -> u32 {
    p.wk * rows.clamp(1, 8) as u32 * 128 * 4
}

/// Bytes of the rotated fp16 input of `rows` rows.
pub fn xh_bytes(rows: usize, k: usize) -> usize {
    rows * k * 2
}

/// Bytes of the fp32 split partials `[sk, rows, n]` (zero without a split).
pub fn z_bytes(p: Plan, rows: usize, n: usize) -> usize {
    if p.sk > 1 {
        p.sk as usize * rows * n * 4
    } else {
        0
    }
}

/// `i32` arrival counters of one weight: one per (pass, 128-column block), 8 passes.
pub fn counter_slots(n: usize) -> usize {
    COUNTER_PASSES * (n / HADAMARD_DIM)
}

/// Where a call's scratch sits inside `W4Scratch::ws`: the rotated input at 0, the split partials after
/// it (both 256-byte aligned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    pub z_offset: usize,
    pub total: usize,
}

/// The scratch layout of a call with `rows` rows (clamped to one launch).
pub fn scratch_layout(p: Plan, rows: usize, k: usize, n: usize) -> ScratchLayout {
    let rows = rows.clamp(1, MAX_KERNEL_ROWS);
    let z_offset = align(xh_bytes(rows, k));
    ScratchLayout {
        z_offset,
        total: z_offset + align(z_bytes(p, rows, n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_reproduces_upstreams_table() {
        // exl3.zig:468-473, at GB10's 192 blocks and 8 tiles a warp.
        let p = |k, n| plan(k, n, 192, 8);
        assert_eq!(p(4096, 1024), Plan { sk: 8, wk: 4 });
        assert_eq!(p(1024, 4096), Plan { sk: 2, wk: 4 });
        assert_eq!(p(2048, 512), Plan { sk: 4, wk: 4 });
        assert_eq!(p(512, 1024), Plan { sk: 1, wk: 4 });
        assert_eq!(p(512, 512), Plan { sk: 1, wk: 4 });
        assert_eq!(p(2048, 4096), Plan { sk: 4, wk: 4 });
    }

    #[test]
    fn every_plan_divides_the_k_tiles() {
        for sms in [1u32, 24, 34, 48, 132] {
            for k in (128..=20480).step_by(128) {
                for n in [
                    128usize, 256, 1024, 2560, 4096, 5120, 8192, 9216, 17408, 248320,
                ] {
                    let pl = plan_for_device(k, n, sms);
                    let kt = k / 16;
                    assert!(
                        kt.is_multiple_of((pl.sk * pl.wk) as usize),
                        "k={k} n={n} sms={sms} {pl:?}"
                    );
                    assert!(pl.wk == 4 || pl.wk == 8, "{pl:?}");
                    assert!(pl.sk.is_power_of_two() && pl.sk <= 64, "{pl:?}");
                }
            }
        }
    }

    #[test]
    fn the_4060_ti_plans_of_the_target_shapes() {
        // 34 SMs: 136 blocks. Wide outputs take 8 warps and no split.
        assert_eq!(plan_for_device(2560, 9216, 34), Plan { sk: 1, wk: 8 });
        assert_eq!(plan_for_device(5120, 17408, 34), Plan { sk: 1, wk: 8 });
        // Narrow outputs split K.
        assert_eq!(plan_for_device(9216, 2560, 34), Plan { sk: 8, wk: 4 });
        assert_eq!(plan_for_device(17408, 5120, 34), Plan { sk: 4, wk: 4 });
    }

    #[test]
    fn strides_match_the_stored_layout() {
        // halves 6 (3 bits): 24 words a tile; n = 256: 16 tiles a k row.
        assert_eq!(strides(256, 6), (16 * 24, 8 * 24));
    }

    #[test]
    fn scratch_layout_is_aligned_and_clamped() {
        let p = Plan { sk: 4, wk: 4 };
        let l = scratch_layout(p, 3, 1024, 512);
        assert_eq!(l.z_offset, align(3 * 1024 * 2));
        assert_eq!(l.total, l.z_offset + align(4 * 3 * 512 * 4));
        assert_eq!(l.z_offset % SCRATCH_ALIGN, 0);
        // Above one launch the layout stops growing: a declined prefill needs none of it.
        assert_eq!(
            scratch_layout(p, 2048, 1024, 512),
            scratch_layout(p, MAX_KERNEL_ROWS, 1024, 512)
        );
        let unsplit = scratch_layout(Plan { sk: 1, wk: 8 }, 16, 1024, 512);
        assert_eq!(unsplit.total, unsplit.z_offset);
    }

    #[test]
    fn shared_memory_stays_under_the_static_ceiling() {
        for rows in 1..=MAX_KERNEL_ROWS {
            assert!(linear_smem(Plan { sk: 1, wk: 8 }, rows) <= 32 * 1024);
        }
        assert_eq!(linear_smem(Plan { sk: 1, wk: 4 }, 3), 4 * 3 * 128 * 4);
        assert_eq!(DEQUANT_SMEM, 66_048);
        assert_eq!(counter_slots(17408), 8 * 136);
    }
}
