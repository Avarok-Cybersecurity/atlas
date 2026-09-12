// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN chunked-prefill state spine's compile-time tile, its shared-memory
//! budget, and the grammar of the two levers that select an arm (#928).
//!
//! Extracted from `ssm_gdn_a3.rs` when the value-split twin was added: that
//! file is the LAUNCHER and is at the 500-line cap, and the guards below are
//! pure functions with no GPU in them — which is the whole reason they can be
//! graded from a CPU test on any host.
//!
//! Two levers live here and they compose in one direction only:
//!   * `[defaults] gdn_prefill_tc` picks the tensor-core FAMILY (the spine plus
//!     the two Hopper remnant twins). [`gdn_tc_spine_reject`].
//!   * `[defaults] gdn_spine_vsplit` picks how many CTAs share one value head's
//!     state inside that spine — 1 (the parent), 2 or 4.
//!     [`gdn_spine_vsplit_reject`]. It is meaningless without the first, and
//!     says so rather than silently doing nothing.

use super::ssm_gdn_tc_route::{GDN_SPINE_VSPLIT2_ENTRY, GDN_SPINE_VSPLIT4_ENTRY};

/// The tensor-core spine's compile-time tile: `K_DIM == V_DIM` in
/// `kernels/gb10/common/gated_delta_rule_chunk_tc.cu`.
pub const GDN_TC_DIM: u32 = 128;
/// That kernel's `CHUNK`.
pub const GDN_TC_CHUNK: u32 = 64;
/// Padded smem row stride for the 128-column tiles (W, U, St), in bf16
/// elements — `TCF_SW` / `GDNH_SW`. The padding is what makes the MMA fragment
/// reads bank-conflict-free; it is not slack.
pub const GDN_TC_SW: u32 = 136;
/// Padded smem row stride for the 64-column tiles (Kt, ducT) — `TCF_SC`.
pub const GDN_TC_SC: u32 = 72;

/// SSOT mirror of `TCF_SMEM` in `gated_delta_rule_chunk_tc.cu`:
///   St[128][136] + Wp[64][136] + Up[64][136] + ducT[128][72] + dec[65] f32
///   = 34816 + 17408 + 17408 + 18432 + 260 = 88 324 B.
/// Under-sizing this reads a tile out of bounds, so the launcher and the
/// kernel must not be able to disagree about it.
pub const GDN_TC_SMEM: u32 = GDN_TC_DIM * GDN_TC_SW * 2
    + 2 * (GDN_TC_CHUNK * GDN_TC_SW * 2)
    + GDN_TC_DIM * GDN_TC_SC * 2
    + (GDN_TC_CHUNK + 1) * 4;

/// Why the `ATLAS_GDN_PREFILL_TC` spine is NOT running — `None` means it is.
///
/// Pure so the grammar is testable without a GPU or the process environment.
/// NAME THE GUARD THAT REJECTED: a perf path that asks to be enabled and
/// silently is not measures as "no effect" (PR #296 shipped exactly that, an
/// ldmatrix GEMM that fell back with no error while both gates stayed green).
///
/// The tile guards are not defensive padding. The kernel's descriptors, smem
/// layout and fragment maps are all compile-time 128/128/64, and its K staging
/// reads 16 bytes at a time, so a narrower head or an odd `qk_stride` would
/// load the wrong columns or fault rather than run slowly.
pub fn gdn_tc_spine_reject(
    requested: bool,
    kernel_present: bool,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
    qk_stride: u32,
) -> Option<&'static str> {
    if !requested {
        Some("not requested")
    } else if !kernel_present {
        Some("kernel absent from this image")
    } else if k_dim != GDN_TC_DIM || v_dim != GDN_TC_DIM || chunk != GDN_TC_CHUNK {
        Some("head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)")
    } else if !qk_stride.is_multiple_of(8) {
        Some("qk_stride is not a multiple of 8 (the K staging uses 16-byte vector loads)")
    } else {
        None
    }
}

// ── `[defaults] gdn_spine_vsplit` ──────────────────────────────────────────

/// The declared value that means "run the parent, one CTA per value head".
pub const GDN_SPINE_VSPLIT_OFF: u32 = 1;

/// Every split the Hopper twin emits an entry point for, `1` included.
///
/// A LIST and not a range: each value is a compiled `__global__`, and a split
/// nothing compiled would resolve to a handle of 0 and fall back silently —
/// which is the failure `gdn_spine_vsplit_reject` exists to name out loud.
/// 8-way is deliberately absent: at nv=48 it would be 384 CTAs on 132 SMs (a
/// three-wave launch) while re-reading W and K eight times, and the value tile
/// would fall to 16 columns, below the 2 n-tiles Phase A's warp split needs.
pub const GDN_SPINE_VSPLIT_VALUES: [u32; 3] = [1, 2, 4];

/// `[defaults] gdn_spine_vsplit`, with `ATLAS_GDN_SPINE_VSPLIT` overriding.
///
/// Returns `(split, from_env)`. The rule is a pure function here, beside the
/// guards that consume it, for the reason `attn_splitk::resolve_policy` is one
/// in `atlas-kernels`: `target_defaults` REPORTS the resolved table, it does
/// not own each lever's grammar.
///
/// An unparseable or unsupported value keeps the TARGET's declaration and is
/// reported as target-sourced — the same disposition `resolve_batchm_max`
/// takes, and for the same reason: a typo in a launch recipe must not silently
/// become a launch geometry nobody chose. A declaration the code does not
/// support is clamped to [`GDN_SPINE_VSPLIT_OFF`] rather than trusted, so a
/// HARDWARE.toml can never name a split with no kernel behind it.
pub fn resolve_spine_vsplit(declared: u32, raw: Option<&str>) -> (u32, bool) {
    let supported = |v: u32| GDN_SPINE_VSPLIT_VALUES.contains(&v);
    let target = if supported(declared) {
        declared
    } else {
        GDN_SPINE_VSPLIT_OFF
    };
    match raw
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&v| supported(v))
    {
        Some(v) => (v, true),
        None => (target, false),
    }
}

/// Dynamic shared memory the value-split twin needs at `split`, in bytes.
///
/// SSOT mirror of `GDNV_SMEM` in
/// `kernels/hopper/common/gdn_chunk_delta_h_vsplit_hopper.cu`, which
/// `static_assert`s both values this function returns for 2 and 4.
///
/// Only three of the five tiles shrink with the split. `Wp` is k-space
/// ([64][136] of W) and `Kt` is k-space ([128][72] of Kᵀ); both are re-read by
/// every CTA and neither narrows. `Kt` ALIASES `St`, so that region is the
/// larger of the two — and past 1-way it is always `Kt`, which is why the
/// budget flattens out rather than halving again:
///
/// | split | St/Kt | Wp | Up | ducT | dec | total | CTAs/SM by smem (228 KB) |
/// |---|---|---|---|---|---|---|---|
/// | 1 (parent) | 34 816 | 17 408 | 17 408 | 18 432 | 260 | **88 324** | 2 |
/// | 2 | 18 432 | 17 408 | 9 216 | 9 216 | 260 | **54 532** | 4 |
/// | 4 | 18 432 | 17 408 | 5 120 | 4 608 | 260 | **45 828** | 5 |
///
/// `split == 1` returns [`GDN_TC_SMEM`]: at one way the parent IS the arm, and
/// a launcher that asked this function for the parent's footprint must get the
/// parent's number rather than a fourth spelling of it.
pub fn gdn_spine_vsplit_smem(split: u32) -> u32 {
    if split <= 1 {
        return GDN_TC_SMEM;
    }
    let vd_l = GDN_TC_DIM / split;
    let st = (vd_l * GDN_TC_SW).max(GDN_TC_DIM * GDN_TC_SC);
    st * 2
        + GDN_TC_CHUNK * GDN_TC_SW * 2
        + GDN_TC_CHUNK * (vd_l + 8) * 2
        + vd_l * GDN_TC_SC * 2
        + (GDN_TC_CHUNK + 1) * 4
}

/// `grid.y` for the value-split twin: one CTA per (stream, value-column block).
///
/// The shape `gated_delta_rule_chunk_delta_h_tc_vblock` already uses for its
/// DV blocks, so the kernel's `b = blockIdx.y / split` and
/// `vs = blockIdx.y % split` are not a new convention here.
pub fn gdn_spine_vsplit_grid_y(batch_size: u32, split: u32) -> u32 {
    batch_size.saturating_mul(split.max(1))
}

/// The entry point a split launches — the SSOT names, never a local string.
pub fn gdn_spine_vsplit_entry(split: u32) -> Option<&'static str> {
    match split {
        2 => Some(GDN_SPINE_VSPLIT2_ENTRY),
        4 => Some(GDN_SPINE_VSPLIT4_ENTRY),
        _ => None,
    }
}

/// Why the value-split twin is NOT running — `None` means it is.
///
/// Pure, and it NAMES THE GUARD THAT REFUSED, for the reason
/// [`gdn_tc_spine_reject`] does. The guards, in the order a reader meets them:
///   * `split == 1` is the declared OFF value, not a failure — the parent runs
///     and the message says which lever chose that;
///   * `!tc_ok` means the tensor-core FAMILY is not running at all, so there is
///     no spine for this lever to split. It is checked SECOND so that the
///     common "nobody asked" case does not read as a family problem;
///   * a split with no compiled entry, and a handle of 0, are separate
///     messages: the first is a declaration the code does not implement, the
///     second is a Hopper-only kernel absent from a gb10 or b200 image;
///   * `num_v_heads == 0` would launch an empty grid;
///   * `batch_size * split` must not overflow the grid — `u16::MAX` is well
///     inside the CUDA limit and well outside any serving batch, so this is a
///     sanity rail, not a tuned bound.
///
/// The tile guards are NOT repeated: this lever can only be reached once
/// [`gdn_tc_spine_reject`] returned `None`, which already established
/// 128/128/64 and the `qk_stride` alignment the K staging needs.
pub fn gdn_spine_vsplit_reject(
    split: u32,
    tc_ok: bool,
    kernel_present: bool,
    num_v_heads: u32,
    batch_size: u32,
) -> Option<&'static str> {
    if split <= 1 {
        Some("[defaults] gdn_spine_vsplit = 1 (the unsplit parent spine)")
    } else if !tc_ok {
        Some("the tensor-core spine is not running, so there is no state tile to split")
    } else if gdn_spine_vsplit_entry(split).is_none() {
        Some("no entry point is compiled for this split (supported: 2, 4)")
    } else if !kernel_present {
        Some("kernel absent from this image (kernels/hopper only)")
    } else if num_v_heads == 0 {
        Some("no value heads")
    } else if gdn_spine_vsplit_grid_y(batch_size, split) > u16::MAX as u32 {
        Some("batch_size * split overflows a sane grid.y")
    } else {
        None
    }
}

/// Which spine kernel this launch runs, with its grid, smem and the reason the
/// other arm did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpineVsplitPick {
    /// `1` when the unsplit parent runs; 2 or 4 when the twin does.
    pub split: u32,
    /// `grid.y` for the launch — `batch_size` at split 1.
    pub grid_y: u32,
    /// Dynamic shared memory, bytes.
    pub smem: u32,
    /// `None` when the twin runs; the named guard when the parent does.
    pub reject: Option<&'static str>,
}

/// [`gdn_spine_vsplit_reject`] applied, with the geometry that follows from it.
///
/// One function so the three answers (which kernel, which grid, how much smem)
/// cannot be derived in three places and disagree — the class of defect
/// `ssm_gdn_a3`'s own `smem_fused` comment records for the scalar spine.
pub fn gdn_spine_vsplit_pick(
    split: u32,
    tc_ok: bool,
    kernel_present: bool,
    num_v_heads: u32,
    batch_size: u32,
) -> SpineVsplitPick {
    match gdn_spine_vsplit_reject(split, tc_ok, kernel_present, num_v_heads, batch_size) {
        None => SpineVsplitPick {
            split,
            grid_y: gdn_spine_vsplit_grid_y(batch_size, split),
            smem: gdn_spine_vsplit_smem(split),
            reject: None,
        },
        why => SpineVsplitPick {
            split: GDN_SPINE_VSPLIT_OFF,
            grid_y: batch_size,
            smem: GDN_TC_SMEM,
            reject: why,
        },
    }
}

#[cfg(test)]
#[path = "ssm_gdn_tc_tests.rs"]
mod ssm_gdn_tc_tests;

#[cfg(test)]
#[path = "ssm_gdn_vsplit_tests.rs"]
mod ssm_gdn_vsplit_tests;
