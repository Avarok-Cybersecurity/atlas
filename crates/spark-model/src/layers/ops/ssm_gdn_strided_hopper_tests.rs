// SPDX-License-Identifier: AGPL-3.0-only

//! Host unit tests for the one-read strided Hopper GDN decode twin (#927).
//!
//! ORACLE: the kernel header of
//! `kernels/hopper/common/gdn_decode_strided_hopper.cu` and the ptxas receipt
//! quoted in it. These tests cannot run the kernel — they pin the maps and the
//! rules the kernel's bit-identity and occupancy arguments rest on, so a later
//! edit to either side shows up as a red test rather than as a silent
//! divergence between a `.cu` literal and a serve log.

use super::{
    GDN_STRIDED_SMEM_BLOCK, GDN_STRIDED_SMEM_BYTES, GDN_STRIDED_SMEM_CTAS_PER_SM,
    GDN_STRIDED_SMEM_DIM, GDN_STRIDED_SMEM_ENTRY, GDN_STRIDED_SMEM_MIN_ROWS,
    GDN_STRIDED_SMEM_MODULE, GDN_STRIDED_SMEM_REG_ROWS, GDN_STRIDED_SMEM_REREAD_ROWS,
    GDN_STRIDED_SMEM_STAGED_ROWS, GdnStridedRowHome, gdn_decode_strided_smem_accept,
    gdn_decode_strided_smem_route_line, gdn_strided_smem_dims_ok, gdn_strided_smem_elem,
    gdn_strided_smem_row_home,
};

/// Qwen3.8-27B GDN geometry, from `kernels/hopper/qwen3.8-27b/MODEL.toml` via
/// `r13-native_gdn_chunk_prefill_microtest.log`: `nk=16 nv=48 kd=128 vd=128`.
const NV: u32 = 48;
/// H100/H200 SXM5 — `kernels/hopper/HARDWARE.toml` `[hardware] sm_count`.
const H100_SMS: u32 = 132;

// ── the tile map ──────────────────────────────────────────────────────────

/// THE BIT-IDENTITY PRECONDITION, as a property of the map: the three row
/// segments cover `[0, 128)` exactly once each. A gap would drop a term from
/// the `kd` chain and an overlap would apply the update to one row twice;
/// neither is visible by reading three consecutive loops.
#[test]
fn the_three_row_segments_partition_the_tile_exactly_once() {
    let mut seen = vec![0u8; GDN_STRIDED_SMEM_DIM as usize];
    let (mut smem, mut reg, mut reread) = (0u32, 0u32, 0u32);
    for j in 0..GDN_STRIDED_SMEM_DIM {
        match gdn_strided_smem_row_home(j).expect("row inside the tile") {
            GdnStridedRowHome::Smem => smem += 1,
            GdnStridedRowHome::Register => reg += 1,
            GdnStridedRowHome::Reread => reread += 1,
        }
        seen[j as usize] += 1;
    }
    assert!(seen.iter().all(|&n| n == 1), "rows covered {seen:?} times");
    assert_eq!(
        (smem, reg, reread),
        (72, 24, 32),
        "the shipped 72/24/32 split"
    );
    assert_eq!(
        smem + reg + reread,
        GDN_STRIDED_SMEM_DIM,
        "the segments must be the whole tile"
    );
    assert_eq!(gdn_strided_smem_row_home(GDN_STRIDED_SMEM_DIM), None);
}

/// …and the segments are CONTIGUOUS and ASCENDING, in that order. The
/// accumulator is carried across the two boundaries, so re-ordering the
/// segments would re-bracket `hk_dot` and `q_dot` and break bit-identity even
/// though the same 128 terms are summed.
#[test]
fn the_segments_are_ascending_and_contiguous() {
    let order = [
        GdnStridedRowHome::Smem,
        GdnStridedRowHome::Register,
        GdnStridedRowHome::Reread,
    ];
    let mut idx = 0usize;
    for j in 0..GDN_STRIDED_SMEM_DIM {
        let home = gdn_strided_smem_row_home(j).unwrap();
        if home != order[idx] {
            idx += 1;
            assert!(idx < order.len(), "segment order changed at row {j}");
            assert_eq!(home, order[idx], "segment order changed at row {j}");
        }
    }
    assert_eq!(idx, 2, "all three segments must be present");
    // Every segment boundary is a multiple of 4: the loops step by 4 and the
    // parent's group shape `h0*k0 + h1*k1 + h2*k2 + h3*k3` is what a boundary
    // off a multiple of 4 would split.
    for b in [
        GDN_STRIDED_SMEM_STAGED_ROWS,
        GDN_STRIDED_SMEM_STAGED_ROWS + GDN_STRIDED_SMEM_REG_ROWS,
    ] {
        assert_eq!(b % 4, 0, "segment boundary {b} splits a group of four");
    }
}

/// The lane map is the PARENT's: thread `tid` owns state column `tid` and
/// walks every row itself. Checked as addresses, because that is what makes
/// the twin's write set the parent's write set element for element.
#[test]
fn one_thread_owns_one_column_and_the_addresses_are_the_parents() {
    for (vh, b) in [(0u32, 0u32), (47, 15), (13, 3)] {
        for j in [0u32, 71, 72, 95, 96, 127] {
            for tid in [0u32, 1, 31, 127] {
                let got = gdn_strided_smem_elem(vh, b, NV, j, tid);
                // The parent's own expression:
                //   H = h_state + (b * num_v_heads + vh) * k_dim * v_dim
                //   H[j * v_dim + tid]
                let want = u64::from((b * NV + vh) * 128 * 128) + u64::from(j * 128 + tid);
                assert_eq!(got, want, "vh={vh} b={b} j={j} tid={tid}");
            }
        }
    }
    // Two threads of one CTA never alias, at any row — the property that makes
    // the shared-memory staging safe without a per-column lock.
    let mut all = std::collections::BTreeSet::new();
    for j in 0..GDN_STRIDED_SMEM_DIM {
        for tid in 0..GDN_STRIDED_SMEM_BLOCK {
            assert!(
                all.insert(gdn_strided_smem_elem(0, 0, NV, j, tid)),
                "row {j} thread {tid} aliases another element"
            );
        }
    }
    assert_eq!(all.len(), 128 * 128, "the CTA covers its tile exactly");
}

/// The shared-memory figure the route line prints is the one ptxas reports:
/// 72 rows x 128 columns x 4 B + `smem_k` + `smem_q` + the clamp's
/// `norm_sums[4]` = 37 904 B, and six of those fit in a GH100 SM's 233 472 B.
#[test]
fn the_shared_memory_footprint_is_the_ptxas_one() {
    assert_eq!(GDN_STRIDED_SMEM_BYTES, 37_904);
    // A GH100 SM offers 233 472 B of shared memory; six CTAs of this kernel
    // must fit, or the `__launch_bounds__(128, 6)` contract is a fiction.
    const { assert!(GDN_STRIDED_SMEM_BYTES * GDN_STRIDED_SMEM_CTAS_PER_SM <= 233_472) };
    // 80 registers x 128 threads x 6 CTAs must fit the 65 536-register file —
    // the other half of the `__launch_bounds__(128, 6)` contract.
    const { assert!(80 * GDN_STRIDED_SMEM_BLOCK * GDN_STRIDED_SMEM_CTAS_PER_SM <= 65_536) };
    // Static shared memory is capped at 48 KB per block on sm_90 without an
    // opt-in the launcher does not make.
    const { assert!(GDN_STRIDED_SMEM_BYTES < 49_152) };
    assert_eq!(GDN_STRIDED_SMEM_REREAD_ROWS, 32);
}

// ── the width guard ───────────────────────────────────────────────────────

/// THE GUARD, at the shapes the campaign actually serves. n=16 is the step
/// this twin was written for (768 CTAs); n=1 is the one the round-12 loss
/// belongs to and stays on the parent.
#[test]
fn the_width_guard_takes_n16_and_declines_n1() {
    assert!(gdn_decode_strided_smem_accept(16, NV, H100_SMS), "n=16");
    assert!(gdn_decode_strided_smem_accept(4, NV, H100_SMS), "n=4");
    assert!(!gdn_decode_strided_smem_accept(1, NV, H100_SMS), "n=1");
    assert!(!gdn_decode_strided_smem_accept(2, NV, H100_SMS), "n=2");
    assert!(
        !gdn_decode_strided_smem_accept(3, NV, H100_SMS),
        "n=3 is 144 CTAs but below the row threshold"
    );
    assert_eq!(GDN_STRIDED_SMEM_MIN_ROWS, 4);
}

/// BOTH conditions are load-bearing: a wide batch of a narrow head count can
/// still leave SMs idle, and that shape must stay on the parent even though it
/// clears the row threshold.
#[test]
fn a_grid_under_the_sm_count_declines_however_many_rows() {
    assert!(
        !gdn_decode_strided_smem_accept(8, 4, H100_SMS),
        "8 rows x 4 heads = 32 CTAs on 132 SMs"
    );
    assert!(
        gdn_decode_strided_smem_accept(8, 4, 32),
        "…but fills a 32-SM part"
    );
    // A zero SM count (a backend that cannot answer) must not divide by nothing
    // nor accept everything: it is clamped to 1, so the row threshold decides.
    assert!(gdn_decode_strided_smem_accept(4, 1, 0));
    assert!(!gdn_decode_strided_smem_accept(1, 1, 0));
    // No overflow panic at absurd widths.
    assert!(gdn_decode_strided_smem_accept(u32::MAX, u32::MAX, H100_SMS));
}

/// The kernel's dimension contract, which is not negotiable: the staging
/// buffer is statically sized and the clamp reduces across the whole head.
#[test]
fn only_the_128x128_head_is_accepted() {
    assert!(gdn_strided_smem_dims_ok(128, 128));
    for (k, v) in [(64u32, 128u32), (128, 64), (256, 128), (128, 256), (0, 0)] {
        assert!(!gdn_strided_smem_dims_ok(k, v), "k={k} v={v}");
    }
}

// ── the route line ────────────────────────────────────────────────────────

/// The line NAMES THE ENTRY, not the family — the round-12 stage-4b nit, which
/// this file inherits because it ships a second kernel whose name differs from
/// the first by a suffix. `…_strided_hopper` and `…_strided_hopper_smem` are
/// one keystroke apart in a log.
#[test]
fn the_route_line_names_the_entry_that_is_launched() {
    let line = gdn_decode_strided_smem_route_line(true, NV, 16, H100_SMS);
    assert!(line.contains(GDN_STRIDED_SMEM_ENTRY), "{line}");
    assert!(
        !line.contains("gated_delta_rule_decode_f32_strided_hopper "),
        "the line must not name the OTHER Hopper twin:\n{line}"
    );
    for field in [
        "grid=[48,16]",
        "block=128",
        "smem=37904B",
        "ctas=768",
        "sm_count=132",
        "gdn_decode_strided_hopper",
    ] {
        assert!(line.contains(field), "missing `{field}` in:\n{line}");
    }
}

/// …and the declined arm names the PARENT, and says why, so a reader who
/// opened the log to answer "did the lever engage?" gets an answer either way.
#[test]
fn the_declined_route_line_names_the_parent_and_the_reason() {
    let line = gdn_decode_strided_smem_route_line(false, NV, 1, H100_SMS);
    assert!(
        line.contains("gated_delta_rule_decode_f32_strided"),
        "{line}"
    );
    assert!(!line.contains(GDN_STRIDED_SMEM_ENTRY), "{line}");
    assert!(line.contains("ctas=48"), "{line}");
}

/// The module and entry the launcher, the probe and the microtest all spell —
/// one string each, so the handle and the log cannot drift apart.
#[test]
fn the_module_and_entry_are_the_files_own_names() {
    assert_eq!(GDN_STRIDED_SMEM_MODULE, "gdn_decode_strided_hopper");
    assert_eq!(
        GDN_STRIDED_SMEM_ENTRY,
        "gated_delta_rule_decode_f32_strided_hopper_smem"
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/hopper/common")
        .join(format!("{GDN_STRIDED_SMEM_MODULE}.cu"));
    let src = std::fs::read_to_string(&root).unwrap_or_else(|e| panic!("{}: {e}", root.display()));
    assert!(
        src.contains(&format!("void {GDN_STRIDED_SMEM_ENTRY}(")),
        "the kernel does not declare `{GDN_STRIDED_SMEM_ENTRY}`"
    );
    // The host's mirror of the tile map is the kernel's own literals.
    for lit in [
        format!("#define GDN_STR_SMEM_ROWS {GDN_STRIDED_SMEM_STAGED_ROWS}"),
        format!("#define GDN_STR_REG_ROWS {GDN_STRIDED_SMEM_REG_ROWS}"),
        format!("#define GDN_STR_MAX_KD {GDN_STRIDED_SMEM_DIM}"),
        format!("__launch_bounds__({GDN_STRIDED_SMEM_BLOCK}, {GDN_STRIDED_SMEM_CTAS_PER_SM})"),
    ] {
        assert!(src.contains(&lit), "kernel is missing `{lit}`");
    }
}
