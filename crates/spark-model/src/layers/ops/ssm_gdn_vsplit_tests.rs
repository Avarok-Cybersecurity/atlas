// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of the value-SPLIT spine twin's column-block index maps,
//! plus the `gdn_spine_vsplit` lever grammar (#928).
//!
//! Companion to `ssm_gdn_tc_tests`, and deliberately a different question.
//! That file asks whether the parent's maps compose to the recurrence. This
//! one asks whether SPLITTING the value dimension across CTAs changes any
//! per-element operation — because the claim the whole lever rests on is that
//! it does not:
//!
//!   * Phase A contracts over `k`, Phase B over `i`; NEITHER contracts over
//!     `v`. So state column `v` depends on operand column `v` and on the
//!     k-space operands `W`, `K` and the decay row, which are SHARED and
//!     re-read rather than reduced.
//!   * An `mma.sync.m16n8k16` output element is a fixed k-tree over the `ks`
//!     loop, and a warp's n-tiles are INDEPENDENT accumulators. So narrowing a
//!     C fragment from 16 n-tiles to 8 or 4 — which is all a split does —
//!     re-partitions accumulators without reassociating any one of them.
//!
//! The simulation below is therefore run in f64 and compared for **exact
//! equality**, not a tolerance: the split arm and the parent arm must execute
//! the same multiplies and the same additions in the same order per element,
//! and anything less than bit equality on a deterministic f64 replay means
//! they do not. A tolerance here would hide exactly the defect the microtest's
//! byte-equality gate exists to catch on device.

use super::{
    GDN_SPINE_VSPLIT_OFF, GDN_SPINE_VSPLIT_VALUES, GDN_TC_SMEM, gdn_spine_vsplit_entry,
    gdn_spine_vsplit_grid_y, gdn_spine_vsplit_pick, gdn_spine_vsplit_reject, gdn_spine_vsplit_smem,
    resolve_spine_vsplit,
};
use crate::layers::ops::{GDN_SPINE_VSPLIT2_ENTRY, GDN_SPINE_VSPLIT4_ENTRY};

const KD: usize = 128;
const VD: usize = 128;
const C: usize = 64;
const SW: usize = 136; // W / U / St padded row stride, in bf16 elements
const SC: usize = 72; //  Kt / ducT padded row stride
const THREADS: usize = 256;

// ── 1. the column-block maps ───────────────────────────────────────────────

/// Phase-B accumulator slot `(tid, nt, e)` -> the state element `S[k][v]` it
/// holds, for a CTA that owns value block `vs` of `split`. Mirrors the
/// kernel's `m0/m1` and `vbase + nt*8 + q*2`.
fn acc_slot(split: usize, vs: usize, tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q) = (lane >> 2, lane & 3);
    let k = warp * 16 + grp + if e >= 2 { 8 } else { 0 };
    let v = vs * (VD / split) + nt * 8 + q * 2 + (e & 1);
    (k, v)
}

/// Phase-A slot `(tid, nt, e)` -> the `ws[i][v]` element it holds, with `v`
/// LOCAL to the CTA's block. `a_n = (warp >> 2) * (VD_L / 2)` is the kernel's.
fn ws_slot(split: usize, tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q) = (lane >> 2, lane & 3);
    let vd_l = VD / split;
    let (a_m, a_n) = ((warp & 3) * 16, (warp >> 2) * (vd_l / 2));
    let i = a_m + grp + if e >= 2 { 8 } else { 0 };
    let v = a_n + nt * 8 + q * 2 + (e & 1);
    (i, v)
}

/// Phase-A n-tiles per warp at `split` — the kernel's `NTA`.
fn nta(split: usize) -> usize {
    VD / split / 16
}
/// Phase-B n-tiles per warp at `split` — the kernel's `NTB`.
fn ntb(split: usize) -> usize {
    VD / split / 8
}

#[test]
fn the_accumulator_map_tiles_the_state_exactly_once_at_every_split() {
    for split in [1usize, 2, 4] {
        let mut seen = vec![0u8; KD * VD];
        for vs in 0..split {
            for tid in 0..THREADS {
                for nt in 0..ntb(split) {
                    for e in 0..4 {
                        let (k, v) = acc_slot(split, vs, tid, nt, e);
                        assert!(k < KD && v < VD, "split={split} vs={vs} -> ({k},{v})");
                        seen[k * VD + v] += 1;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "the {split} CTAs must partition S[k][v]; holes={} duplicates={}",
            seen.iter().filter(|&&c| c == 0).count(),
            seen.iter().filter(|&&c| c > 1).count()
        );
        // The register budget claim in the kernel header: state f32 registers
        // per thread fall with the split, 64 -> 32 -> 16.
        assert_eq!(ntb(split) * 4, KD * VD / split / THREADS);
    }
}

#[test]
fn the_phase_a_map_covers_every_token_column_of_its_block_once() {
    for split in [1usize, 2, 4] {
        let vd_l = VD / split;
        let mut seen = vec![0u8; C * vd_l];
        for tid in 0..THREADS {
            for nt in 0..nta(split) {
                for e in 0..4 {
                    let (i, v) = ws_slot(split, tid, nt, e);
                    assert!(i < C && v < vd_l, "split={split} -> ({i},{v})");
                    seen[i * vd_l + v] += 1;
                }
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "split={split}: the 4 m-tile x 2 n-half warp split must neither \
             overlap nor leave a hole; holes={} duplicates={}",
            seen.iter().filter(|&&c| c == 0).count(),
            seen.iter().filter(|&&c| c > 1).count()
        );
    }
}

/// `Kt` aliases `St`, and `Kt` is k-space at every split — which is why the
/// smem model takes a max and why the budget flattens instead of halving.
#[test]
fn every_padded_address_stays_inside_the_split_buffers() {
    for split in [2usize, 4] {
        let vd_l = VD / split;
        let st_elems = (vd_l * SW).max(KD * SC);
        assert!(KD * SC <= st_elems, "Kt overruns St at split={split}");
        let su = vd_l + 8;
        for tid in 0..THREADS {
            for nt in 0..ntb(split) {
                for e in 0..4 {
                    let (_, v) = acc_slot(split, 0, tid, nt, e);
                    assert!(v * SW + KD - 1 < st_elems, "St[v][k] overflow");
                    assert!(v * SC + C - 1 < vd_l * SC, "ducT[v][i] overflow");
                    // The duc lo limb aliases the dead `Wp` at stride 68.
                    assert!(v * 68 + C - 1 < C * SW, "ducL overruns Wp");
                }
            }
            for nt in 0..nta(split) {
                for e in 0..4 {
                    let (i, v) = ws_slot(split, tid, nt, e);
                    assert!(i * su + v < C * su, "Up[i][v] overflow");
                }
            }
        }
        // ...and the launcher's byte count is the sum the kernel lays out.
        assert_eq!(
            gdn_spine_vsplit_smem(split as u32) as usize,
            st_elems * 2 + C * SW * 2 + C * su * 2 + vd_l * SC * 2 + (C + 1) * 4
        );
    }
    assert_eq!(gdn_spine_vsplit_smem(2), 54_532);
    assert_eq!(gdn_spine_vsplit_smem(4), 45_828);
    // Split 1 IS the parent, so it must answer with the parent's footprint.
    assert_eq!(gdn_spine_vsplit_smem(1), GDN_TC_SMEM);
    assert_eq!(GDN_TC_SMEM, 88_324);
}

// ── 2. the split is EXACT, operation for operation ─────────────────────────

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
}

/// T=150 -> chunks of 64, 64, 22. The partial tail is the point: `i >= ce` is
/// the one place the MMA form does work the scalar spine skipped, and the
/// split has to zero the same rows of two tiles with two different strides.
const T: usize = 150;

struct Fixture {
    w: Vec<f64>,   // [nchunks][C][KD]
    u: Vec<f64>,   // [nchunks][C][VD]
    key: Vec<f64>, // [T][KD]
    gc: Vec<f64>,  // [nchunks][C]
    h0: Vec<f64>,  // [KD][VD]
}

fn fixture() -> (Fixture, usize) {
    let nt = T.div_ceil(C);
    let mut r = Rng(0x0928_5F17_2026);
    let f = Fixture {
        w: (0..nt * C * KD).map(|_| r.f()).collect(),
        u: (0..nt * C * VD).map(|_| r.f()).collect(),
        key: (0..T * KD).map(|_| r.f()).collect(),
        gc: (0..nt * C).map(|i| -0.02 * (i % C) as f64).collect(),
        h0: (0..KD * VD).map(|_| r.f() * 0.2).collect(),
    };
    (f, nt)
}

/// The recurrence driven through the kernel's maps at `split` ways, with the
/// k-contraction evaluated in the MMA's `ks` order so the per-element addition
/// SEQUENCE is the one the hardware runs. Each of the `split` CTAs is
/// simulated independently and the results are gathered — which is exactly
/// what the device does, since nothing is reduced across them.
fn simulated(f: &Fixture, nchunks: usize, split: usize) -> Vec<f64> {
    simulated_biased(f, nchunks, split, 0)
}

/// [`simulated`] with `vbias` columns of deliberate offset in the U read — the
/// single most plausible slip in a column-block map, and one that leaves every
/// shape, stride and bound legal. Used only by the negative control.
fn simulated_biased(f: &Fixture, nchunks: usize, split: usize, vbias: usize) -> Vec<f64> {
    let vd_l = VD / split;
    let mut out = vec![0.0f64; KD * VD];
    for vs in 0..split {
        let st_elems = (vd_l * SW).max(KD * SC);
        let mut acc = vec![0.0f64; THREADS * 16 * 4];
        for tid in 0..THREADS {
            for nt in 0..ntb(split) {
                for e in 0..4 {
                    let (k, v) = acc_slot(split, vs, tid, nt, e);
                    acc[(tid * 16 + nt) * 4 + e] = f.h0[k * VD + v];
                }
            }
        }
        let mut st = vec![0.0f64; st_elems];
        let mut kt = vec![0.0f64; KD * SC];
        let mut duct = vec![0.0f64; vd_l * SC];
        let mut wp = vec![0.0f64; C * SW];
        let mut up = vec![0.0f64; C * (vd_l + 8)];
        let su = vd_l + 8;

        for c in 0..nchunks {
            let ce = (T - c * C).min(C);
            let gl = f.gc[c * C + ce - 1];
            let mut dec = vec![0.0f64; C + 1];
            dec[0] = gl.exp();
            for (i, d) in dec.iter_mut().skip(1).take(ce).enumerate() {
                *d = (gl - f.gc[c * C + i]).exp();
            }
            // (1) stage W (full) and this CTA's COLUMN BLOCK of U.
            for i in 0..C {
                for x in 0..KD {
                    wp[i * SW + x] = if i < ce {
                        f.w[(c * C + i) * KD + x]
                    } else {
                        0.0
                    };
                }
                for x in 0..vd_l {
                    up[i * su + x] = if i < ce {
                        f.u[(c * C + i) * VD + (vs * vd_l + x + vbias) % VD]
                    } else {
                        0.0
                    };
                }
            }
            // (2) snapshot S -> St[v_local][k].
            for tid in 0..THREADS {
                for nt in 0..ntb(split) {
                    for e in 0..4 {
                        let (k, v) = acc_slot(split, vs, tid, nt, e);
                        st[(v - vs * vd_l) * SW + k] = acc[(tid * 16 + nt) * 4 + e];
                    }
                }
            }
            // (3) K^T staging — k-space, identical at every split.
            for k in 0..KD {
                for i in 0..C {
                    kt[k * SC + i] = if i < ce {
                        f.key[(c * C + i) * KD + k]
                    } else {
                        0.0
                    };
                }
            }
            // (4)+(5) Phase A in `ks` order, then the transposed duc epilogue.
            for tid in 0..THREADS {
                for nt in 0..nta(split) {
                    for e in 0..4 {
                        let (i, v) = ws_slot(split, tid, nt, e);
                        let mut ws = 0.0;
                        for ks in (0..KD).step_by(16) {
                            for k in ks..ks + 16 {
                                ws += wp[i * SW + k] * st[v * SW + k];
                            }
                        }
                        let uci = up[i * su + v] - ws;
                        duct[v * SC + i] = if i < ce { dec[1 + i] * uci } else { 0.0 };
                    }
                }
            }
            // (7) Phase B, accumulating into the same registers.
            for tid in 0..THREADS {
                for nt in 0..ntb(split) {
                    for e in 0..4 {
                        let (k, v) = acc_slot(split, vs, tid, nt, e);
                        let slot = (tid * 16 + nt) * 4 + e;
                        let mut a = dec[0] * acc[slot];
                        for is in (0..C).step_by(16) {
                            for i in is..is + 16 {
                                a += kt[k * SC + i] * duct[(v - vs * vd_l) * SC + i];
                            }
                        }
                        acc[slot] = a;
                    }
                }
            }
        }
        for tid in 0..THREADS {
            for nt in 0..ntb(split) {
                for e in 0..4 {
                    let (k, v) = acc_slot(split, vs, tid, nt, e);
                    out[k * VD + v] = acc[(tid * 16 + nt) * 4 + e];
                }
            }
        }
    }
    out
}

/// THE CLAIM: the 2- and 4-way splits are BIT-IDENTICAL to the unsplit arm,
/// not merely close. Exact `==` on f64, because a split that reassociated
/// anything would show up here as a last-bit difference and a tolerance would
/// swallow it.
#[test]
fn the_value_split_is_bit_identical_to_the_unsplit_spine() {
    let (f, nchunks) = fixture();
    let base = simulated(&f, nchunks, 1);
    for split in [2usize, 4] {
        let got = simulated(&f, nchunks, split);
        let bad = base
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "split={split}: {bad} of {} state elements differ from the unsplit \
             arm. The value dimension is not contracted over in either phase, so \
             ANY difference is a map defect, not a numerics trade",
            base.len()
        );
    }
}

/// NEGATIVE CONTROL: the test above is only evidence if a wrong column map
/// fails it. Offsetting one CTA's `vbase` by a single column leaves every
/// shape, bound and stride legal and must still break the comparison.
#[test]
fn a_shifted_column_block_breaks_the_bit_identity() {
    let (f, nchunks) = fixture();
    let base = simulated(&f, nchunks, 1);
    for split in [2usize, 4] {
        let got = simulated_biased(&f, nchunks, split, 1);
        assert!(
            base.iter()
                .zip(got.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "split={split}: a one-column offset in the U read must move the \
             result, or the bit-identity assertion above is not evidence"
        );
    }
}

// ── 3. the lever grammar ───────────────────────────────────────────────────

/// Production geometry with the tensor-core family live: both splits accepted,
/// and the grid is the whole claim — 48 CTAs become 96 and 192.
#[test]
fn the_production_geometry_accepts_both_splits() {
    for (split, want_y) in [(2u32, 2u32), (4, 4)] {
        let pick = gdn_spine_vsplit_pick(split, true, true, 48, 1);
        assert_eq!(pick.reject, None, "split={split}");
        assert_eq!(pick.split, split);
        assert_eq!(pick.grid_y, want_y);
        assert_eq!(pick.smem, gdn_spine_vsplit_smem(split));
        assert_eq!(48 * pick.grid_y, 48 * split, "CTAs = nv * split");
    }
}

/// The shipped default is OFF on every target, and OFF means the parent: the
/// parent's grid, the parent's smem, and a message naming the lever that chose
/// it rather than silence.
#[test]
fn the_default_is_the_unsplit_parent_and_says_so() {
    let pick = gdn_spine_vsplit_pick(GDN_SPINE_VSPLIT_OFF, true, true, 48, 1);
    assert_eq!(pick.split, 1);
    assert_eq!(pick.grid_y, 1);
    assert_eq!(pick.smem, GDN_TC_SMEM);
    assert_eq!(
        pick.reject,
        Some("[defaults] gdn_spine_vsplit = 1 (the unsplit parent spine)")
    );
}

/// Every refusal NAMES its guard, so an A/B that silently fell back cannot be
/// mistaken for a lever with no effect (PR #296).
#[test]
fn every_refusal_names_its_guard() {
    for (case, want) in [
        (
            gdn_spine_vsplit_reject(2, false, true, 48, 1),
            "the tensor-core spine is not running, so there is no state tile to split",
        ),
        (
            gdn_spine_vsplit_reject(3, true, true, 48, 1),
            "no entry point is compiled for this split (supported: 2, 4)",
        ),
        (
            gdn_spine_vsplit_reject(2, true, false, 48, 1),
            "kernel absent from this image (kernels/hopper only)",
        ),
        (
            gdn_spine_vsplit_reject(2, true, true, 0, 1),
            "no value heads",
        ),
        (
            gdn_spine_vsplit_reject(4, true, true, 48, 65_535),
            "batch_size * split overflows a sane grid.y",
        ),
    ] {
        assert_eq!(case, Some(want));
    }
}

/// A gb10 or b200 image has no such kernel, so the lever cannot reach one even
/// if a recipe sets it — the handle of 0 IS the "not on this target" answer.
#[test]
fn a_target_without_the_kernel_keeps_the_unsplit_spine() {
    let pick = gdn_spine_vsplit_pick(2, true, false, 48, 1);
    assert_eq!(pick.split, GDN_SPINE_VSPLIT_OFF);
    assert_eq!(pick.grid_y, 1);
    assert_eq!(pick.smem, GDN_TC_SMEM);
}

/// The override grammar: the environment names a SUPPORTED split or it is
/// ignored, and an ignored value keeps the target's declaration and reports as
/// target-sourced — a typo in a launch recipe must not become a launch
/// geometry nobody chose.
#[test]
fn the_override_grammar_only_accepts_compiled_splits() {
    assert_eq!(resolve_spine_vsplit(1, None), (1, false));
    assert_eq!(resolve_spine_vsplit(1, Some("2")), (2, true));
    assert_eq!(resolve_spine_vsplit(1, Some(" 4 ")), (4, true));
    assert_eq!(resolve_spine_vsplit(2, Some("1")), (1, true));
    for junk in ["3", "8", "0", "", "on", "true", "-2"] {
        assert_eq!(
            resolve_spine_vsplit(2, Some(junk)),
            (2, false),
            "`{junk}` must keep the target's declaration"
        );
    }
    // A declaration the code does not implement is clamped to OFF, not trusted.
    assert_eq!(resolve_spine_vsplit(3, None), (GDN_SPINE_VSPLIT_OFF, false));
    assert_eq!(resolve_spine_vsplit(0, None), (GDN_SPINE_VSPLIT_OFF, false));
}

/// Split -> entry name is a total function over the supported set and names
/// the SSOT constants, so the route line and the handle cannot disagree.
#[test]
fn every_supported_split_maps_to_a_compiled_entry() {
    assert_eq!(
        gdn_spine_vsplit_entry(1),
        None,
        "1 is the parent, not a twin"
    );
    assert_eq!(gdn_spine_vsplit_entry(2), Some(GDN_SPINE_VSPLIT2_ENTRY));
    assert_eq!(gdn_spine_vsplit_entry(4), Some(GDN_SPINE_VSPLIT4_ENTRY));
    assert_eq!(gdn_spine_vsplit_entry(3), None);
    for v in GDN_SPINE_VSPLIT_VALUES {
        assert_eq!(gdn_spine_vsplit_entry(v).is_some(), v > 1, "split={v}");
    }
    assert_eq!(gdn_spine_vsplit_grid_y(4, 2), 8);
    assert_eq!(gdn_spine_vsplit_grid_y(4, 1), 4);
}
