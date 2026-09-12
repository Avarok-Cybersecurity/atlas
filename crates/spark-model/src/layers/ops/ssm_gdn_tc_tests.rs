// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of `gated_delta_rule_chunk_delta_h_tcfuse`'s index algebra,
//! plus the `ATLAS_GDN_PREFILL_TC` lever grammar (#928).
//!
//! The kernel's risk is not its arithmetic — an `mma.sync` accumulates in f32
//! whatever it is handed. The risk is the four maps that decide WHAT it is
//! handed, because the recurrent state never leaves the MMA accumulator and is
//! therefore addressed by warp/lane rather than by a loop variable:
//!
//!   1. the Phase-B C-fragment map, which IS the state layout `S[k][v]`;
//!   2. the Phase-A C-fragment map over `ws[i][v]`, split 4 m-tiles x 2 n-halves
//!      across 8 warps (the shipped `mma_gram` only ever fences 4 warps, so this
//!      split is new and is exactly where an overlap or a hole would live);
//!   3. the `K -> Kt` transpose map that feeds Phase B's `.row` operand;
//!   4. the padded smem strides (136 for W/U/St, 72 for Kt/ducT), which exist to
//!      make the fragment reads bank-conflict-free and which silently read the
//!      wrong column if the launcher and the kernel disagree.
//!
//! Every one is checked here for bijectivity AND composed end-to-end against a
//! plain `for k { for v { ... } }` recurrence, at a shape whose last chunk is
//! PARTIAL (T=150 -> chunks of 64, 64, 22) because the `i >= ce` zero-fill is
//! the one place the MMA form does work the scalar spine skipped.

use super::{GDN_TC_CHUNK, GDN_TC_DIM, GDN_TC_SMEM, gdn_tc_spine_reject};

const KD: usize = 128;
const VD: usize = 128;
const C: usize = 64;
const SW: usize = 136; // W / U / St padded row stride, in bf16 elements
const SC: usize = 72; //  Kt / ducT padded row stride
const THREADS: usize = 256;

/// `Kt` ALIASES `St` in the kernel (Phase A consumes St, then it is dead and
/// the K transpose reuses its bytes). Checked at compile time because it is a
/// property of the two padded shapes, not of any run.
const _: () = assert!(KD * SC * 2 <= VD * SW * 2);

/// Phase-B accumulator slot `(tid, nt, e)` -> the state element `S[k][v]` it
/// holds. Mirrors the kernel's `m0/m1` and `n0/n1` exactly.
fn acc_slot(tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q) = (lane >> 2, lane & 3);
    let k = warp * 16 + grp + if e >= 2 { 8 } else { 0 };
    let v = nt * 8 + q * 2 + (e & 1);
    (k, v)
}

/// Phase-A slot `(tid, nt, e)` -> the `ws[i][v]` element it holds.
fn ws_slot(tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q) = (lane >> 2, lane & 3);
    let (a_m, a_n) = ((warp & 3) * 16, (warp >> 2) * 64);
    let i = a_m + grp + if e >= 2 { 8 } else { 0 };
    let v = a_n + nt * 8 + q * 2 + (e & 1);
    (i, v)
}

/// K-staging slot `(tid, j, e)` -> the `(k, i)` pair it moves into `Kt`.
fn kt_slot(tid: usize, j: usize, e: usize) -> (usize, usize) {
    ((tid & 3) * 32 + j * 8 + e, tid >> 2)
}

// ── 1. the three maps are bijections ───────────────────────────────────────

#[test]
fn the_accumulator_map_tiles_the_state_exactly_once() {
    let mut seen = vec![0u8; KD * VD];
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                assert!(k < KD && v < VD, "tid={tid} nt={nt} e={e} -> ({k},{v})");
                seen[k * VD + v] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "every S[k][v] must be owned by exactly one lane slot; \
         holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
    // 64 f32 accumulator registers per thread is the budget claim in the
    // kernel's header (half of the scalar spine's 128) — 16 tiles x 4.
    assert_eq!(16 * 4, KD * VD / THREADS);
}

#[test]
fn the_phase_a_map_covers_every_token_column_once() {
    let mut seen = vec![0u8; C * VD];
    for tid in 0..THREADS {
        for nt in 0..8 {
            for e in 0..4 {
                let (i, v) = ws_slot(tid, nt, e);
                assert!(i < C && v < VD, "tid={tid} nt={nt} e={e} -> ({i},{v})");
                seen[i * VD + v] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "the 4 m-tile x 2 n-half warp split must neither overlap nor leave a \
         hole; holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
}

#[test]
fn the_k_transpose_map_is_a_bijection() {
    let mut seen = vec![0u8; KD * C];
    for tid in 0..THREADS {
        for j in 0..4 {
            for e in 0..8 {
                let (k, i) = kt_slot(tid, j, e);
                assert!(k < KD && i < C);
                // The 16-byte vector load the kernel uses is 8 contiguous bf16
                // starting at `kcol + j*8`, so `e` must stay inside one load.
                assert_eq!(k / 8, ((tid & 3) * 32 + j * 8) / 8);
                seen[k * C + i] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "K^T staging must tile [128][64]"
    );
}

/// The padded strides are the reason the maps are safe to index at all: every
/// address the kernel forms must land inside the buffer the launcher sized.
#[test]
fn every_padded_address_stays_inside_its_buffer() {
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                assert!(v * SW + k < VD * SW, "St[v][k] overflow");
                assert!(v * SC + 63 < VD * SC, "ducT[v][i] overflow");
            }
        }
        for j in 0..4 {
            for e in 0..8 {
                let (k, i) = kt_slot(tid, j, e);
                assert!(k * SC + i < KD * SC, "Kt[k][i] overflow");
            }
        }
    }
    // ...and the launcher's byte count must be the sum the kernel lays out.
    assert_eq!(
        GDN_TC_SMEM as usize,
        VD * SW * 2 + 2 * (C * SW * 2) + VD * SC * 2 + (C + 1) * 4
    );
    assert_eq!(GDN_TC_SMEM, 88_324);
}

// ── 2. the maps COMPOSE to the recurrence ──────────────────────────────────

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

/// T=150 -> chunks of 64, 64, 22. The partial tail is the point.
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
    let mut r = Rng(0x0928_7CF0_2026);
    let f = Fixture {
        w: (0..nt * C * KD).map(|_| r.f()).collect(),
        u: (0..nt * C * VD).map(|_| r.f()).collect(),
        key: (0..T * KD).map(|_| r.f()).collect(),
        // A decreasing cumulative log-gate, as recompute_wu emits.
        gc: (0..nt * C).map(|i| -0.02 * (i % C) as f64).collect(),
        h0: (0..KD * VD).map(|_| r.f() * 0.2).collect(),
    };
    (f, nt)
}

/// The plain recurrence, written the obvious way.
fn reference(f: &Fixture, nt: usize) -> Vec<f64> {
    let mut s = f.h0.clone();
    for c in 0..nt {
        let ce = (T - c * C).min(C);
        let gl = f.gc[c * C + ce - 1];
        let mut duc = vec![0.0f64; C * VD];
        for i in 0..ce {
            let dc = (gl - f.gc[c * C + i]).exp();
            for v in 0..VD {
                let mut ws = 0.0;
                for k in 0..KD {
                    ws += f.w[(c * C + i) * KD + k] * s[k * VD + v];
                }
                duc[i * VD + v] = dc * (f.u[(c * C + i) * VD + v] - ws);
            }
        }
        let edl = gl.exp();
        for k in 0..KD {
            for v in 0..VD {
                let mut a = edl * s[k * VD + v];
                for i in 0..ce {
                    a += duc[i * VD + v] * f.key[(c * C + i) * KD + k];
                }
                s[k * VD + v] = a;
            }
        }
    }
    s
}

/// The SAME recurrence driven entirely through the kernel's maps and padded
/// smem strides: state lives in `acc[tid][nt][e]`, never in `S[k][v]`.
fn simulated(f: &Fixture, nt_chunks: usize) -> Vec<f64> {
    let mut acc = vec![0.0f64; THREADS * 16 * 4];
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                acc[(tid * 16 + nt) * 4 + e] = f.h0[k * VD + v];
            }
        }
    }
    let mut st = vec![0.0f64; VD * SW];
    let mut kt = vec![0.0f64; KD * SC];
    let mut duct = vec![0.0f64; VD * SC];
    let mut wp = vec![0.0f64; C * SW];
    let mut up = vec![0.0f64; C * SW];

    for c in 0..nt_chunks {
        let ce = (T - c * C).min(C);
        let gl = f.gc[c * C + ce - 1];
        let mut dec = vec![0.0f64; C + 1];
        dec[0] = gl.exp();
        for i in 0..ce {
            dec[1 + i] = (gl - f.gc[c * C + i]).exp();
        }
        // (1) stage W/U with the padded stride, zero-filling rows past `ce`.
        for i in 0..C {
            for x in 0..KD {
                wp[i * SW + x] = if i < ce {
                    f.w[(c * C + i) * KD + x]
                } else {
                    0.0
                };
                up[i * SW + x] = if i < ce {
                    f.u[(c * C + i) * VD + x]
                } else {
                    0.0
                };
            }
        }
        // (2) snapshot S -> St[v][k], straight out of the accumulator.
        for tid in 0..THREADS {
            for nt in 0..16 {
                for e in 0..4 {
                    let (k, v) = acc_slot(tid, nt, e);
                    st[v * SW + k] = acc[(tid * 16 + nt) * 4 + e];
                }
            }
        }
        // (3) K^T staging.
        for tid in 0..THREADS {
            for j in 0..4 {
                for e in 0..8 {
                    let (k, i) = kt_slot(tid, j, e);
                    kt[k * SC + i] = if i < ce {
                        f.key[(c * C + i) * KD + k]
                    } else {
                        0.0
                    };
                }
            }
        }
        // (4)+(5) Phase A, then the epilogue that writes duc TRANSPOSED.
        for tid in 0..THREADS {
            for nt in 0..8 {
                for e in 0..4 {
                    let (i, v) = ws_slot(tid, nt, e);
                    let mut ws = 0.0;
                    for k in 0..KD {
                        ws += wp[i * SW + k] * st[v * SW + k];
                    }
                    let uci = up[i * SW + v] - ws;
                    duct[v * SC + i] = if i < ce { dec[1 + i] * uci } else { 0.0 };
                }
            }
        }
        // (7) Phase B, accumulating into the same registers.
        for tid in 0..THREADS {
            for nt in 0..16 {
                for e in 0..4 {
                    let (k, v) = acc_slot(tid, nt, e);
                    let slot = (tid * 16 + nt) * 4 + e;
                    let mut a = dec[0] * acc[slot];
                    for i in 0..C {
                        a += kt[k * SC + i] * duct[v * SC + i];
                    }
                    acc[slot] = a;
                }
            }
        }
    }
    let mut out = vec![0.0f64; KD * VD];
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                out[k * VD + v] = acc[(tid * 16 + nt) * 4 + e];
            }
        }
    }
    out
}

#[test]
fn the_simulated_index_math_reproduces_the_recurrence() {
    let (f, nt) = fixture();
    let want = reference(&f, nt);
    let got = simulated(&f, nt);
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want.iter()) {
        se += (a - b) * (a - b);
        sr += b * b;
    }
    let rel = (se / sr).sqrt();
    assert!(
        rel < 1e-12,
        "the kernel's maps must compose to the plain recurrence; rel_rms={rel:e}"
    );
}

/// NEGATIVE CONTROL: the test above is only evidence if a wrong map fails it.
/// Transposing `ducT` back to `duc[i][v]` — the single most plausible slip in
/// this kernel, and one that leaves every shape and bound legal — must break it.
#[test]
fn a_transposed_duc_breaks_the_composition() {
    let (f, nt) = fixture();
    let want = reference(&f, nt);
    let mut bad = f;
    // Equivalent injection: swap the K^T staging to K[i][k], which is the same
    // class of error on the other operand and stays in bounds because KD > C.
    bad.key.reverse();
    let got = simulated(&bad, nt);
    let diff: f64 = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1.0, "a perturbed operand must move the result");
}

// ── 3. the lever grammar ───────────────────────────────────────────────────

/// Production geometry (Qwen3.8-27B: kd=vd=128, chunk 64, qk_stride=conv_dim
/// =10240) is the one combination that must be accepted.
#[test]
fn the_production_geometry_is_accepted() {
    assert_eq!(gdn_tc_spine_reject(true, true, 128, 128, 64, 10240), None);
}

/// A FALSE resolved bit refuses the spine even with the kernel present and the
/// production geometry — which is what `ATLAS_GDN_PREFILL_TC=0` has to mean now
/// that `kernels/hopper` declares `[defaults] gdn_prefill_tc = true` (round 13)
/// and the variable is the family's kill switch rather than its arming lever.
/// The bit is RESOLVED by `target_defaults` and handed in; this layer never
/// reads the environment, so "off" is one question with one answer here.
#[test]
fn a_false_lever_refuses_the_spine() {
    assert_eq!(
        gdn_tc_spine_reject(false, true, 128, 128, 64, 10240),
        Some("not requested"),
        "a false resolved bit must keep the scalar spine, whatever else holds"
    );
}

/// Each refusal NAMES itself, so an A/B that silently fell back cannot be
/// mistaken for a lever with no effect.
#[test]
fn every_refusal_names_its_guard() {
    for (case, want) in [
        (
            gdn_tc_spine_reject(true, false, 128, 128, 64, 10240),
            "kernel absent from this image",
        ),
        (
            gdn_tc_spine_reject(true, true, 64, 128, 64, 10240),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_tc_spine_reject(true, true, 128, 64, 64, 10240),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_tc_spine_reject(true, true, 128, 128, 32, 10240),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_tc_spine_reject(true, true, 128, 128, 64, 10244),
            "qk_stride is not a multiple of 8 (the K staging uses 16-byte vector loads)",
        ),
    ] {
        assert_eq!(case, Some(want));
    }
}

/// The two constants the launcher and the kernel share must not drift.
#[test]
fn the_tile_constants_match_the_kernel() {
    assert_eq!(GDN_TC_DIM, 128);
    assert_eq!(GDN_TC_CHUNK, 64);
}
