// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests for GDN write-on-accept on the batched verify
//! (port lane L10, row `gdn_woa_mtp`): the engage decision at every width, its
//! agreement with the fold-kernel choice and the model's stash sizing, and a
//! CPU oracle of the record-and-fold algebra over the exact stash layout of
//! `kernels/ada/common/gated_delta_rule_wyn_woa.cu` (and, at K = 4, of the
//! gb10 pair).
//!
//! The oracle reproduces the parent wyN kernel's expressions in f32 (Rust
//! never contracts a multiply-add, which is the CPU twin of the kernels'
//! `--fmad=false`), runs the twin (outputs plus stash, no state writes) and
//! the fold for every accepted count, and requires the folded state to be
//! bit-equal to the parent's intermediate. It proves the layout and the fold
//! recurrence; the GPU oracles (`examples/gdn_woa_oracle.rs`,
//! `examples/gdn_wyn_woa_oracle.rs`) prove the kernels.

// The oracle indexes rows, heads and columns the way the kernels do, which
// reads clearer than iterator chains over four nested buffers.
#![allow(clippy::needless_range_loop)]

use super::woa::{WoaRequest, stash_seq_floats, woa_decision};
use super::woa_wyn::{FoldKernel, fold_choice};
use crate::port::gdn_woa::{scale_k4_stash_floats, stash_rows_cap, stash_seq_floats_k};

// ── Decision table ───────────────────────────────────────────────────────

const GO: WoaRequest = WoaRequest {
    requested: true,
    kk: 4,
    enabled: true,
    h_f16: false,
    kd: 128,
    vd: 128,
    kernels_linked: true,
    bound_seqs: 16,
    n: 8,
    generalized: false,
    wyn_twin_linked: false,
};

/// Row off: the decision is exactly the shipped K = 4 rule at every width.
#[test]
fn row_off_is_the_shipped_k4_rule() {
    for kk in 0..=20 {
        for linked in [false, true] {
            let r = WoaRequest {
                kk,
                wyn_twin_linked: linked,
                ..GO
            };
            assert_eq!(woa_decision(r), kk == 4, "kk={kk} linked={linked}");
        }
    }
}

/// Row on: every width 2..=16 engages once its twin is linked; K = 4 does
/// not need the K-generic twin; nothing outside 2..=16 ever engages.
#[test]
fn row_on_engages_every_linked_width() {
    let on = WoaRequest {
        generalized: true,
        ..GO
    };
    for kk in 0..=20 {
        let linked = WoaRequest {
            kk,
            wyn_twin_linked: true,
            ..on
        };
        assert_eq!(woa_decision(linked), (2..=16).contains(&kk), "kk={kk}");
        let unlinked = WoaRequest {
            kk,
            wyn_twin_linked: false,
            ..on
        };
        assert_eq!(woa_decision(unlinked), kk == 4, "kk={kk} unlinked");
    }
}

/// The K-generic arm keeps every shipped gate: request, kill switch, f16,
/// head dims, the K = 4 pair's clear kernel, the bound stash and its width.
#[test]
fn the_generic_arm_keeps_every_gate() {
    let on = WoaRequest {
        kk: 2,
        generalized: true,
        wyn_twin_linked: true,
        ..GO
    };
    assert!(woa_decision(on));
    assert!(!woa_decision(WoaRequest {
        requested: false,
        ..on
    }));
    assert!(!woa_decision(WoaRequest {
        enabled: false,
        ..on
    }));
    assert!(!woa_decision(WoaRequest { h_f16: true, ..on }));
    assert!(!woa_decision(WoaRequest { kd: 64, ..on }));
    assert!(!woa_decision(WoaRequest { vd: 64, ..on }));
    assert!(!woa_decision(WoaRequest {
        kernels_linked: false,
        ..on
    }));
    assert!(!woa_decision(WoaRequest {
        bound_seqs: 0,
        ..on
    }));
    assert!(!woa_decision(WoaRequest {
        bound_seqs: 4,
        n: 5,
        ..on
    }));
}

/// Every engaged twin is committed by the fold that reads ITS stash layout:
/// K = 4 by the wy4 fold, any other width by the K-generic fold (which is
/// linked whenever a K-generic twin is offered, `WynWoaKernels::twin_for`).
#[test]
fn every_engaged_twin_is_folded_by_its_own_kernel() {
    for generalized in [false, true] {
        for wyn_fold_linked in [false, true] {
            for kk in 0..=17 {
                // A K-generic twin is only ever offered with its fold.
                let wyn_twin_linked = wyn_fold_linked && kk != 4;
                let r = WoaRequest {
                    kk,
                    generalized,
                    wyn_twin_linked,
                    ..GO
                };
                if !woa_decision(r) {
                    continue;
                }
                let want = if kk == 4 {
                    FoldKernel::Wy4
                } else {
                    FoldKernel::Wyn
                };
                assert_eq!(fold_choice(kk, wyn_fold_linked), want, "kk={kk}");
            }
        }
    }
}

/// The model's stash covers every width its batched verify can ask for: the
/// MTP batched verify (K <= 4) on any model, and the DFlash band on a DFlash
/// model. Row off it is the shipped four-row stash.
#[test]
fn the_bound_stash_covers_every_requestable_width() {
    let (nk, nv, kd, vd) = (16, 32, 128, 128);
    let k4 = stash_seq_floats(nk, nv, kd, vd);
    assert_eq!(
        scale_k4_stash_floats(k4, stash_rows_cap(false, Some(16))),
        k4
    );
    for dflash in [None, Some(2), Some(3), Some(4), Some(8), Some(12), Some(16)] {
        let bound = scale_k4_stash_floats(k4, stash_rows_cap(true, dflash));
        let widest = dflash.unwrap_or(4).max(4);
        for k in 2..=widest {
            assert!(
                stash_seq_floats_k(k, nk, nv, kd, vd) <= bound,
                "k={k} dflash={dflash:?}"
            );
        }
    }
}

// ── CPU oracle of the record-and-fold algebra ───────────────────────────

/// Stash offsets per sequence for width `k`: the CUDA `WYNW_VN`, `WYNW_G`
/// and `WYNW_SK` macros (and the gb10 `WOA_*` macros at k = 4).
#[derive(Clone, Copy)]
struct Layout {
    k: usize,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
}

impl Layout {
    fn vn(&self, t: usize, vh: usize) -> usize {
        (t * self.nv + vh) * self.vd
    }
    fn g(&self, t: usize, vh: usize) -> usize {
        self.k * self.nv * self.vd + t * self.nv + vh
    }
    fn sk(&self, t: usize, kh: usize) -> usize {
        self.k * self.nv * self.vd + self.k * self.nv + (t * self.nk + kh) * self.kd
    }
    fn seq_floats(&self) -> usize {
        stash_seq_floats_k(self.k, self.nk, self.nv, self.kd, self.vd)
    }
}

/// One verify launch's inputs for one sequence: rows t = 0..k.
struct Inputs {
    q: Vec<Vec<f32>>,  // [k][nk*kd]
    kk: Vec<Vec<f32>>, // [k][nk*kd]
    v: Vec<Vec<f32>>,  // [k][nv*vd]
    g: Vec<Vec<f32>>,  // [k][nv], clamped
    b: Vec<Vec<f32>>,  // [k][nv]
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn inputs(l: Layout, rng: &mut Lcg) -> Inputs {
    let rows = |n: usize, s: f32, rng: &mut Lcg| -> Vec<Vec<f32>> {
        (0..l.k)
            .map(|_| (0..n).map(|_| rng.f() * s).collect())
            .collect()
    };
    let q = rows(l.nk * l.kd, 0.5, rng);
    let kk = rows(l.nk * l.kd, 0.5, rng);
    let v = rows(l.nv * l.vd, 0.5, rng);
    let g = (0..l.k)
        .map(|_| {
            (0..l.nv)
                .map(|_| (0.5 + 0.49 * rng.f()).clamp(1e-6, 1.0 - 1e-6))
                .collect()
        })
        .collect();
    let b = (0..l.k)
        .map(|_| (0..l.nv).map(|_| 0.5 + 0.5 * rng.f().abs()).collect())
        .collect();
    Inputs { q, kk, v, g, b }
}

/// The shared front half (k-dots, pass-1 dots, WY correction) for one head,
/// in the kernel's order. Returns `vn[t][vd]`.
fn wy_front(l: Layout, x: &Inputs, h: &[f32], vh: usize) -> Vec<Vec<f32>> {
    let kh = vh / (l.nv / l.nk);
    let key = |t: usize, j: usize| x.kk[t][kh * l.kd + j];
    let mut kdot = vec![vec![0f32; l.k]; l.k];
    for t in 1..l.k {
        for s in 0..t {
            kdot[t][s] = (0..l.kd).map(|j| key(t, j) * key(s, j)).sum();
        }
    }
    let sg: Vec<f32> = (0..l.k).map(|t| x.g[t][vh]).collect();
    let mut vn = vec![vec![0f32; l.vd]; l.k];
    for c in 0..l.vd {
        let mut hk = vec![0f32; l.k];
        for j in (0..l.kd).step_by(4) {
            let hh = |i: usize| h[(j + i) * l.vd + c];
            for t in 0..l.k {
                hk[t] += hh(0) * key(t, j)
                    + hh(1) * key(t, j + 1)
                    + hh(2) * key(t, j + 2)
                    + hh(3) * key(t, j + 3);
            }
        }
        let vi = |t: usize| x.v[t][vh * l.vd + c];
        let bt = |t: usize| x.b[t][vh];
        vn[0][c] = (vi(0) - sg[0] * hk[0]) * bt(0);
        for t in 1..l.k {
            let lead: f32 = (0..t).fold(1.0f32, |p, u| p * sg[u]);
            let mut corrected = lead * hk[t];
            for s in 0..t {
                let gprod: f32 = (s + 1..t).fold(1.0f32, |p, u| p * sg[u]);
                corrected += gprod * kdot[t][s] * vn[s][c];
            }
            vn[t][c] = (vi(t) - sg[t] * corrected) * bt(t);
        }
    }
    vn
}

/// Per-head buffers: `[head][kd * vd]` for a state, `[head][vd]` for an output row.
type Heads = Vec<Vec<f32>>;

/// The parent: per head, every intermediate state and the final one, plus
/// the outputs. Returns (`states[t]` = state after row t, `out[t][vh][vd]`).
fn parent(l: Layout, x: &Inputs, h0: &[Vec<f32>]) -> (Vec<Heads>, Vec<Heads>) {
    let mut states = vec![h0.to_vec(); l.k];
    let mut out = vec![vec![vec![0f32; l.vd]; l.nv]; l.k];
    for vh in 0..l.nv {
        let kh = vh / (l.nv / l.nk);
        let vn = wy_front(l, x, &h0[vh], vh);
        let scale = 1.0 / (l.kd as f32).sqrt();
        for c in 0..l.vd {
            let mut qd = vec![0f32; l.k];
            let mut hcol: Vec<f32> = (0..l.kd).map(|j| h0[vh][j * l.vd + c]).collect();
            for chunk in (0..l.kd).step_by(4) {
                for t in 0..l.k {
                    for i in 0..4 {
                        let j = chunk + i;
                        hcol[j] = x.g[t][vh] * hcol[j] + x.kk[t][kh * l.kd + j] * vn[t][c];
                        states[t][vh][j * l.vd + c] = hcol[j];
                    }
                    let qs = |i: usize| x.q[t][kh * l.kd + chunk + i];
                    qd[t] += hcol[chunk] * qs(0)
                        + hcol[chunk + 1] * qs(1)
                        + hcol[chunk + 2] * qs(2)
                        + hcol[chunk + 3] * qs(3);
                }
            }
            for t in 0..l.k {
                out[t][vh][c] = qd[t] * scale;
            }
        }
    }
    (states, out)
}

/// The twin: the parent's outputs, no state writes, and the stash (keys
/// written by the first v-head of each k-head only, as in the kernel).
fn twin(l: Layout, x: &Inputs, h0: &[Vec<f32>], stash: &mut [f32]) -> Vec<Heads> {
    let hr = l.nv / l.nk;
    for vh in 0..l.nv {
        let kh = vh / hr;
        let vn = wy_front(l, x, &h0[vh], vh);
        for t in 0..l.k {
            stash[l.g(t, vh)] = x.g[t][vh];
            stash[l.vn(t, vh)..l.vn(t, vh) + l.vd].copy_from_slice(&vn[t]);
            if vh % hr == 0 {
                let src = &x.kk[t][kh * l.kd..(kh + 1) * l.kd];
                stash[l.sk(t, kh)..l.sk(t, kh) + l.kd].copy_from_slice(src);
            }
        }
    }
    parent(l, x, h0).1
}

/// The fold of rows 0..na, from the stash alone (the kernel's update order).
fn fold(l: Layout, h: &mut [Vec<f32>], stash: &[f32], na: usize) {
    for (vh, head) in h.iter_mut().enumerate() {
        let kh = vh / (l.nv / l.nk);
        for c in 0..l.vd {
            for j in 0..l.kd {
                let mut x = head[j * l.vd + c];
                for t in 0..na {
                    x = stash[l.g(t, vh)] * x + stash[l.sk(t, kh) + j] * stash[l.vn(t, vh) + c];
                }
                head[j * l.vd + c] = x;
            }
        }
    }
}

fn bits(a: &[Vec<f32>]) -> Vec<u32> {
    a.iter().flatten().map(|x| x.to_bits()).collect()
}

/// Two sequences at every width 2..=16 (the K-generic twin's range plus the
/// gb10 pair's K = 4), every accepted count: outputs bit-equal, the twin
/// leaves the state untouched, and the fold lands on the parent's state
/// after row na-1 bit for bit. Small dims keep it fast; the algebra and the
/// layout are dimension-independent.
#[test]
fn fold_equals_the_parent_intermediate_at_every_width_and_count() {
    let mut rng = Lcg(0x5eed);
    for k in 2..=16 {
        let l = Layout {
            k,
            nk: 2,
            nv: 4,
            kd: 8,
            vd: 8,
        };
        let seqs = 2;
        let mut stash = vec![f32::NAN; seqs * l.seq_floats()];
        for s in 0..seqs {
            let x = inputs(l, &mut rng);
            let h0: Vec<Vec<f32>> = (0..l.nv)
                .map(|_| (0..l.kd * l.vd).map(|_| rng.f() * 0.1).collect())
                .collect();
            let (states, out_p) = parent(l, &x, &h0);
            let sb = &mut stash[s * l.seq_floats()..(s + 1) * l.seq_floats()];
            let out_t = twin(l, &x, &h0, sb);
            for t in 0..k {
                assert_eq!(
                    bits(&out_t[t]),
                    bits(&out_p[t]),
                    "k={k} seq={s} out row {t}"
                );
            }
            assert!(sb.iter().all(|v| !v.is_nan()), "k={k}: stash fully written");
            for na in 1..=k {
                let mut h = h0.clone();
                fold(l, &mut h, sb, na);
                assert_eq!(bits(&h), bits(&states[na - 1]), "k={k} seq={s} na={na}");
            }
        }
    }
}

/// The stash regions tile the per-sequence slab exactly: no overlap, no gap,
/// so one sequence's fold can never read another's rows or another region.
#[test]
fn stash_regions_tile_the_slab() {
    for k in 2..=16 {
        let l = Layout {
            k,
            nk: 2,
            nv: 6,
            kd: 4,
            vd: 4,
        };
        let mut hits = vec![0u8; l.seq_floats()];
        for t in 0..k {
            for vh in 0..l.nv {
                hits[l.g(t, vh)] += 1;
                for c in 0..l.vd {
                    hits[l.vn(t, vh) + c] += 1;
                }
            }
            for kh in 0..l.nk {
                for j in 0..l.kd {
                    hits[l.sk(t, kh) + j] += 1;
                }
            }
        }
        assert!(hits.iter().all(|&h| h == 1), "k={k}: {hits:?}");
    }
}

/// At K = 4 the K-generic layout is the gb10 pair's (`WOA_VN`, `WOA_G`,
/// `WOA_SK` in gated_delta_rule_wy4_woa.cu), so the wy4 fold and the
/// K-generic fold agree on what a K = 4 stash means.
#[test]
fn k4_layout_is_the_gb10_pairs() {
    let (nk, nv, kd, vd) = (16, 48, 128, 128);
    let l = Layout {
        k: 4,
        nk,
        nv,
        kd,
        vd,
    };
    assert_eq!(l.seq_floats(), stash_seq_floats(nk, nv, kd, vd));
    for t in 0..4 {
        for vh in [0, 1, 47] {
            assert_eq!(l.vn(t, vh), (t * nv + vh) * vd);
            assert_eq!(l.g(t, vh), 4 * nv * vd + t * nv + vh);
        }
        for kh in [0, 15] {
            assert_eq!(l.sk(t, kh), 4 * nv * vd + 4 * nv + (t * nk + kh) * kd);
        }
    }
}
