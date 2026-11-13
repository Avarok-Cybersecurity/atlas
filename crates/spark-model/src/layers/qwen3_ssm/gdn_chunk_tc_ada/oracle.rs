// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the FP64 reference for the gated delta rule that the Ada
//! tensor-core chunked prefill (`gdn_chunk_tc_ada.cu`) is qualified against,
//! plus the suite-owned acceptance criteria and a deterministic input
//! generator. Self-contained (no `crate::` paths) because the GPU microtest
//! `examples/gdn_chunk_tc_ada_microtest.rs` includes this file by `#[path]`.
//!
//! Two independent FP64 evaluations of the same contract:
//!   * [`recurrent`]: token by token, the scalar spine's formula
//!     (`gated_delta_rule_prefill`): `v' = (v - g S^T k) beta`,
//!     `S = g S + k v'^T`, `o = S^T q / sqrt(k_dim)`;
//!   * [`chunked`]: the WY chunk form the kernel computes (log-space gate
//!     cumsum with Atlas's `1e-30` floor, `(I + L) U = beta V`,
//!     `(I + L) W = beta exp(gc) K`, `v_new = U - W S_c`, the state update and
//!     the output), with a partial last chunk.
//!
//! Their agreement to rounding proves the chunk algebra; the kernel is then
//! held to [`OUTPUT`] and [`STATE`] against [`recurrent`].
//!
//! Layouts (row major): `q`, `k` `[t][nk][kd]`; `v` `[t][nv][vd]`; `gate`,
//! `beta` `[t][nv]`; state `[nv][kd][vd]`; output `[t][nv][vd]`. Value head
//! `vh` reads key head `vh / (nv / nk)`, as every Atlas GDN kernel does.

// Index loops are the reference formula written out; iterator chains over four
// strided tensors would hide it.
#![allow(clippy::needless_range_loop)]

/// Atlas's FLA gate floor before the log (`GATE_FLOOR` in gated_delta_rule_fla.cu).
pub const GATE_FLOOR: f64 = 1e-30;

/// One prefill problem. Inputs are the values the kernel sees: q/k/v already
/// rounded to BF16, gate/beta and the initial state in FP32.
#[derive(Clone, Debug)]
pub struct Problem {
    pub t: usize,
    pub nk: usize,
    pub nv: usize,
    pub kd: usize,
    pub vd: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
    pub h0: Vec<f32>,
}

impl Problem {
    fn kh(&self, vh: usize) -> usize {
        vh / (self.nv / self.nk)
    }
    fn qv(&self, t: usize, vh: usize) -> &[f32] {
        let o = (t * self.nk + self.kh(vh)) * self.kd;
        &self.q[o..o + self.kd]
    }
    fn kv(&self, t: usize, vh: usize) -> &[f32] {
        let o = (t * self.nk + self.kh(vh)) * self.kd;
        &self.k[o..o + self.kd]
    }
    fn vv(&self, t: usize, vh: usize) -> &[f32] {
        let o = (t * self.nv + vh) * self.vd;
        &self.v[o..o + self.vd]
    }
    fn g(&self, t: usize, vh: usize) -> f64 {
        self.gate[t * self.nv + vh] as f64
    }
    fn b(&self, t: usize, vh: usize) -> f64 {
        self.beta[t * self.nv + vh] as f64
    }
}

/// Output `[t][nv][vd]` and final state `[nv][kd][vd]`, both FP64.
#[derive(Clone, Debug)]
pub struct Reference {
    pub out: Vec<f64>,
    pub state: Vec<f64>,
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// Token-sequential FP64 recurrence (the scalar spine's formula).
pub fn recurrent(p: &Problem) -> Reference {
    let (kd, vd) = (p.kd, p.vd);
    let inv = 1.0 / (kd as f64).sqrt();
    let mut out = vec![0f64; p.t * p.nv * vd];
    let mut state: Vec<f64> = p.h0.iter().map(|x| *x as f64).collect();
    for vh in 0..p.nv {
        let s = &mut state[vh * kd * vd..(vh + 1) * kd * vd];
        for t in 0..p.t {
            let (q, k, v) = (p.qv(t, vh), p.kv(t, vh), p.vv(t, vh));
            let (g, b) = (p.g(t, vh), p.b(t, vh));
            for c in 0..vd {
                let hk: f64 = (0..kd).map(|r| s[r * vd + c] * k[r] as f64).sum();
                let vn = (v[c] as f64 - g * hk) * b;
                let mut qd = 0f64;
                for r in 0..kd {
                    let h = g * s[r * vd + c] + k[r] as f64 * vn;
                    s[r * vd + c] = h;
                    qd += h * q[r] as f64;
                }
                out[(t * p.nv + vh) * vd + c] = qd * inv;
            }
        }
    }
    Reference { out, state }
}

/// The WY chunk form in FP64, `chunk` tokens per chunk, last chunk partial.
pub fn chunked(p: &Problem, chunk: usize) -> Reference {
    let (kd, vd) = (p.kd, p.vd);
    let inv = 1.0 / (kd as f64).sqrt();
    let mut out = vec![0f64; p.t * p.nv * vd];
    let mut state: Vec<f64> = p.h0.iter().map(|x| *x as f64).collect();
    for vh in 0..p.nv {
        let s = &mut state[vh * kd * vd..(vh + 1) * kd * vd];
        let mut cs = 0;
        while cs < p.t {
            let ce = (p.t - cs).min(chunk);
            let mut gc = vec![0f64; ce];
            let mut acc = 0f64;
            for (i, g) in gc.iter_mut().enumerate() {
                acc += p.g(cs + i, vh).max(GATE_FLOOR).ln();
                *g = acc;
            }
            let ks: Vec<&[f32]> = (0..ce).map(|i| p.kv(cs + i, vh)).collect();
            // L[i][l] = beta_i exp(gc_i - gc_l) <k_l, k_i>, l < i.
            let mut l_mat = vec![0f64; ce * ce];
            for i in 0..ce {
                for l in 0..i {
                    l_mat[i * ce + l] = p.b(cs + i, vh) * (gc[i] - gc[l]).exp() * dot(ks[l], ks[i]);
                }
            }
            // Forward substitution: (I + L) U = beta V, (I + L) W = beta exp(gc) K.
            let mut u = vec![0f64; ce * vd];
            let mut w = vec![0f64; ce * kd];
            for i in 0..ce {
                let bi = p.b(cs + i, vh);
                let vi = p.vv(cs + i, vh);
                for c in 0..vd {
                    let mut x = bi * vi[c] as f64;
                    for l in 0..i {
                        x -= l_mat[i * ce + l] * u[l * vd + c];
                    }
                    u[i * vd + c] = x;
                }
                let e = bi * gc[i].exp();
                for r in 0..kd {
                    let mut x = e * ks[i][r] as f64;
                    for l in 0..i {
                        x -= l_mat[i * ce + l] * w[l * kd + r];
                    }
                    w[i * kd + r] = x;
                }
            }
            // v_new = U - W S_c (entry state).
            let mut vn = vec![0f64; ce * vd];
            for i in 0..ce {
                for c in 0..vd {
                    let ws: f64 = (0..kd).map(|r| w[i * kd + r] * s[r * vd + c]).sum();
                    vn[i * vd + c] = u[i * vd + c] - ws;
                }
            }
            // O_i = (exp(gc_i) q_i^T S_c + sum_{l<=i} exp(gc_i - gc_l) <q_i, k_l> v_new_l) / sqrt(kd).
            for i in 0..ce {
                let qi = p.qv(cs + i, vh);
                let qk: Vec<f64> = (0..=i)
                    .map(|l| (gc[i] - gc[l]).exp() * dot(qi, ks[l]))
                    .collect();
                let eg = gc[i].exp();
                for c in 0..vd {
                    let qs: f64 = (0..kd).map(|r| qi[r] as f64 * s[r * vd + c]).sum();
                    let av: f64 = (0..=i).map(|l| qk[l] * vn[l * vd + c]).sum();
                    out[((cs + i) * p.nv + vh) * vd + c] = (eg * qs + av) * inv;
                }
            }
            // S_{c+1} = exp(gc_last) S_c + sum_i exp(gc_last - gc_i) k_i v_new_i^T.
            let gl = gc[ce - 1];
            let gamma = gl.exp();
            for x in s.iter_mut() {
                *x *= gamma;
            }
            for i in 0..ce {
                let d = (gl - gc[i]).exp();
                for r in 0..kd {
                    let kr = ks[i][r] as f64 * d;
                    for c in 0..vd {
                        s[r * vd + c] += kr * vn[i * vd + c];
                    }
                }
            }
            cs += ce;
        }
    }
    Reference { out, state }
}

/// A suite-owned acceptance criterion (Cinference op-development.md 6.3): a
/// normwise relative bound with a finite gross pointwise cap, so cancellation
/// does not demand elementwise agreement and isolated corruption cannot hide in
/// the norm. Non-finite policy: any NaN or infinity fails.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Criterion {
    /// `||got - ref||_2 / ||ref||_2` must not exceed this.
    pub normwise: f64,
    /// `max |got - ref|` must not exceed `pointwise_cap * max |ref|`.
    pub pointwise_cap: f64,
}

/// BF16 output. The arithmetic profile: BF16 W, U, v_new and S_c
/// intermediates, TF32 products with FP32 accumulation, BF16 output rounding
/// (Atlas's FLA route rounds the same intermediates to BF16).
pub const OUTPUT: Criterion = Criterion {
    normwise: 2.0e-2,
    pointwise_cap: 0.25,
};

/// FP32 final state: same profile without the output rounding.
pub const STATE: Criterion = Criterion {
    normwise: 2.0e-2,
    pointwise_cap: 0.25,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Errors {
    pub normwise: f64,
    pub max_abs: f64,
    pub ref_max: f64,
    pub finite: bool,
}

/// Errors of `got` against `reference` (same length).
pub fn compare(got: &[f64], reference: &[f64]) -> Errors {
    assert_eq!(got.len(), reference.len(), "compare: length mismatch");
    let (mut num, mut den, mut max_abs, mut ref_max) = (0f64, 0f64, 0f64, 0f64);
    let mut finite = true;
    for (g, r) in got.iter().zip(reference) {
        if !g.is_finite() || !r.is_finite() {
            finite = false;
            continue;
        }
        let d = g - r;
        num += d * d;
        den += r * r;
        max_abs = max_abs.max(d.abs());
        ref_max = ref_max.max(r.abs());
    }
    let normwise = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    Errors {
        normwise,
        max_abs,
        ref_max,
        finite,
    }
}

impl Criterion {
    pub fn accepts(&self, e: &Errors) -> bool {
        e.finite
            && e.normwise <= self.normwise
            && e.max_abs <= self.pointwise_cap * e.ref_max.max(f64::MIN_POSITIVE)
    }
}

/// Round to the nearest BF16 (ties to even), returned as f32.
pub fn bf16_round(x: f32) -> f32 {
    f32::from_bits((bf16_bits(x) as u32) << 16)
}

/// BF16 bit pattern of `x`, round to nearest even.
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) as u16
}

/// SplitMix64: deterministic inputs without a dependency.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

/// Inputs shaped like a Qwen3.5 GDN layer's: q and k L2-normalized per head
/// (the model's `use_qk_l2norm`), v in [-1, 1), gate (linear decay) in
/// [0.85, 0.999), beta in [0.05, 0.95), a small initial state. q/k/v are
/// rounded to BF16, the values the kernel reads.
pub fn generate(seed: u64, t: usize, nk: usize, nv: usize, kd: usize, vd: usize) -> Problem {
    assert!(
        nk > 0 && nv.is_multiple_of(nk),
        "nv must be a multiple of nk"
    );
    let mut rng = Rng(seed);
    let unit_rows = |rows: usize, dim: usize, rng: &mut Rng| -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let row: Vec<f32> = (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            out.extend(row.iter().map(|x| bf16_round(x / n)));
        }
        out
    };
    let q = unit_rows(t * nk, kd, &mut rng);
    let k = unit_rows(t * nk, kd, &mut rng);
    let v: Vec<f32> = (0..t * nv * vd)
        .map(|_| bf16_round(rng.uniform(-1.0, 1.0)))
        .collect();
    let gate: Vec<f32> = (0..t * nv).map(|_| rng.uniform(0.85, 0.999)).collect();
    let beta: Vec<f32> = (0..t * nv).map(|_| rng.uniform(0.05, 0.95)).collect();
    let h0: Vec<f32> = (0..nv * kd * vd)
        .map(|_| rng.uniform(-0.05, 0.05))
        .collect();
    Problem {
        t,
        nk,
        nv,
        kd,
        vd,
        q,
        k,
        v,
        gate,
        beta,
        h0,
    }
}
