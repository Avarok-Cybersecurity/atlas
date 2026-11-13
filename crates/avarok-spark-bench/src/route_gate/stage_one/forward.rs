// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from rayline-ai/vllm-arc (vllm_arc/arc_heads.py, commit 4ee6a9f), Copyright 2026 Atlas Futures,
// Inc., licensed under the Apache License 2.0. Translated to Rust; see stage_one/mod.rs.
//! Responsibility: the Stage-1 forward pass in f64: the logit and probability of every candidate per ensemble member,
//! the four usage legs, the usage orderer (ratio and recalibration) and the Platt-calibrated ensemble.

use rayon::prelude::*;

use super::blocks::{ETA_CLAMP, interp_row, order_ratio_one, platt};
use super::{StageOne, StageOut};
use crate::route_gate::error::{GResult, value_err};
use crate::route_gate::mat::{Mat, dot, np_clip, np_max, np_min, sigmoid};

impl StageOne {
    /// Runs the head on readout rows `x` (`[n, d]`) with one regime name per row.
    pub fn forward(&self, x: &Mat, regimes: &[&str]) -> GResult<StageOut> {
        let dm = self.dims;
        if x.cols != dm.d {
            return value_err(format!(
                "readout rows must be [n, {}], got [{}, {}]",
                dm.d, x.rows, x.cols
            ));
        }
        if regimes.len() != x.rows {
            return value_err("one regime per row");
        }
        let idx = self.regime_index(regimes)?;
        let t = &self.t;
        let n = x.rows;
        let z: Vec<f64> = (0..n * dm.d)
            .map(|q| (x.data[q] - t.mean[q % dm.d]) * t.scale[q % dm.d])
            .collect();
        let (c, l) = (dm.c, dm.l);
        let mut p_all = vec![0.0; dm.m * n * c];
        let mut usage = vec![0.0; dm.m * n * c * l];
        for m in 0..dm.m {
            let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
                .into_par_iter()
                .map(|i| self.member_row(m, &z[i * dm.d..(i + 1) * dm.d], idx[i]))
                .collect();
            for (i, (p, u)) in rows.into_iter().enumerate() {
                p_all[(m * n + i) * c..(m * n + i + 1) * c].copy_from_slice(&p);
                usage[((m * n + i) * c) * l..((m * n + i + 1) * c) * l].copy_from_slice(&u);
            }
        }
        let usage = self.order(&z, n, usage);
        Ok(self.ensemble(n, &p_all, &usage))
    }

    /// Success probability of every candidate for each row, `[n, C]` (the ensemble point row).
    pub fn point_p(&self, x: &Mat, regimes: &[&str]) -> GResult<Mat> {
        let out = self.forward(x, regimes)?;
        Mat::new(out.n, out.c, out.p[..out.n * out.c].to_vec())
    }

    /// `point_p` with one regime for every row.
    pub fn point_p_one(&self, x: &Mat, regime: &str) -> GResult<Mat> {
        self.point_p(x, &vec![regime; x.rows])
    }

    /// One member's probabilities `[C]` and raw usage legs `[C * L]` for one standardized row.
    fn member_row(&self, m: usize, z: &[f64], regime: usize) -> (Vec<f64>, Vec<f64>) {
        let (d, c, l, h, e, r, reg) = (
            self.dims.d,
            self.dims.c,
            self.dims.l,
            self.dims.h,
            self.dims.e,
            self.dims.r,
            self.dims.reg,
        );
        let t = &self.t;
        let dif = dot(z, &t.difficulty[m * d..(m + 1) * d]);
        let vp: Vec<f64> = (0..r)
            .map(|q| dot(z, &t.v[(m * r + q) * d..(m * r + q + 1) * d]))
            .collect();
        let mut hidden = t.c1[m * h..(m + 1) * h].to_vec();
        for (k, &zk) in z.iter().enumerate() {
            let w = &t.w1[(m * d + k) * h..(m * d + k + 1) * h];
            for (a, b) in hidden.iter_mut().zip(w) {
                *a += zk * b;
            }
        }
        for a in &mut hidden {
            *a = np_max(*a, 0.0);
        }
        let mut h2 = t.c2[m * e..(m + 1) * e].to_vec();
        for (q, &hq) in hidden.iter().enumerate() {
            let w = &t.w2[(m * h + q) * e..(m * h + q + 1) * e];
            for (a, b) in h2.iter_mut().zip(w) {
                *a += hq * b;
            }
        }
        let mut p = vec![0.0; c];
        let mut u = vec![0.0; c * l];
        for ci in 0..c {
            let mut eta = t.logit_bias[m * c + ci] + t.regime[m * reg + regime] - dif;
            let ur = &t.u[(m * c + ci) * r..(m * c + ci + 1) * r];
            eta += vp.iter().zip(ur).map(|(a, b)| a * b).sum::<f64>();
            eta += dot(&h2, &t.embed[(m * c + ci) * e..(m * c + ci + 1) * e]);
            p[ci] = if t.p_ok[m * c + ci] {
                sigmoid(np_clip(eta, -ETA_CLAMP, ETA_CLAMP))
            } else {
                f64::NAN
            };
            for k in 0..l {
                let q = (m * c + ci) * l + k;
                let exponent = dot(z, &t.w_usage[q * d..(q + 1) * d]) + t.b_usage[q];
                let mean = exponent.exp() * t.smear[q] - 1.0;
                let clamped = np_min(np_max(mean, t.low[q]), t.high[q]);
                u[ci * l + k] = if t.usage_ok[q] { clamped } else { f64::NAN };
            }
        }
        (p, u)
    }

    /// The usage orderer: a net-token estimate per `(leg, candidate)`, the pool-quantile ratio, then the recalibration.
    fn order(&self, z: &[f64], n: usize, usage: Vec<f64>) -> Vec<f64> {
        let dm = self.dims;
        let (d, c, l, dd, ho, p, k) = (dm.d, dm.c, dm.l, dm.dd, dm.ho, dm.p, dm.k);
        let t = &self.t;
        // net[(i * L + leg) * C + cand]
        let net_rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let zi = &z[i * d..(i + 1) * d];
                let projected: Vec<f64> = (0..dd)
                    .map(|q| {
                        (0..d)
                            .map(|a| (zi[a] - t.ord_mean[a]) * t.ord_axes[q * d + a])
                            .sum::<f64>()
                    })
                    .collect();
                let mut net = vec![f64::NAN; l * c];
                for leg in 0..l {
                    let x: Vec<f64> = (0..dd)
                        .map(|q| {
                            (projected[q] - t.ord_center[leg * dd + q]) / t.ord_spread[leg * dd + q]
                        })
                        .collect();
                    let mut base = t.ord_b1[leg * ho..(leg + 1) * ho].to_vec();
                    for (q, &xq) in x.iter().enumerate() {
                        let w = &t.ord_w1[(leg * dd + q) * ho..(leg * dd + q + 1) * ho];
                        for (a, b) in base.iter_mut().zip(w) {
                            *a += xq * b;
                        }
                    }
                    for ci in 0..c {
                        if !t.ord_seen[leg * c + ci] {
                            continue;
                        }
                        let e1 = &t.ord_e1[(leg * c + ci) * ho..(leg * c + ci + 1) * ho];
                        let mut eta = 0.0;
                        for q in 0..ho {
                            let hid = np_max(base[q] + e1[q], 0.0);
                            eta += hid * t.ord_w2[leg * ho + q];
                        }
                        eta += t.ord_level[leg * c + ci] + t.ord_b[leg];
                        net[leg * c + ci] = eta.exp();
                    }
                }
                net
            })
            .collect();
        let alpha = |ci: usize, leg: usize| {
            if t.ord_alpha.len() == 1 {
                t.ord_alpha[0]
            } else {
                t.ord_alpha[ci * l + leg]
            }
        };
        let mut out = usage;
        out.par_chunks_mut(c * l)
            .enumerate()
            .for_each(|(row, chunk)| {
                let i = row % n;
                for ci in 0..c {
                    for leg in 0..l {
                        let cl = ci * l + leg;
                        let glm = chunk[cl];
                        let pool_n = t.ord_pool_n[cl];
                        let ratio = order_ratio_one(
                            glm,
                            net_rows[i][leg * c + ci],
                            &t.ord_pool_glm[cl * p..(cl + 1) * p],
                            &t.ord_pool_net[cl * p..(cl + 1) * p],
                            pool_n,
                            alpha(ci, leg),
                        );
                        let v = glm * ratio;
                        let ok = v.is_finite() && v > 0.0 && t.ord_recal_n[cl] > 0;
                        chunk[cl] = if ok {
                            let lp = v.ln();
                            let shift = interp_row(
                                lp,
                                &t.ord_recal_x[cl * k..(cl + 1) * k],
                                &t.ord_recal_c[cl * k..(cl + 1) * k],
                                t.ord_recal_n[cl],
                            );
                            (lp + shift).exp()
                        } else {
                            v
                        };
                    }
                }
            });
        out
    }

    /// Platt per point member, the mean over the finite members (row 0), then the Platt-mapped draws.
    fn ensemble(&self, n: usize, p: &[f64], usage: &[f64]) -> StageOut {
        let dm = self.dims;
        let (c, l) = (dm.c, dm.l);
        let t = &self.t;
        let rows = 1 + dm.draws;
        let mut p_out = vec![0.0; rows * n * c];
        let mut u_out = vec![0.0; rows * n * c * l];
        for i in 0..n {
            for ci in 0..c {
                let (mut total, mut count) = (0.0, 0usize);
                for (s, &m) in t.ens_point.iter().enumerate() {
                    let v = platt(
                        p[(m * n + i) * c + ci],
                        t.cal_platt[s * c + ci],
                        t.cal_a[s * c + ci],
                        t.cal_b[s * c + ci],
                    );
                    if v.is_finite() {
                        total += v;
                        count += 1;
                    }
                }
                p_out[i * c + ci] = if count > 0 {
                    total / count as f64
                } else {
                    f64::NAN
                };
                for k in 0..l {
                    let sum: f64 = t
                        .ens_point
                        .iter()
                        .map(|&m| usage[((m * n + i) * c + ci) * l + k])
                        .sum();
                    u_out[(i * c + ci) * l + k] = sum / t.ens_point.len() as f64;
                }
            }
        }
        for (dr, (&m, &owner)) in t.ens_draws.iter().zip(&t.ens_owner).enumerate() {
            let row = 1 + dr;
            for i in 0..n {
                for ci in 0..c {
                    let v = platt(
                        p[(m * n + i) * c + ci],
                        t.cal_platt[owner * c + ci],
                        t.cal_a[owner * c + ci],
                        t.cal_b[owner * c + ci],
                    );
                    p_out[(row * n + i) * c + ci] = v;
                    let src = ((m * n + i) * c + ci) * l;
                    let dst = ((row * n + i) * c + ci) * l;
                    u_out[dst..dst + l].copy_from_slice(&usage[src..src + l]);
                }
            }
        }
        StageOut {
            rows,
            n,
            c,
            l,
            p: p_out,
            usage: u_out,
        }
    }
}
