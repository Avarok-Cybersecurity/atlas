// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of the Stage-1 head formulas and orderer building blocks (port of the Python
//! `StageOneTests` that need no forward shape, usage or loader checks); also hosts the toy-release fixture that
//! `tests_stage_one_head` reuses. Oracles are hand-written scalar loops and naive NumPy-equivalent routines.

use std::collections::BTreeMap;

use super::error::GateError;
use super::mat::{Mat, dot};
use super::mt::NumpyRandom;
use super::stage_one::{
    MIN_POOL, PLATT_EPS, Side, StageOne, batched_searchsorted, interp_row, linear_quantile_row,
    order_ratio_one, platt, search_sorted,
};
use super::tensorio::Tensor;

pub(super) const M: usize = 3;
pub(super) const C: usize = 4;
pub(super) const D: usize = 6;
pub(super) const L: usize = 4;
pub(super) const H: usize = 5;
pub(super) const E: usize = 3;
/// Number of regimes (the Python `R`).
pub(super) const R: usize = 2;
/// Rank of the `v`/`u` term (the middle axis of `v[M, 1, D]`).
pub(super) const RK: usize = 1;
pub(super) const S: usize = 3;
pub(super) const DD: usize = 3;
pub(super) const HO: usize = 4;
pub(super) const P: usize = 30;
pub(super) const K: usize = 5;

pub(super) const REGIMES: [&str; 2] = ["r0", "r1"];

fn put(t: &mut BTreeMap<String, Tensor>, name: &str, shape: &[usize], data: Vec<f64>) {
    t.insert(name.to_string(), Tensor::new(shape.to_vec(), data));
}

fn rn(rng: &mut NumpyRandom, n: usize, s: f64) -> Vec<f64> {
    rng.randn_vec(n).into_iter().map(|v| v * s).collect()
}

/// Random tensors with the exact names and rank structure of the shipped release, at toy sizes.
pub(super) fn tiny_stage_one(seed: u64, mlp: bool) -> BTreeMap<String, Tensor> {
    let mut rng = NumpyRandom::new(seed);
    let mut t: BTreeMap<String, Tensor> = BTreeMap::new();
    let ms = if mlp { 1.0 } else { 0.0 };
    let pools = |rng: &mut NumpyRandom| -> Vec<f64> {
        let mut out = Vec::with_capacity(C * L * P);
        for _ in 0..C * L {
            let mut row: Vec<f64> = rng
                .rand_vec(P)
                .into_iter()
                .map(|v| v * 10.0 + 1.0)
                .collect();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.extend(row);
        }
        out
    };
    let pool_glm = pools(&mut rng);
    let pool_net = pools(&mut rng);
    put(&mut t, "mean", &[D], rn(&mut rng, D, 0.1));
    let scale: Vec<f64> = rng
        .randn_vec(D)
        .into_iter()
        .map(|v| v.abs() + 0.5)
        .collect();
    put(&mut t, "scale", &[D], scale);
    put(&mut t, "difficulty", &[M, D], rn(&mut rng, M * D, 0.3));
    put(&mut t, "logit_bias", &[M, C], rn(&mut rng, M * C, 1.0));
    put(&mut t, "regime", &[M, R], rn(&mut rng, M * R, 1.0));
    put(&mut t, "v", &[M, RK, D], rn(&mut rng, M * RK * D, 0.2));
    put(&mut t, "u", &[M, C, RK], rn(&mut rng, M * C * RK, 0.2));
    put(&mut t, "w1", &[M, D, H], rn(&mut rng, M * D * H, 0.3 * ms));
    put(&mut t, "c1", &[M, H], rn(&mut rng, M * H, 0.1 * ms));
    put(&mut t, "w2", &[M, H, E], rn(&mut rng, M * H * E, 0.3 * ms));
    put(&mut t, "c2", &[M, E], rn(&mut rng, M * E, 0.1 * ms));
    put(&mut t, "embed", &[M, C, E], rn(&mut rng, M * C * E, 0.3));
    put(
        &mut t,
        "w_usage",
        &[M, C, L, D],
        rn(&mut rng, M * C * L * D, 0.1),
    );
    put(&mut t, "b_usage", &[M, C, L], rn(&mut rng, M * C * L, 0.1));
    put(&mut t, "smear", &[M, C, L], vec![1.0; M * C * L]);
    put(&mut t, "low", &[M, C, L], vec![0.0; M * C * L]);
    put(&mut t, "high", &[M, C, L], vec![1e6; M * C * L]);
    put(&mut t, "usage_ok", &[M, C, L], vec![1.0; M * C * L]);
    put(&mut t, "p_ok", &[M, C], vec![1.0; M * C]);
    put(&mut t, "ord_mean", &[D], rn(&mut rng, D, 0.1));
    put(&mut t, "ord_axes", &[DD, D], rn(&mut rng, DD * D, 0.3));
    put(&mut t, "ord_center", &[L, DD], rn(&mut rng, L * DD, 0.1));
    put(&mut t, "ord_spread", &[L, DD], vec![1.0; L * DD]);
    put(
        &mut t,
        "ord_w1",
        &[L, DD, HO],
        rn(&mut rng, L * DD * HO, 0.3),
    );
    put(&mut t, "ord_e1", &[L, C, HO], rn(&mut rng, L * C * HO, 0.1));
    put(&mut t, "ord_b1", &[L, HO], rn(&mut rng, L * HO, 0.1));
    put(&mut t, "ord_w2", &[L, HO], rn(&mut rng, L * HO, 0.3));
    let level: Vec<f64> = rng
        .randn_vec(L * C)
        .into_iter()
        .map(|v| v * 0.1 + 1.0)
        .collect();
    put(&mut t, "ord_level", &[L, C], level);
    put(&mut t, "ord_b", &[L], rn(&mut rng, L, 0.1));
    put(&mut t, "ord_seen", &[L, C], vec![1.0; L * C]);
    put(&mut t, "ord_pool_glm", &[C, L, P], pool_glm);
    put(&mut t, "ord_pool_net", &[C, L, P], pool_net);
    put(&mut t, "ord_pool_n", &[C, L], vec![P as f64; C * L]);
    let knots = [-1.0, 0.0, 1.0, 2.0, 3.0];
    put(
        &mut t,
        "ord_recal_x",
        &[C, L, K],
        (0..C * L).flat_map(|_| knots).collect(),
    );
    put(
        &mut t,
        "ord_recal_c",
        &[C, L, K],
        rn(&mut rng, C * L * K, 0.05),
    );
    put(&mut t, "ord_recal_n", &[C, L], vec![K as f64; C * L]);
    put(&mut t, "ord_alpha", &[1], vec![0.25]);
    put(&mut t, "cal_a", &[S, C], vec![1.0; S * C]);
    put(&mut t, "cal_b", &[S, C], vec![0.0; S * C]);
    put(&mut t, "cal_platt", &[S, C], vec![0.0; S * C]);
    put(
        &mut t,
        "ens_point",
        &[S],
        (0..S).map(|i| i as f64).collect(),
    );
    put(&mut t, "ens_draws", &[0], vec![]);
    put(&mut t, "ens_owner", &[0], vec![]);
    put(&mut t, "auxiliary", &[C], vec![0.0; C]);
    t
}

pub(super) fn regime_names() -> Vec<String> {
    REGIMES.iter().map(|s| s.to_string()).collect()
}

/// Builds the head from a tensor map (cloned) with the toy regime names.
pub(super) fn head(t: &BTreeMap<String, Tensor>) -> StageOne {
    StageOne::new(t.clone(), None, Some(regime_names())).unwrap()
}

pub(super) fn at<'a>(t: &'a BTreeMap<String, Tensor>, name: &str) -> &'a [f64] {
    &t[name].data
}

/// Replaces tensor `name` with zeros of the same shape.
pub(super) fn zero(t: &mut BTreeMap<String, Tensor>, name: &str) {
    let shape = t[name].shape.clone();
    let n: usize = shape.iter().product();
    t.insert(name.to_string(), Tensor::new(shape, vec![0.0; n]));
}

pub(super) fn randn_mat(seed: u64, n: usize) -> Mat {
    Mat::new(n, D, NumpyRandom::new(seed).randn_vec(n * D)).unwrap()
}

/// `assert_allclose`: `|got - want| <= atol + rtol |want|`.
pub(super) fn assert_close(got: f64, want: f64, rtol: f64, atol: f64) {
    assert!(
        (got - want).abs() <= atol + rtol * want.abs(),
        "got {got:e}, want {want:e} (rtol {rtol:e}, atol {atol:e})"
    );
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn standardize(t: &BTreeMap<String, Tensor>, x: &Mat, n: usize) -> Vec<f64> {
    let (mean, scale) = (at(t, "mean"), at(t, "scale"));
    (0..D).map(|k| (x.at(n, k) - mean[k]) * scale[k]).collect()
}

fn regime_pos(name: &str) -> usize {
    REGIMES.iter().position(|r| *r == name).unwrap()
}

/// The MLP embedding `h[E]` of member `m` for standardized row `z`.
fn mlp_h(t: &BTreeMap<String, Tensor>, m: usize, z: &[f64]) -> Vec<f64> {
    let (w1, c1, w2, c2) = (at(t, "w1"), at(t, "c1"), at(t, "w2"), at(t, "c2"));
    let hidden: Vec<f64> = (0..H)
        .map(|j| {
            let s: f64 =
                (0..D).map(|k| z[k] * w1[(m * D + k) * H + j]).sum::<f64>() + c1[m * H + j];
            s.max(0.0)
        })
        .collect();
    (0..E)
        .map(|e| {
            (0..H)
                .map(|j| hidden[j] * w2[(m * H + j) * E + e])
                .sum::<f64>()
                + c2[m * E + e]
        })
        .collect()
}

/// NumPy `quantile(method="linear")` of an ascending slice.
fn np_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virt = (n as f64 - 1.0) * q;
    let lo = virt.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let g = virt - virt.floor();
    sorted[lo] + (sorted[hi] - sorted[lo]) * g
}

/// NumPy `interp` with constant tails, written as a naive scan.
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = (0..n - 1).find(|&j| x >= xp[j] && x < xp[j + 1]).unwrap();
    fp[j] + (fp[j + 1] - fp[j]) * (x - xp[j]) / (xp[j + 1] - xp[j])
}

#[test]
fn formula_without_mlp_matches_scalar_loops() {
    let mut t = tiny_stage_one(1, true);
    for k in ["w1", "c1", "w2", "c2", "v"] {
        zero(&mut t, k);
    }
    let h = head(&t);
    let x = randn_mat(5, 3);
    let regs = ["r0", "r1", "r0"];
    let got = h.point_p(&x, &regs).unwrap();
    let (dif, lb, rg) = (at(&t, "difficulty"), at(&t, "logit_bias"), at(&t, "regime"));
    for n in 0..3 {
        let z = standardize(&t, &x, n);
        for c in 0..C {
            let mut sum = 0.0;
            for m in 0..M {
                let eta = lb[m * C + c] + rg[m * R + regime_pos(regs[n])]
                    - dot(&z, &dif[m * D..(m + 1) * D]);
                sum += sig(eta.clamp(-40.0, 40.0));
            }
            assert_close(got.at(n, c), sum / M as f64, 0.0, 5e-13);
        }
    }
}

#[test]
fn mlp_embed_term() {
    let mut t = tiny_stage_one(2, true);
    zero(&mut t, "v");
    let h = head(&t);
    let x = randn_mat(6, 2);
    let got = h.point_p_one(&x, "r1").unwrap();
    let (dif, lb, rg, embed) = (
        at(&t, "difficulty"),
        at(&t, "logit_bias"),
        at(&t, "regime"),
        at(&t, "embed"),
    );
    for n in 0..2 {
        let z = standardize(&t, &x, n);
        for c in 0..C {
            let mut sum = 0.0;
            for m in 0..M {
                let hh = mlp_h(&t, m, &z);
                let eta = lb[m * C + c] + rg[m * R + 1] - dot(&z, &dif[m * D..(m + 1) * D])
                    + dot(&hh, &embed[(m * C + c) * E..(m * C + c + 1) * E]);
                sum += sig(eta.clamp(-40.0, 40.0));
            }
            assert_close(got.at(n, c), sum / M as f64, 0.0, 5e-13);
        }
    }
}

#[test]
fn v_u_term() {
    let t = tiny_stage_one(3, false);
    let h = head(&t);
    let x = randn_mat(7, 1);
    let z = standardize(&t, &x, 0);
    let got = h.point_p_one(&x, "r0").unwrap();
    let (dif, lb, rg, v, u) = (
        at(&t, "difficulty"),
        at(&t, "logit_bias"),
        at(&t, "regime"),
        at(&t, "v"),
        at(&t, "u"),
    );
    for c in 0..C {
        let mut sum = 0.0;
        for m in 0..M {
            let eta = lb[m * C + c] + rg[m * R] - dot(&z, &dif[m * D..(m + 1) * D])
                + dot(&z, &v[m * RK * D..m * RK * D + D]) * u[m * C * RK + c * RK];
            sum += sig(eta);
        }
        assert_close(got.at(0, c), sum / M as f64, 0.0, 5e-13);
    }
}

#[test]
fn eta_is_clamped_to_plus_minus_40() {
    let mut t = tiny_stage_one(4, false);
    for k in ["v", "difficulty", "regime"] {
        zero(&mut t, k);
    }
    let mut lb = vec![0.0; M * C];
    for m in 0..M {
        lb[m * C] = 1000.0;
        lb[m * C + 1] = -1000.0;
    }
    t.insert("logit_bias".to_string(), Tensor::new(vec![M, C], lb));
    let h = head(&t);
    let p = h.point_p_one(&Mat::zeros(1, D), "r0").unwrap();
    assert!((p.at(0, 0) - sig(40.0)).abs() < 5e-16);
    assert!((p.at(0, 1) - sig(-40.0)).abs() < 5e-31);
    assert!(p.at(0, 1) > 0.0);
}

#[test]
fn p_ok_masks_members() {
    let mut t = tiny_stage_one(5, false);
    let mut p_ok = vec![1.0; M * C];
    p_ok[C + 2] = 0.0; // member 1 does not score candidate 2
    for m in 0..M {
        p_ok[m * C + 3] = 0.0; // nobody scores candidate 3
    }
    t.insert("p_ok".to_string(), Tensor::new(vec![M, C], p_ok));
    let h = head(&t);
    let x = randn_mat(8, 2);
    let out = h.forward(&x, &["r0", "r0"]).unwrap();
    for i in 0..2 {
        assert!(out.p[i * C + 3].is_nan());
    }
    // candidate 2 is the mean of members 0 and 2 only: recompute each member alone with the mask all ones
    let mut per_member: Vec<Mat> = Vec::new();
    for m in 0..M {
        let mut tm = t.clone();
        tm.insert(
            "p_ok".to_string(),
            Tensor::new(vec![M, C], vec![1.0; M * C]),
        );
        tm.insert(
            "ens_point".to_string(),
            Tensor::new(vec![S], vec![m as f64; S]),
        );
        per_member.push(head(&tm).point_p_one(&x, "r0").unwrap());
    }
    for i in 0..2 {
        let want = (per_member[0].at(i, 2) + per_member[2].at(i, 2)) / 2.0;
        assert_close(out.p[i * C + 2], want, 0.0, 1e-12);
    }
}

#[test]
fn platt_formula() {
    let (a, b) = (2.0, 0.5);
    let p = [0.7, 0.0, 1.0, f64::NAN];
    for &pv in &p[..3] {
        let got = platt(pv, true, a, b);
        let q = pv.clamp(PLATT_EPS, 1.0 - PLATT_EPS);
        let want = 1.0 / (1.0 + (-(a * (q / (1.0 - q)).ln() + b)).exp());
        assert_close(got, want, 1e-12, 0.0);
    }
    assert!(platt(p[3], true, a, b).is_nan());
    for &pv in &p[..3] {
        assert_eq!(platt(pv, false, a, b), pv);
    }
}

#[test]
fn linear_quantile_matches_numpy_definition() {
    let mut rng = NumpyRandom::new(0);
    let n = [[25usize, 40], [3, 1]];
    for row in n {
        for len in row {
            let mut v: Vec<f64> = rng.rand_vec(len).into_iter().map(|x| x * 100.0).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut pool = v.clone();
            pool.resize(40, f64::INFINITY);
            let mut q = rng.rand_vec(7);
            q[0] = 0.0;
            q[1] = 1.0;
            for &qq in &q {
                assert_close(
                    linear_quantile_row(&pool, len, qq),
                    np_quantile(&v, qq),
                    1e-12,
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn interp_matches_numpy_definition() {
    let mut rng = NumpyRandom::new(1);
    let n = [[5usize, 3], [1, 2]];
    for row in n {
        for k in row {
            let mut xs: Vec<f64> = rng.rand_vec(k).into_iter().map(|v| v * 4.0 - 2.0).collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, v) in xs.iter_mut().enumerate() {
                *v += i as f64 * 1e-3;
            }
            let f = rng.randn_vec(k);
            let mut xp = xs.clone();
            xp.resize(5, xs[k - 1]);
            let mut fp = f.clone();
            fp.resize(5, f[k - 1]);
            for xv in rng.rand_vec(9).into_iter().map(|v| v * 6.0 - 3.0) {
                assert_close(
                    interp_row(xv, &xp, &fp, k),
                    np_interp(xv, &xs, &f),
                    1e-12,
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn batched_searchsorted_matches_counts() {
    let mut rng = NumpyRandom::new(2);
    let (lead, p, k) = (6usize, 10usize, 4usize);
    let mut rows = Vec::new();
    for _ in 0..lead {
        let mut r = rng.rand_vec(p);
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rows.extend(r);
    }
    let vals = rng.rand_vec(lead * k);
    for side in [Side::Left, Side::Right] {
        let got = batched_searchsorted(&rows, lead, p, &vals, lead, k, side).unwrap();
        for i in 0..lead {
            let row = &rows[i * p..(i + 1) * p];
            for j in 0..k {
                let v = vals[i * k + j];
                let want = match side {
                    Side::Left => row.iter().filter(|&&x| x < v).count(),
                    Side::Right => row.iter().filter(|&&x| x <= v).count(),
                };
                assert_eq!(got[i * k + j], want);
                assert_eq!(search_sorted(row, v, side), want);
            }
        }
    }
    let bad = batched_searchsorted(&rows, lead, p, &vals[..4 * k], 4, k, Side::Right);
    assert!(matches!(bad, Err(GateError::Value(_))));
}

#[test]
fn order_ratio() {
    let pool_glm: Vec<f64> = (1..=40).map(f64::from).collect();
    let pool_net: Vec<f64> = (1..=40).map(|v| 2.0 * f64::from(v)).collect();
    let alpha = 0.25;
    let got = order_ratio_one(10.5, 40.0, &pool_glm, &pool_net, 40, alpha);
    let top = np_quantile(&pool_glm, 20.0 / 40.0); // net 40 has 20 pool values <= it
    let bottom = np_quantile(&pool_glm, 10.0 / 40.0); // glm 10.5 has 10 pool values <= it
    assert!((got - (top / bottom).powf(alpha)).abs() < 5e-13);
    assert_eq!(
        order_ratio_one(f64::NAN, 40.0, &pool_glm, &pool_net, 40, alpha),
        1.0
    );
    assert_eq!(
        order_ratio_one(10.5, 40.0, &pool_glm, &pool_net, MIN_POOL - 1, alpha),
        1.0
    );
}
