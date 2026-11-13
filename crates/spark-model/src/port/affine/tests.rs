// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/affine4_host.zig:68-83 (the quantizer and bf16 rounding tests). Copyright 2026 TensorFold
// contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is
// also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust; the MLX quantizer's edge groups, the clip search and
// the planted controls are new.

//! Responsibility: CPU tests of the affine quantizer mirror and the bf16 rounding (`reference.rs`), with
//! a planted control per gate. Layout, pack and GEMV-bound tests are in `tests_layout.rs`; the launch
//! geometry and the `W4Linear` contract on the mock backend in `tests_mock.rs`.

use super::reference::{
    Mlx, bf16_of, code_of, dequantize_minmax, f32_of, group_err, group_params, mlx_params,
    quantize_minmax, quantize_mlx, warp_sum,
};

/// A small deterministic generator (the microtest's).
pub(super) struct Lcg(pub u64);

impl Lcg {
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// bf16-exact values, roughly normal with standard deviation `sigma`.
    pub fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                f32_of(bf16_of(x))
            })
            .collect()
    }
    pub fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}

#[test]
fn bf16_rounding_ties_to_even_and_keeps_nan() {
    // affine4_host.zig:79-82.
    assert_eq!(bf16_of(1.0), 0x3f80);
    assert_eq!(bf16_of(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(bf16_of(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert!(f32_of(bf16_of(f32::NAN)).is_nan());
    // Every finite value agrees with the half crate's round-to-nearest-even.
    let mut r = Lcg(7);
    for _ in 0..200_000 {
        let x = f32::from_bits(r.u32());
        if x.is_finite() {
            assert_eq!(bf16_of(x), half::bf16::from_f32(x).to_bits(), "{x:e}");
        }
    }
}

#[test]
fn upstream_minmax_quantizer_lands_within_half_a_step() {
    // affine4_host.zig:68-78: a sine group and a flat group.
    let mut values = [0f32; 128];
    for (i, v) in values.iter_mut().enumerate().take(64) {
        *v = (i as f32 * 0.37).sin() * 0.05;
    }
    values[64..].fill(0.02);
    let (mut words, mut scales, mut biases) = ([0u32; 16], [0u16; 2], [0u16; 2]);
    quantize_minmax(&values, &mut words, &mut scales, &mut biases);
    let mut back = [0f32; 128];
    dequantize_minmax(&words, &scales, &biases, &mut back);
    for (i, (v, b)) in values.iter().zip(&back).enumerate() {
        assert!(
            (v - b).abs() <= f32_of(scales[i / 64]) * 0.51 + 1e-3,
            "{i}: {v} vs {b}"
        );
    }
}

fn dequant(q: &Mlx, row: usize, col: usize) -> f32 {
    let gi = row * (q.k / q.group) + col / q.group;
    f32_of(q.scales[gi]) * q.code(row, col) as f32 + f32_of(q.biases[gi])
}

/// Every value within one step of its code, plus the bf16 rounding of the scale and bias. Half a step
/// inside the grid; MLX keeps the larger-magnitude edge exact, so the far edge can clamp by up to one
/// step (a symmetric [-1, 1] group: s = -1/8, b = 1, and -1 lands on 1 - 15/8).
fn within_one_step(values: &[f32], q: &Mlx) {
    for row in 0..q.n {
        for col in 0..q.k {
            let gi = row * (q.k / q.group) + col / q.group;
            let (s, b) = (f32_of(q.scales[gi]).abs(), f32_of(q.biases[gi]).abs());
            let tol = 1.01 * s + (b + 15.0 * s) / 128.0 + 1e-30;
            let (v, d) = (values[row * q.k + col], dequant(q, row, col));
            assert!((v - d).abs() <= tol, "({row},{col}) {v} vs {d}, tol {tol}");
        }
    }
}

#[test]
fn mlx_quantizer_edge_groups() {
    for group in [32usize, 64, 128] {
        let mut r = Lcg(group as u64);
        let mut v = Vec::new();
        // Random.
        v.extend(r.bf16s(group, 0.02));
        // Flat: lands exactly on its bias.
        v.extend(std::iter::repeat_n(f32_of(bf16_of(0.0213)), group));
        // One outlier.
        let mut g = r.bf16s(group, 0.001);
        g[group / 3] = 0.75;
        v.extend(g);
        // All negative.
        v.extend(r.bf16s(group, 0.01).into_iter().map(|x| -x.abs() - 0.003));
        // A range whose step underflows: the 1e-7 floor.
        v.extend((0..group).map(|i| f32_of(bf16_of(1e-38 * (i % 3) as f32))));
        let rows = v.len() / group;
        let q = quantize_mlx(&v, rows, group, group, 0).expect("finite groups");
        within_one_step(&v, &q);
        for col in 0..group {
            assert_eq!(dequant(&q, 1, col), v[group], "flat group, g{group}");
        }
    }
}

#[test]
fn mlx_quantizer_flags_the_first_non_finite_group() {
    let mut r = Lcg(3);
    let mut v = r.bf16s(64 * 6, 0.02);
    v[64 * 4 + 7] = f32::INFINITY;
    v[64 * 2 + 1] = f32::NAN;
    assert_eq!(quantize_mlx(&v, 2, 192, 64, 0), Err(2));
    assert!(group_params(&v[64 * 4..64 * 5], 3).is_none());
    // A range that overflows f32 is unrepresentable too.
    let mut w = vec![0f32; 64];
    w[0] = -3.0e38;
    w[1] = 3.0e38;
    assert_eq!(quantize_mlx(&w, 1, 64, 64, 0), Err(0));
}

#[test]
fn mlx_params_follow_upstream() {
    // The larger-magnitude edge lands on the grid: [-1, 3] -> edge 3, step 4/15, q0 = round(3 / -step).
    let (s, b) = mlx_params(-1.0, 3.0).unwrap();
    let step = 4.0f32 / 15.0;
    let q0 = (3.0 / -step).round();
    assert_eq!((s, b), (3.0 / q0, 3.0));
    assert_eq!(code_of(3.0, s, b), 0);
    // Low edge.
    let (s, b) = mlx_params(-3.0, 1.0).unwrap();
    assert_eq!(b, -3.0);
    assert!(s > 0.0);
    assert_eq!(code_of(-3.0, s, b), 0);
}

#[test]
fn clip_search_off_is_plain_mlx_and_never_worse() {
    let (n, k) = (8usize, 256usize);
    for group in [32usize, 64, 128] {
        let mut r = Lcg(11 + group as u64);
        // Heavy tails (a cubed normal), where clipping the range has something to win.
        let v: Vec<f32> = r
            .bf16s(n * k, 1.0)
            .into_iter()
            .map(|x| f32_of(bf16_of(x * x * x * 0.01)))
            .collect();
        let plain = quantize_mlx(&v, n, k, group, 0).unwrap();
        let clipped = quantize_mlx(&v, n, k, group, 10).unwrap();
        let mut improved = 0;
        for gid in 0..n * k / group {
            let g = &v[gid * group..(gid + 1) * group];
            let (ps, pb) = group_params(g, 0).unwrap();
            let (cs, cb) = group_params(g, 10).unwrap();
            assert!(
                group_err(g, cs, cb) <= group_err(g, ps, pb),
                "g{group} group {gid}"
            );
            let moved = (clipped.scales[gid], clipped.biases[gid])
                != (plain.scales[gid], plain.biases[gid]);
            improved += usize::from(moved);
        }
        assert!(
            improved > 0,
            "g{group}: the clip search never chose a candidate"
        );
        assert_eq!(
            quantize_mlx(&v, n, k, group, 0).unwrap(),
            plain,
            "deterministic"
        );
    }
}

#[test]
fn warp_sum_is_the_xor_butterfly() {
    let mut p = [0f32; 32];
    for (i, x) in p.iter_mut().enumerate() {
        *x = 1.0 + (i as f32) * 1e-7;
    }
    // ((p0 + p16) + (p8 + p24)) + ... : compute the tree explicitly for lane 0.
    let mut t = p;
    for off in [16usize, 8, 4, 2, 1] {
        let prev = t;
        for l in 0..32 {
            t[l] = prev[l] + prev[l ^ off];
        }
    }
    assert_eq!(warp_sum(p).to_bits(), t[0].to_bits());
    assert!(
        t.iter().all(|x| x.to_bits() == t[0].to_bits()),
        "every lane holds the sum"
    );
}

/// PLANTED CONTROL for the quantizer bytes gate: a round-half-to-even quantizer must differ from the
/// kernel's round-half-away on a group built to hit ties.
#[test]
fn planted_half_even_quantizer_differs_on_ties() {
    // [0, 15] with halves: edge 15, step 1, s = -1, b = 15, so (x - 15) / -1 = 15 - x hits .5 exactly.
    let mut g: Vec<f32> = (0..64).map(|i| (i % 16) as f32).collect();
    for (i, x) in g.iter_mut().enumerate().skip(1).step_by(4) {
        *x = (i % 15) as f32 + 0.5;
    }
    g[0] = 0.0;
    g[63] = 15.0;
    let (s, b) = group_params(&g, 0).unwrap();
    let away: Vec<u32> = g.iter().map(|&x| code_of(x, s, b)).collect();
    let even: Vec<u32> = g
        .iter()
        .map(|&x| ((x - b) / s).round_ties_even().clamp(0.0, 15.0) as u32)
        .collect();
    assert_ne!(away, even, "the control must fail the bytes gate");
}
