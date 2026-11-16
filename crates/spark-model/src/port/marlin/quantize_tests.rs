// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the RTN quantizer (`quantize.rs`) and the reference (`reference.rs`):
//! exact groups, edge groups (zero, all negative, one outlier), the clip search never losing to plain
//! RTN, the BF16 helpers, the kernel-exact dequant, and the GEMM oracle with planted controls.
#![allow(clippy::useless_vec)]

use super::quantize::{self, RtnOpts, quantize_group, rtn_int4_sym};
use super::reference::{
    self, LogicalInt4, bf16_to_f32, check_output, dequant, f16_to_bf16, f32_to_bf16, ulp_bf16,
};
use super::repack_tests::{Rng, random_logical};

fn group_err(vals: &[f32], q: &[u8], s: u16) -> f64 {
    vals.iter()
        .zip(q)
        .map(|(&v, &c)| {
            let d = f64::from(v) - f64::from(bf16_to_f32(dequant(c, 8, s)));
            d * d
        })
        .sum()
}

#[test]
fn bf16_helpers_round_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3F80);
    // Halfway between 1.0 and the next BF16: ties to the even significand.
    assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3F80);
    assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
    assert_eq!(bf16_to_f32(0xC000), -2.0);
    assert_eq!(f16_to_bf16(0x3C00), 0x3F80);
    assert_eq!(f16_to_bf16(0xBC00), 0xBF80);
    assert_eq!(ulp_bf16(1.0), 2f64.powi(-7));
    assert_eq!(ulp_bf16(3.0), 2f64.powi(-6));
    assert_eq!(ulp_bf16(0.0), f64::from(f32::MIN_POSITIVE) * 2f64.powi(-7));
}

#[test]
fn dequant_is_the_kernels_arithmetic() {
    // (q - 8) * s, one rounding; a zero difference times a negative scale is -0 (as `__hmul2` gives).
    assert_eq!(dequant(15, 8, f32_to_bf16(0.5)), f32_to_bf16(3.5));
    assert_eq!(dequant(0, 8, f32_to_bf16(0.25)), f32_to_bf16(-2.0));
    assert_eq!(dequant(8, 8, 0xBF80), 0x8000);
    assert_eq!(dequant(8, 8, 0x3F80), 0x0000);
    // With a zero point: (q - z) * s.
    assert_eq!(dequant(3, 5, f32_to_bf16(1.5)), f32_to_bf16(-3.0));
    // A scale with a full 8-bit significand: the product rounds once.
    let s = 0x3FFF; // 1.9921875
    assert_eq!(dequant(15, 0, s), f32_to_bf16(15.0 * 1.992_187_5));
}

#[test]
fn an_exact_group_quantizes_without_error() {
    let s = 0.5f32;
    let vals: Vec<f32> = (0..128).map(|i| ((i % 16) as f32 - 8.0) * s).collect();
    let mut q = vec![0u8; 128];
    let scale = quantize_group(&vals, 0, &mut q);
    assert_eq!(scale, f32_to_bf16(s));
    assert_eq!(group_err(&vals, &q, scale), 0.0);
    for (i, &c) in q.iter().enumerate() {
        assert_eq!(c as usize, i % 16);
    }
}

#[test]
fn edge_groups() {
    let mut q = vec![0u8; 128];
    // All zero: scale 0, every code the zero offset.
    assert_eq!(quantize_group(&[0.0; 128], 3, &mut q), 0);
    assert!(q.iter().all(|&c| c == 8));
    // All negative: the scale comes from the minimum over 8.
    let neg: Vec<f32> = (0..128).map(|i| -1.0 - i as f32 / 128.0).collect();
    let s = quantize_group(&neg, 0, &mut q);
    let mn = neg.iter().copied().fold(f32::INFINITY, f32::min);
    assert_eq!(s, f32_to_bf16(mn.abs() / 8.0));
    assert!(q.iter().all(|&c| c <= 8));
    // One outlier: plain RTN spends the range on it.
    let mut r = Rng::new(17);
    let mut out: Vec<f32> = (0..128).map(|_| 0.01 * r.unit()).collect();
    out[40] = 1.0;
    let s = quantize_group(&out, 0, &mut q);
    assert_eq!(s, f32_to_bf16(1.0 / 7.0));
    assert_eq!(q[40], 15);
}

#[test]
fn the_clip_search_never_loses_to_plain_rtn() {
    let mut r = Rng::new(23);
    let mut wins = 0;
    for g in 0..64 {
        let mut vals: Vec<f32> = (0..128).map(|_| r.unit()).collect();
        if g % 4 == 0 {
            vals[g % 128] *= 6.0;
        }
        let mut q0 = vec![0u8; 128];
        let s0 = quantize_group(&vals, 0, &mut q0);
        let mut q1 = vec![0u8; 128];
        let s1 = quantize_group(&vals, quantize::MAX_CLIP_SEARCH, &mut q1);
        let (e0, e1) = (group_err(&vals, &q0, s0), group_err(&vals, &q1, s1));
        assert!(e1 <= e0, "group {g}: clip {e1} > plain {e0}");
        if e1 < e0 {
            wins += 1;
        }
    }
    assert!(
        wins > 0,
        "the search must find a better scale somewhere (else it is vacuous)"
    );
}

#[test]
fn rtn_matches_the_per_group_rule_in_marlin_orientation() {
    let (n, k, group) = (96, 384, 128);
    let mut r = Rng::new(31);
    let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(r.unit())).collect();
    let opts = RtnOpts {
        group,
        clip_search: 2,
    };
    let l = rtn_int4_sym(&w, n, k, &opts).unwrap();
    assert_eq!(l.zeros, None);
    assert_eq!((l.n, l.k, l.group), (n, k, group));
    for nn in 0..n {
        for g in 0..k / group {
            let vals: Vec<f32> = (0..group)
                .map(|i| bf16_to_f32(w[nn * k + g * group + i]))
                .collect();
            let mut q = vec![0u8; group];
            let s = quantize_group(&vals, 2, &mut q);
            assert_eq!(l.scales[g * n + nn], s);
            for i in 0..group {
                assert_eq!(l.q[(g * group + i) * n + nn], q[i]);
            }
        }
    }
    // Every reconstructed weight is within half a step of the source (no clipping at clip_search 0),
    // plus the BF16 rounding of the scale (the top code may clamp by 7 half-ulps of it) and of the
    // dequantized value: at most 10% of the half step.
    let plain = rtn_int4_sym(
        &w,
        n,
        k,
        &RtnOpts {
            group,
            clip_search: 0,
        },
    )
    .unwrap();
    let deq = plain.dequant_nk();
    for nn in 0..n {
        for kk in 0..k {
            let s = f64::from(bf16_to_f32(plain.scales[(kk / group) * n + nn]));
            let d = (f64::from(bf16_to_f32(w[nn * k + kk]))
                - f64::from(bf16_to_f32(deq[nn * k + kk])))
            .abs();
            assert!(d <= 0.5 * s * 1.1 + 1e-6, "[{nn}, {kk}] err {d} step {s}");
        }
    }
}

#[test]
fn rtn_refuses_bad_input() {
    let mut w = vec![f32_to_bf16(0.5); 64 * 128];
    let opts = RtnOpts {
        group: 128,
        clip_search: 0,
    };
    assert!(rtn_int4_sym(&w, 64, 128, &opts).is_ok());
    w[77] = 0x7FC0; // NaN
    let e = rtn_int4_sym(&w, 64, 128, &opts).unwrap_err().to_string();
    assert!(e.contains("[0, 77]"), "{e}");
    assert!(rtn_int4_sym(&w[..100], 64, 128, &opts).is_err());
    let bad_group = RtnOpts {
        group: 96,
        clip_search: 0,
    };
    assert!(rtn_int4_sym(&vec![0u16; 64 * 128], 64, 128, &bad_group).is_err());
}

#[test]
fn dequant_nk_is_the_transposed_logical_weight() {
    let l = random_logical(64, 256, 128, true, 41);
    let nk = l.dequant_nk();
    for nn in 0..l.n {
        for kk in 0..l.k {
            let g = kk / l.group;
            let want = dequant(l.q[kk * l.n + nn], l.zero(g, nn), l.scales[g * l.n + nn]);
            assert_eq!(nk[nn * l.k + kk], want);
        }
    }
}

#[test]
fn the_gemm_oracle_accepts_the_rounded_reference_and_rejects_planted_errors() {
    let l: LogicalInt4 = random_logical(64, 256, 128, false, 43);
    let w = l.dequant_nk();
    let (m, n, k) = (5, l.n, l.k);
    let mut r = Rng::new(47);
    let x: Vec<u16> = (0..m * k).map(|_| f32_to_bf16(r.unit())).collect();
    let (val, _) = reference::reference_gemm(&x, m, &w, n, k);
    let got: Vec<u16> = val.iter().map(|&v| f32_to_bf16(v as f32)).collect();
    let rep = check_output(&got, &x, m, &w, n, k);
    assert!(rep.ok(), "{rep:?}");
    // Control 1: four ulps off on one output.
    let mut bad = got.clone();
    let i = 3 * n + 7;
    bad[i] = f32_to_bf16(
        bf16_to_f32(bad[i]) + 4.0 * ulp_bf16(f64::from(bf16_to_f32(bad[i]))) as f32 + 0.05,
    );
    assert!(!check_output(&bad, &x, m, &w, n, k).ok());
    // Control 2: the right numbers against a weight with one doubled scale group.
    let mut l2 = l.clone();
    l2.scales[0] = f32_to_bf16(2.0 * bf16_to_f32(l2.scales[0]));
    let w2 = l2.dequant_nk();
    let rep2 = check_output(&got, &x, m, &w2, n, k);
    assert!(!rep2.ok());
    assert_eq!(
        rep2.first.map(|f| f.1),
        Some(0),
        "the doubled scale is column 0's"
    );
    // A non-finite output always fails.
    let mut nan = got;
    nan[0] = 0x7FC0;
    assert!(!check_output(&nan, &x, m, &w, n, k).ok());
}
