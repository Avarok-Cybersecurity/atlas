// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracle tests of the Hadamard side of the EXL3 route: the fp32 butterfly, the
//! rotation and dequant mirrors against FP64, the algebra the kernels rely on (rotating the input and
//! un-rotating the output equals multiplying by the dequantized weight), and the GEMV bound's power to
//! accept a correct fp32 product and reject a planted fault.

use super::format::Exl3Codebook;
use super::reference::unpack_parts;
use super::reference_math::{
    GemvBound, HAD_SCALE, bf16_to_f32, dense_f64, dequant_bf16_mirror, dequant_f64, f16_to_f32,
    f32_to_bf16, fwht128_f32, fwht128_f64, gemv_row_ref, had_scale_f64, rot_in_f64, rot_in_mirror,
};
use super::tests::{Lcg, OwnedParts};

fn bf16_rows(r: &mut Lcg, len: usize) -> Vec<u16> {
    (0..len).map(|_| f32_to_bf16(r.normal(1.0))).collect()
}

#[test]
fn had_scale_is_one_over_sqrt_128_in_f32() {
    assert_eq!(HAD_SCALE, had_scale_f64() as f32);
}

#[test]
fn the_butterfly_is_its_own_inverse_and_has_the_sylvester_signs() {
    let mut r = Lcg(1);
    let mut v = [0f32; 128];
    for x in v.iter_mut() {
        *x = r.normal(1.0);
    }
    let before = v;
    fwht128_f32(&mut v);
    fwht128_f32(&mut v);
    for (a, b) in v.iter().zip(&before) {
        assert!((a / 128.0 - b).abs() < 1e-5, "{a} {b}");
    }
    let mut imp = [0f64; 128];
    imp[67] = 1.0;
    fwht128_f64(&mut imp);
    for (i, &h) in imp.iter().enumerate() {
        let want = if (i & 67).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        assert_eq!(h, want, "{i}");
    }
    // The fp32 transform agrees with f64 to fp32 rounding.
    let mut a = before;
    let mut b: Vec<f64> = before.iter().map(|&x| f64::from(x)).collect();
    fwht128_f32(&mut a);
    fwht128_f64(&mut b);
    let l1: f64 = before.iter().map(|x| f64::from(x.abs())).sum();
    for (x, y) in a.iter().zip(&b) {
        assert!((f64::from(*x) - y).abs() <= 8.0 * 2f64.powi(-24) * l1);
    }
}

#[test]
fn the_rotation_mirror_is_the_f64_rotation_rounded_once() {
    let mut r = Lcg(2);
    let (rows, k) = (3, 512);
    let p = OwnedParts::random(&mut r, k, 128, 4, Exl3Codebook::Mcg, false);
    let suh = p.parts().suh_bits();
    let x = bf16_rows(&mut r, rows * k);
    let got = rot_in_mirror(&x, rows, k, &suh);
    let xf: Vec<f64> = x.iter().map(|&b| f64::from(bf16_to_f32(b))).collect();
    let want = rot_in_f64(&xf, rows, k, &suh);
    let l1 = xf.iter().map(|v| v.abs()).fold(0.0, f64::max) * 128.0;
    for (g, w) in got.iter().zip(&want) {
        let gv = f64::from(f16_to_f32(*g));
        assert!(
            (gv - w).abs() <= w.abs() * 2f64.powi(-11) + l1 * 2f64.powi(-22),
            "{gv} {w}"
        );
    }
}

/// The Hadamard-domain product in f64 with no rounding at all: rotate, multiply by `Wq`, un-rotate.
fn rotated_linear_f64(p: &OwnedParts, x: &[f64], rows: usize) -> Vec<f64> {
    let parts = p.parts();
    let (k, n) = (p.k, p.n);
    let wq: Vec<f64> = unpack_parts(&parts)
        .iter()
        .map(|&b| f64::from(f16_to_f32(b)))
        .collect();
    let svh = parts.svh_bits();
    let xh = rot_in_f64(x, rows, k, &parts.suh_bits());
    let mut y = dense_f64(&xh, rows, &wq, k, n);
    for r in 0..rows {
        for nb in 0..n / 128 {
            let blk = &mut y[r * n + nb * 128..r * n + nb * 128 + 128];
            fwht128_f64(blk);
            for (j, v) in blk.iter_mut().enumerate() {
                *v *= had_scale_f64() * f64::from(f16_to_f32(svh[nb * 128 + j]));
            }
        }
    }
    y
}

#[test]
fn rotating_the_input_and_the_output_equals_the_dequantized_weight() {
    let mut r = Lcg(3);
    for (halves, cb, signs) in [
        (4, Exl3Codebook::Mcg, false),
        (6, Exl3Codebook::Mul1, true),
        (7, Exl3Codebook::Mul1, false),
        (8, Exl3Codebook::Inst3, false),
    ] {
        let p = OwnedParts::random(&mut r, 256, 384, halves, cb, signs);
        let rows = 3;
        let x: Vec<f64> = (0..rows * p.k).map(|_| f64::from(r.normal(1.0))).collect();
        let parts = p.parts();
        let w = dequant_f64(
            &unpack_parts(&parts),
            p.k,
            p.n,
            &parts.suh_bits(),
            &parts.svh_bits(),
        );
        let want = dense_f64(&x, rows, &w, p.k, p.n);
        let got = rotated_linear_f64(&p, &x, rows);
        let scale = want.iter().map(|v| v.abs()).fold(1.0, f64::max);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-9 * scale, "{halves} {cb:?}: {g} vs {w}");
        }
    }
}

#[test]
fn the_bf16_dequant_mirror_is_the_f64_weight_rounded() {
    let mut r = Lcg(4);
    for (halves, cb) in [
        (3, Exl3Codebook::Mul1),
        (8, Exl3Codebook::Mcg),
        (16, Exl3Codebook::Inst3),
    ] {
        let p = OwnedParts::random(&mut r, 256, 256, halves, cb, false);
        let parts = p.parts();
        let (suh, svh) = (parts.suh_bits(), parts.svh_bits());
        let wq = unpack_parts(&parts);
        let got = dequant_bf16_mirror(&wq, p.k, p.n, &suh, &svh);
        let want = dequant_f64(&wq, p.k, p.n, &suh, &svh);
        let l = wq.iter().map(|&b| f16_to_f32(b).abs()).fold(0f32, f32::max) as f64;
        for kk in 0..p.k {
            for nn in 0..p.n {
                let w = want[kk * p.n + nn];
                let g = f64::from(bf16_to_f32(got[nn * p.k + kk]));
                let s = f64::from(f16_to_f32(suh[kk]) * f16_to_f32(svh[nn])).abs();
                let tol = w.abs() * 2f64.powi(-8) + 2f64.powi(-12) * l * s;
                assert!(
                    (g - w).abs() <= tol,
                    "{halves} {cb:?} [{kk},{nn}]: {g} vs {w}"
                );
            }
        }
    }
}

/// An fp32 model of the linear kernel (sequential sums in k order, then the epilogue).
fn fp32_linear_row(xh: &[u16], wq: &[u16], k: usize, n: usize, svh: &[u16]) -> Vec<u16> {
    let mut acc = vec![0f32; n];
    for kk in 0..k {
        let xv = f16_to_f32(xh[kk]);
        for c in 0..n {
            acc[c] += xv * f16_to_f32(wq[kk * n + c]);
        }
    }
    let mut out = vec![0u16; n];
    let mut v = [0f32; 128];
    for nb in 0..n / 128 {
        v.copy_from_slice(&acc[nb * 128..nb * 128 + 128]);
        fwht128_f32(&mut v);
        for (j, vj) in v.iter().enumerate() {
            out[nb * 128 + j] = f32_to_bf16(vj * HAD_SCALE * f16_to_f32(svh[nb * 128 + j]));
        }
    }
    out
}

#[test]
fn the_gemv_bound_accepts_an_fp32_product_and_rejects_a_dropped_tile() {
    let mut r = Lcg(5);
    let p = OwnedParts::random(&mut r, 1024, 256, 6, Exl3Codebook::Mul1, false);
    let parts = p.parts();
    let (k, n) = (p.k, p.n);
    let wq = unpack_parts(&parts);
    let (suh, svh) = (parts.suh_bits(), parts.svh_bits());
    let x = bf16_rows(&mut r, k);
    let xh = rot_in_mirror(&x, 1, k, &suh);
    let b: GemvBound = gemv_row_ref(&xh, &wq, k, n, &svh);
    let good = fp32_linear_row(&xh, &wq, k, n, &svh);
    assert_eq!(b.violations(&good), 0, "a correct fp32 product passes");
    // Planted: the last k tile never reaches the sums.
    let mut short = xh.clone();
    for v in &mut short[k - 16..] {
        *v = 0;
    }
    let bad = fp32_linear_row(&short, &wq, k, n, &svh);
    assert!(
        b.violations(&bad) > n / 4,
        "a dropped k tile fails the bound: {}",
        b.violations(&bad)
    );
    // Planted: one wrong output.
    let mut one = good.clone();
    one[17] = f32_to_bf16(bf16_to_f32(one[17]).abs() * 2.0 + 4.0);
    assert_eq!(b.violations(&one), 1);
}
