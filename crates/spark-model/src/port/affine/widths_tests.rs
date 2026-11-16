// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:452-471 (the AffineN layout and split_k tests) and
// zig/src/core/affine4_host.zig:68-83 (the round-trip and bf16 tie tests). Copyright 2026 TensorFold
// contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0
// is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md. Changes for Atlas: Rust
// tests over the widths module's CPU oracle and the mock GPU backend; edge groups, clip search, pack
// round trips, lane emulation, GEMV model bounds and planted controls are Atlas additions.
//! Responsibility: CPU tests of the 2/3/5/6/8-bit affine path (lane L17c): layout numbers, the
//! quantizer's bounds and edge groups, the packers and the lane reads against the MLX layout, and the
//! GEMV model against the float64 bound (with planted controls that must fail it). The host launch
//! path on the mock backend is in `widths/tests_mock.rs`.

use super::reference::{self as r, MlxWeight};
use super::{kernels, layout};

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// bf16 values, roughly normal with standard deviation `sigma`.
    fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                r::round_bf16(
                    (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma,
                )
            })
            .collect()
    }
}

fn weight(rng: &mut Lcg, n: usize, k: usize, bits: u8) -> MlxWeight {
    let v = rng.bf16s(n * k, 0.02);
    r::quantize_mlx(&v, n, k, bits, 0).expect("finite weights quantize")
}

#[test]
fn smem_numbers_match_the_tile_formula() {
    for bits in layout::BITS {
        let stage = 16 * 128 + 64 * 64 * u32::from(bits) / 8 + 2 * 128 + 64;
        assert_eq!(layout::gemv_smem(bits), 6 * stage, "gemv {bits}");
        assert_eq!(layout::split_smem(bits), 4 * stage, "split {bits}");
        assert!(
            layout::gemv_smem(bits) < 48 * 1024,
            "no opt-in needed at {bits} bits"
        );
        assert!(layout::resident_per_sm(bits) >= 1);
    }
    assert_eq!(layout::gemv_smem(8), 38_784);
    assert_eq!(layout::split_smem(8), 25_856);
    assert_eq!(layout::gemv_smem(2), 14_208 + 3_072 * 2);
}

#[test]
fn split_k_follows_the_upstream_shapes_and_atlas_shapes() {
    assert_eq!(layout::split_k(10304, 2688), 2);
    assert_eq!(layout::split_k(2688, 4096), 8);
    assert_eq!(layout::split_k(4608, 2688), 2);
    assert_eq!(layout::split_k(2688, 5376), 4);
    assert_eq!(layout::split_k(131072, 2688), 1);
    assert_eq!(layout::split_k(9216, 2560), 2, "4B gate/up");
    assert_eq!(layout::split_k(2560, 9216), 8, "4B down");
    for (n, k) in [
        (9216, 2560),
        (2560, 9216),
        (17408, 5120),
        (5120, 17408),
        (129, 1024),
        (9, 64),
    ] {
        let sk = layout::split_k(n, k);
        assert!(
            layout::groups(k).is_multiple_of(sk),
            "{n}x{k}: sk {sk} divides the groups"
        );
        if sk > 1 {
            assert!(layout::tiles(n) < layout::SPLIT_TILES_MAX && layout::groups(k) / sk >= 8);
        }
    }
}

#[test]
fn layout_sizes_match_upstream() {
    assert_eq!(layout::npad(10304), 10368);
    assert_eq!(layout::tile_words(10304, 2688, 8) * 4, 10368 * 2688);
    assert_eq!(layout::tile_words(10304, 2688, 3) * 4, 10368 * 2688 * 3 / 8);
    assert_eq!(layout::table_elems(10304, 2688) * 2, 2688 / 64 * 10368 * 2);
    for bits in layout::BITS {
        let blocks = layout::npad(10304) / 64 * layout::groups(2688);
        assert_eq!(
            blocks * layout::block_words(bits),
            layout::tile_words(10304, 2688, bits),
            "{bits}"
        );
    }
    assert!(layout::supports(2, 64) && layout::supports(8, 64));
    assert!(
        !layout::supports(4, 64)
            && !layout::supports(7, 64)
            && !layout::supports(3, 32)
            && !layout::supports(8, 128)
    );
    assert!(layout::check_shape(10, 96, 3).is_err() && layout::check_shape(10, 128, 4).is_err());
    assert!(layout::check_shape(10, 128, 5).is_ok());
}

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(r::bf16_bits(1.0), 0x3f80);
    assert_eq!(r::bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(r::bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert!(r::bf16_f32(r::bf16_bits(f32::NAN)).is_nan());
    assert_eq!(r::bf16_bits(f32::INFINITY), 0x7f80);
}

#[test]
fn code_at_reads_codes_that_cross_words() {
    let mut rng = Lcg(7);
    for bits in [2u8, 3, 5, 6, 8] {
        let codes: Vec<u32> = (0..128)
            .map(|_| (rng.unit() * 255.0) as u32 & ((1 << bits) - 1))
            .collect();
        let mut row = vec![0u32; 128 * usize::from(bits) / 32];
        for (i, &c) in codes.iter().enumerate() {
            for at in 0..usize::from(bits) {
                let bit = i * usize::from(bits) + at;
                row[bit / 32] |= ((c >> at) & 1) << (bit % 32);
            }
        }
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(r::code_at(&row, i, bits), c, "bits {bits} code {i}");
        }
    }
}

#[test]
fn dequantized_values_stay_on_the_grid() {
    let mut rng = Lcg(11);
    for bits in layout::BITS {
        let (n, k) = (5, 256);
        let v = rng.bf16s(n * k, 0.05);
        let w = r::quantize_mlx(&v, n, k, bits, 0).unwrap();
        let back = r::dequant_mlx(&w);
        let levels = (1u32 << bits) - 1;
        let rw = layout::mlx_row_words(k, bits);
        for (i, (&x, &y)) in v.iter().zip(&back).enumerate() {
            let g = (i / k) * (k / 64) + (i % k) / 64;
            let (s, b) = (
                r::bf16_f32(w.scales[g]).abs(),
                r::bf16_f32(w.biases[g]).abs(),
            );
            let q = r::code_at(&w.words[(i / k) * rw..(i / k + 1) * rw], i % k, bits);
            // An interior code is within half a step; a clamped one within the grid's reach (MLX
            // fits the larger-magnitude edge exactly, so the other end may fall short by up to about
            // a step). Plus the bf16 storage error of s (times the code) and of b.
            let reach = if q > 0 && q < levels { 0.5005 } else { 1.5 };
            let bound = reach * s + (levels as f32 * s + b) * 2f32.powi(-8) + 1e-6;
            assert!(
                (x - y).abs() <= bound,
                "bits {bits} at {i}: {x} vs {y} (code {q}, bound {bound})"
            );
        }
    }
}

#[test]
fn edge_groups_quantize_as_mlx_does() {
    let k = 64;
    for bits in layout::BITS {
        // flat nonzero, all zero, one outlier, all negative, subnormal-scale (tiny) groups
        let mut v = vec![0.375f32; k];
        v.extend(vec![0.0f32; k]);
        v.extend((0..k).map(|i| {
            if i == 17 {
                4.0
            } else {
                r::round_bf16(0.01 * (i as f32 - 32.0) / 32.0)
            }
        }));
        v.extend((0..k).map(|i| r::round_bf16(-0.5 - 0.01 * i as f32)));
        v.extend((0..k).map(|i| r::round_bf16(1e-38 * (1.0 + i as f32 / 64.0))));
        let w = r::quantize_mlx(&v, 5, k, bits, 0).unwrap();
        let back = r::dequant_mlx(&w);
        assert!(
            back[..k].iter().all(|&y| y == 0.375),
            "flat group is exact at {bits} bits"
        );
        assert!(
            back[k..2 * k].iter().all(|&y| y == 0.0),
            "zero group is exact at {bits} bits"
        );
        for row in 0..4 {
            let g = &v[row * k..(row + 1) * k];
            let edge = g
                .iter()
                .copied()
                .fold(0f32, |m, x| if x.abs() > m.abs() { x } else { m });
            let at = g.iter().position(|&x| x == edge).unwrap();
            assert_eq!(
                back[row * k + at],
                edge,
                "row {row} at {bits} bits: the larger edge lands on the grid"
            );
        }
        assert!(back[2 * k + 17] == 4.0, "the outlier is the edge");
        // Below the 1e-7 step floor the edge rounds to code 0 with bias 0: everything is near zero.
        assert!(
            back[4 * k..]
                .iter()
                .zip(&v[4 * k..])
                .all(|(y, x)| (y - x).abs() <= 1e-7),
            "tiny group"
        );
    }
}

#[test]
fn non_finite_groups_are_reported_by_index() {
    let (n, k) = (3, 256);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = vec![0.01f32; n * k];
        v[k + 3 * 64 + 5] = bad;
        assert_eq!(
            r::quantize_mlx(&v, n, k, 3, 0),
            Err(4 + 3),
            "row 1 group 3 of {bad}"
        );
    }
}

#[test]
fn clip_search_never_raises_the_squared_error() {
    let mut rng = Lcg(23);
    let (n, k) = (8, 512);
    let mut v = rng.bf16s(n * k, 0.02);
    for i in (0..v.len()).step_by(97) {
        v[i] = r::round_bf16(v[i] * 12.0); // outliers, where clipping helps
    }
    for bits in [2u8, 3, 5] {
        let full = r::dequant_mlx(&r::quantize_mlx(&v, n, k, bits, 0).unwrap());
        let clipped_w = r::quantize_mlx(&v, n, k, bits, 4).unwrap();
        let clipped = r::dequant_mlx(&clipped_w);
        assert_eq!(
            r::quantize_mlx(&v, n, k, bits, 4).unwrap(),
            clipped_w,
            "deterministic"
        );
        let mut improved = 0;
        for g in 0..n * k / 64 {
            let sse = |d: &[f32]| -> f64 {
                (g * 64..(g + 1) * 64)
                    .map(|i| f64::from(v[i] - d[i]).powi(2))
                    .sum()
            };
            let (a, b) = (sse(&full), sse(&clipped));
            // The quantizer compares fp32 butterfly sums; allow their rounding against this fp64 sum.
            assert!(
                b <= a * (1.0 + 1e-5),
                "bits {bits} group {g}: clip {b} > full {a}"
            );
            improved += usize::from(b < a);
        }
        // At 2 and 3 bits an outlier group gains from a shorter range; at 5 bits it rarely does.
        assert!(
            bits > 3 || improved > 0,
            "bits {bits}: the clip search chose a clipped range somewhere"
        );
    }
}

#[test]
fn packs_round_trip_through_the_dequant_index_math_and_the_lane_reads() {
    let mut rng = Lcg(31);
    for bits in layout::BITS {
        for (n, k) in [(9usize, 64usize), (129, 128), (64, 192), (200, 256)] {
            let w = weight(&mut rng, n, k, bits);
            let t = r::pack(&w);
            let rw = layout::mlx_row_words(k, bits);
            for row in 0..n {
                for kk in 0..k {
                    let want = r::code_at(&w.words[row * rw..(row + 1) * rw], kk, bits);
                    assert_eq!(
                        r::tile_code(&t, row, kk),
                        want,
                        "{bits}b {n}x{k} ({row},{kk})"
                    );
                }
            }
            for tile in 0..layout::npad(n) / 64 {
                for g in 0..k / 64 {
                    let lanes = r::lane_codes(&t, tile, g);
                    for (cc, col) in lanes.iter().enumerate() {
                        let row = tile * 64 + cc;
                        for (kk, &q) in col.iter().enumerate() {
                            let want = if row < n {
                                r::code_at(&w.words[row * rw..(row + 1) * rw], g * 64 + kk, bits)
                            } else {
                                0
                            };
                            assert_eq!(
                                u32::from(q),
                                want,
                                "{bits}b lane read ({row},{})",
                                g * 64 + kk
                            );
                        }
                    }
                }
            }
            let np = layout::npad(n);
            for g in 0..k / 64 {
                assert!(
                    t.scales_t[g * np + n..(g + 1) * np].iter().all(|&s| s == 0),
                    "padded scales zero"
                );
                for col in 0..n {
                    assert_eq!(t.scales_t[g * np + col], w.scales[col * (k / 64) + g]);
                    assert_eq!(t.biases_t[g * np + col], w.biases[col * (k / 64) + g]);
                }
            }
        }
    }
}

/// Count outputs of `got` (fp32, before bf16 rounding) outside the float64 bound of `w`.
fn violations(w: &MlxWeight, x: &[f32], rows: usize, got: &[f32]) -> usize {
    let (want, tol) = r::gemv_f64(w, x, rows);
    got.iter()
        .zip(want.iter().zip(&tol))
        .filter(|(g, (y, t))| (f64::from(r::round_bf16(**g)) - **y).abs() > **t)
        .count()
}

#[test]
fn gemv_model_meets_the_float64_bound_and_split_matches_unsplit() {
    let mut rng = Lcg(41);
    for bits in layout::BITS {
        for (n, k) in [(129usize, 1024usize), (64, 4096), (70, 256)] {
            let w = weight(&mut rng, n, k, bits);
            let t = r::pack(&w);
            let rows = 16;
            let x = rng.bf16s(rows * k, 1.0);
            let unsplit = r::gemv_model(&t, &x, rows, false);
            let split = r::gemv_model(&t, &x, rows, true);
            assert!(
                unsplit
                    .iter()
                    .zip(&split)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{bits}b {n}x{k}: split order"
            );
            assert_eq!(
                violations(&w, &x, rows, &unsplit),
                0,
                "{bits}b {n}x{k} sk {}",
                layout::split_k(n, k)
            );
            let one = r::gemv_model(&t, &x[..k], 1, false);
            assert!(
                one.iter()
                    .zip(&unsplit[..n])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "row 0 alone"
            );
        }
    }
}

#[test]
fn planted_controls_fail_the_bound() {
    let mut rng = Lcg(43);
    let (n, k, rows) = (96usize, 1024usize, 4usize);
    for bits in [3u8, 8] {
        let w = weight(&mut rng, n, k, bits);
        let x = rng.bf16s(rows * k, 1.0);
        let good = r::gemv_model(&r::pack(&w), &x, rows, false);
        assert_eq!(violations(&w, &x, rows, &good), 0);
        // Control 1: a GEMV that drops the last group.
        let mut dropped = w.clone();
        let kg = k / 64;
        for c in 0..n {
            dropped.scales[c * kg + kg - 1] = 0;
            dropped.biases[c * kg + kg - 1] = 0;
        }
        let bad = r::gemv_model(&r::pack(&dropped), &x, rows, false);
        assert!(
            violations(&w, &x, rows, &bad) > n * rows / 2,
            "{bits}b: a dropped group must fail"
        );
        // Control 2: a GEMV that ignores the biases (no b * sum(x) term).
        let mut nobias = w.clone();
        nobias.biases.iter_mut().for_each(|b| *b = 0);
        let bad = r::gemv_model(&r::pack(&nobias), &x, rows, false);
        assert!(
            violations(&w, &x, rows, &bad) > n * rows / 2,
            "{bits}b: a missing bias term must fail"
        );
    }
}

#[test]
fn kernel_names_exist_in_the_ada_source() {
    let src = include_str!("../../../../../kernels/ada/common/affine_n_ada.cu");
    for name in [
        kernels::SUMS,
        kernels::QUANTIZE,
        kernels::PACK_NARROW,
        kernels::PACK_WIDE,
        kernels::TRANSPOSE,
        kernels::DEQUANT,
    ] {
        assert!(
            src.contains(&format!("{name}(")),
            "{name} missing from affine_n_ada.cu"
        );
    }
    for bits in [2u8, 3, 5, 6] {
        assert!(
            src.contains(&format!("AVAROK_AFFINE_NARROW({bits})")),
            "narrow instance {bits}"
        );
    }
    assert!(src.contains("AVAROK_AFFINE_WIDTH(8)"));
    assert!(src.contains("affine##B##_gemv_ada") && src.contains("affine##B##_split_ada"));
    assert_eq!(kernels::gemv_name(3), "affine3_gemv_ada");
    assert_eq!(kernels::split_name(8), "affine8_split_ada");
}
