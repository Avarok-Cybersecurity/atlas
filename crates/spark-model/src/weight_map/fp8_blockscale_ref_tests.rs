// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Deterministic pseudo-random values with a heavy-ish tail, so a 128x128
/// tile has a real maximum and most elements sit well below it (the shape of
/// a trained weight, not of a uniform draw).
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Sum of four uniforms minus 2: roughly normal, sigma ~0.58.
    fn normalish(&mut self) -> f32 {
        self.unit() + self.unit() + self.unit() + self.unit() - 2.0
    }
    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|_| round_bf16(self.normalish() * scale))
            .collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn every_finite_e4m3_byte_round_trips_through_the_encoder() {
    for b in 0u8..=255 {
        if b & 0x7F == 0x7F {
            continue; // the two NaN encodings
        }
        let v = fp8_e4m3_to_f32(b);
        let back = e4m3_encode_satfinite(v);
        // -0.0 decodes to -0.0 and must come back as 0x80.
        assert_eq!(back, b, "byte {b:#04x} -> {v} -> {back:#04x}");
    }
}

#[test]
fn rounding_is_to_nearest_even_in_normal_and_subnormal_range() {
    // 1.0 is 0x38 (mantissa 0), 1.125 is 0x39, 1.25 is 0x3A.
    assert_eq!(
        e4m3_encode_satfinite(1.0625),
        0x38,
        "tie between 0x38 and 0x39 goes to even 0x38"
    );
    assert_eq!(
        e4m3_encode_satfinite(1.1875),
        0x3A,
        "tie between 0x39 and 0x3A goes to even 0x3A"
    );
    assert_eq!(
        e4m3_encode_satfinite(1.07),
        0x39,
        "above the midpoint rounds up"
    );
    // Subnormals are m * 2^-9.
    let q = 1.0 / 512.0;
    assert_eq!(
        e4m3_encode_satfinite(0.5 * q),
        0x00,
        "half of the smallest subnormal ties to 0"
    );
    assert_eq!(
        e4m3_encode_satfinite(1.5 * q),
        0x02,
        "1.5 ulp ties to even 2"
    );
    assert_eq!(
        e4m3_encode_satfinite(7.5 * q),
        0x08,
        "the top subnormal tie carries into 2^-6"
    );
    // A mantissa that rounds to 8 carries into the next binade: 1.9375 -> 2.0.
    assert_eq!(e4m3_encode_satfinite(1.9375), 0x40);
}

#[test]
fn encoding_saturates_and_never_emits_nan() {
    assert_eq!(e4m3_encode_satfinite(448.0), 0x7E);
    assert_eq!(e4m3_encode_satfinite(1.0e9), 0x7E);
    assert_eq!(e4m3_encode_satfinite(-1.0e9), 0xFE);
    assert_eq!(e4m3_encode_satfinite(f32::INFINITY), 0x7E);
    let mut last = 0u8;
    let mut x = 1.0e-6f32;
    while x < 1000.0 {
        let b = e4m3_encode_satfinite(x);
        assert_ne!(b & 0x7F, 0x7F, "{x} encoded to a NaN byte");
        assert!(
            b >= last,
            "encoding is not monotonic at {x}: {last:#04x} then {b:#04x}"
        );
        last = b;
        x *= 1.003;
    }
}

#[test]
fn a_tile_maximum_maps_to_the_e4m3_maximum() {
    let mut rng = Lcg(7);
    let (n, k) = (128, 256);
    let mut x = rng.vec(n * k, 0.02);
    x[5 * k + 200] = -0.5; // the max of tile (0, 1), negative on purpose
    let q = quantize_blockscaled(&x, n, k);
    assert_eq!(q.scales.len(), 2);
    assert_eq!(q.scales[1], 0.5 / 448.0);
    assert_eq!(q.bytes[5 * k + 200], 0xFE, "-max must encode to -448");
    // 448 * (0.5 / 448) is -0.5 to within the f32 rounding of the scale.
    assert!((q.value_at(5, 200) + 0.5).abs() <= 0.5 * f32::EPSILON);
}

#[test]
fn edge_tiles_get_their_own_scales_and_the_grid_rounds_up() {
    let mut rng = Lcg(11);
    let (n, k) = (130, 200); // a 2-row and a 72-column partial edge
    let mut x = rng.vec(n * k, 0.01);
    x[129 * k + 199] = 3.0; // only element of the corner tile that is large
    let q = quantize_blockscaled(&x, n, k);
    assert_eq!(q.k_blocks(), 2);
    assert_eq!(q.scales.len(), 4, "ceil(130/128) * ceil(200/128)");
    assert_eq!(
        q.scales[3],
        3.0 / 448.0,
        "the corner tile sees only its own elements"
    );
    assert!(
        q.scales[0] < q.scales[3] / 10.0,
        "the corner's maximum must not leak into tile (0,0)"
    );
}

#[test]
fn an_all_zero_tile_is_floored_and_encodes_zeros() {
    let q = quantize_blockscaled(&vec![0.0; 128 * 128], 128, 128);
    assert_eq!(q.scales, vec![SCALE_FLOOR]);
    assert!(q.bytes.iter().all(|&b| b == 0));
    assert!(q.dequant().iter().all(|&v| v == 0.0));
}

#[test]
fn dequant_error_is_within_half_an_e4m3_ulp() {
    let mut rng = Lcg(3);
    let (n, k) = (256, 384);
    let x = rng.vec(n * k, 0.05);
    let q = quantize_blockscaled(&x, n, k);
    for r in 0..n {
        for c in 0..k {
            let s = q.scale_at(r, c);
            let xs = (x[r * k + c] / s).abs();
            let err = (q.value_at(r, c) - x[r * k + c]).abs();
            // Normal range: 3 mantissa bits, half an ulp is 2^-4 relative.
            // Subnormal range: absolute half-ulp 2^-10 of the scale.
            let bound = if xs >= 0.015_625 {
                x[r * k + c].abs() / 16.0
            } else {
                s / 1024.0
            };
            assert!(
                err <= bound * 1.0001,
                "[{r},{c}] x={} err={err} bound={bound}",
                x[r * k + c]
            );
        }
    }
}

#[test]
fn the_exact_gemv_model_agrees_with_a_float64_dot_of_the_dequantized_weight() {
    let mut rng = Lcg(5);
    for &(n, k) in &[(8usize, 1024usize), (12, 2560), (4, 9216)] {
        let w = quantize_blockscaled(&rng.vec(n * k, 0.03), n, k);
        let a = rng.vec(k, 1.0);
        let got = gemv_w8a16_exact(&a, &w);
        let deq = w.dequant();
        for row in 0..n {
            let want: f64 = (0..k).map(|c| a[c] as f64 * deq[row * k + c] as f64).sum();
            let tol = 1e-4
                * (0..k)
                    .map(|c| (a[c] * deq[row * k + c]).abs() as f64)
                    .sum::<f64>();
            assert!(
                (got[row] as f64 - want).abs() <= tol,
                "[{n}x{k}] row {row}: {} vs {want}",
                got[row]
            );
        }
    }
}

#[test]
fn fp8_weights_track_the_bf16_product_on_a_decode_shape() {
    // What the route trades: BF16 weights for E4M3 ones. The output of one
    // 2560-wide row against BF16 activations must stay close to the BF16
    // product (this is the number a logit-agreement check aggregates).
    let mut rng = Lcg(9);
    let (n, k) = (64, 2560);
    let wf = rng.vec(n * k, 0.03);
    let w = quantize_blockscaled(&wf, n, k);
    let a = rng.vec(k, 1.0);
    let got = gemv_w8a16_exact(&a, &w);
    let want: Vec<f32> = (0..n)
        .map(|r| {
            (0..k)
                .map(|c| a[c] as f64 * wf[r * k + c] as f64)
                .sum::<f64>() as f32
        })
        .collect();
    let cos = cosine(&got, &want);
    eprintln!("W8A16 (E4M3 weights) vs BF16 product, 64x2560: cosine {cos:.6}");
    assert!(cos > 0.999, "cosine {cos}");
}

#[test]
fn the_w8a8_reference_matches_dequantized_math_and_tracks_bf16() {
    let mut rng = Lcg(13);
    let (m, n, k) = (16, 256, 512);
    let wf = rng.vec(n * k, 0.03);
    let af = rng.vec(m * k, 1.0);
    let w = quantize_blockscaled(&wf, n, k);
    let (ab, asc) = per_token_group_quant(&af, m, k);
    let got = gemm_w8a8_ref(&ab, &asc, m, &w);

    // Same arithmetic in float64 on dequantized operands.
    let groups = k / FP8_BLOCK;
    let deq_w = w.dequant();
    for row in 0..m {
        for col in 0..n {
            let want: f64 = (0..k)
                .map(|c| {
                    let av = fp8_e4m3_to_f32(ab[row * k + c]) as f64
                        * asc[row * groups + c / FP8_BLOCK] as f64;
                    av * deq_w[col * k + c] as f64
                })
                .sum();
            let mag: f64 = (0..k)
                .map(|c| (af[row * k + c] * wf[col * k + c]).abs() as f64)
                .sum();
            assert!(
                (got[row * n + col] as f64 - want).abs() <= 1e-5 * mag,
                "[{row},{col}] {} vs {want}",
                got[row * n + col]
            );
        }
    }

    // And against the unquantized product: W8A8 is lossier than W8A16 by
    // construction; the dense-FFN arm is gated at cosine >= 0.999.
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|c| af[row * k + c] as f64 * wf[col * k + c] as f64)
                .sum::<f64>() as f32
        })
        .collect();
    let cos = cosine(&got, &want);
    eprintln!("W8A8 (E4M3 weights and activations) vs BF16 product, 16x256x512: cosine {cos:.6}");
    assert!(cos > 0.999, "cosine {cos}");
}

#[test]
fn the_w8a8_term_bound_is_the_dequantized_absolute_sum() {
    let mut rng = Lcg(29);
    let (m, n, k) = (3, 128, 256);
    let w = quantize_blockscaled(&rng.vec(n * k, 0.03), n, k);
    let (ab, asc) = per_token_group_quant(&rng.vec(m * k, 1.0), m, k);
    let got = gemm_w8a8_ref(&ab, &asc, m, &w);
    let mag = gemm_w8a8_term_mag(&ab, &asc, m, &w);
    let groups = k / FP8_BLOCK;
    for row in 0..m {
        for col in 0..n {
            let want: f64 = (0..k)
                .map(|c| {
                    let av = fp8_e4m3_to_f32(ab[row * k + c]) * asc[row * groups + c / FP8_BLOCK];
                    (f64::from(av) * f64::from(w.value_at(col, c))).abs()
                })
                .sum();
            let i = row * n + col;
            assert!(
                (f64::from(mag[i]) - want).abs() <= 1e-5 * want,
                "[{row},{col}] {} vs {want}",
                mag[i]
            );
            assert!(
                got[i].abs() <= mag[i] * (1.0 + 1e-5),
                "[{row},{col}] |{}| > {}",
                got[i],
                mag[i]
            );
        }
    }
}

#[test]
fn per_token_groups_are_independent_rows_and_groups() {
    let mut x = vec![0.01f32; 2 * 256];
    x[3] = 2.0; // row 0, group 0
    x[256 + 130] = -4.0; // row 1, group 1
    let (b, s) = per_token_group_quant(&x, 2, 256);
    assert_eq!(
        s,
        vec![2.0 / 448.0, 0.01 / 448.0, 0.01 / 448.0, 4.0 / 448.0]
    );
    assert_eq!(b[3], 0x7E);
    assert_eq!(b[256 + 130], 0xFE);
}

#[test]
fn fp16_accumulation_would_overflow_on_one_saturated_product() {
    // The prefill GEMM accumulates in FP32 (fp8_gemm_ada.cu). Both operands
    // are scaled so their group maximum is 448, so a single product can be
    // 448 * 448, already past FP16's largest finite value.
    let one = 448.0f32 * 448.0;
    assert!(one > half::f16::MAX.to_f32());
    assert!(half::f16::from_f32(one).is_infinite());
    // Typical operands do not saturate, but a 128-term group sum of the
    // e4m3 values of an outlier channel does not need many large terms.
    let eight_large = 8.0 * 96.0 * 96.0;
    assert!(eight_large > half::f16::MAX.to_f32());
}
