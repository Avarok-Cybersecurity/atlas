// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::weight_map::fp8_blockscale_ref::{E4M3_MAX, SCALE_FLOOR, group_scale};

fn values(len: usize, seed: u64, sigma: f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((s >> 40) as f32) / ((1u64 << 24) as f32);
            round_bf16((unit * 2.0 - 1.0) * 3.0 * sigma)
        })
        .collect()
}

#[test]
fn the_fused_reference_is_the_quantizer_applied_to_the_bf16_product() {
    let (m, k) = (3, 384);
    let (g, u) = (values(m * k, 1, 2.0), values(m * k, 2, 1.0));
    let product = swiglu_product(&g, &u);
    assert!(product.iter().all(|x| *x == round_bf16(*x)));
    assert_eq!(
        swiglu_quant(&g, &u, m, k),
        per_token_group_quant(&product, m, k)
    );
}

#[test]
fn each_group_scale_is_the_max_of_the_rounded_product_over_448() {
    let (m, k) = (2, 256);
    let (g, u) = (values(m * k, 3, 2.0), values(m * k, 4, 1.5));
    let product = swiglu_product(&g, &u);
    let (bytes, scales) = swiglu_quant(&g, &u, m, k);
    for (i, s) in scales.iter().enumerate() {
        let span = i * 128..(i + 1) * 128;
        assert_eq!(
            s.to_bits(),
            group_scale(product[span.clone()].iter().copied()).to_bits()
        );
        // The group's largest element lands on E4M3's largest finite code.
        let at = span
            .clone()
            .max_by(|a, b| product[*a].abs().total_cmp(&product[*b].abs()))
            .expect("non-empty group");
        assert_eq!(bytes[at] & 0x7F, 0x7E, "group {i}: max element not at 448");
    }
}

#[test]
fn a_zero_gate_gives_a_zero_group_at_the_scale_floor() {
    let (m, k) = (1, 128);
    let g = vec![0.0f32; k];
    let u = values(k, 5, 4.0);
    let (bytes, scales) = swiglu_quant(&g, &u, m, k);
    assert_eq!(scales, vec![SCALE_FLOOR]);
    assert!(
        bytes.iter().all(|b| *b & 0x7F == 0),
        "silu(0) * u must encode to +-0"
    );
}

#[test]
fn the_silu_model_matches_the_textbook_formula_to_a_bf16_ulp() {
    for (g, u) in [(1.0f32, 1.0f32), (-2.5, 0.75), (6.0, -3.0), (0.125, 8.0)] {
        let want = g / (1.0 + (-g).exp()) * u;
        let got = swiglu_bf16(g, u);
        assert!(
            (got - want).abs() <= want.abs() / 128.0,
            "swiglu({g}, {u}) = {got}, textbook {want}"
        );
    }
    // A very negative gate drives exp2 to +inf and the product to zero, as on
    // the device (`ex2.approx` of a large argument is +inf).
    assert_eq!(swiglu_bf16(-200.0, 5.0), 0.0);
}

#[test]
fn the_exp2_step_is_the_only_free_choice_in_the_model() {
    // Same exp2 -> same bits; this is what lets the oracle attribute any host
    // vs device product mismatch to `ex2.approx` alone.
    let g = values(512, 6, 3.0);
    let u = values(512, 7, 1.0);
    for (a, b) in g.iter().zip(&u) {
        let x = swiglu_bf16_with(*a, *b, f32::exp2);
        assert_eq!(x.to_bits(), swiglu_bf16(*a, *b).to_bits());
    }
}

#[test]
fn non_finite_products_quantize_as_the_device_does() {
    // Group 0 holds a NaN product (dropped by the max, as PTX `max.f32` drops
    // it) and group 1 an infinite one (scale +inf, so inf/inf = NaN). Both NaN
    // quotients saturate to +448 through `fmaxf(fminf(v, 448), -448)`.
    let mut product = vec![0.5f32; 256];
    product[3] = f32::NAN;
    product[129] = f32::INFINITY;
    let (bytes, scales) = quant_of_product(&product, 1, 256);
    assert_eq!(scales[0].to_bits(), (0.5f32 / E4M3_MAX).to_bits());
    assert_eq!(scales[1], f32::INFINITY);
    assert_eq!((bytes[3], bytes[129]), (0x7E, 0x7E));
    assert_eq!(bytes[130], 0x00, "finite / inf is +0");
}

#[test]
#[should_panic(expected = "not a multiple of 128")]
fn a_k_off_the_group_grid_is_refused() {
    let _ = swiglu_quant(&[0.0; 100], &[0.0; 100], 1, 100);
}
