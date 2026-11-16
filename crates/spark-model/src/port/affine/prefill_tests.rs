// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the affine prompt GEMMs (lane L17b) on the CPU: the layout identities the copied
//! kernels rely on (slot order, `bytes4`, `pair`, the tiled words, which must be lane L17a's pack), the
//! activation quantizer's edge groups, the int8 model against its derived bound and the BF16 reference bound
//! (each with a planted control that must fail), and the SwiGLU fusion against SwiGLU then quantize. The host
//! dispatch on the mock backend is in `prefill_tests_mock.rs`, the dense FFN path in `prefill_ffn_tests.rs`.

use super::oracle::{
    self, GROUP as G, HostAffine, Lcg, bf16_bits, bytes4, int8_gemm, pair, quantize_rows,
    slot_input, swiglu_bf16, swiglu_quantize, violations,
};
use super::*;

fn random_x(rng: &mut Lcg, m: usize, ldx: usize, sigma: f32) -> Vec<f32> {
    // Row scales spread over two decades so the per-group scale matters.
    (0..m * ldx)
        .map(|j| rng.bf16_normal(sigma * (1.0 + 9.0 * ((j / ldx) % 3) as f32 / 2.0)))
        .collect()
}

#[test]
fn shared_memory_and_descriptor_sizes_match_the_kernel_tiles() {
    assert_eq!(
        BF16_SMEM_BYTES, 62_976,
        "qlinear.zig prefill_smem, Tile<64,128,128,2,4,3>"
    );
    assert_eq!(
        INT8_SMEM_BYTES, 55_296,
        "w4a8.zig smem, Tile8<128,128,2,4,4>"
    );
    assert_eq!(
        PARTS_BYTES,
        MAX_PARTS * PART_BYTES + 8,
        "count plus padding to 8"
    );
    // Both sizes are constants, so the bound is checked at compile time.
    const {
        assert!(
            BF16_SMEM_BYTES <= 101_376 && INT8_SMEM_BYTES <= 101_376,
            "sm_89 per-block opt-in"
        );
    }
}

#[test]
fn row_bands_follow_the_upstream_formula() {
    assert_eq!(
        row_band(3921, 2560),
        19,
        "12 MiB / (128 * 2560 * 2) = 19 row tiles"
    );
    assert_eq!(row_band(3921, 9216), 5);
    assert_eq!(row_band(17, 2560), 1, "never more than the row tiles");
    assert_eq!(row_band(1 << 20, 64), 768);
    assert_eq!(row_band(1, 1 << 30), 1, "never zero");
    assert_eq!(grid_blocks(3921, 9216), 31 * 72);
    assert_eq!(grid_blocks(17, 129), 2);
}

#[test]
fn the_int8_buffers_fit_the_w4_scratch_regions() {
    use crate::port::w4::scratch;
    for (m, k) in [(2048, 9216), (2048, 2560), (17, 64), (1, 64), (3, 128)] {
        assert_eq!(xq_bytes(m, k), scratch::xq_bytes(m, k));
        assert!(
            xs_bytes(m, k) <= scratch::xs_bytes(m, k).max(scratch::ALIGN),
            "m {m} k {k}"
        );
    }
}

#[test]
fn slot_order_is_a_permutation_with_an_inverse() {
    let mut seen = [false; 32];
    for s in 0..32 {
        seen[slot_input(s)] = true;
        assert_eq!(oracle::slot_of(slot_input(s)), s);
    }
    assert!(seen.iter().all(|b| *b));
}

/// The int8 GEMM's view of one (row, column, group): the lanes of the column's n8 slice read the tiled words,
/// `bytes4` gives B slots 4c + i (b0) and 16 + 4c + i (b1) of each 32-input step, the A side reads the
/// quantizer's slot-ordered bytes. Must equal the logical integer dot.
fn kernel_view_dot(
    words: &[u32],
    kg: usize,
    q: &oracle::QuantRows,
    r: usize,
    col: usize,
    g: usize,
) -> i32 {
    let (t, slice) = (col / 64, (col % 64) / 8);
    let mut dot = 0i32;
    for c in 0..4 {
        let lane = (col % 8) * 4 + c;
        for kt in 0..2 {
            let word = words[((t * kg + g) * 8 + slice) * 64 + lane * 2 + kt];
            let base = r * q.k + g * G + kt * 32;
            for (half, b) in [bytes4(word, 0), bytes4(word, 8)].iter().enumerate() {
                for (i, code) in b.iter().enumerate() {
                    let x = q.xq[base + 16 * half + 4 * c + i] as i8;
                    dot += i32::from(*code) * i32::from(x);
                }
            }
        }
    }
    dot
}

#[test]
fn the_int8_kernel_view_dot_equals_the_logical_dot() {
    let mut rng = Lcg(0x17b0_0001);
    let (n, k, m) = (129, 192, 3);
    let w = HostAffine::random(&mut rng, n, k, 0.02);
    let x = random_x(&mut rng, m, k, 1.0);
    let q = quantize_rows(&x, k, m, k);
    let words = w.tiled_words();
    assert_eq!(words.len(), w.npad() / 64 * w.kg() * 512);
    for r in 0..m {
        let xr = q.row_codes(r);
        for col in [0, 1, 7, 8, 63, 64, 100, 127, 128] {
            for g in 0..w.kg() {
                let want: i32 = (0..G)
                    .map(|i| i32::from(w.codes[col * k + g * G + i]) * xr[g * G + i])
                    .sum();
                assert_eq!(
                    kernel_view_dot(&words, w.kg(), &q, r, col, g),
                    want,
                    "r {r} col {col} g {g}"
                );
            }
        }
    }
}

#[test]
fn the_bf16_kernel_view_operands_are_the_logical_codes() {
    // m16n8k16 B fragment: b0 holds k = 2c, 2c + 1 and b1 holds k = 2c + 8, 2c + 9 of each 16-input step.
    let mut rng = Lcg(0x17b0_0002);
    let w = HostAffine::random(&mut rng, 64, 128, 0.02);
    let words = w.tiled_words();
    for col in [0, 9, 63] {
        let (slice, c_lane) = ((col % 64) / 8, (col % 8) * 4);
        for g in 0..w.kg() {
            for c in 0..4 {
                let lane = c_lane + c;
                for kt in 0..4 {
                    let word = words[(g * 8 + slice) * 64 + lane * 2 + kt / 2];
                    let sh = (kt as u32 & 1) * 8;
                    let at = |off: usize| w.codes[col * w.k + g * G + kt * 16 + 2 * c + off];
                    assert_eq!(pair(word, sh), [at(0), at(1)], "b0 col {col} g {g} kt {kt}");
                    assert_eq!(
                        pair(word, sh + 4),
                        [at(8), at(9)],
                        "b1 col {col} g {g} kt {kt}"
                    );
                }
            }
        }
    }
    // Padded columns past n hold zero codes and zero scales.
    let w = HostAffine::random(&mut rng, 70, 64, 0.02);
    let sc = w.tiled_scales();
    assert_eq!(sc.len(), w.npad());
    assert!(sc[70..].iter().all(|v| *v == 0) && sc[..70].iter().all(|v| *v != 0));
}

#[test]
fn quantizer_edge_groups() {
    let k = 6 * G;
    let mut x = vec![0.0f32; k];
    x[..G].fill(0.5); // flat
    // x[G..2G] all zero
    for (i, v) in x[2 * G..3 * G].iter_mut().enumerate() {
        *v = [0.5, 1.5, 2.5, -0.5][i % 4]; // ties, with amax 127 below
    }
    x[2 * G + 63] = 127.0;
    for (i, v) in x[3 * G..4 * G].iter_mut().enumerate() {
        *v = oracle::round_bf16(1e-39 * (1.0 + i as f32)); // tiny amax: 127 / amax is infinite
    }
    x[4 * G..5 * G].fill(3.0e38); // finite inputs, overflowing sum
    x[5 * G + 17] = f32::NAN;
    let q = quantize_rows(&x, k, 1, k);
    assert!(q.exceptional);
    assert_eq!(q.group_ok, vec![true, true, true, true, false, false]);
    assert!((0..G).all(|i| q.code(0, i) == 127));
    assert_eq!((q.xa[0], q.xs[0]), (0.5 / 127.0, 32.0));
    assert!((G..2 * G).all(|i| q.code(0, i) == 0));
    assert_eq!((q.xa[1], q.xs[1]), (1.0, 0.0), "amax 0: scale 1");
    let ties: Vec<i32> = (0..4).map(|i| q.code(0, 2 * G + i)).collect();
    assert_eq!(
        ties,
        vec![0, 2, 2, 0],
        "round half to even, as __float2int_rn"
    );
    let a = q.xa[3];
    assert!(
        a > 0.0 && a.is_finite(),
        "the division path keeps the group finite"
    );
    for (i, xi) in x.iter().enumerate().take(4 * G).skip(3 * G) {
        let c = q.code(0, i);
        assert!((-127..=127).contains(&c));
        // a is an FP32 subnormal here, so its own rounding is coarser than in a normal group.
        let err = (f64::from(*xi) - f64::from(a) * f64::from(c)).abs();
        assert!(err <= f64::from(a) * 0.501, "tiny group input {i}");
    }
    // The exceptional groups are left unwritten (zero here), as the kernel leaves them.
    assert_eq!((q.xa[4], q.xa[5]), (0.0, 0.0));
}

#[test]
fn the_int8_model_is_within_the_derived_bound_and_a_dropped_group_is_not() {
    let mut rng = Lcg(0x17b0_0003);
    for (n, k, m) in [(129, 256, 5), (64, 64, 3), (200, 512, 2)] {
        let w = HostAffine::random(&mut rng, n, k, 0.03);
        let x = random_x(&mut rng, m, k, 1.0);
        let q = quantize_rows(&x, k, m, k);
        assert!(!q.exceptional);
        let rows: Vec<usize> = (0..m).collect();
        let want = oracle::exact(&w, &x, k, &rows);
        let bound = oracle::int8_bound(&w, &x, k, &q, &rows, &want);
        let got = int8_gemm(&w, &q, &rows);
        assert_eq!(violations(&got, &want, &bound), 0, "n {n} k {k}");
        // Planted control: the same chain without the last group must break the bound somewhere.
        let mut dropped = w.clone();
        for col in 0..n {
            let g = col * dropped.kg() + dropped.kg() - 1;
            dropped.scales[g] = 0.0;
            dropped.biases[g] = 0.0;
        }
        let bad = int8_gemm(&dropped, &q, &rows);
        assert!(
            violations(&bad, &want, &bound) > 0,
            "control passed: the bound is vacuous"
        );
    }
}

#[test]
fn the_int8_model_is_row_invariant() {
    let mut rng = Lcg(0x17b0_0004);
    let w = HostAffine::random(&mut rng, 96, 128, 0.02);
    let x = random_x(&mut rng, 6, 128, 1.0);
    let all = int8_gemm(&w, &quantize_rows(&x, 128, 6, 128), &[4]);
    let one = int8_gemm(&w, &quantize_rows(&x[4 * 128..], 128, 1, 128), &[0]);
    assert_eq!(all, one, "a row's bits do not depend on the other rows");
}

#[test]
fn the_bf16_reference_bound_holds_and_catches_a_dropped_group() {
    let mut rng = Lcg(0x17b0_0005);
    let (n, k, m) = (130, 256, 4);
    let w = HostAffine::random(&mut rng, n, k, 0.03);
    let x = random_x(&mut rng, m, k, 1.0);
    let rows: Vec<usize> = (0..m).collect();
    let (want, bound) = oracle::bf16_reference(&w, &x, k, &rows);
    let exact: Vec<u16> = want.iter().map(|v| bf16_bits(*v as f32)).collect();
    assert_eq!(violations(&exact, &want, &bound), 0);
    let mut dropped = w.clone();
    for col in 0..n {
        let g = col * dropped.kg() + dropped.kg() - 1;
        dropped.scales[g] = 0.0;
        dropped.biases[g] = 0.0;
    }
    let (bad, _) = oracle::bf16_reference(&dropped, &x, k, &rows);
    let bad: Vec<u16> = bad.iter().map(|v| bf16_bits(*v as f32)).collect();
    assert!(violations(&bad, &want, &bound) > 0, "control passed");
    // The BF16 operand is one rounding of the exact affine value.
    let (v, b) = (w.weight(3, 70), w.weight_bf16(3, 70));
    assert!((v - b).abs() <= v.abs() * oracle::U_BF16 + oracle::BF16_TINY);
}

#[test]
fn the_fused_swiglu_quantizer_equals_swiglu_then_quantize() {
    let mut rng = Lcg(0x17b0_0006);
    let (m, k, ld) = (3, 192, 200);
    let mut gate: Vec<f32> = (0..m * ld).map(|_| rng.bf16_normal(2.0)).collect();
    let up: Vec<f32> = (0..m * ld).map(|_| rng.bf16_normal(1.0)).collect();
    gate[2 * ld + 70] = f32::NAN; // row 2, group 1
    let (h, fused) = swiglu_quantize(&gate, &up, ld, m, k);
    let composed_h: Vec<f32> = (0..m * k)
        .map(|j| swiglu_bf16(gate[(j / k) * ld + j % k], up[(j / k) * ld + j % k]))
        .collect();
    let same = |a: &f32, b: &f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    assert!(h.iter().zip(&composed_h).all(|(a, b)| same(a, b)));
    assert_eq!(fused, quantize_rows(&composed_h, k, m, k));
    assert!(fused.exceptional && !fused.group_ok[2 * 3 + 1] && fused.group_ok[2 * 3]);
}

#[test]
fn the_oracle_layout_is_the_l17a_pack() {
    // The microtest uploads the oracle's own tiles; the serve path's come from lane L17a's pack. Both must be the
    // same bytes, or a green microtest would prove nothing about the route.
    use crate::port::affine::reference::{pack_tiles, transpose_pad16};
    let mut rng = Lcg(0x17b0_0007);
    for (n, k) in [(130, 192), (64, 64), (257, 128)] {
        let w = HostAffine::random(&mut rng, n, k, 0.02);
        let mut mlx = vec![0u32; n * k / 8];
        for (i, c) in w.codes.iter().enumerate() {
            mlx[i / 8] |= u32::from(*c) << (4 * (i % 8));
        }
        assert_eq!(
            w.tiled_words(),
            pack_tiles(&mlx, n, k, G),
            "words n {n} k {k}"
        );
        let bits = |v: &[f32]| -> Vec<u16> { v.iter().map(|x| bf16_bits(*x)).collect() };
        assert_eq!(
            w.tiled_scales(),
            transpose_pad16(&bits(&w.scales), n, k / G, w.npad()),
            "scales"
        );
        assert_eq!(
            w.tiled_biases(),
            transpose_pad16(&bits(&w.biases), n, k / G, w.npad()),
            "biases"
        );
    }
}
