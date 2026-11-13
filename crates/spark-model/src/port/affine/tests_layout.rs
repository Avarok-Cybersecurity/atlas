// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the affine layout (pack, transpose, dequant, group sums), the shape
//! arithmetic (`split_k`, shared memory, scratch) and the float64 GEMV bound, each gate with a planted
//! control that must fail it.

use super::reference::{
    GemvRef, Mlx, OFFS, bf16_of, count_outside, dequant_bf16, f32_of, gemv_bound, gemv_f64,
    group_sums, npad_of, pack_tiles, quantize_mlx, tiled_position, transpose_pad16, unpack_tiles,
    untranspose16,
};
use super::tests::Lcg;
use super::weight::{Layout, align_scratch, blocks_per_sm, lane_smem, lane_stages, split_k};

fn random_words(r: &mut Lcg, len: usize) -> Vec<u32> {
    (0..len).map(|_| r.u32()).collect()
}

#[test]
fn pack_round_trips_and_pads_with_zero() {
    for group in [32usize, 64, 128] {
        for (n, k) in [(129usize, 256usize), (64, 128), (200, 384)] {
            let mut r = Lcg((group * n + k) as u64);
            let words = random_words(&mut r, n * k / 8);
            let tiled = pack_tiles(&words, n, k, group);
            assert_eq!(tiled.len(), npad_of(n) * k / 8);
            assert_eq!(
                unpack_tiles(&tiled, n, k, group),
                words,
                "g{group} [{n}x{k}]"
            );
            // Rows past n are zero: every tiled word whose row is >= n.
            let lanes = group / 32;
            for (idx, w) in tiled.iter().enumerate() {
                let lane = (idx / lanes) % 32;
                let j = (idx / (lanes * 32)) % 8;
                let t = idx / (lanes * 256) / (k / group);
                if t * 64 + j * 8 + lane / 4 >= n {
                    assert_eq!(*w, 0, "padding word {idx}");
                }
            }
        }
    }
}

#[test]
fn pack_at_group_64_is_upstreams_index_map() {
    // tf_pack_dense: v = idx & 1, lane = (idx >> 1) & 31, j = (idx >> 6) & 7, tg = idx >> 9.
    let (n, k) = (128usize, 128usize);
    let mut r = Lcg(5);
    let words = random_words(&mut r, n * k / 8);
    let tiled = pack_tiles(&words, n, k, 64);
    for (idx, got) in tiled.iter().enumerate() {
        let (v, lane, j, tg) = (idx & 1, (idx >> 1) & 31, (idx >> 6) & 7, idx >> 9);
        let (g, t) = (tg % (k / 64), tg / (k / 64));
        let row = t * 64 + j * 8 + (lane >> 2);
        let c = lane & 3;
        let mut want = 0u32;
        for (p, off) in OFFS.iter().enumerate() {
            let input = g * 64 + 32 * v + 2 * c + off;
            want |= ((words[row * (k / 8) + (input >> 3)] >> (4 * (input & 7))) & 0xF) << (4 * p);
        }
        assert_eq!(*got, want, "word {idx}");
    }
}

/// PLANTED CONTROL for the pack round trip: a pack with the nibble order of a plain (non-MMA) layout
/// must not round-trip through the real inverse.
#[test]
fn planted_wrong_nibble_order_fails_the_round_trip() {
    let (n, k, group) = (64usize, 128usize, 64usize);
    let mut r = Lcg(9);
    let words = random_words(&mut r, n * k / 8);
    let mut wrong = pack_tiles(&words, n, k, group);
    for w in wrong.iter_mut() {
        // Swap nibbles 1 and 4 (inputs +8 and +1): a layout that is not the B fragment's.
        let (a, b) = ((*w >> 4) & 0xF, (*w >> 16) & 0xF);
        *w = (*w & !0x000F_00F0) | (b << 4) | (a << 16);
    }
    assert_ne!(unpack_tiles(&wrong, n, k, group), words);
}

#[test]
fn tiled_position_inverts_the_pack() {
    for group in [32usize, 64, 128] {
        let (n, k) = (70usize, 256usize);
        let mut r = Lcg(group as u64 + 1);
        let words = random_words(&mut r, n * k / 8);
        let tiled = pack_tiles(&words, n, k, group);
        for row in 0..n {
            for col in 0..k {
                let (wi, p) = tiled_position(row, col, k, group);
                let i = row * k + col;
                let want = (words[i / 8] >> (4 * (i % 8))) & 0xF;
                assert_eq!((tiled[wi] >> (4 * p)) & 0xF, want, "g{group} ({row},{col})");
            }
        }
    }
}

#[test]
fn scale_transpose_round_trips_and_pads() {
    let (n, kg) = (129usize, 5usize);
    let npad = npad_of(n);
    let mut r = Lcg(13);
    let x: Vec<u16> = (0..n * kg).map(|_| r.u32() as u16).collect();
    let t = transpose_pad16(&x, n, kg, npad);
    assert_eq!(untranspose16(&t, n, kg, npad), x);
    for g in 0..kg {
        assert!(t[g * npad + n..(g + 1) * npad].iter().all(|v| *v == 0));
    }
}

#[test]
fn tiled_dequant_matches_the_mlx_dequant() {
    let (n, k, group) = (96usize, 256usize, 64usize);
    let mut r = Lcg(17);
    let v = r.bf16s(n * k, 0.02);
    let q = quantize_mlx(&v, n, k, group, 0).unwrap();
    let tiled = pack_tiles(&q.words, n, k, group);
    let npad = npad_of(n);
    let (st, bt) = (
        transpose_pad16(&q.scales, n, k / group, npad),
        transpose_pad16(&q.biases, n, k / group, npad),
    );
    // The dequant kernel's map: tiled word and nibble, (g, row) scale and bias.
    let mut from_tiles = vec![0u16; n * k];
    for row in 0..n {
        for col in 0..k {
            let (wi, p) = tiled_position(row, col, k, group);
            let code = ((tiled[wi] >> (4 * p)) & 0xF) as f32;
            let sb = (col / group) * npad + row;
            from_tiles[row * k + col] = bf16_of(f32_of(st[sb]) * code + f32_of(bt[sb]));
        }
    }
    assert_eq!(from_tiles, dequant_bf16(&q));
}

#[test]
fn group_sums_are_in_order() {
    let (rows, k, group) = (3usize, 256usize, 32usize);
    let mut r = Lcg(19);
    // Inputs spread over 25 binades (an exact power-of-two scale of a bf16 value stays bf16-exact): with
    // the plain sigma-3 values every group sum is exact in f32, so no summation order could be told apart.
    let base = r.bf16s(rows * k, 3.0);
    let mut scale = || 2f32.powi((r.u32() % 25) as i32 - 12);
    let x: Vec<u16> = base.iter().map(|v| bf16_of(v * scale())).collect();
    let s = group_sums(&x, rows, k, group);
    let mut reversed_differs = 0;
    for row in 0..rows {
        for g in 0..k / group {
            let e = &x[row * k + g * group..row * k + (g + 1) * group];
            let fwd = e.iter().fold(0f32, |a, b| a + f32_of(*b));
            let rev = e.iter().rev().fold(0f32, |a, b| a + f32_of(*b));
            assert_eq!(s[row * (k / group) + g].to_bits(), fwd.to_bits());
            reversed_differs += usize::from(rev.to_bits() != fwd.to_bits());
        }
    }
    // PLANTED CONTROL: the reversed order is a different result somewhere, so the bytes gate can fail.
    assert!(reversed_differs > 0);
}

#[test]
fn split_k_is_upstreams_at_group_64() {
    // (n, k) -> sk for the 4B shapes and the edge shape; qlinear.zig:308-314.
    for (n, k, sk) in [
        (9216usize, 2560usize, 2usize),
        (2560, 9216, 8),
        (12288, 2560, 1),
        (2560, 4096, 8),
        (4096, 2560, 4),
        (129, 2560, 4),
        (64, 512, 1),
    ] {
        assert_eq!(split_k(n, k, 64), sk, "[{n}x{k}]");
    }
    // Slices always hold whole groups, at least 8 of them, and never more than 8 slices.
    for group in [32usize, 64, 128] {
        for n in [129usize, 2560, 9216] {
            for k in [1024usize, 2560, 4096, 9216] {
                let sk = split_k(n, k, group);
                assert!(sk.is_power_of_two() && sk <= 8);
                assert!((k / group).is_multiple_of(sk));
                assert!(sk == 1 || k / group / sk >= 8);
            }
        }
    }
}

#[test]
fn lane_shared_memory_matches_upstream_and_the_kernel_comments() {
    assert_eq!(lane_smem(64, 8), 35_328); // qlinear.zig:271 group_smem
    assert_eq!(lane_smem(64, 4), 17_664); // qlinear.zig:275 split_smem
    assert_eq!(lane_smem(32, lane_stages(32).0), 18_944);
    assert_eq!(lane_smem(32, lane_stages(32).1), 9_472);
    assert_eq!(lane_smem(128, lane_stages(128).0), 34_048);
    assert_eq!(lane_smem(128, lane_stages(128).1), 25_536);
    for group in [32usize, 64, 128] {
        let (a, b) = lane_stages(group);
        assert!(lane_smem(group, a) <= 48 * 1024 && lane_smem(group, b) <= 48 * 1024);
    }
    assert_eq!(blocks_per_sm(35_328), 2);
    assert_eq!(blocks_per_sm(9_472), 9);
    assert_eq!(blocks_per_sm(1), 12);
}

#[test]
fn layout_sizes() {
    let l = Layout::of(9216, 2560, 64);
    assert_eq!(l.npad, 9216);
    assert_eq!(l.words, 9216 * 2560 / 2);
    assert_eq!(l.scales, 40 * 9216 * 2);
    // 4.5 bits a weight at group 64.
    assert_eq!(l.total() * 8, 9216 * 2560 * 9 / 2);
    assert_eq!(Layout::of(129, 64, 32).npad, 256);
    assert_eq!(align_scratch(1), 256);
    assert_eq!(align_scratch(512), 512);
}

/// The lane kernel's arithmetic on the host in f32: per group a dot and a sum, two fused multiply-adds,
/// slices added in order, one bf16 rounding. `drop_last`: the planted control that loses the last group.
fn lane_emulate(x: &[u16], rows: usize, w: &Mlx, sk: usize, drop_last: bool) -> Vec<u16> {
    let (n, k, group) = (w.n, w.k, w.group);
    let kg = k / group;
    let per = kg / sk;
    let mut out = vec![0u16; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let mut tot = 0f32;
            for s in 0..sk {
                let mut acc = 0f32;
                for g in s * per..(s + 1) * per {
                    if drop_last && g == kg - 1 {
                        continue;
                    }
                    let (mut dot, mut sum) = (0f32, 0f32);
                    for i in g * group..(g + 1) * group {
                        let xi = f32_of(x[r * k + i]);
                        dot += xi * w.code(c, i) as f32;
                        sum += xi;
                    }
                    let (sc, bi) = (f32_of(w.scales[c * kg + g]), f32_of(w.biases[c * kg + g]));
                    acc = sum.mul_add(bi, dot.mul_add(sc, acc));
                }
                tot = if s == 0 { acc } else { tot + acc };
            }
            out[r * n + c] = bf16_of(tot);
        }
    }
    out
}

#[test]
fn the_f32_lane_chain_meets_the_bound_and_a_dropped_group_does_not() {
    for (n, k, group) in [
        (129usize, 2560usize, 64usize),
        (64, 1024, 32),
        (80, 2048, 128),
    ] {
        let mut r = Lcg((n + k + group) as u64);
        let wv = r.bf16s(n * k, 0.02);
        let w = quantize_mlx(&wv, n, k, group, 0).unwrap();
        let rows = 5;
        let x: Vec<u16> = r.bf16s(rows * k, 1.0).iter().map(|v| bf16_of(*v)).collect();
        let sk = split_k(n, k, group);
        let refr: GemvRef = gemv_f64(&x, rows, &w, 4);
        let good = lane_emulate(&x, rows, &w, sk, false);
        let (bad, worst) = count_outside(&good, &refr, k, group, sk);
        assert_eq!(bad, 0, "g{group} [{n}x{k}] worst {worst}");
        // PLANTED CONTROL: dropping the last group must fail the bound somewhere.
        let planted = lane_emulate(&x, rows, &w, sk, true);
        let (pbad, _) = count_outside(&planted, &refr, k, group, sk);
        assert!(
            pbad > 0,
            "g{group} [{n}x{k}]: the dropped-group control passed"
        );
        // Rows are independent of the row count: the first row alone gives the same bits.
        assert_eq!(lane_emulate(&x[..k], 1, &w, sk, false), good[..n].to_vec());
    }
}

#[test]
fn the_reference_is_thread_count_invariant_and_bounds_are_positive() {
    let (n, k, group) = (37usize, 256usize, 64usize);
    let mut r = Lcg(23);
    let w = quantize_mlx(&r.bf16s(n * k, 0.02), n, k, group, 0).unwrap();
    let x: Vec<u16> = r.bf16s(2 * k, 1.0).iter().map(|v| bf16_of(*v)).collect();
    let (a, b) = (gemv_f64(&x, 2, &w, 1), gemv_f64(&x, 2, &w, 5));
    assert_eq!(a.want, b.want);
    assert_eq!(a.mag, b.mag);
    assert!(gemv_bound(0.0, 1.0, k, group, 1) > 0.0);
    assert!(gemv_bound(1.0, 0.0, k, group, 1) >= 1.0 / 256.0);
}
