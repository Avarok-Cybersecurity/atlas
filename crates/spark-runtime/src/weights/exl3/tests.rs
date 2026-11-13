// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the EXL3 format and its FP64 reference: exact fp16 rounding, the three
//! codebooks against an independent f32 evaluation, window ends and tile states against the
//! shift-register reading of the format, tile placement, unpack, packed signs, the Hadamard butterfly
//! against the explicit matrix, and dequantization and matvec against a dense FP64 reconstruction.

use super::dequant::{self, Axis, HADAMARD_DIM, hadamard_128};
use super::fixture::{Rng, new_bits, pack_stream, shift_register_states};
use super::format::{self, Codebook, TILE_PLACEMENT, Width};
use super::fp16;

fn widths() -> impl Iterator<Item = Width> {
    (2..=16).map(|h| Width::from_halves(h).expect("2..=16 is a width"))
}

#[test]
fn fp16_roundtrips_every_pattern_and_matches_half() {
    for b in 0..=u16::MAX {
        let nan = (b >> 10) & 0x1F == 0x1F && b & 0x3FF != 0;
        let v = fp16::to_f64(b);
        if nan {
            assert!(v.is_nan());
            assert_eq!(fp16::from_f64(v), 0x7E00);
            continue;
        }
        assert_eq!(fp16::from_f64(v), b, "pattern {b:#06x}");
        assert_eq!(
            v.to_bits(),
            half::f16::from_bits(b).to_f64().to_bits(),
            "{b:#06x}"
        );
    }
}

#[test]
fn fp16_rounds_once_to_nearest_even() {
    // Every gap between neighbors: the midpoint ties to the even pattern, and the next f64 on either
    // side of it goes to the near neighbor (the case a double rounding or a truncated mantissa gets wrong).
    for b in 0u16..0x7BFF {
        let (lo, hi) = (fp16::to_f64(b), fp16::to_f64(b + 1));
        let mid = (lo + hi) / 2.0;
        let even = if b.is_multiple_of(2) { b } else { b + 1 };
        assert_eq!(fp16::from_f64(mid), even, "tie above {b:#06x}");
        assert_eq!(
            fp16::from_f64(mid.next_up()),
            b + 1,
            "just above the tie at {b:#06x}"
        );
        assert_eq!(
            fp16::from_f64(mid.next_down()),
            b,
            "just below the tie at {b:#06x}"
        );
        assert_eq!(fp16::from_f64(-mid), 0x8000 | even);
    }
    assert_eq!(fp16::from_f64(65504.0), 0x7BFF);
    assert_eq!(fp16::from_f64(65520.0f64.next_down()), 0x7BFF);
    assert_eq!(
        fp16::from_f64(65520.0),
        0x7C00,
        "the tie at the top rounds to infinity"
    );
    assert_eq!(fp16::from_f64(f64::MAX), 0x7C00);
    assert_eq!(fp16::from_f64(f64::NEG_INFINITY), 0xFC00);
    assert_eq!(fp16::from_f64(-0.0), 0x8000);
    assert_eq!(
        fp16::from_f64(2f64.powi(-25)),
        0,
        "half the smallest subnormal ties to zero"
    );
    assert_eq!(fp16::from_f64(2f64.powi(-25).next_up()), 1);
    assert_eq!(fp16::from_f64(1e-300), 0);
    assert_eq!(fp16::from_f64(f64::MIN_POSITIVE / 4.0), 0, "f64 subnormal");
    assert_eq!(fp16::from_f64(2f64.powi(-14)), 0x0400, "smallest normal");
    assert_eq!(fp16::from_f64(1.0), fp16::ONE);
    assert_eq!(fp16::from_f64(-1.0), fp16::MINUS_ONE);
}

/// The codebooks evaluated from the recipe's constants in f32, where every intermediate is exact
/// (3inst/mcg: two fp16 values of exponent 12..15 sum in at most 15 bits; mul1: `h * scale + bias`
/// is an integer below 2^20 times 2^-18), then rounded by the `half` crate.
fn codebook_reference(cb: Codebook, state: u16) -> u16 {
    let s = u32::from(state);
    let x = match cb {
        Codebook::Inst3 => s.wrapping_mul(89_226_354).wrapping_add(64_248_484),
        Codebook::Mcg => s.wrapping_mul(0xCBAC_1FED),
        Codebook::Mul1 => s.wrapping_mul(0x83DC_D12D),
    };
    let f = |bits: u32| half::f16::from_bits(bits as u16).to_f32();
    let v = if cb == Codebook::Mul1 {
        let h = 1024 + x.to_le_bytes().iter().map(|&b| u32::from(b)).sum::<u32>();
        h as f32 * f(0x1EEE) + f(0xC931)
    } else {
        let m = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
        f(m & 0xFFFF) + f(m >> 16)
    };
    half::f16::from_f32(v).to_bits()
}

#[test]
fn codebooks_match_an_independent_f32_evaluation_on_every_state() {
    for cb in Codebook::ALL {
        let (mut sum, mut sq) = (0f64, 0f64);
        for s in 0..=u16::MAX {
            let got = format::codebook_value(cb, s);
            assert_eq!(got, codebook_reference(cb, s), "{} state {s}", cb.name());
            let v = fp16::to_f64(got);
            assert!(
                v.is_finite() && v.abs() < 4.0,
                "{} state {s}: {v}",
                cb.name()
            );
            sum += v;
            sq += v * v;
        }
        let count = 65536.0;
        let mean = sum / count;
        let sd = (sq / count - mean * mean).sqrt();
        // Each codebook is a roughly zero-mean, unit-scale Gaussian-like table.
        assert!(mean.abs() < 0.15, "{} mean {mean}", cb.name());
        assert!(sd > 0.6 && sd < 2.0, "{} sd {sd}", cb.name());
    }
    let table = format::codebook_table(Codebook::Mcg);
    assert_eq!(table.len(), 65536);
    assert_eq!(table[1234], format::codebook_value_f64(Codebook::Mcg, 1234));
}

#[test]
fn codebook_markers() {
    assert_eq!(Codebook::from_marker(0xCBAC_1FED), Some(Codebook::Mcg));
    assert_eq!(Codebook::from_marker(0x83DC_D12D), Some(Codebook::Mul1));
    assert_eq!(Codebook::from_marker(7), None);
    for cb in Codebook::ALL {
        assert_eq!(
            cb.marker_value().and_then(Codebook::from_marker),
            cb.marker_value().map(|_| cb)
        );
        assert_eq!(cb.marker_suffix().is_some(), cb != Codebook::Inst3);
    }
}

#[test]
fn widths_from_halves_and_words() {
    assert_eq!(Width::from_halves(1), None);
    assert_eq!(Width::from_halves(17), None);
    assert_eq!(Width::from_tile_words(16).map(Width::halves), Some(2));
    assert_eq!(Width::from_tile_words(128).map(Width::halves), Some(16));
    assert_eq!(Width::from_tile_words(56).map(Width::halves), Some(7));
    for bad in [0, 8, 12, 30, 136, 256] {
        assert_eq!(Width::from_tile_words(bad), None, "{bad} words");
    }
    let w = Width::from_halves(7).unwrap();
    assert!(w.is_half());
    assert_eq!(w.bits(), 3.5);
    assert_eq!(w.bpw_x100(), 350);
    assert_eq!(w.tile_words(), 56);
    assert_eq!(w.ring_bits(), 896);
    assert!(!Width::from_halves(6).unwrap().is_half());
}

#[test]
fn stream_ends_are_the_running_sum_of_new_bits() {
    for w in widths() {
        let ends = format::stream_ends(w);
        let mut at = 0;
        for (p, &end) in ends.iter().enumerate() {
            at += new_bits(w, p);
            assert_eq!(end, at, "halves {} p {p}", w.halves());
        }
        assert_eq!(
            ends[255],
            w.ring_bits(),
            "the last window ends at the ring's end"
        );
    }
    // The documented half-width example: 2.5 bits alternates 2 and 3 new bits.
    let ends = format::stream_ends(Width::from_halves(5).unwrap());
    assert_eq!(&ends[..6], &[2, 5, 7, 10, 12, 15]);
}

#[test]
fn tile_states_match_the_shift_register_reading() {
    let mut rng = Rng::new(0x51A7E5);
    for w in widths() {
        for _ in 0..4 {
            let bits = rng.bits(w.ring_bits() as usize);
            let words = pack_stream(&bits);
            assert_eq!(words.len(), w.tile_words());
            let got = format::tile_states(w, &words).unwrap();
            assert_eq!(
                got,
                shift_register_states(w, &bits),
                "halves {}",
                w.halves()
            );
        }
    }
    // A stream of one set bit at index 0 (the MSB of the first 32-bit word, i.e. bit 15 of word 1).
    let w = Width::from_halves(4).unwrap();
    let mut words = vec![0i16; w.tile_words()];
    words[1] = i16::MIN;
    let states = format::tile_states(w, &words).unwrap();
    // E(0) = 2: value 0's window is ring bits [498, 512) ++ [0, 2), so the set bit (index 0) is the
    // window's 15th bit of 16, weight 2^1. E(1) = 4 puts it at weight 2^3.
    assert_eq!(states[0], 0b10);
    assert_eq!(states[1], 0b1000);
    assert!(format::tile_states(w, &words[1..]).is_err());
}

#[test]
fn placement_is_the_two_mma_b_fragments_of_each_lane() {
    let mut seen = [false; 256];
    for (p, place) in TILE_PLACEMENT.iter().enumerate() {
        let (lane, j) = (p / 8, p % 8);
        assert_eq!(
            usize::from(place.col),
            lane / 4 + if j >= 4 { 8 } else { 0 }
        );
        let k = usize::from(place.row);
        assert_eq!(
            k % 8 / 2,
            lane % 4,
            "p {p}: fragment row pair is the lane's thread-in-group"
        );
        assert_eq!(k % 2, j % 2);
        assert_eq!(k / 8, (j / 2) % 2);
        let at = k * 16 + usize::from(place.col);
        assert!(!seen[at], "two values at one position");
        seen[at] = true;
    }
    let spot = |p: usize| (TILE_PLACEMENT[p].row, TILE_PLACEMENT[p].col);
    assert_eq!(
        [spot(0), spot(1), spot(2), spot(3), spot(4)],
        [(0, 0), (1, 0), (8, 0), (9, 0), (0, 8)]
    );
    assert_eq!(spot(8), (2, 0));
    assert_eq!(spot(32), (0, 1));
    assert_eq!(spot(255), (15, 15));
}

#[test]
fn unpack_places_each_tiles_values() {
    let mut rng = Rng::new(7);
    let w = Width::from_halves(6).unwrap();
    let (tk, tn) = (2usize, 3usize);
    let trellis = rng.i16s(tk * tn * w.tile_words());
    for cb in Codebook::ALL {
        let wq = format::unpack(w, cb, &trellis, tk, tn).unwrap();
        assert_eq!(wq.len(), tk * 16 * tn * 16);
        for a in 0..tk {
            for b in 0..tn {
                let at = (a * tn + b) * w.tile_words();
                let states = format::tile_states(w, &trellis[at..at + w.tile_words()]).unwrap();
                for (p, place) in TILE_PLACEMENT.iter().enumerate() {
                    let r = a * 16 + usize::from(place.row);
                    let c = b * 16 + usize::from(place.col);
                    assert_eq!(wq[r * tn * 16 + c], format::codebook_value(cb, states[p]));
                }
            }
        }
    }
    let zeros = vec![0i16; w.tile_words()];
    let wq = format::unpack(w, Codebook::Mul1, &zeros, 1, 1).unwrap();
    assert!(
        wq.iter()
            .all(|&v| v == format::codebook_value(Codebook::Mul1, 0))
    );
    assert!(format::unpack(w, Codebook::Mcg, &trellis, tk, tn + 1).is_err());
}

#[test]
fn packed_signs_expand_bit_by_bit() {
    let out = format::unpack_signs(&[0b101, -1, 0]);
    assert_eq!(out.len(), 48);
    assert_eq!(out[0], fp16::MINUS_ONE);
    assert_eq!(out[1], fp16::ONE);
    assert_eq!(out[2], fp16::MINUS_ONE);
    assert!(out[3..16].iter().all(|&v| v == fp16::ONE));
    assert!(out[16..32].iter().all(|&v| v == fp16::MINUS_ONE));
    assert!(out[32..].iter().all(|&v| v == fp16::ONE));
}

fn hadamard_entry(i: usize, j: usize) -> f64 {
    let sign = if (i & j).count_ones() % 2 == 1 {
        -1.0
    } else {
        1.0
    };
    sign / (HADAMARD_DIM as f64).sqrt()
}

#[test]
fn butterfly_equals_the_explicit_sylvester_matrix() {
    let mut rng = Rng::new(11);
    let x: Vec<f64> = (0..HADAMARD_DIM).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let mut fast = x.clone();
    hadamard_128(&mut fast);
    for (i, &f) in fast.iter().enumerate() {
        let want: f64 = (0..HADAMARD_DIM).map(|j| hadamard_entry(i, j) * x[j]).sum();
        assert!((f - want).abs() < 1e-12, "row {i}: {f} vs {want}");
    }
    hadamard_128(&mut fast);
    for (a, b) in fast.iter().zip(x.iter()) {
        assert!((a - b).abs() < 1e-12, "H is its own inverse");
    }
    let mut m = vec![0f64; 128 * 3];
    assert!(dequant::rotate(&mut m, 128, 3, Axis::Rows).is_ok());
    assert!(dequant::rotate(&mut m, 128, 3, Axis::Cols).is_err());
    assert!(dequant::rotate(&mut m, 3, 128, Axis::Rows).is_err());
}

/// `diag(suh) @ H_K @ W_q @ H_N @ diag(svh)` with explicit Hadamard matrices.
/// `H_K` and `H_N` are block diagonal, so only the 128 entries of the element's own block are nonzero.
fn dense_reference(wq: &[f64], k: usize, n: usize, suh: &[u16], svh: &[u16]) -> Vec<f64> {
    let d = HADAMARD_DIM;
    let mut left = vec![0f64; k * n];
    for r in 0..k {
        let base = r / d * d;
        for c in 0..n {
            let s: f64 = (base..base + d)
                .map(|j| hadamard_entry(r % d, j % d) * wq[j * n + c])
                .sum();
            left[r * n + c] = s * fp16::to_f64(suh[r]);
        }
    }
    let mut out = vec![0f64; k * n];
    for r in 0..k {
        for c in 0..n {
            let base = c / d * d;
            let s: f64 = (base..base + d)
                .map(|j| left[r * n + j] * hadamard_entry(j % d, c % d))
                .sum();
            out[r * n + c] = s * fp16::to_f64(svh[c]);
        }
    }
    out
}

#[test]
fn dequantize_blocks_and_matvec_agree_with_a_dense_reconstruction() {
    let mut rng = Rng::new(99);
    for (cb, halves) in [
        (Codebook::Mcg, 4u16),
        (Codebook::Mul1, 5),
        (Codebook::Inst3, 6),
    ] {
        let w = Width::from_halves(halves).unwrap();
        let (k, n) = (256usize, 384usize);
        let (tk, tn) = (k / 16, n / 16);
        let trellis = rng.i16s(tk * tn * w.tile_words());
        let suh = rng.fp16s(k, 0.5, 1.5);
        let svh = rng.fp16s(n, 0.5, 1.5);
        let bias = rng.fp16s(n, -0.5, 0.5);
        let wq: Vec<f64> = format::unpack(w, cb, &trellis, tk, tn)
            .unwrap()
            .into_iter()
            .map(fp16::to_f64)
            .collect();
        let want = dense_reference(&wq, k, n, &suh, &svh);
        let got = dequant::dequantize(w, cb, &trellis, tk, tn, &suh, &svh).unwrap();
        let mag = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        for (g, r) in got.iter().zip(want.iter()) {
            assert!(
                (g - r).abs() <= 1e-12 * (1.0 + mag),
                "{}: {g} vs {r}",
                cb.name()
            );
        }
        for kb in 0..k / HADAMARD_DIM {
            for nb in 0..n / HADAMARD_DIM {
                let block =
                    dequant::dequantize_block(w, cb, &trellis, tk, tn, &suh, &svh, kb, nb).unwrap();
                for i in 0..HADAMARD_DIM {
                    let row = (kb * HADAMARD_DIM + i) * n + nb * HADAMARD_DIM;
                    assert_eq!(
                        &block[i * HADAMARD_DIM..(i + 1) * HADAMARD_DIM],
                        &got[row..row + HADAMARD_DIM],
                        "block ({kb}, {nb}) row {i} is bit-equal to the full decode"
                    );
                }
            }
        }
        assert!(dequant::dequantize_block(w, cb, &trellis, tk, tn, &suh, &svh, 2, 0).is_err());
        let x: Vec<f64> = (0..k).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let y = dequant::matvec(w, cb, &trellis, tk, tn, &suh, &svh, Some(&bias), &x).unwrap();
        for c in 0..n {
            let dot: f64 =
                (0..k).map(|r| x[r] * got[r * n + c]).sum::<f64>() + fp16::to_f64(bias[c]);
            assert!(
                (y[c] - dot).abs() <= 1e-9 * (1.0 + dot.abs()),
                "column {c}: {} vs {dot}",
                y[c]
            );
        }
        assert!(dequant::matvec(w, cb, &trellis, tk, tn, &suh, &svh, None, &x[1..]).is_err());
        assert!(
            dequant::matvec(w, cb, &trellis, tk, tn, &suh, &svh, Some(&bias[1..]), &x).is_err()
        );
    }
}
