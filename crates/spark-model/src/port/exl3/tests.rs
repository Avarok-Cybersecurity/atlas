// SPDX-License-Identifier: AGPL-3.0-only
// Test design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/exl3_projection_decode.zig:64-105 (the circular extraction at every width and pattern,
// all 65536 states of every codebook, the placement bijection) and zig/src/core/exl3_layout.zig:196-206
// (width rules). Apache-2.0; see EXL3-ATTRIBUTION.md. Changes: no code copied; the tests are written for
// Atlas's Rust mirror of the Ada kernels' lane decoder.

//! Responsibility: CPU oracle tests of the EXL3 format side: hand-computed codebook values and stream
//! states, the kernels' lane extraction against the scalar stream decoder at every width, the placement
//! bijection, tile unpacking, sign words and the admission rules. Shared test helpers ([`Lcg`],
//! [`OwnedParts`]) live here.

use super::format::{
    Exl3Codebook, Exl3Parts, Exl3Scales, F16_MINUS_ONE, F16_ONE, all_widths, tile_elems,
    trellis_bytes, unpack_signs, width_supported,
};
use super::reference::{
    codebook_value, decode_tile, lane_states, lane_words, placement, stream_end, tile_states,
    tile_u32, unpack,
};

/// A deterministic generator for the tests.
pub(super) struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Roughly normal, standard deviation about `sigma`.
    pub fn normal(&mut self, sigma: f32) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma
    }
    pub fn u16s(&mut self, len: usize) -> Vec<u16> {
        (0..len).map(|_| self.next_u32() as u16).collect()
    }
}

/// One random EXL3 matrix with owned bytes.
pub(super) struct OwnedParts {
    pub k: usize,
    pub n: usize,
    pub halves: u16,
    pub codebook: Exl3Codebook,
    pub trellis: Vec<u8>,
    pub su: Vec<u8>,
    pub sv: Vec<u8>,
    pub signs: bool,
}

impl OwnedParts {
    /// Random trellis words and scales of magnitude 0.5 to 1.5 with random signs (or sign words).
    pub fn random(
        r: &mut Lcg,
        k: usize,
        n: usize,
        halves: u16,
        codebook: Exl3Codebook,
        signs: bool,
    ) -> Self {
        let trellis = super::format::u16_bytes(&r.u16s(trellis_bytes(k, n, halves) / 2));
        let mut scales = |len: usize| -> Vec<u8> {
            if signs {
                super::format::u16_bytes(&r.u16s(len / 16))
            } else {
                let v: Vec<u16> = (0..len)
                    .map(|_| {
                        let m = 0.5 + r.unit();
                        let s = if r.next_u32() & 1 == 1 { -m } else { m };
                        half::f16::from_f32(s).to_bits()
                    })
                    .collect();
                super::format::u16_bytes(&v)
            }
        };
        let su = scales(k);
        let sv = scales(n);
        Self {
            k,
            n,
            halves,
            codebook,
            trellis,
            su,
            sv,
            signs,
        }
    }

    pub fn parts(&self) -> Exl3Parts<'_> {
        Exl3Parts {
            k: self.k,
            n: self.n,
            halves: self.halves,
            codebook: self.codebook,
            trellis: &self.trellis,
            su: scales_of(self.signs, &self.su),
            sv: scales_of(self.signs, &self.sv),
            bias: None,
        }
    }
}

fn scales_of(signs: bool, bytes: &[u8]) -> Exl3Scales<'_> {
    if signs {
        Exl3Scales::Signs(bytes)
    } else {
        Exl3Scales::F16(bytes)
    }
}

#[test]
fn codebook_values_computed_by_hand() {
    // MCG, state 0: masked = 0x3B603B60, 0x3B60 + 0x3B60 = 0.921875 * 2 = 1.84375 = 0x3F60.
    assert_eq!(codebook_value(Exl3Codebook::Mcg, 0), 0x3F60);
    // 3INST, state 0: x = 64248484 = 0x03D45AA4 -> 0x38B4 (0.587890625) + 0x31C4 (0.18017578125)
    // = 0.76806640625, exact in fp16: 0x3A25.
    assert_eq!(codebook_value(Exl3Codebook::Inst3, 0), 0x3A25);
    // MUL1, state 0: 1024 * 0x1EEE (6.9296875) + 0xC931 (-10.3828125) = -3.453125 = 0xC2E8.
    assert_eq!(codebook_value(Exl3Codebook::Mul1, 0), 0xC2E8);
}

#[test]
fn the_f64_to_fp16_rounding_is_one_round_to_nearest_even() {
    use super::reference::f16_from_f64;
    for b in 0u16..0x7BFF {
        let v = half::f16::from_bits(b).to_f64();
        assert_eq!(f16_from_f64(v), b, "{b:#06x}");
        assert_eq!(f16_from_f64(-v), b | 0x8000, "-{b:#06x}");
        let next = half::f16::from_bits(b + 1).to_f64();
        let mid = (v + next) / 2.0;
        let even = if b % 2 == 0 { b } else { b + 1 };
        assert_eq!(f16_from_f64(mid), even, "the tie above {b:#06x}");
        let eps = (next - v) * 2f64.powi(-30);
        assert_eq!(
            f16_from_f64(mid + eps),
            b + 1,
            "just above the tie of {b:#06x}"
        );
        assert_eq!(f16_from_f64(mid - eps), b, "just below the tie of {b:#06x}");
    }
    assert_eq!(f16_from_f64(65504.0), 0x7BFF);
    assert_eq!(f16_from_f64(65519.99), 0x7BFF);
    assert_eq!(f16_from_f64(65520.0), 0x7C00);
    assert_eq!(f16_from_f64(f64::INFINITY), 0x7C00);
    assert_eq!(f16_from_f64(f64::NAN) & 0x7C00, 0x7C00);
}

#[test]
fn every_state_decodes_finite_and_one_rounding_holds_in_f32_too() {
    // The kernels round once in fp16 hardware; an fp32 route with one final fp16 rounding (the Metal
    // form) must give the same bits for all 65536 states, which shows no double rounding can occur.
    use super::reference::{FLIP, INST3_ADD, INST3_MUL, MASK, MCG_MUL, MUL1_MUL};
    let f16 = |b: u16| half::f16::from_bits(b).to_f32();
    for cb in [Exl3Codebook::Inst3, Exl3Codebook::Mcg, Exl3Codebook::Mul1] {
        for s in 0..=u16::MAX {
            let got = codebook_value(cb, s);
            assert!(half::f16::from_bits(got).is_finite(), "{cb:?} {s}");
            let s32 = u32::from(s);
            let via_f32 = match cb {
                Exl3Codebook::Mul1 => {
                    let x = s32.wrapping_mul(MUL1_MUL);
                    let h = (1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24))
                        as f32;
                    half::f16::from_f32(h.mul_add(f16(0x1EEE), f16(0xC931))).to_bits()
                }
                _ => {
                    let x = if cb == Exl3Codebook::Mcg {
                        s32.wrapping_mul(MCG_MUL)
                    } else {
                        s32.wrapping_mul(INST3_MUL).wrapping_add(INST3_ADD)
                    };
                    let m = (x & MASK) ^ FLIP;
                    half::f16::from_f32(f16((m & 0xFFFF) as u16) + f16((m >> 16) as u16)).to_bits()
                }
            };
            assert_eq!(got, via_f32, "{cb:?} state {s}");
        }
    }
}

#[test]
fn stream_ends_close_the_ring_and_grow() {
    for (halves, _) in all_widths() {
        assert_eq!(stream_end(halves, 255), 128 * halves as usize, "{halves}");
        for p in 1..256 {
            assert!(stream_end(halves, p) > stream_end(halves, p - 1));
        }
    }
    // 3 half-bits: the value windows advance 1, 2, 1, 2, ... bits.
    let e: Vec<usize> = (0..6).map(|p| stream_end(3, p)).collect();
    assert_eq!(e, vec![1, 3, 4, 6, 7, 9]);
}

#[test]
fn a_single_set_bit_appears_in_the_first_sixteen_windows_at_one_bit() {
    // 1 bit a value: window p is bits [p - 15, p] of the ring. Stream bit 0 is the MSB of word 0, i.e.
    // bit 15 of int16 element 1. It sits at position 15 - p of window p, so state p = 1 << p.
    let mut tile = vec![0u16; tile_elems(2)];
    tile[1] = 0x8000;
    let s = tile_states(2, &tile);
    for (p, &v) in s.iter().enumerate() {
        let want = if p < 16 { 1u16 << p } else { 0 };
        assert_eq!(v, want, "state {p}");
    }
}

#[test]
fn widths_follow_the_format_rules() {
    assert_eq!(all_widths().len(), 3 * 8 + 6);
    for cb in [Exl3Codebook::Inst3, Exl3Codebook::Mcg, Exl3Codebook::Mul1] {
        assert!(!width_supported(1, cb) && !width_supported(15, cb) && !width_supported(17, cb));
        assert!(width_supported(4, cb) && width_supported(16, cb));
    }
    assert!(width_supported(7, Exl3Codebook::Mul1));
    assert!(!width_supported(7, Exl3Codebook::Mcg) && !width_supported(7, Exl3Codebook::Inst3));
    assert_eq!(
        Exl3Codebook::from_marker(0xCBAC_1FED).unwrap(),
        Exl3Codebook::Mcg
    );
    assert_eq!(
        Exl3Codebook::from_marker(0x83DC_D12D).unwrap(),
        Exl3Codebook::Mul1
    );
    assert!(Exl3Codebook::from_marker(7).is_err());
}

#[test]
fn every_lane_reads_two_or_three_words() {
    for (halves, _) in all_widths() {
        let lw = lane_words(halves);
        assert!((2..=3).contains(&lw), "{halves} half-bits: {lw} words");
    }
    assert_eq!(lane_words(2), 2);
    assert_eq!(lane_words(16), 3);
}

#[test]
fn the_kernels_lane_extraction_equals_the_scalar_stream_at_every_width() {
    let mut r = Lcg(0x4558_4c33_5354_4154);
    for (halves, _) in all_widths() {
        let elems = tile_elems(halves);
        for pattern in 0..12 {
            let tile: Vec<u16> = (0..elems)
                .map(|j| match pattern {
                    0 => 0,
                    1 => 0xFFFF,
                    2 if j % 2 == 0 => 0x8001,
                    2 => 0x1234,
                    _ => r.next_u32() as u16,
                })
                .collect();
            let want = tile_states(halves, &tile);
            let words = tile_u32(&tile);
            for lane in 0..32 {
                let got = lane_states(halves, &words, lane);
                assert_eq!(
                    &got[..],
                    &want[8 * lane..8 * lane + 8],
                    "{halves} half-bits, pattern {pattern}, lane {lane}"
                );
            }
        }
    }
}

#[test]
fn placement_is_a_bijection_onto_the_tile() {
    let mut seen = [[false; 16]; 16];
    for p in 0..256 {
        let (row, col) = placement(p);
        assert!(!seen[row][col], "value {p}");
        seen[row][col] = true;
    }
    // Lane 0's values 0..3 are column 0, rows 0, 1, 8, 9: the B fragment k pairs of mma.m16n8k16.
    assert_eq!(
        (0..4).map(placement).collect::<Vec<_>>(),
        vec![(0, 0), (1, 0), (8, 0), (9, 0)]
    );
    assert_eq!(placement(4), (0, 8));
    assert_eq!(placement(8), (2, 0));
}

#[test]
fn unpack_puts_each_tile_at_its_place() {
    let mut r = Lcg(7);
    let (k, n, halves, cb) = (32, 48, 6, Exl3Codebook::Mul1);
    let trellis = r.u16s(trellis_bytes(k, n, halves) / 2);
    let wq = unpack(halves, cb, &trellis, k, n);
    let words = tile_elems(halves);
    for tk in 0..k / 16 {
        for tn in 0..n / 16 {
            let at = (tk * (n / 16) + tn) * words;
            let tile = decode_tile(halves, cb, &trellis[at..at + words]);
            for (i, row) in tile.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    assert_eq!(wq[(tk * 16 + i) * n + tn * 16 + j], v);
                }
            }
        }
    }
}

#[test]
fn the_fast_lane_unpack_equals_the_scalar_unpack() {
    use super::reference::{codebook_lut, unpack_tile_rows};
    let mut r = Lcg(8);
    for (halves, cb) in [
        (3, Exl3Codebook::Mul1),
        (8, Exl3Codebook::Mcg),
        (16, Exl3Codebook::Inst3),
    ] {
        let (k, n) = (64, 32);
        let trellis = r.u16s(trellis_bytes(k, n, halves) / 2);
        let want = unpack(halves, cb, &trellis, k, n);
        let lut = codebook_lut(cb);
        let mut got = vec![0u16; k * n];
        // in two row bands, as the microtest's threads do
        let (top, bottom) = got.split_at_mut(16 * n);
        unpack_tile_rows(halves, &lut, &trellis, n, 0, 1, top);
        unpack_tile_rows(halves, &lut, &trellis, n, 1, k / 16, bottom);
        assert_eq!(got, want, "{halves} {cb:?}");
    }
}

#[test]
fn sign_words_expand_bit_by_bit() {
    let words = super::format::u16_bytes(&[0b101, 0x8000]);
    let v = unpack_signs(&words, 32);
    assert_eq!(&v[..3], &[F16_MINUS_ONE, F16_ONE, F16_MINUS_ONE]);
    assert!(v[3..31].iter().all(|&b| b == F16_ONE));
    assert_eq!(v[31], F16_MINUS_ONE);
}

#[test]
fn the_admission_rules_refuse_each_broken_part() {
    let mut r = Lcg(11);
    let good = OwnedParts::random(&mut r, 256, 128, 8, Exl3Codebook::Mcg, false);
    good.parts().check().unwrap();
    assert_eq!(good.parts().bpw_x100(), 400);
    let signs = OwnedParts::random(&mut r, 256, 128, 7, Exl3Codebook::Mul1, true);
    signs.parts().check().unwrap();
    assert_eq!(signs.parts().suh_bits().len(), 256);

    let mut p = good.parts();
    p.k = 200;
    assert!(
        p.check().is_err(),
        "k not a whole number of Hadamard blocks"
    );
    let mut p = good.parts();
    p.halves = 7;
    assert!(
        p.check().is_err(),
        "odd width without mul1 (and the wrong trellis length)"
    );
    let mut p = good.parts();
    p.trellis = &good.trellis[2..];
    assert!(p.check().is_err(), "short trellis");
    let mut p = good.parts();
    p.su = Exl3Scales::F16(&good.su[..10]);
    assert!(p.check().is_err(), "short su");
    let mut p = good.parts();
    p.sv = Exl3Scales::Signs(&good.sv);
    assert!(p.check().is_err(), "fp16 bytes passed as sign words");
    let bias = vec![0u8; 256];
    let mut p = good.parts();
    p.bias = Some(&bias[..]);
    let err = p.check().unwrap_err().to_string();
    assert!(err.contains("bias"), "{err}");
}
