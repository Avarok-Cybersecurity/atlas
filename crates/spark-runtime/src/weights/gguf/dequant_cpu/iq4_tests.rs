// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-quants.c (dequantize_row_iq4_xs, dequantize_row_iq4_nl, dequantize_row_q5_K), as
// the literal reference loops below. Changes: translated from C to Rust for a test oracle.
#![allow(clippy::needless_range_loop)]

//! Responsibility: the oracle of the IQ4 decoders (port lane L19) and of the Q5_K decoder the Ada GPU dequant
//! mirrors.
//!
//! Three independent checks: (1) a second implementation that walks the block the way ggml's
//! `dequantize_row_*` does (pointer offsets, `half` for the f16 scale) compared BIT FOR BIT with
//! [`super::iq4`] on random blocks, scales and sub-normal `d` included; (2) hand-built blocks whose values
//! are known in closed form; (3) the loader entry points ([`super::GgmlType::from_id`],
//! [`super::dequant_to_f32`], [`super::to_bf16_bytes`]) on multi-block tensors.

use super::iq4::{IQ4_NL_BLOCK_BYTES, IQ4_XS_BLOCK_BYTES, KVALUES_IQ4NL, iq4_xs_sub_scale};
use super::{GgmlType, dequant_to_bf16, dequant_to_f32, f32_to_bf16_bits, supports, to_bf16_bytes};

/// xorshift64*: deterministic test bytes without a dev-dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    /// A finite f16 bit pattern: any sign, exponent 0 (zero and sub-normals) to 30, any mantissa.
    fn f16_finite(&mut self) -> u16 {
        let r = self.next();
        let sign = ((r >> 63) as u16) << 15;
        let exp = ((r >> 32) % 31) as u16;
        let mant = (r & 0x3ff) as u16;
        sign | (exp << 10) | mant
    }
}

fn f16(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// ggml `dequantize_row_iq4_xs`, walked as the C does: `qs` advances 16 bytes a sub-block, `y` 32 values.
fn ggml_iq4_xs(x: &[u8], y: &mut [f32]) {
    for (i, blk) in x.chunks_exact(IQ4_XS_BLOCK_BYTES).enumerate() {
        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let mut qs = 8usize;
        let mut yo = 256 * i;
        for ib in 0..8usize {
            let ls = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (i32::from((scales_h >> (2 * ib)) & 3) << 4);
            let dl = d * (ls - 32) as f32;
            for j in 0..16 {
                y[yo + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(blk[qs + j] & 0xf)]);
                y[yo + j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(blk[qs + j] >> 4)]);
            }
            yo += 32;
            qs += 16;
        }
    }
}

/// ggml `dequantize_row_iq4_nl`.
fn ggml_iq4_nl(x: &[u8], y: &mut [f32]) {
    for (i, blk) in x.chunks_exact(IQ4_NL_BLOCK_BYTES).enumerate() {
        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..16 {
            y[32 * i + j] = d * f32::from(KVALUES_IQ4NL[usize::from(blk[2 + j] & 0xf)]);
            y[32 * i + j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(blk[2 + j] >> 4)]);
        }
    }
}

/// ggml `get_scale_min_k4`.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// ggml `dequantize_row_q5_K`, the literal loop (`u1`, `u2` bit masks shifted two a chunk).
fn ggml_q5_k(x: &[u8], y: &mut [f32]) {
    for (i, blk) in x.chunks_exact(176).enumerate() {
        let d = f16(u16::from_le_bytes([blk[0], blk[1]]));
        let min = f16(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let mut ql = 48usize;
        let mut yo = 256 * i;
        let (mut is, mut u1, mut u2) = (0usize, 1u8, 2u8);
        for _ in (0..256).step_by(64) {
            let (sc, m) = scale_min_k4(is, scales);
            let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
            let (sc, m) = scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
            for l in 0..32 {
                let q = i32::from(blk[ql + l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                y[yo] = d1 * q as f32 - m1;
                yo += 1;
            }
            for l in 0..32 {
                let q = i32::from(blk[ql + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                y[yo] = d2 * q as f32 - m2;
                yo += 1;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// `n` random blocks of `bb` bytes whose leading f16 scale(s) are finite.
fn random_blocks(rng: &mut Rng, n: usize, bb: usize, f16_fields: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * bb);
    for _ in 0..n {
        for _ in 0..f16_fields {
            v.extend_from_slice(&rng.f16_finite().to_le_bytes());
        }
        for _ in 2 * f16_fields..bb {
            v.push(rng.byte());
        }
    }
    v
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn iq4_xs_matches_the_ggml_loop_bit_for_bit_on_random_blocks() {
    let mut rng = Rng(0x1914_0023_cafe_f00d);
    let n = 257;
    let raw = random_blocks(&mut rng, n, IQ4_XS_BLOCK_BYTES, 1);
    let mut want = vec![0f32; n * 256];
    ggml_iq4_xs(&raw, &mut want);
    let mut got = vec![0f32; n * 256];
    dequant_to_f32(GgmlType::Iq4Xs, &raw, n * 256, &mut got).unwrap();
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn iq4_nl_matches_the_ggml_loop_bit_for_bit_on_random_blocks() {
    let mut rng = Rng(0x1914_0020_0bad_beef);
    let n = 333;
    let raw = random_blocks(&mut rng, n, IQ4_NL_BLOCK_BYTES, 1);
    let mut want = vec![0f32; n * 32];
    ggml_iq4_nl(&raw, &mut want);
    let mut got = vec![0f32; n * 32];
    dequant_to_f32(GgmlType::Iq4Nl, &raw, n * 32, &mut got).unwrap();
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn q5_k_matches_the_ggml_loop_bit_for_bit_on_random_blocks() {
    // The Ada GPU Q5_K dequant mirrors this decoder; pin it to ggml's loop first.
    let mut rng = Rng(0x1914_0013_5eed_0005);
    let n = 129;
    let raw = random_blocks(&mut rng, n, 176, 2);
    let mut want = vec![0f32; n * 256];
    ggml_q5_k(&raw, &mut want);
    let mut got = vec![0f32; n * 256];
    dequant_to_f32(GgmlType::Q5K, &raw, n * 256, &mut got).unwrap();
    assert_eq!(bits(&got), bits(&want));
}

/// An IQ4_XS block with scale `d` (f16 bits), sub-block scales `ls[ib]` (0..64) and codes `code(i)` for
/// weight `i` (0..256).
fn iq4_xs_block(d: u16, ls: [u8; 8], code: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut b = vec![0u8; IQ4_XS_BLOCK_BYTES];
    b[0..2].copy_from_slice(&d.to_le_bytes());
    let mut scales_h = 0u16;
    for (ib, &s) in ls.iter().enumerate() {
        b[4 + ib / 2] |= (s & 0x0F) << (4 * (ib % 2));
        scales_h |= u16::from(s >> 4) << (2 * ib);
    }
    b[2..4].copy_from_slice(&scales_h.to_le_bytes());
    for ib in 0..8 {
        for j in 0..16 {
            let lo = code(32 * ib + j) & 0x0F;
            let hi = code(32 * ib + j + 16) & 0x0F;
            b[8 + 16 * ib + j] = lo | (hi << 4);
        }
    }
    b
}

#[test]
fn iq4_xs_closed_form_values() {
    // d = 1.0; ls = 33 everywhere gives dl = 1, so every value is its code-book entry.
    let blk = iq4_xs_block(0x3C00, [33; 8], |i| (i % 16) as u8);
    let mut out = [0f32; 256];
    dequant_to_f32(GgmlType::Iq4Xs, &blk, 256, &mut out).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, f32::from(KVALUES_IQ4NL[i % 16]), "weight {i}");
    }

    // Per-sub-block scales over the whole 6-bit range: ls = 0 (dl = -32 d), 32 (zero), 63 (31 d).
    let ls = [0u8, 1, 31, 32, 33, 47, 62, 63];
    let blk = iq4_xs_block(0x3800, ls, |_| 15); // d = 0.5, code 15 = +113
    dequant_to_f32(GgmlType::Iq4Xs, &blk, 256, &mut out).unwrap();
    for (ib, &s) in ls.iter().enumerate() {
        let want = 0.5 * (i32::from(s) - 32) as f32 * 113.0;
        for v in &out[32 * ib..32 * ib + 32] {
            assert_eq!(*v, want, "sub-block {ib}");
        }
    }
}

#[test]
fn sub_scale_unpack_covers_every_bit() {
    for ib in 0..8 {
        for s in 0u8..64 {
            let mut ls = [0u8; 8];
            ls[ib] = s;
            let blk = iq4_xs_block(0, ls, |_| 0);
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            for jb in 0..8 {
                let want = if jb == ib { i32::from(s) } else { 0 };
                assert_eq!(
                    iq4_xs_sub_scale(scales_h, &blk[4..8], jb),
                    want,
                    "ib {ib} jb {jb} s {s}"
                );
            }
        }
    }
}

#[test]
fn iq4_nl_closed_form_values() {
    let mut blk = vec![0u8; IQ4_NL_BLOCK_BYTES];
    blk[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // d = 2.0
    for j in 0..16 {
        blk[2 + j] = (j as u8) | ((15 - j as u8) << 4);
    }
    let mut out = [0f32; 32];
    dequant_to_f32(GgmlType::Iq4Nl, &blk, 32, &mut out).unwrap();
    for j in 0..16 {
        assert_eq!(out[j], 2.0 * f32::from(KVALUES_IQ4NL[j]));
        assert_eq!(out[j + 16], 2.0 * f32::from(KVALUES_IQ4NL[15 - j]));
    }
}

#[test]
fn ids_sizes_and_loader_entry_points() {
    assert_eq!(GgmlType::from_id(23, 128).unwrap(), GgmlType::Iq4Xs);
    assert_eq!(GgmlType::from_id(20, 128).unwrap(), GgmlType::Iq4Nl);
    assert!(supports(23) && supports(20));
    assert_eq!(GgmlType::Iq4Xs.block_size(), 256);
    assert_eq!(GgmlType::Iq4Xs.block_bytes().unwrap(), 136);
    assert_eq!(GgmlType::Iq4Nl.block_size(), 32);
    assert_eq!(GgmlType::Iq4Nl.block_bytes().unwrap(), 18);

    // Short input and a partial block are errors, never a panic or a silent truncation.
    let mut out = vec![0f32; 512];
    assert!(dequant_to_f32(GgmlType::Iq4Xs, &[0u8; 136], 512, &mut out).is_err());
    assert!(dequant_to_f32(GgmlType::Iq4Xs, &[0u8; 272], 300, &mut out).is_err());

    // The loader's BF16 host bytes are the RNE rounding of the f32 values.
    let mut rng = Rng(0x0b16_0023_0000_0001);
    let raw = random_blocks(&mut rng, 3, IQ4_XS_BLOCK_BYTES, 1);
    let mut f = vec![0f32; 768];
    dequant_to_f32(GgmlType::Iq4Xs, &raw, 768, &mut f).unwrap();
    let host = to_bf16_bytes(23, 128, &raw, 768).unwrap();
    let mut b16 = vec![0u16; 768];
    dequant_to_bf16(GgmlType::Iq4Xs, &raw, 768, &mut b16).unwrap();
    for (i, v) in f.iter().enumerate() {
        let want = f32_to_bf16_bits(*v);
        assert_eq!(b16[i], want, "value {i}");
        assert_eq!(
            u16::from_le_bytes([host[2 * i], host[2 * i + 1]]),
            want,
            "byte pair {i}"
        );
    }
}
