// SPDX-License-Identifier: AGPL-3.0-only
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::identity_op,
    clippy::cast_lossless
)]
//! Per-block CPU dequant routines, one per ggml type. Every function consumes
//! exactly one block's bytes (`blk`) and writes `block_size` f32 outputs into
//! `out`. Index arithmetic mirrors llama.cpp `ggml-quants.c` verbatim.

use super::{bf16_to_f32, rd_f16};

// ---- passthrough / widen (whole-run) --------------------------------------

pub(super) fn widen_f32(src: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = f32::from_le_bytes([src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]]);
    }
}

pub(super) fn widen_f16(src: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = rd_f16(src, i * 2);
    }
}

pub(super) fn widen_bf16(src: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = bf16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
    }
}

// ---- legacy QK=32 quants --------------------------------------------------

/// Q8_0 — `{ f16 d; i8 qs[32] }`, 34 B.
pub(super) fn dequant_q8_0(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    for j in 0..32 {
        out[j] = (blk[2 + j] as i8) as f32 * d;
    }
}

/// Q8_1 — `{ f16 d; f16 s; i8 qs[32] }`, 36 B. `s` is a dot-product sum,
/// irrelevant to weight dequant.
pub(super) fn dequant_q8_1(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    for j in 0..32 {
        out[j] = (blk[4 + j] as i8) as f32 * d;
    }
}

/// Q4_0 — `{ f16 d; u8 qs[16] }`, 18 B. Symmetric, nibble − 8. Byte `j` holds
/// element `j` (low) and `j+16` (high).
pub(super) fn dequant_q4_0(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    for j in 0..16 {
        out[j] = ((blk[2 + j] & 0x0F) as i32 - 8) as f32 * d;
        out[j + 16] = ((blk[2 + j] >> 4) as i32 - 8) as f32 * d;
    }
}

/// Q4_1 — `{ f16 d; f16 m; u8 qs[16] }`, 20 B. Asymmetric, `y = nibble·d + m`.
pub(super) fn dequant_q4_1(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let m = rd_f16(blk, 2);
    for j in 0..16 {
        out[j] = (blk[4 + j] & 0x0F) as f32 * d + m;
        out[j + 16] = (blk[4 + j] >> 4) as f32 * d + m;
    }
}

/// Q5_0 — `{ f16 d; u8 qh[4]; u8 qs[16] }`, 22 B. 5th bit from packed u32 qh;
/// value − 16. Note asymmetric extraction (`>>j` low, `>>(j+12)` high).
pub(super) fn dequant_q5_0(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
    let qs = &blk[6..22];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as i32;
        let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
        out[j] = (((qs[j] & 0x0F) as i32 | xh0) - 16) as f32 * d;
        out[j + 16] = (((qs[j] >> 4) as i32 | xh1) - 16) as f32 * d;
    }
}

/// Q5_1 — `{ f16 d; f16 m; u8 qh[4]; u8 qs[16] }`, 24 B. Asymmetric with min.
pub(super) fn dequant_q5_1(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let m = rd_f16(blk, 2);
    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
    let qs = &blk[8..24];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as i32;
        let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
        out[j] = ((qs[j] & 0x0F) as i32 | xh0) as f32 * d + m;
        out[j + 16] = ((qs[j] >> 4) as i32 | xh1) as f32 * d + m;
    }
}

// ---- K-quants (QK_K = 256) ------------------------------------------------

/// 6-bit packed scale/min unpack shared by Q4_K and Q5_K.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Q6_K — `{ u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }`, 210 B.
pub(super) fn dequant_q6_k(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 208);
    let scale = |i: usize| (blk[192 + i] as i8) as f32;
    for n in 0..2 {
        let ql = &blk[n * 64..];
        let qh = &blk[128 + n * 32..];
        let sco = n * 8;
        let yo = n * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0F) as i32 | ((((qh[l] >> 0) & 3) as i32) << 4)) - 32;
            let q2 = ((ql[l + 32] & 0x0F) as i32 | ((((qh[l] >> 2) & 3) as i32) << 4)) - 32;
            let q3 = ((ql[l] >> 4) as i32 | ((((qh[l] >> 4) & 3) as i32) << 4)) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | ((((qh[l] >> 6) & 3) as i32) << 4)) - 32;
            out[yo + l] = d * scale(sco + is) * q1 as f32;
            out[yo + l + 32] = d * scale(sco + is + 2) * q2 as f32;
            out[yo + l + 64] = d * scale(sco + is + 4) * q3 as f32;
            out[yo + l + 96] = d * scale(sco + is + 6) * q4 as f32;
        }
    }
}

/// Q4_K — `{ f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }`, 144 B.
pub(super) fn dequant_q4_k(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let dmin = rd_f16(blk, 2);
    let scales = &blk[4..16];
    let qs = &blk[16..144];

    let mut y = 0usize;
    let mut is = 0usize;
    let mut qoff = 0usize;
    for _ in 0..(256 / 64) {
        let (sc, m) = get_scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * m as f32);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * m as f32);
        for l in 0..32 {
            out[y] = d1 * (qs[qoff + l] & 0x0F) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            out[y] = d2 * (qs[qoff + l] >> 4) as f32 - m2;
            y += 1;
        }
        qoff += 32;
        is += 2;
    }
}

/// Q5_K — `{ f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }`, 176 B.
pub(super) fn dequant_q5_k(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let dmin = rd_f16(blk, 2);
    let scales = &blk[4..16];
    let qh = &blk[16..48];
    let ql = &blk[48..176];

    let mut y = 0usize;
    let mut is = 0usize;
    let mut qoff = 0usize;
    let (mut u1, mut u2) = (1u8, 2u8);
    for _ in 0..(256 / 64) {
        let (sc, m) = get_scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * m as f32);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * m as f32);
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out[y] = d1 * ((ql[qoff + l] & 0x0F) as i32 + hi) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[y] = d2 * ((ql[qoff + l] >> 4) as i32 + hi) as f32 - m2;
            y += 1;
        }
        qoff += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Q2_K — `{ u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }`, 84 B.
/// Field order differs: scales/qs first, d/dmin last.
pub(super) fn dequant_q2_k(blk: &[u8], out: &mut [f32]) {
    let scales = &blk[0..16];
    let qs = &blk[16..80];
    let d = rd_f16(blk, 80);
    let dmin = rd_f16(blk, 82);

    let mut y = 0usize;
    let mut is = 0usize;
    for n in (0..256).step_by(128) {
        let q = &qs[(n / 128) * 32..];
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = scales[is];
            is += 1;
            let (dl, ml) = (d * (sc & 0x0F) as f32, dmin * (sc >> 4) as f32);
            for l in 0..16 {
                out[y] = dl * ((q[l] >> shift) & 3) as f32 - ml;
                y += 1;
            }
            let sc = scales[is];
            is += 1;
            let (dl, ml) = (d * (sc & 0x0F) as f32, dmin * (sc >> 4) as f32);
            for l in 0..16 {
                out[y] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                y += 1;
            }
            shift += 2;
        }
    }
}

/// Q3_K — `{ u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }`, 110 B.
pub(super) fn dequant_q3_k(blk: &[u8], out: &mut [f32]) {
    let hmask = &blk[0..32];
    let qs = &blk[32..96];
    let raw = &blk[96..108];
    let d_all = rd_f16(blk, 108);

    const KM1: u32 = 0x0303_0303;
    const KM2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    aux[1] = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    aux[2] = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
    aux[3] = ((aux[1] >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
    aux[0] = (aux[0] & KM2) | (((tmp >> 0) & KM1) << 4);
    aux[1] = (aux[1] & KM2) | (((tmp >> 2) & KM1) << 4);
    let mut sc = [0i8; 16];
    let b = [
        aux[0].to_le_bytes(),
        aux[1].to_le_bytes(),
        aux[2].to_le_bytes(),
        aux[3].to_le_bytes(),
    ];
    for i in 0..16 {
        sc[i] = b[i / 4][i % 4] as i8;
    }

    let mut y = 0usize;
    let mut is = 0usize;
    let mut m: u8 = 1;
    for n in (0..256).step_by(128) {
        let q = &qs[(n / 128) * 32..];
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (sc[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let h = if hmask[l] & m != 0 { 0 } else { 4 };
                out[y] = dl * (((q[l] >> shift) & 3) as i32 - h) as f32;
                y += 1;
            }
            let dl = d_all * (sc[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let h = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                out[y] = dl * (((q[l + 16] >> shift) & 3) as i32 - h) as f32;
                y += 1;
            }
            shift += 2;
            m <<= 1;
        }
    }
}

/// Q8_K — `{ f32 d; i8 qs[256]; i16 bsums[16] }`, 292 B. `d` is f32; bsums
/// are unused for dequant.
pub(super) fn dequant_q8_k(blk: &[u8], out: &mut [f32]) {
    let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
    for j in 0..256 {
        out[j] = (blk[4 + j] as i8) as f32 * d;
    }
}

// ---- ternary types --------------------------------------------------------

/// TQ2_0 (id 35) — `{ u8 qs[64]; f16 d }`, 66 B. Planar 2-bit, scale at END.
pub(super) fn dequant_tq2_0(blk: &[u8], out: &mut [f32]) {
    let qs = &blk[0..64];
    let d = rd_f16(blk, 64);
    let mut y = 0usize;
    for j in (0..64).step_by(32) {
        for l in 0..4 {
            for m in 0..32 {
                let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                out[y] = (q - 1) as f32 * d;
                y += 1;
            }
        }
    }
}

/// TQ1_0 (id 34) — `{ u8 qs[48]; u8 qh[4]; f16 d }`, 54 B. Base-3 packing;
/// the `(u8)(byte*pow3[n])` truncation before `*3>>8` is load-bearing.
pub(super) fn dequant_tq1_0(blk: &[u8], out: &mut [f32]) {
    const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];
    let qs = &blk[0..48];
    let qh = &blk[48..52];
    let d = rd_f16(blk, 52);
    let mut y = 0usize;
    let dig = |byte: u8, n: usize| -> i32 {
        let q = (byte as u16).wrapping_mul(POW3[n]) & 0xFF;
        ((q * 3) >> 8) as i32
    };
    for j in (0..32).step_by(32) {
        for n in 0..5 {
            for m in 0..32 {
                out[y] = (dig(qs[j + m], n) - 1) as f32 * d;
                y += 1;
            }
        }
    }
    for j in (32..48).step_by(16) {
        for n in 0..5 {
            for m in 0..16 {
                out[y] = (dig(qs[j + m], n) - 1) as f32 * d;
                y += 1;
            }
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            out[y] = (dig(qh[j], n) - 1) as f32 * d;
            y += 1;
        }
    }
}

/// PrismML Q2_0 (id 42), parameterized on group size `g` ∈ {128, 64}.
/// Block = `[f16 d (FRONT)][u8 qs[g/4]]`. Contiguous low-bits-first codes;
/// `value = (code − 1)·d`. Symmetric, no min.
pub(super) fn dequant_q2_0_gn(blk: &[u8], out: &mut [f32], g: usize) {
    let d = rd_f16(blk, 0);
    let qs = &blk[2..2 + g / 4];
    for j in 0..g {
        let code = ((qs[j / 4] >> (2 * (j % 4))) & 3) as i32;
        out[j] = (code - 1) as f32 * d;
    }
}
