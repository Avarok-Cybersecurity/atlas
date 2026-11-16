// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/exl3_format.zig:5-123 (codebook constants, codebookValue, streamEnds, streamBit,
// tileStates, tilePlacement, unpack) and zig/kernels/cuda/exl3_decode.cuh:14-86 and 129-131 (tile_words,
// stream_end, lane_words, lane_start, window16, lane_states, value_row, value_col), after ExLlamaV3
// (turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig and CUDA C++ to Rust; the CUDA lane extraction is mirrored on
// the host with an explicit funnel shift so the CPU tests can prove it equal to the scalar stream decoder.

//! Responsibility: the CPU decoder of the EXL3 trellis format, written twice on purpose:
//!
//! * the SCALAR oracle ([`tile_states`], bit by bit from the format's definition, as TensorFold's
//!   `exl3_format.zig` does), with the codebooks ([`codebook_value`]) and the fragment placement, which
//!   together give the decoded matrix `Wq` ([`unpack`]);
//! * the GPU's LANE extraction ([`lane_states`]: each lane's 2 or 3 words, a funnel shift and 16-bit
//!   windows, line for line `exl3_decode_ada.cuh`), so a CPU test pins the kernel's bit arithmetic
//!   against the scalar oracle at every width.
//!
//! The Hadamard side (rotation, dequant, the GEMV oracle and its bound) is `reference_math.rs`.

use super::format::{Exl3Codebook, Exl3Parts, tile_elems};

pub const INST3_MUL: u32 = 89_226_354;
pub const INST3_ADD: u32 = 64_248_484;
pub const MCG_MUL: u32 = 0xCBAC_1FED;
pub const MUL1_MUL: u32 = 0x83DC_D12D;
pub const MASK: u32 = 0x8FFF_8FFF;
pub const FLIP: u32 = 0x3B60_3B60;
/// fp16 bit patterns of the mul1 scale and bias.
pub const MUL1_SCALE_BITS: u16 = 0x1EEE;
pub const MUL1_BIAS_BITS: u16 = 0xC931;

fn f16_f64(bits: u16) -> f64 {
    half::f16::from_bits(bits).to_f64()
}

/// Round an f64 to the nearest fp16, ties to even, in ONE step from the full 53-bit significand. (The
/// `half` crate's `f16::from_f64` may go through f32 on F16C hosts, which can round twice.)
pub fn f16_from_f64(v: f64) -> u16 {
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a.is_nan() {
        return sign | 0x7E00;
    }
    if a >= 65520.0 {
        return sign | 0x7C00; // at or past the tie above the largest finite fp16: infinity
    }
    // The f64 exponent (f64 subnormals are far below the fp16 range and fall to the subnormal step).
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let e = e.max(-14);
    // Units of the fp16 step at this exponent: exact, the step is a power of two.
    let q = (a / 2f64.powi(e - 10)).round_ties_even() as u32;
    if q < 1024 {
        return sign | q as u16; // subnormal (only reachable at e = -14)
    }
    let (mut m, mut exp) = (q, e + 15);
    if m == 2048 {
        m = 1024;
        exp += 1;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | (m - 1024) as u16
}

/// The fp16 bits of one 16-bit state under `cb`, with exactly one fp16 rounding (`codebookValue`). The
/// float64 sums are exact, so this equals the kernel's `__hadd2` (3INST, MCG) and `__hfma2` (MUL1).
pub fn codebook_value(cb: Exl3Codebook, state: u16) -> u16 {
    let s = u32::from(state);
    let x = match cb {
        Exl3Codebook::Mul1 => s.wrapping_mul(MUL1_MUL),
        Exl3Codebook::Mcg => s.wrapping_mul(MCG_MUL),
        Exl3Codebook::Inst3 => s.wrapping_mul(INST3_MUL).wrapping_add(INST3_ADD),
    };
    if cb == Exl3Codebook::Mul1 {
        let bytes = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
        let h = f64::from(1024 + bytes);
        return f16_from_f64(h * f16_f64(MUL1_SCALE_BITS) + f16_f64(MUL1_BIAS_BITS));
    }
    let masked = (x & MASK) ^ FLIP;
    let lo = f16_f64((masked & 0xFFFF) as u16);
    let hi = f16_f64((masked >> 16) as u16);
    f16_from_f64(lo + hi)
}

/// E(p) for p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive).
pub fn stream_end(halves: u16, p: usize) -> usize {
    let k2 = halves as usize;
    if k2.is_multiple_of(2) {
        (p + 1) * k2 / 2
    } else {
        ((p + 1) * k2 - ((p + 1) & 1)) / 2
    }
}

/// Bit `index` of the circular stream: MSB first within little-endian pairs of int16 words.
fn stream_bit(tile: &[u16], index: usize) -> u32 {
    let w = index / 32;
    let within = 31 - (index % 32);
    let word = u32::from(tile[2 * w]) | (u32::from(tile[2 * w + 1]) << 16);
    (word >> within) & 1
}

/// The 16-bit state of each of a tile's 256 values, in stream order, bit by bit (`tileStates`).
pub fn tile_states(halves: u16, tile: &[u16]) -> [u16; 256] {
    assert_eq!(
        tile.len(),
        tile_elems(halves),
        "a tile at {halves} half-bits"
    );
    let ring = 128 * halves as usize;
    let mut out = [0u16; 256];
    for (p, o) in out.iter_mut().enumerate() {
        let start = (stream_end(halves, p) + ring - 16) % ring;
        let mut s = 0u32;
        for b in 0..16 {
            s = (s << 1) | stream_bit(tile, (start + b) % ring);
        }
        *o = s as u16;
    }
    out
}

/// `(row, col)` = `(k, n)` in the 16 x 16 tile of stream value `p` (lane `p / 8`, value `p % 8`).
pub fn placement(p: usize) -> (usize, usize) {
    let (lane, j) = (p / 8, p % 8);
    (value_row(lane, j), value_col(lane, j))
}

pub fn value_row(lane: usize, j: usize) -> usize {
    2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)
}

pub fn value_col(lane: usize, j: usize) -> usize {
    (lane >> 2) + 8 * (j >> 2)
}

/// The decoded tile, fp16 bits `[16][16]` (row = k, col = n).
pub fn decode_tile(halves: u16, cb: Exl3Codebook, tile: &[u16]) -> [[u16; 16]; 16] {
    let states = tile_states(halves, tile);
    let mut out = [[0u16; 16]; 16];
    for (p, &s) in states.iter().enumerate() {
        let (r, c) = placement(p);
        out[r][c] = codebook_value(cb, s);
    }
    out
}

/// `Wq [k, n]` fp16 bits, row-major, in the rotated domain, from the trellis int16 `[k/16, n/16, words]`.
pub fn unpack(halves: u16, cb: Exl3Codebook, trellis: &[u16], k: usize, n: usize) -> Vec<u16> {
    let words = tile_elems(halves);
    let (kt, nt) = (k / 16, n / 16);
    assert_eq!(trellis.len(), kt * nt * words, "trellis length");
    let mut out = vec![0u16; k * n];
    for tk in 0..kt {
        for tn in 0..nt {
            let at = (tk * nt + tn) * words;
            let tile = decode_tile(halves, cb, &trellis[at..at + words]);
            for (r, row) in tile.iter().enumerate() {
                let base = (tk * 16 + r) * n + tn * 16;
                out[base..base + 16].copy_from_slice(row);
            }
        }
    }
    out
}

/// [`unpack`] of a parts view.
pub fn unpack_parts(p: &Exl3Parts<'_>) -> Vec<u16> {
    unpack(p.halves, p.codebook, &p.trellis_u16(), p.k, p.n)
}

// ---- the GPU's lane extraction (exl3_decode_ada.cuh), mirrored ----------------------------------------

/// 32-bit words of a tile.
pub fn tile_words(halves: u16) -> usize {
    4 * halves as usize
}

/// Words a lane reads (`lane_words<K2>()`).
pub fn lane_words(halves: u16) -> usize {
    let k2 = halves as isize;
    let mut most = 0isize;
    for l in 0..32usize {
        let e0 = stream_end(halves, 8 * l) as isize;
        let first = (e0 - 16 + 128 * k2).rem_euclid(32);
        let need = first + stream_end(halves, 8 * l + 7) as isize - e0 + 16;
        most = most.max(need);
    }
    (most as usize).div_ceil(32)
}

/// `(word, offset)` of the lane's first window (`lane_start<K2>`).
pub fn lane_start(halves: u16, lane: usize) -> (usize, u32) {
    let k2 = halves as usize;
    let first = 4 * lane * k2 + k2 / 2 + 128 * k2 - 16;
    ((first >> 5) % tile_words(halves), (first & 31) as u32)
}

/// `__funnelshift_l(lo, hi, s)`: the high 32 bits of `(hi:lo) << (s & 31)`.
pub fn funnelshift_l(lo: u32, hi: u32, s: u32) -> u32 {
    let wide = (u64::from(hi) << 32) | u64::from(lo);
    ((wide << (s & 31)) >> 32) as u32
}

/// `window16`: bits `[d, d + 16)` of the 96-bit stream `hi:mid:lo`.
pub fn window16(hi: u32, mid: u32, lo: u32, d: u32) -> u32 {
    if d <= 16 {
        (hi >> (16 - d)) & 0xffff
    } else if d < 32 {
        funnelshift_l(mid, hi, d) >> 16
    } else if d <= 48 {
        (mid >> (48 - d)) & 0xffff
    } else {
        funnelshift_l(lo, mid, d - 32) >> 16
    }
}

/// The tile's int16 elements as the kernels' `u32` words.
pub fn tile_u32(tile: &[u16]) -> Vec<u32> {
    tile.chunks_exact(2)
        .map(|c| u32::from(c[0]) | (u32::from(c[1]) << 16))
        .collect()
}

/// The 16-bit states of lane `lane`'s 8 values (stream values `8 lane + j`), the way the kernel extracts
/// them from its 2 or 3 words (`load_lane_words` then `lane_states`).
pub fn lane_states(halves: u16, words: &[u32], lane: usize) -> [u16; 8] {
    let tw = tile_words(halves);
    let lw = lane_words(halves);
    assert!(
        lw <= 3,
        "a lane spans at most 3 words, {halves} half-bits needs {lw}"
    );
    let (word, offset) = lane_start(halves, lane);
    let mut w = [0u32; 3];
    for (i, wi) in w.iter_mut().enumerate().take(lw) {
        *wi = words[(word + i) % tw];
    }
    let c = if lw > 2 { w[2] } else { 0 };
    let hi = funnelshift_l(w[1], w[0], offset);
    let mid = funnelshift_l(c, w[1], offset);
    let lo = c << offset;
    let e0 = stream_end(halves, 0) as u32;
    let mut s = [0u16; 8];
    for (j, v) in s.iter_mut().enumerate() {
        *v = window16(hi, mid, lo, stream_end(halves, j) as u32 - e0) as u16;
    }
    s
}

/// [`codebook_value`] of every state (a 128 KiB table, for decoding large matrices on the host).
pub fn codebook_lut(cb: Exl3Codebook) -> Vec<u16> {
    (0..=u16::MAX).map(|s| codebook_value(cb, s)).collect()
}

/// Rows `[16 tk0, 16 tk1)` of `Wq [k, n]` into `out` (`(tk1 - tk0) * 16 * n` fp16 bits), through the
/// kernels' lane extraction and a codebook table: the same values as [`unpack`] (the tests pin the lane
/// extraction to the scalar stream), fast enough for the microtest's 27B shapes.
pub fn unpack_tile_rows(
    halves: u16,
    lut: &[u16],
    trellis: &[u16],
    n: usize,
    tk0: usize,
    tk1: usize,
    out: &mut [u16],
) {
    let elems = tile_elems(halves);
    let nt = n / 16;
    assert_eq!(out.len(), (tk1 - tk0) * 16 * n, "output rows");
    for tk in tk0..tk1 {
        for tn in 0..nt {
            let at = (tk * nt + tn) * elems;
            let words = tile_u32(&trellis[at..at + elems]);
            for lane in 0..32 {
                let s = lane_states(halves, &words, lane);
                for (j, &st) in s.iter().enumerate() {
                    let row = (tk - tk0) * 16 + value_row(lane, j);
                    out[row * n + tn * 16 + value_col(lane, j)] = lut[st as usize];
                }
            }
        }
    }
}
