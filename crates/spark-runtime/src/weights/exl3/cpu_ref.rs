// SPDX-License-Identifier: AGPL-3.0-only

//! CPU reference implementation. Bit-exact to the GPU kernel — every fp16
//! operation replicates the kernel's op order and per-op rounding.
//!
//! Child module of `weights/exl3.rs` (`exl3::cpu_ref`), split out for the
//! ≤500 LoC cap; the module path is unchanged.

use half::f16;

use super::Exl3Codebook;

const R_SCALE: f32 = 0.088_388_35_f32; // 0.08838834764831845f in the kernel

fn f16b(bits: u16) -> f16 {
    f16::from_bits(bits)
}

/// f16 add with CUDA `__hadd` semantics (exact in f32, one rounding).
fn hadd(a: f16, b: f16) -> f16 {
    f16::from_f32(a.to_f32() + b.to_f32())
}
fn hsub(a: f16, b: f16) -> f16 {
    f16::from_f32(a.to_f32() - b.to_f32())
}
fn hmul(a: f16, b: f16) -> f16 {
    f16::from_f32(a.to_f32() * b.to_f32())
}
/// f16 fused multiply-add with CUDA `__hfma` semantics: a*b+c computed
/// exactly, ONE rounding. f64 holds the exact product+sum of f16 inputs.
fn hfma(a: f16, b: f16, c: f16) -> f16 {
    f16::from_f64(a.to_f64() * b.to_f64() + c.to_f64())
}

/// Decode one 16-bit code window through codebook `cb`.
/// (`decode_3inst` in upstream codebook.cuh; the lop3 imm 0x6a with
/// those masks is `(x & 0x8fff8fff) ^ 0x3b603b60`.)
pub fn decode_code(w: u16, cb: Exl3Codebook) -> f16 {
    let w = w as u32;
    match cb {
        Exl3Codebook::Inst3 => {
            let x = w.wrapping_mul(89226354).wrapping_add(64248484);
            let x = (x & 0x8fff8fff) ^ 0x3b603b60;
            hadd(f16b(x as u16), f16b((x >> 16) as u16))
        }
        Exl3Codebook::Mcg => {
            let x = w.wrapping_mul(0xCBAC1FED);
            let x = (x & 0x8fff8fff) ^ 0x3b603b60;
            hadd(f16b(x as u16), f16b((x >> 16) as u16))
        }
        Exl3Codebook::Mul1 => {
            let x = w.wrapping_mul(0x83DCD12D);
            // __dp4a(x, 0x01010101, 0x6400): sum of the 4 bytes + bias.
            let sum: u32 =
                0x6400 + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
            let k_inv = f16b(0x1eee); //  0.00677 = 1/147.7
            let k_bias = f16b(0xc931); // -10.39
            hfma(f16b(sum as u16), k_inv, k_bias)
        }
    }
}

/// Decode one 16x16 tile (256 codes at `16*k` packed u16 words) into
/// `tile[row][col]`.
///
/// Bitstream: the u16 words pair into little-endian u32s
/// (`u32[i] = (u16[2i+1] << 16) | u16[2i]`), and stream bit `x` is bit
/// `31 - x%32` of u32 `x/32` — MSB-first WITHIN each u32, ascending
/// u32 order. (Derived from the kernel's funnel-shift indexing:
/// `s0 = (i1+1)*32 - b1` aligns the window END to the u32's LOW bits,
/// which puts earlier stream bits at higher bit positions.)
///
/// Code `t`'s decode window is stream bits `[(t+1)*k - 16, (t+1)*k)`
/// mod `256*k` — K fresh bits below the previous window (the trellis
/// overlap); the window value is read MSB-first: bit 15 of `w` is the
/// OLDEST stream bit in the window.
///
/// Position mapping `t -> (row, col)` follows the m16n8k16 B-fragment
/// layout the packer wrote (verified against the GPU kernel by the
/// parity example): with `l = t/8`, `j = t%8`:
///   row = (l%4)*2 + (j&1) + ((j>>1)&1)*8
///   col = ((l & !4)/8)*2 + ((l>>2)&1) + ((j>>2)&1)*8
pub fn decode_tile(packed: &[u16], k: u32, cb: Exl3Codebook) -> [[f16; 16]; 16] {
    let k = k as usize;
    assert_eq!(packed.len(), 16 * k);
    let total_bits = 256 * k;
    let stream_bit = |idx: usize| -> u16 {
        let idx = idx % total_bits;
        let w32 = idx / 32;
        let bit = 31 - (idx % 32); // MSB-first within the u32
        let word = if bit >= 16 {
            packed[w32 * 2 + 1] // u32 high half = second u16
        } else {
            packed[w32 * 2]
        };
        (word >> (bit % 16)) & 1
    };
    let mut tile = [[f16::from_f32(0.0); 16]; 16];
    for t in 0..256 {
        let end = (t + 1) * k + total_bits; // + total_bits avoids underflow
        let mut w: u16 = 0;
        for b in 0..16 {
            // w bit 15 = oldest stream bit of the window
            w |= stream_bit(end - 16 + b) << (15 - b);
        }
        let l = t / 8;
        let j = t % 8;
        let row = (l % 4) * 2 + (j & 1) + ((j >> 1) & 1) * 8;
        let col = ((l & !4) / 8) * 2 + ((l >> 2) & 1) + ((j >> 2) & 1) * 8;
        tile[row][col] = decode_code(w, cb);
    }
    tile
}

/// One in-place FWHT butterfly stage in f16 over stride `s`, matching
/// `shuffle_had_h2x32`: index with the bit clear gets `self + partner`,
/// index with the bit set gets `partner - self`.
fn fwht_stage_f16(v: &mut [f16; 128], group: usize, s: usize) {
    // `group` values per index share the transform (the 4-wide chunks);
    // s indexes in units of groups.
    let mut out = *v;
    for a in 0..(128 / group) {
        let p = a ^ s;
        for g in 0..group {
            let own = v[a * group + g];
            let partner = v[p * group + g];
            out[a * group + g] = if a & s == 0 {
                hadd(own, partner)
            } else {
                hsub(partner, own)
            };
        }
    }
    *v = out;
}

/// Full 128x128-block reconstruction with the both-side Hadamard, exactly
/// replicating the GPU kernel's op order:
///  1. decode tiles -> W_hat
///  2. per column: 4-point butterfly over row groups (f16, then *rs),
///     then 5 FWHT stages over the 32 groups (f16)
///  3. per row: 4-point butterfly over col groups in f32 (exact), one
///     rounding to f16, *rs (f16), then 5 FWHT stages over groups (f16),
///     then *suh[row] then *svh[col] (two f16 muls)
#[allow(clippy::needless_range_loop)]
pub fn reconstruct_had_block(
    trellis_block: impl Fn(usize, usize) -> Vec<u16>, // (tile_r, tile_c) -> 16*k words
    suh: &[u16],                                      // 128 f16 bits for this block's rows
    svh: &[u16],                                      // 128 f16 bits for this block's cols
    k: u32,
    cb: Exl3Codebook,
) -> Vec<u16> {
    let rs = f16::from_f32(R_SCALE);
    // 1. decode
    let mut w = vec![[f16::from_f32(0.0); 128]; 128];
    for tr in 0..8 {
        for tc in 0..8 {
            let words = trellis_block(tr, tc);
            let tile = decode_tile(&words, k, cb);
            for r in 0..16 {
                for c in 0..16 {
                    w[tr * 16 + r][tc * 16 + c] = tile[r][c];
                }
            }
        }
    }
    // 2. column-direction (H . W): FWHT down each column over rows
    for c in 0..128 {
        let mut col = [f16::from_f32(0.0); 128];
        for r in 0..128 {
            col[r] = w[r][c];
        }
        // 4-point butterfly within each row group of 4, then *rs
        for a in 0..32 {
            let v0 = col[a * 4];
            let v1 = col[a * 4 + 1];
            let v2 = col[a * 4 + 2];
            let v3 = col[a * 4 + 3];
            let s0 = hadd(v0, v1);
            let d0 = hsub(v0, v1);
            let s1 = hadd(v2, v3);
            let d1 = hsub(v2, v3);
            col[a * 4] = hmul(hadd(s0, s1), rs);
            col[a * 4 + 1] = hmul(hadd(d0, d1), rs);
            col[a * 4 + 2] = hmul(hsub(s0, s1), rs);
            col[a * 4 + 3] = hmul(hsub(d0, d1), rs);
        }
        for s in [1usize, 2, 4, 8, 16] {
            fwht_stage_f16(&mut col, 4, s);
        }
        for r in 0..128 {
            w[r][c] = col[r];
        }
    }
    // 3. row-direction (W . H) + scales, fused as in the kernel store
    let mut out = vec![0u16; 128 * 128];
    for r in 0..128 {
        let mut row = [f16::from_f32(0.0); 128];
        // 4-point butterfly in f32 (exact), one rounding, then *rs in f16
        for g in 0..32 {
            let v0 = w[r][g * 4].to_f32();
            let v1 = w[r][g * 4 + 1].to_f32();
            let v2 = w[r][g * 4 + 2].to_f32();
            let v3 = w[r][g * 4 + 3].to_f32();
            let s0 = v0 + v1;
            let d0 = v0 - v1;
            let s1 = v2 + v3;
            let d1 = v2 - v3;
            row[g * 4] = hmul(f16::from_f32(s0 + s1), rs);
            row[g * 4 + 1] = hmul(f16::from_f32(d0 + d1), rs);
            row[g * 4 + 2] = hmul(f16::from_f32(s0 - s1), rs);
            row[g * 4 + 3] = hmul(f16::from_f32(d0 - d1), rs);
        }
        for s in [1usize, 2, 4, 8, 16] {
            fwht_stage_f16(&mut row, 4, s);
        }
        let su = f16b(suh[r]);
        for c in 0..128 {
            let v = hmul(hmul(row[c], su), f16b(svh[c]));
            out[r * 128 + c] = v.to_bits();
        }
    }
    out
}

/// Whole-tensor CPU reconstruction: trellis `[in/16, out/16, 16*k]` u16
/// words -> f16 bits `[in, out]` row-major (the GPU kernel's pre-transpose
/// layout, which is what the parity example compares against).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_had_f16(
    trellis: &[u16],
    suh: &[u16],
    svh: &[u16],
    in_dim: usize,
    out_dim: usize,
    k: u32,
    cb: Exl3Codebook,
) -> Vec<u16> {
    assert_eq!(in_dim % 128, 0);
    assert_eq!(out_dim % 128, 0);
    let kt = 16 * k as usize;
    assert_eq!(trellis.len(), (in_dim / 16) * (out_dim / 16) * kt);
    assert_eq!(suh.len(), in_dim);
    assert_eq!(svh.len(), out_dim);
    let tiles_n = out_dim / 16;
    let mut out = vec![0u16; in_dim * out_dim];
    for kb in 0..in_dim / 128 {
        for nb in 0..out_dim / 128 {
            let block = reconstruct_had_block(
                |tr, tc| {
                    let tile_r = kb * 8 + tr;
                    let tile_c = nb * 8 + tc;
                    let base = (tile_r * tiles_n + tile_c) * kt;
                    trellis[base..base + kt].to_vec()
                },
                &suh[kb * 128..kb * 128 + 128],
                &svh[nb * 128..nb * 128 + 128],
                k,
                cb,
            );
            for r in 0..128 {
                let dst = (kb * 128 + r) * out_dim + nb * 128;
                out[dst..dst + 128].copy_from_slice(&block[r * 128..r * 128 + 128]);
            }
        }
    }
    out
}

/// Dimensionality of one PLE n-gram embedding row (fixed by the format).
pub const NGRAM_ROW_DIM: usize = 160;

/// Words per packed `exl3_ngram_trellis` row: 1 scale word + the
/// `160*K`-bit ring.
pub fn ngram_words_per_row(k: u32) -> usize {
    1 + NGRAM_ROW_DIM * k as usize / 16
}

/// Decode one packed `exl3_ngram_trellis` row (see upstream
/// exllamav3 `ngram_codec.py`, snapshotted in
/// `.research/exllamav3_ref/`): word 0 is the fp16 row scale's bits,
/// words 1.. hold the tail-biting ring where stream bits
/// `[i*K, (i+1)*K)` are the LOW K bits of position i's 16-bit state,
/// LSB-first within each u16 (NOT the tile format's MSB-first order).
/// `state_i` bit `m` lives at stream bit
/// `((i - m/K) mod 160)*K + (m%K)`.
///
///     row[i] = decode_mul1(state_i) as f32 * scale as f32
///              [+ head_bias[i] as f32]
///
/// f32 math mirrors both the upstream reference and the
/// `batched_embed_exl3` kernel; output is the f32 value (callers round
/// to their target dtype).
pub fn decode_ngram_row(row_words: &[u16], k: u32, head_bias: Option<&[u16]>) -> Vec<f32> {
    let kk = k as usize;
    assert_eq!(row_words.len(), ngram_words_per_row(k));
    let scale = f16::from_bits(row_words[0]).to_f32();
    let stream = &row_words[1..];
    let mut out = Vec::with_capacity(NGRAM_ROW_DIM);
    for i in 0..NGRAM_ROW_DIM {
        let mut state: u32 = 0;
        for m in 0..16usize {
            let pos = (i + NGRAM_ROW_DIM - m / kk) % NGRAM_ROW_DIM;
            let bit = pos * kk + m % kk;
            let b = (stream[bit / 16] >> (bit % 16)) & 1;
            state |= u32::from(b) << m;
        }
        let mut v = decode_code(state as u16, Exl3Codebook::Mul1).to_f32() * scale;
        if let Some(bias) = head_bias {
            v += f16::from_bits(bias[i]).to_f32();
        }
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // decode_code spot values, computed independently from the format
    // spec (u32 arithmetic + f16 reinterpretation done by hand):
    //   mul1(0): x = 0, dp4a sum = 0x6400 -> f16 1024.0;
    //            1024 * 0.006775 - 10.3906 = -3.4531 (one fma rounding)
    #[test]
    fn mul1_code_zero() {
        let v = decode_code(0, Exl3Codebook::Mul1);
        let expect = f16::from_f64(
            f16::from_bits(0x6400).to_f64() * f16::from_bits(0x1eee).to_f64()
                + f16::from_bits(0xc931).to_f64(),
        );
        assert_eq!(v.to_bits(), expect.to_bits());
    }

    // The mcg decode of code 0 is exactly 0x3b60-as-f16 + 0x3b60-as-f16
    // (x=0 -> masked 0 -> xor pattern in both halves).
    #[test]
    fn mcg_code_zero() {
        let v = decode_code(0, Exl3Codebook::Mcg);
        let half = f16::from_bits(0x3b60);
        let expect = f16::from_f32(half.to_f32() + half.to_f32());
        assert_eq!(v.to_bits(), expect.to_bits());
    }

    // A trellis of all-zero words must decode every position to the same
    // value (every 16-bit window is 0), so after the symmetric Hadamard
    // sandwich with unit scales the block is rank-1-ish but crucially
    // FINITE everywhere. Sanity: no NaN/Inf anywhere at any K.
    #[test]
    fn zero_trellis_finite() {
        for k in 1..=8u32 {
            for cb in [Exl3Codebook::Inst3, Exl3Codebook::Mcg, Exl3Codebook::Mul1] {
                let one = f16::from_f32(1.0).to_bits();
                let out = reconstruct_had_f16(
                    &vec![0u16; (128 / 16) * (128 / 16) * 16 * k as usize],
                    &vec![one; 128],
                    &vec![one; 128],
                    128,
                    128,
                    k,
                    cb,
                );
                for &bits in &out {
                    let v = f16::from_bits(bits).to_f32();
                    assert!(v.is_finite(), "K={k} cb={cb:?} produced {v}");
                }
            }
        }
    }

    // Ngram ring decode: an all-zero ring at scale 1.0 decodes every
    // position to decode_mul1(0); a bias shifts it.
    #[test]
    fn ngram_zero_ring() {
        let k = 6u32;
        let mut row = vec![0u16; ngram_words_per_row(k)];
        row[0] = f16::from_f32(1.0).to_bits();
        let base = decode_code(0, Exl3Codebook::Mul1).to_f32();
        let out = decode_ngram_row(&row, k, None);
        assert_eq!(out.len(), NGRAM_ROW_DIM);
        for &v in &out {
            assert_eq!(v, base);
        }
        let bias = vec![f16::from_f32(0.5).to_bits(); NGRAM_ROW_DIM];
        let out_b = decode_ngram_row(&row, k, Some(&bias));
        for &v in &out_b {
            assert_eq!(v, base + 0.5);
        }
    }

    // Ring state reconstruction: pack per the upstream formula (stream
    // bits [j*K, (j+1)*K) = low K bits of state_j, LSB-first u16s) and
    // assert the decoder rebuilds the CLOSED-FORM full state
    //   state_i = (s_i & 63) | ((s_{i-1} & 63) << 6) | ((s_{i-2} & 15) << 12)
    // (K=6) — two independent derivations of the tail-biting ring.
    #[test]
    fn ngram_ring_state_reconstruction_k6() {
        let k = 6u32;
        let kk = 6usize;
        // Deterministic pseudo-random low-6-bit chunks.
        let s: Vec<u16> = (0..NGRAM_ROW_DIM)
            .map(|i| (((i as u32).wrapping_mul(2654435761) >> 13) & 63) as u16)
            .collect();
        let mut row = vec![0u16; ngram_words_per_row(k)];
        row[0] = f16::from_f32(1.0).to_bits();
        for (j, &sj) in s.iter().enumerate() {
            for b in 0..kk {
                let bit = j * kk + b;
                row[1 + bit / 16] |= ((sj >> b) & 1) << (bit % 16);
            }
        }
        let out = decode_ngram_row(&row, k, None);
        for i in 0..NGRAM_ROW_DIM {
            let prev1 = s[(i + NGRAM_ROW_DIM - 1) % NGRAM_ROW_DIM];
            let prev2 = s[(i + NGRAM_ROW_DIM - 2) % NGRAM_ROW_DIM];
            let expect_state =
                u32::from(s[i] & 63) | (u32::from(prev1 & 63) << 6) | (u32::from(prev2 & 15) << 12);
            let expect = decode_code(expect_state as u16, Exl3Codebook::Mul1).to_f32();
            assert_eq!(out[i], expect, "position {i}");
        }
    }

    // Bit-window extraction must match the kernel's aligned-K=4 fast
    // path (`dq8_aligned_4bits`), whose first-lane extraction reduces
    // to: window(t=7) = u32word0 & 0xffff = u16[0], and
    // window(t=3) = u32word0 >> 16 = u16[1] (derived by hand from the
    // funnel-shift indexing; the GPU parity example is the ground
    // truth).
    #[test]
    fn window_alignment_k4() {
        let k = 4usize;
        let mut words = vec![0u16; 16 * k];
        words[0] = 0xABCD; // u32 word0 low half
        words[1] = 0x1234; // u32 word0 high half
        let total = 256 * k;
        let stream_bit = |idx: usize| -> u16 {
            let idx = idx % total;
            let w32 = idx / 32;
            let bit = 31 - (idx % 32);
            let word = if bit >= 16 {
                words[w32 * 2 + 1]
            } else {
                words[w32 * 2]
            };
            (word >> (bit % 16)) & 1
        };
        let get = |t: usize| {
            let end = (t + 1) * k + total;
            let mut w = 0u16;
            for b in 0..16 {
                w |= stream_bit(end - 16 + b) << (15 - b);
            }
            w
        };
        assert_eq!(get(3), 0x1234);
        assert_eq!(get(7), 0xABCD);
    }
}
