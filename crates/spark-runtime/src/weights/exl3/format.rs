// SPDX-License-Identifier: AGPL-3.0-only
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/src/core/exl3_format.zig:1-134 (Apache-2.0, ashhart/TensorFold, commit 3885dce; Copyright 2026
// TensorFold contributors). Changes: translated from Zig to Rust; states returned as u16; explicit
// length and width checks returning errors instead of debug asserts; the placement table built once
// as a const; fp16 rounding through the exact `fp16` module. See EXL3-ATTRIBUTION.md.

//! Responsibility: the EXL3 bitstream and codebooks: bit widths (whole and half), the three codebooks
//! (`3inst`, `mcg`, `mul1`) mapping a 16-bit trellis state to an fp16 value, the circular per-tile
//! bitstream and its window ends `E(p)`, the tensor-core placement of a tile's 256 values, the
//! trellis unpack to `W_q` in the rotated domain, and the packed sign words of older checkpoints.
//!
//! The format (TensorFold `docs/recipes/exl3.md`, "The format, in one paragraph"): a linear layer with
//! K inputs and N outputs stores `trellis` int16 `[K/16, N/16, 16 * bits]`. Each 16x16 tile holds 256
//! values in a circular bitstream of `256 * bits` bits; its int16 words, read in little-endian pairs
//! as 32-bit words, are read from the most significant bit. Value `p` is the 16 bits of the stream
//! that end at `E(p)` (exclusive, wrapping), taken as an unsigned state. Half widths (`bits = KA + 1/2`,
//! `mul1` only) alternate KA and KA + 1 new bits a value. Dequantization (the rotations and scales) is
//! in [`super::dequant`].

use anyhow::{Result, bail};

use super::fp16;

/// `3inst` multiplier: `x = s * 89226354 + 64248484 (mod 2^32)`.
pub const INST3_MUL: u32 = 89_226_354;
/// `3inst` addend.
pub const INST3_ADD: u32 = 64_248_484;
/// `mcg` multiplier, also the value of the `.mcg` marker tensor.
pub const MCG_MUL: u32 = 0xCBAC_1FED;
/// `mul1` multiplier, also the value of the `.mul1` marker tensor.
pub const MUL1_MUL: u32 = 0x83DC_D12D;
/// Mask applied to `x` by `3inst` and `mcg` before the flip.
pub const MASK: u32 = 0x8FFF_8FFF;
/// XOR applied after the mask: the two halves become fp16 values with exponents 12 to 15.
pub const FLIP: u32 = 0x3B60_3B60;
/// fp16 bit pattern of the `mul1` scale (about 1/147.7).
pub const MUL1_SCALE_BITS: u16 = 0x1EEE;
/// fp16 bit pattern of the `mul1` bias (about -10.39).
pub const MUL1_BIAS_BITS: u16 = 0xC931;
/// Values a tile holds (16 x 16).
pub const TILE_VALUES: usize = 256;
/// Edge of a tile.
pub const TILE_DIM: usize = 16;

/// The codebook a group's trellis decodes with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codebook {
    /// The original 3-instruction codebook (no marker tensor).
    Inst3,
    /// Multiplicative congruential codebook (marker tensor `.mcg`).
    Mcg,
    /// Byte-sum codebook (marker tensor `.mul1`); the only one with half-bit widths.
    Mul1,
}

impl Codebook {
    /// The name ExLlamaV3 and TensorFold use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Inst3 => "3inst",
            Self::Mcg => "mcg",
            Self::Mul1 => "mul1",
        }
    }

    /// The marker tensor suffix naming this codebook (`{prefix}.mcg` / `{prefix}.mul1`); `3inst` has none.
    pub fn marker_suffix(self) -> Option<&'static str> {
        match self {
            Self::Inst3 => None,
            Self::Mcg => Some("mcg"),
            Self::Mul1 => Some("mul1"),
        }
    }

    /// The value the marker tensor must hold (its multiplier).
    pub fn marker_value(self) -> Option<u32> {
        match self {
            Self::Inst3 => None,
            Self::Mcg => Some(MCG_MUL),
            Self::Mul1 => Some(MUL1_MUL),
        }
    }

    /// The codebook a marker value names (`exl3_layout.zig:9-15`).
    pub fn from_marker(value: u32) -> Option<Self> {
        match value {
            MUL1_MUL => Some(Self::Mul1),
            MCG_MUL => Some(Self::Mcg),
            _ => None,
        }
    }

    /// Every codebook, in TensorFold's enum order (`inst3`, `mcg`, `mul1`).
    pub const ALL: [Codebook; 3] = [Self::Inst3, Self::Mcg, Self::Mul1];
}

/// A bit width in half bits: 2 (1 bit) to 16 (8 bits). Odd values are the `mul1` half widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Width {
    halves: u16,
}

impl Width {
    /// Smallest width the format admits (1 bit a value).
    pub const MIN_HALVES: u16 = 2;
    /// Largest width the format admits (8 bits a value).
    pub const MAX_HALVES: u16 = 16;

    /// A width from its half-bit count, if it is in 2..=16.
    pub fn from_halves(halves: u16) -> Option<Self> {
        (Self::MIN_HALVES..=Self::MAX_HALVES)
            .contains(&halves)
            .then_some(Self { halves })
    }

    /// The width a trellis whose last dimension is `words` int16 words a tile stores (`16 * bits`).
    pub fn from_tile_words(words: usize) -> Option<Self> {
        if !words.is_multiple_of(8) {
            return None;
        }
        u16::try_from(words / 8).ok().and_then(Self::from_halves)
    }

    /// Half bits a value (the kernels' K2): 4 for 2 bits, 16 for 8 bits.
    pub fn halves(self) -> u16 {
        self.halves
    }

    /// Bits a value (exact in halves).
    pub fn bits(self) -> f32 {
        f32::from(self.halves) / 2.0
    }

    /// Bits a value times 100 (the `W4Format::Exl3 { bpw_x100 }` value): 250 for 2.5 bits.
    pub fn bpw_x100(self) -> u16 {
        self.halves * 50
    }

    /// int16 words a 16x16 tile: `16 * bits`.
    pub fn tile_words(self) -> usize {
        usize::from(self.halves) * 8
    }

    /// True for a half width (`KA + 1/2` bits).
    pub fn is_half(self) -> bool {
        self.halves % 2 == 1
    }

    /// Bits in a tile's circular bitstream: `256 * bits`.
    pub fn ring_bits(self) -> u32 {
        128 * u32::from(self.halves)
    }
}

/// The fp16 value (as a bit pattern) of one 16-bit state under `codebook`, with exactly one rounding.
///
/// `3inst` and `mcg`: the masked, flipped word is two fp16 values whose float64 sum is exact, then
/// rounded once. `mul1`: `h = 1024 + (sum of x's four bytes)`, `h * scale + bias` in float64 (exact),
/// then rounded once (`exl3_format.zig:35-52`).
pub fn codebook_value(codebook: Codebook, state: u16) -> u16 {
    let s = u32::from(state);
    let x = match codebook {
        Codebook::Mul1 => s.wrapping_mul(MUL1_MUL),
        Codebook::Mcg => s.wrapping_mul(MCG_MUL),
        Codebook::Inst3 => s.wrapping_mul(INST3_MUL).wrapping_add(INST3_ADD),
    };
    if codebook == Codebook::Mul1 {
        let byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF);
        let h = f64::from(1024 + byte_sum);
        return fp16::from_f64(h * fp16::to_f64(MUL1_SCALE_BITS) + fp16::to_f64(MUL1_BIAS_BITS));
    }
    let masked = (x & MASK) ^ FLIP;
    let lo = fp16::to_f64((masked & 0xFFFF) as u16);
    let hi = fp16::to_f64((masked >> 16) as u16);
    fp16::from_f64(lo + hi)
}

/// [`codebook_value`] as `f64`.
pub fn codebook_value_f64(codebook: Codebook, state: u16) -> f64 {
    fp16::to_f64(codebook_value(codebook, state))
}

/// The `f64` value of every state under `codebook` (65536 entries), for decoders that look values up.
pub fn codebook_table(codebook: Codebook) -> Vec<f64> {
    (0..=u16::MAX)
        .map(|s| codebook_value_f64(codebook, s))
        .collect()
}

/// `E(p)` for `p = 0..255`: where value `p`'s 16-bit window ends in the tile's bitstream (exclusive).
///
/// Whole widths: `E(p) = (p + 1) * bits`. Half widths `bits = KA + 1/2`:
/// `E(p) = ((p + 1) * (2 * KA + 1) - ((p + 1) % 2)) / 2`, so positions alternate KA and KA + 1 new bits
/// and the odd positions take the extra bit.
pub fn stream_ends(w: Width) -> [u32; TILE_VALUES] {
    let mut ends = [0u32; TILE_VALUES];
    let halves = u32::from(w.halves());
    for (p, end) in ends.iter_mut().enumerate() {
        let p1 = p as u32 + 1;
        *end = if !w.is_half() {
            p1 * halves / 2
        } else {
            let ka = (halves - 1) / 2;
            (p1 * (2 * ka + 1) - (p1 % 2)) / 2
        };
    }
    ends
}

/// Bit `index` of a tile's stream: MSB-first within little-endian pairs of int16 words.
fn stream_bit(words: &[i16], index: u32) -> u32 {
    let pair = (index / 32) as usize;
    let within = 31 - (index % 32);
    let lo = u32::from(words[pair * 2] as u16);
    let hi = u32::from(words[pair * 2 + 1] as u16);
    ((lo | (hi << 16)) >> within) & 1
}

/// The 16-bit state of each of a tile's 256 values, in stream order, from the tile's int16 words.
pub fn tile_states(w: Width, tile: &[i16]) -> Result<[u16; TILE_VALUES]> {
    if tile.len() != w.tile_words() {
        bail!(
            "EXL3 tile has {} int16 words; {} bits a value needs {}",
            tile.len(),
            w.bits(),
            w.tile_words()
        );
    }
    let ends = stream_ends(w);
    let ring = w.ring_bits();
    let mut states = [0u16; TILE_VALUES];
    for (state, &end) in states.iter_mut().zip(ends.iter()) {
        let start = (end + ring - 16) % ring;
        let mut s = 0u32;
        for b in 0..16 {
            s = (s << 1) | stream_bit(tile, (start + b) % ring);
        }
        *state = s as u16;
    }
    Ok(states)
}

/// Where stream value `p` lands in its 16x16 tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Row within the tile (the K direction).
    pub row: u8,
    /// Column within the tile (the N direction).
    pub col: u8,
}

const fn build_placement() -> [Placement; TILE_VALUES] {
    let mut out = [Placement { row: 0, col: 0 }; TILE_VALUES];
    let mut p = 0;
    while p < TILE_VALUES {
        let lane = p / 8;
        let j = p % 8;
        out[p] = Placement {
            row: (2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)) as u8,
            col: (lane / 4 + 8 * (j >> 2)) as u8,
        };
        p += 1;
    }
    out
}

/// Stream value `p` -> tile position: lane `l = p / 8` of a warp holds values `8l..8l+7`, exactly its
/// B fragments of the tile's two `mma.m16n8k16` (columns 0-7 and 8-15).
pub const TILE_PLACEMENT: [Placement; TILE_VALUES] = build_placement();

/// `W_q` `[K, N]` as fp16 bit patterns, row-major, in the rotated domain, from a stored-order trellis
/// int16 `[K/16, N/16, 16 * bits]`.
pub fn unpack(
    w: Width,
    codebook: Codebook,
    trellis: &[i16],
    tiles_k: usize,
    tiles_n: usize,
) -> Result<Vec<u16>> {
    check_trellis(w, trellis, tiles_k, tiles_n)?;
    let n = tiles_n * TILE_DIM;
    let words = w.tile_words();
    let mut out = vec![0u16; tiles_k * TILE_DIM * n];
    for tk in 0..tiles_k {
        for tn in 0..tiles_n {
            let at = (tk * tiles_n + tn) * words;
            let states = tile_states(w, &trellis[at..at + words])?;
            for (state, place) in states.iter().zip(TILE_PLACEMENT.iter()) {
                let r = tk * TILE_DIM + usize::from(place.row);
                let c = tn * TILE_DIM + usize::from(place.col);
                out[r * n + c] = codebook_value(codebook, *state);
            }
        }
    }
    Ok(out)
}

/// Checks a stored-order trellis's length against its tile counts and width.
pub fn check_trellis(w: Width, trellis: &[i16], tiles_k: usize, tiles_n: usize) -> Result<()> {
    let want = tiles_k
        .checked_mul(tiles_n)
        .and_then(|t| t.checked_mul(w.tile_words()));
    if want != Some(trellis.len()) {
        bail!(
            "EXL3 trellis has {} int16 words; [{tiles_k}, {tiles_n}] tiles at {} bits need {:?}",
            trellis.len(),
            w.bits(),
            want
        );
    }
    // A half width needs the `mul1` codebook; the width alone cannot say which codebook a trellis
    // uses, so callers that know it check that (`Exl3Tensor::new`, the loader's admission).
    Ok(())
}

/// Packed sign words `su` / `sv` int16 `[len/16]` -> fp16 `[len]` of +1 and -1: bit `b` of word `w`
/// set means element `16w + b` is -1 (`exl3_format.zig:125-134`).
pub fn unpack_signs(packed_words: &[i16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(packed_words.len() * 16);
    for &word in packed_words {
        let word = word as u16;
        for b in 0..16 {
            out.push(if (word >> b) & 1 == 1 {
                fp16::MINUS_ONE
            } else {
                fp16::ONE
            });
        }
    }
    out
}
