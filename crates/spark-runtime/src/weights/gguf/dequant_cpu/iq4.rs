// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-common.h (block_iq4_xs, block_iq4_nl, kvalues_iq4nl) and ggml/src/ggml-quants.c
// (dequantize_row_iq4_xs, dequantize_row_iq4_nl). Changes: translated from C to Rust, one block per call
// over a byte slice, little-endian field reads, by Atlas contributors.
#![allow(clippy::needless_range_loop)]

//! Responsibility: the CPU reference decoders of the IQ4 family (port lane L19): `IQ4_XS` (ggml id 23,
//! 256 weights in 136 bytes, 4.25 bpw) and `IQ4_NL` (id 20, 32 weights in 18 bytes).
//!
//! These are the bit-exact oracle of the Ada GPU dequant (`kernels/ada/common/dequant_iq4xs_q5k_bf16_ada.cu`)
//! and of the keep-packed GEMV's weight values, and the CPU fallback of the GGUF loader for an IQ4 tensor
//! that is neither kept packed nor expanded on the GPU. The arithmetic is ggml's: `dl = d * (ls - 32)` in
//! f32, then `dl * kvalues_iq4nl[nibble]`, two f32 products and nothing fused.

use super::rd_f16;

/// ggml's non-linear 4-bit code book (`kvalues_iq4nl`, `ggml-common.h`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Bytes of one `block_iq4_xs`: `f16 d`, `u16 scales_h`, `u8 scales_l[4]`, `u8 qs[128]`.
pub const IQ4_XS_BLOCK_BYTES: usize = 136;
/// Bytes of one `block_iq4_nl`: `f16 d`, `u8 qs[16]`.
pub const IQ4_NL_BLOCK_BYTES: usize = 18;

/// The 6-bit scale of sub-block `ib` (0..8) of an IQ4_XS block: the low nibble of `scales_l[ib / 2]`
/// (the high nibble for odd `ib`) with two bits of `scales_h` at `2 * ib` above it.
#[inline]
pub fn iq4_xs_sub_scale(scales_h: u16, scales_l: &[u8], ib: usize) -> i32 {
    let lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
    let hi = ((scales_h >> (2 * ib)) & 3) as u8;
    i32::from(lo | (hi << 4))
}

/// IQ4_XS: one 136-byte block into 256 f32 values.
pub(super) fn dequant_iq4_xs(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
    let scales_l = &blk[4..8];
    let qs = &blk[8..136];
    for ib in 0..8 {
        let ls = iq4_xs_sub_scale(scales_h, scales_l, ib);
        let dl = d * (ls - 32) as f32;
        let q = &qs[16 * ib..16 * ib + 16];
        let y = &mut out[32 * ib..32 * ib + 32];
        for j in 0..16 {
            y[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[j] & 0x0F)]);
            y[j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[j] >> 4)]);
        }
    }
}

/// IQ4_NL: one 18-byte block into 32 f32 values.
pub(super) fn dequant_iq4_nl(blk: &[u8], out: &mut [f32]) {
    let d = rd_f16(blk, 0);
    let qs = &blk[2..18];
    for j in 0..16 {
        out[j] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0x0F)]);
        out[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)]);
    }
}
