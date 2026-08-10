// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side helpers for the served NLLB runtime: sinusoidal position tables
//! and byte reinterpret casts. Promoted from `examples/nllb_cuda_bf16/util.rs`.

use half::bf16;

/// One M2M-100 sinusoidal position row into `out[d]` (bf16): `sin` in the lower
/// half, `cos` in the upper half, `freq_j = exp(-j·ln(10000)/(d/2-1))`.
pub(super) fn sinusoid_row(pos: f32, d: usize, out: &mut [bf16]) {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    for j in 0..half {
        let ang = pos * (-(j as f32) * emb_scale).exp();
        out[j] = bf16::from_f32(ang.sin());
        out[half + j] = bf16::from_f32(ang.cos());
    }
}

/// Decoder position table `[max_len, d]` (bf16). Decoder positions are the
/// fairseq/M2M-100 convention: logical position `i` uses sinusoid `i + 2`
/// (offset by `padding_idx + 1` with `padding_idx = 1`).
pub(super) fn decoder_pos_table_bf16(max_len: usize, d: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); max_len * d];
    for i in 0..max_len {
        sinusoid_row((i + 2) as f32, d, &mut t[i * d..i * d + d]);
    }
    t
}

/// Encoder position embeddings `[seq, d]` (bf16) with masked incremental
/// positions: non-pad tokens count from `padding_idx + 1`; pad tokens get the
/// zeroed `padding_idx` row.
pub(super) fn encoder_pos_bf16(ids: &[u32], d: usize, pad: u32) -> Vec<bf16> {
    let seq = ids.len();
    let mut t = vec![bf16::from_f32(0.0); seq * d];
    let mut running = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let p = if id != pad {
            running += 1;
            running + pad
        } else {
            pad
        };
        if p != pad {
            sinusoid_row(p as f32, d, &mut t[i * d..i * d + d]);
        }
    }
    t
}

/// Reinterpret a `&[u32]` as raw little-endian bytes for an H2D copy.
pub(super) fn u32_bytes(v: &[u32]) -> &[u8] {
    // SAFETY: `u32` is `Copy`/POD; we produce a read-only byte view of the same
    // allocation with a correctly scaled length, and the source outlives the use.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Reinterpret a `&[bf16]` as raw little-endian bytes for an H2D copy.
pub(super) fn bf16_bytes(v: &[bf16]) -> &[u8] {
    // SAFETY: `bf16` is a `#[repr(transparent)]` POD wrapper over `u16`; same
    // reasoning as `u32_bytes`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
#[path = "util_tests.rs"]
mod tests;
