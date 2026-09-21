// SPDX-License-Identifier: AGPL-3.0-only
//! CPU IQ3_XXS dequant (ggml `dequantize_row_iq3_xxs`).
//!
//! Block = 98 B / 256 weights: f16 `d`, then `qs[96]`
//! (64 B grid indices + 32 B scales-and-signs).

#![allow(clippy::needless_range_loop)]

use super::iq2_xs::{KMASK_IQ2XS, KSIGNS_IQ2XS};

/// ggml `iq3xxs_grid`: 256 entries, 4 unsigned magnitudes per `u32`.
pub static IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

const QK: usize = 256;
const BLOCK_BYTES: usize = 98;

/// Dequant one IQ3_XXS block into 256 f32 weights.
pub fn dequant_iq3_xxs(blk: &[u8], out: &mut [f32]) {
    debug_assert!(blk.len() >= BLOCK_BYTES);
    debug_assert!(out.len() >= QK);
    let d = {
        let bits = u16::from_le_bytes([blk[0], blk[1]]);
        half::f16::from_bits(bits).to_f32()
    };
    let qs = &blk[2..66];
    let scales = &blk[66..98];
    for ib32 in 0..8 {
        let aux = u32::from_le_bytes(scales[4 * ib32..4 * ib32 + 4].try_into().unwrap());
        let db = d * (0.5 + ((aux >> 28) as f32)) * 0.5;
        let weight_base = ib32 * 32;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
            let group_base = weight_base + l * 8;
            for j in 0..4 {
                let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                out[group_base + j] = db * g1[j] as f32 * s0;
                out[group_base + 4 + j] = db * g2[j] as f32 * s1;
            }
        }
    }
}

/// Dequant a row-major IQ3_XXS tensor (`n * k` weights, `k` multiple of 256).
pub fn dequant_iq3_xxs_tensor(raw: &[u8], n: usize, k: usize, out: &mut [f32]) -> anyhow::Result<()> {
    anyhow::ensure!(k > 0 && k % QK == 0, "IQ3_XXS K={k} must be multiple of {QK}");
    anyhow::ensure!(out.len() >= n * k, "IQ3_XXS out too small");
    let blocks_per_row = k / QK;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    anyhow::ensure!(raw.len() >= n * row_bytes, "IQ3_XXS raw too small");
    for row in 0..n {
        let src = &raw[row * row_bytes..(row + 1) * row_bytes];
        let dst = &mut out[row * k..(row + 1) * k];
        for b in 0..blocks_per_row {
            dequant_iq3_xxs(
                &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
                &mut dst[b * QK..(b + 1) * QK],
            );
        }
    }
    Ok(())
}

/// Column-cover dequant for a TP K-split that is not a multiple of 256.
pub fn dequant_iq3_xxs_column_cover(
    raw: &[u8],
    n: usize,
    full_k: usize,
    k_local: usize,
    rank: usize,
    world: usize,
    out: &mut [f32],
) -> anyhow::Result<()> {
    anyhow::ensure!(world > 0 && rank < world, "bad TP rank");
    anyhow::ensure!(full_k % world == 0, "full_k not divisible by TP");
    anyhow::ensure!(k_local == full_k / world, "k_local mismatch");
    anyhow::ensure!(out.len() >= n * k_local, "out too small");
    let k0 = rank * k_local;
    let k1 = k0 + k_local;
    let b0 = k0 / QK;
    let b1 = k1.div_ceil(QK);
    let n_cover = b1 - b0;
    let row_bytes = n_cover * BLOCK_BYTES;
    anyhow::ensure!(raw.len() >= n * row_bytes, "cover raw too small");
    let skip = k0 - b0 * QK;
    let mut tmp = vec![0f32; n_cover * QK];
    for row in 0..n {
        let src = &raw[row * row_bytes..(row + 1) * row_bytes];
        for b in 0..n_cover {
            dequant_iq3_xxs(
                &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
                &mut tmp[b * QK..(b + 1) * QK],
            );
        }
        let dst = &mut out[row * k_local..(row + 1) * k_local];
        dst.copy_from_slice(&tmp[skip..skip + k_local]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq3_xxs_grid0_zero_signs() {
        let d = 2.0f32;
        let mut blk = [0u8; BLOCK_BYTES];
        let db = half::f16::from_f32(d).to_le_bytes();
        blk[0] = db[0];
        blk[1] = db[1];
        let mut out = [0f32; 256];
        dequant_iq3_xxs(&blk, &mut out);
        let sign0 = KSIGNS_IQ2XS[0];
        let mag = 4.0f32;
        let scale = d * 0.25;
        for (i, &v) in out.iter().enumerate() {
            let sign = if sign0 & KMASK_IQ2XS[i % 8] != 0 { -1.0 } else { 1.0 };
            let expected = scale * mag * sign;
            assert!((v - expected).abs() < 1e-4, "out[{i}]={v} want {expected}");
        }
    }
}
