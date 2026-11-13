// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the keep-packed GGUF microtest's `block_q8_1` readback (port lane L19): parsers
//! for the decode and MMQ activation layouts, and the block comparison against the CPU mirror.

use spark_model::port::gguf_q4::reference::Q8Block;

/// Parse `rows` decode-layout `block_q8_1` rows: `half2 ds` then 32 codes, 36 bytes a block.
pub(super) fn parse_q8_rows(bytes: &[u8], rows: usize, k: usize) -> Vec<Vec<Q8Block>> {
    let per_row = k / 32;
    (0..rows)
        .map(|r| {
            (0..per_row)
                .map(|b| {
                    let blk = &bytes[(r * per_row + b) * 36..][..36];
                    let mut qs = [0i8; 32];
                    for (q, byte) in qs.iter_mut().zip(&blk[4..36]) {
                        *q = *byte as i8;
                    }
                    Q8Block {
                        qs,
                        d: half::f16::from_le_bytes([blk[0], blk[1]]).to_f32(),
                        s: half::f16::from_le_bytes([blk[2], blk[3]]).to_f32(),
                    }
                })
                .collect()
        })
        .collect()
}

/// Parse the MMQ layout of `m` rows: block (kb, r) at index `kb * m + r`, 144 bytes, 16 bytes of
/// scales (`f32 d4[4]` or `half2 ds4[4]`) then 128 codes.
pub(super) fn parse_q8_mmq(bytes: &[u8], m: usize, k: usize, ds4: bool) -> Vec<Vec<Q8Block>> {
    (0..m)
        .map(|r| {
            (0..k / 32)
                .map(|sb| {
                    let blk = &bytes[((sb / 4) * m + r) * 144..][..144];
                    let i = sb % 4;
                    let mut qs = [0i8; 32];
                    for (q, byte) in qs.iter_mut().zip(&blk[16 + 32 * i..16 + 32 * i + 32]) {
                        *q = *byte as i8;
                    }
                    let f = &blk[4 * i..4 * i + 4];
                    let (d, s) = if ds4 {
                        let h = |a: u8, b: u8| half::f16::from_le_bytes([a, b]).to_f32();
                        (h(f[0], f[1]), h(f[2], f[3]))
                    } else {
                        (f32::from_le_bytes([f[0], f[1], f[2], f[3]]), 0.0)
                    };
                    Q8Block { qs, d, s }
                })
                .collect()
        })
        .collect()
}

/// `(code or scale mismatches, partial sums beyond two f16 units)` of GPU blocks against the mirror.
pub(super) fn compare_q8(
    got: &[Vec<Q8Block>],
    want: &[Vec<Q8Block>],
    check_s: bool,
) -> (usize, usize) {
    let (mut bad, mut bad_s) = (0usize, 0usize);
    for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
        bad += usize::from(g.qs != w.qs || g.d.to_bits() != w.d.to_bits());
        if check_s {
            let tol = w.s.abs() * 2f32.powi(-10) + f32::from(half::f16::MIN_POSITIVE_SUBNORMAL);
            bad_s += usize::from((g.s - w.s).abs() > tol);
        }
    }
    (bad, bad_s)
}
