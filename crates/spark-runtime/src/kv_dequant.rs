// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side dequantization of paged KV cache blocks → BF16.
//!
//! Used by `--high-speed-swap` Phase 6.2.c to produce BF16 source data for
//! the orchestrator's tile-streaming attention kernel from any of Atlas's
//! quantized KV layouts. The kernel-side packing/scale layouts mirrored
//! here:
//!
//! | Quant   | data bytes/elem | scale | LUT                             | source kernel                      |
//! |---------|-----------------|-------|---------------------------------|------------------------------------|
//! | BF16    | 2               | none  | identity                        | (direct copy, not in this module)  |
//! | FP8     | 1 (E4M3)        | tensor| `e4m3_lut`                      | reshape_and_cache_fp8.cu           |
//! | NVFP4   | 0.5 (4-bit)     | group | `NVFP4_E2M1_LUT`                | paged_decode_attn_nvfp4.cu         |
//! | Turbo4  | 0.5 (4-bit)     | group | `TURBO4_LUT` (Lloyd-Max 16)     | paged_decode_attn_turbo4.cu        |
//! | Turbo3  | 0.375 (3-bit)   | group | `TURBO3_LUT` (Lloyd-Max 8)      | paged_decode_attn_turbo3.cu        |
//! | Turbo8  | 1 (E4M3)        | group | `e4m3_lut`                      | paged_decode_attn_turbo8.cu        |
//!
//! "Group" scales cover 16 elements per scale byte (`NVFP4_GROUP_SIZE`) and
//! are stored in a separate section after the data section within each
//! block. All LUTs match their CUDA-side counterparts byte-for-byte.

use half::bf16;

/// Group size for per-group FP8 scales. Matches `NVFP4_GROUP_SIZE` in the
/// per-quant attention kernels.
pub const NVFP4_GROUP_SIZE: usize = 16;

/// E2M1 4-bit codebook (NVFP4). Matches
/// `kernels/gb10/nvfp4/paged_decode_attn_nvfp4.cu:118`.
pub const NVFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Turbo4 16-level Lloyd-Max codebook. Matches
/// `kernels/gb10/nvfp4/paged_decode_attn_turbo4.cu:121`.
pub const TURBO4_LUT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];

/// Turbo3 8-level Lloyd-Max codebook. Matches
/// `kernels/gb10/nvfp4/paged_decode_attn_turbo3.cu:137`.
pub const TURBO3_LUT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];

/// E4M3 → f32 LUT (256 entries). Built lazily on first use.
pub fn e4m3_lut() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for byte in 0..256u32 {
            let sign_bit = (byte >> 7) & 1;
            let exp = ((byte >> 3) & 0xF) as i32;
            let mant = byte & 0x7;
            let s: f32 = if sign_bit == 0 { 1.0 } else { -1.0 };
            lut[byte as usize] = if exp == 0 {
                if mant == 0 {
                    s * 0.0
                } else {
                    s * (mant as f32) * 2.0f32.powi(-9)
                }
            } else if exp == 0xF && mant == 0x7 {
                f32::NAN
            } else {
                s * 2.0f32.powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            };
        }
        lut
    })
}

/// Dequant FP8 (E4M3) bytes to BF16, applying a per-tensor scale.
pub fn dequant_fp8_to_bf16(fp8_bytes: &[u8], scale: f32, out: &mut [bf16]) {
    debug_assert_eq!(fp8_bytes.len(), out.len());
    let lut = e4m3_lut();
    for (i, b) in fp8_bytes.iter().enumerate() {
        out[i] = bf16::from_f32(lut[*b as usize] * scale);
    }
}

/// Dequant a 4-bit packed (NVFP4 or Turbo4) KV block to BF16.
///
/// Layout per block:
///   data:   `bs * nkv * (hd / 2)` bytes (2 nibbles/byte).
///   scales: `bs * nkv * (hd / NVFP4_GROUP_SIZE)` bytes (1 FP8 scale per group).
pub fn dequant_4bit_block_to_bf16(
    raw: &[u8],
    bs: usize,
    nkv: usize,
    hd: usize,
    lut: &[f32; 16],
    out: &mut [bf16],
) {
    debug_assert!(hd.is_multiple_of(NVFP4_GROUP_SIZE));
    debug_assert!(hd.is_multiple_of(2));
    debug_assert_eq!(out.len(), bs * nkv * hd);
    let head_data_bytes = hd / 2;
    let head_scale_bytes = hd / NVFP4_GROUP_SIZE;
    let token_data_stride = nkv * head_data_bytes;
    let token_scale_stride = nkv * head_scale_bytes;
    let data_section_bytes = bs * token_data_stride;
    debug_assert!(raw.len() >= data_section_bytes + bs * token_scale_stride);
    let (data, scales) = raw.split_at(data_section_bytes);
    let e4m3 = e4m3_lut();
    for tok in 0..bs {
        for kv_h in 0..nkv {
            let d_off = tok * token_data_stride + kv_h * head_data_bytes;
            let s_off = tok * token_scale_stride + kv_h * head_scale_bytes;
            for byte_idx in 0..head_data_bytes {
                let byte = data[d_off + byte_idx];
                let n0 = (byte & 0x0F) as usize;
                let n1 = ((byte >> 4) & 0x0F) as usize;
                let elem_pair_idx = byte_idx * 2;
                let group_idx = elem_pair_idx / NVFP4_GROUP_SIZE;
                let scale = e4m3[scales[s_off + group_idx] as usize];
                let v0 = lut[n0] * scale;
                let v1 = lut[n1] * scale;
                let out_base = (tok * nkv + kv_h) * hd + elem_pair_idx;
                out[out_base] = bf16::from_f32(v0);
                out[out_base + 1] = bf16::from_f32(v1);
            }
        }
    }
}

/// Dequant a Turbo3 (3-bit packed) KV block to BF16.
///
/// Layout per block:
///   data:   `bs * nkv * (hd * 3 / 8)` bytes — 8 values packed in 3 bytes.
///   scales: `bs * nkv * (hd / NVFP4_GROUP_SIZE)` bytes.
/// Bit packing for 8 vals v0..v7 in 3 bytes b0,b1,b2 (mirrors
/// `kernels/gb10/nvfp4/paged_decode_attn_turbo3.cu:67-75`):
///   v0 = b0 & 0x7
///   v1 = (b0 >> 3) & 0x7
///   v2 = ((b0 >> 6) | (b1 << 2)) & 0x7
///   v3 = (b1 >> 1) & 0x7
///   v4 = (b1 >> 4) & 0x7
///   v5 = ((b1 >> 7) | (b2 << 1)) & 0x7
///   v6 = (b2 >> 2) & 0x7
///   v7 = (b2 >> 5) & 0x7
pub fn dequant_turbo3_block_to_bf16(
    raw: &[u8],
    bs: usize,
    nkv: usize,
    hd: usize,
    out: &mut [bf16],
) {
    debug_assert!(hd.is_multiple_of(8));
    debug_assert!(hd.is_multiple_of(NVFP4_GROUP_SIZE));
    debug_assert_eq!(out.len(), bs * nkv * hd);
    let head_data_bytes = hd * 3 / 8;
    let head_scale_bytes = hd / NVFP4_GROUP_SIZE;
    let token_data_stride = nkv * head_data_bytes;
    let token_scale_stride = nkv * head_scale_bytes;
    let data_section_bytes = bs * token_data_stride;
    debug_assert!(raw.len() >= data_section_bytes + bs * token_scale_stride);
    let (data, scales) = raw.split_at(data_section_bytes);
    let e4m3 = e4m3_lut();
    for tok in 0..bs {
        for kv_h in 0..nkv {
            let d_off = tok * token_data_stride + kv_h * head_data_bytes;
            let s_off = tok * token_scale_stride + kv_h * head_scale_bytes;
            for triplet_idx in 0..hd / 8 {
                let b0 = data[d_off + triplet_idx * 3] as u32;
                let b1 = data[d_off + triplet_idx * 3 + 1] as u32;
                let b2 = data[d_off + triplet_idx * 3 + 2] as u32;
                let nibbles = [
                    (b0) & 0x7,
                    (b0 >> 3) & 0x7,
                    ((b0 >> 6) | (b1 << 2)) & 0x7,
                    (b1 >> 1) & 0x7,
                    (b1 >> 4) & 0x7,
                    ((b1 >> 7) | (b2 << 1)) & 0x7,
                    (b2 >> 2) & 0x7,
                    (b2 >> 5) & 0x7,
                ];
                let elem_base_in_head = triplet_idx * 8;
                for k in 0..8 {
                    let elem = elem_base_in_head + k;
                    let group_idx = elem / NVFP4_GROUP_SIZE;
                    let scale = e4m3[scales[s_off + group_idx] as usize];
                    let v = TURBO3_LUT[nibbles[k] as usize] * scale;
                    let out_idx = (tok * nkv + kv_h) * hd + elem;
                    out[out_idx] = bf16::from_f32(v);
                }
            }
        }
    }
}

/// Dequant a Turbo8 (FP8 E4M3 data + per-group **BF16** scales) KV block to BF16.
///
/// Layout per block (post 2026-04-28 BF16-scale upgrade):
///   data:   `bs * nkv * hd` bytes (1 FP8 byte per element).
///   scales: `bs * nkv * (hd / NVFP4_GROUP_SIZE) * 2` bytes (BF16 = 2 bytes/scale).
/// Mirrors `kernels/gb10/nvfp4/paged_decode_attn_turbo8*.cu` post-upgrade.
pub fn dequant_turbo8_block_to_bf16(
    raw: &[u8],
    bs: usize,
    nkv: usize,
    hd: usize,
    out: &mut [bf16],
) {
    debug_assert!(hd.is_multiple_of(NVFP4_GROUP_SIZE));
    debug_assert_eq!(out.len(), bs * nkv * hd);
    let head_data_bytes = hd;
    let head_scale_bytes = (hd / NVFP4_GROUP_SIZE) * 2; // BF16 scales: 2 bytes per group
    let token_data_stride = nkv * head_data_bytes;
    let token_scale_stride = nkv * head_scale_bytes;
    let data_section_bytes = bs * token_data_stride;
    debug_assert!(raw.len() >= data_section_bytes + bs * token_scale_stride);
    let (data, scales) = raw.split_at(data_section_bytes);
    let e4m3 = e4m3_lut();
    for tok in 0..bs {
        for kv_h in 0..nkv {
            let d_off = tok * token_data_stride + kv_h * head_data_bytes;
            let s_off = tok * token_scale_stride + kv_h * head_scale_bytes;
            for i in 0..hd {
                let byte = data[d_off + i] as usize;
                let group_idx = i / NVFP4_GROUP_SIZE;
                // Read BF16 scale (2 bytes, little-endian).
                let s_byte_off = s_off + group_idx * 2;
                let scale_bf16 = bf16::from_le_bytes([scales[s_byte_off], scales[s_byte_off + 1]]);
                let scale = scale_bf16.to_f32();
                let v = e4m3[byte] * scale;
                let out_idx = (tok * nkv + kv_h) * hd + i;
                out[out_idx] = bf16::from_f32(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FP8 E4M3 byte for +1.0: sign=0, exp=7, mantissa=0 → 0b00111000 = 0x38.
    const FP8_ONE: u8 = 0x38;

    #[test]
    fn e4m3_lut_basics() {
        let lut = e4m3_lut();
        assert_eq!(lut[0x00], 0.0);
        assert_eq!(lut[0x80], -0.0);
        assert!((lut[0x38] - 1.0).abs() < 1e-6); // +1.0
        assert!((lut[0xB8] + 1.0).abs() < 1e-6); // -1.0
        assert!((lut[0x3F] - 1.875).abs() < 1e-6); // mantissa max
        assert!((lut[0x40] - 2.0).abs() < 1e-6); // exp +1
        assert!((lut[0x78] - 256.0).abs() < 1e-6); // exp +8
        assert!((lut[0x7E] - 448.0).abs() < 1e-6); // max finite
        assert!(lut[0x7F].is_nan());
        assert!(lut[0xFF].is_nan());
        assert!((lut[0x01] - (1.0 / 512.0)).abs() < 1e-9); // smallest subnormal
    }

    #[test]
    fn fp8_dequant_with_scale() {
        let bytes = [0x38u8, 0x3F, 0x40, 0xB8];
        let mut out = vec![bf16::ZERO; 4];
        dequant_fp8_to_bf16(&bytes, 0.5, &mut out);
        let f: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert!((f[0] - 0.5).abs() < 1e-2);
        assert!((f[1] - 0.9375).abs() < 1e-2);
        assert!((f[2] - 1.0).abs() < 1e-2);
        assert!((f[3] + 0.5).abs() < 1e-2);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn nvfp4_dequant_layout() {
        let bs = 1;
        let nkv = 1;
        let hd = 16;
        let mut raw = vec![0u8; 8 + 1];
        for i in 0..8 {
            let lo = (2 * i) & 0xF;
            let hi = (2 * i + 1) & 0xF;
            raw[i] = (lo as u8) | ((hi as u8) << 4);
        }
        raw[8] = FP8_ONE;
        let mut out = vec![bf16::ZERO; bs * nkv * hd];
        dequant_4bit_block_to_bf16(&raw, bs, nkv, hd, &NVFP4_E2M1_LUT, &mut out);
        for i in 0..hd {
            let expected = NVFP4_E2M1_LUT[i];
            assert!(
                (out[i].to_f32() - expected).abs() < 1e-2,
                "elem {i}: expected {expected}, got {}",
                out[i].to_f32(),
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn turbo4_dequant_layout() {
        // Same nibble pattern, different LUT — verifies the LUT is honored.
        let bs = 1;
        let nkv = 1;
        let hd = 16;
        let mut raw = vec![0u8; 8 + 1];
        for i in 0..8 {
            raw[i] = ((2 * i) as u8 & 0xF) | (((2 * i + 1) as u8 & 0xF) << 4);
        }
        raw[8] = FP8_ONE;
        let mut out = vec![bf16::ZERO; hd];
        dequant_4bit_block_to_bf16(&raw, bs, nkv, hd, &TURBO4_LUT, &mut out);
        for i in 0..hd {
            assert!(
                (out[i].to_f32() - TURBO4_LUT[i]).abs() < 1e-2,
                "elem {i}: expected {}, got {}",
                TURBO4_LUT[i],
                out[i].to_f32(),
            );
        }
    }

    #[test]
    fn turbo3_unpack_round_trip() {
        let bs = 1;
        let nkv = 1;
        let hd = 16;
        let head_data_bytes = hd * 3 / 8; // 6
        let mut raw = vec![0u8; head_data_bytes + 1];
        let pack8 = |vals: [u8; 8]| -> [u8; 3] {
            let b0 = vals[0] | (vals[1] << 3) | (vals[2] << 6);
            let b1 = (vals[2] >> 2) | (vals[3] << 1) | (vals[4] << 4) | (vals[5] << 7);
            let b2 = (vals[5] >> 1) | (vals[6] << 2) | (vals[7] << 5);
            [b0, b1, b2]
        };
        let t0 = pack8([0, 1, 2, 3, 4, 5, 6, 7]);
        let t1 = pack8([7, 6, 5, 4, 3, 2, 1, 0]);
        raw[..3].copy_from_slice(&t0);
        raw[3..6].copy_from_slice(&t1);
        raw[head_data_bytes] = FP8_ONE;
        let mut out = vec![bf16::ZERO; bs * nkv * hd];
        dequant_turbo3_block_to_bf16(&raw, bs, nkv, hd, &mut out);
        let expect: Vec<f32> = (0..8u32)
            .map(|i| TURBO3_LUT[i as usize])
            .chain((0..8u32).rev().map(|i| TURBO3_LUT[i as usize]))
            .collect();
        for (i, e) in expect.iter().enumerate() {
            assert!(
                (out[i].to_f32() - e).abs() < 1e-2,
                "elem {i}: expected {e}, got {}",
                out[i].to_f32(),
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn turbo8_dequant_layout() {
        // 2026-04-28: Turbo8 scales are BF16 (2 bytes), not FP8 (1 byte).
        // 1 token × 1 kv_head × hd=16, 1 group of 16. Total bytes = 16 (data) + 2 (scale).
        let bs = 1;
        let nkv = 1;
        let hd = 16;
        let mut raw = vec![FP8_ONE; hd + 2];
        // Scale = 1.0 in BF16 (= 0x3F80 little-endian bytes [0x80, 0x3F]).
        let scale_bytes = bf16::from_f32(1.0).to_le_bytes();
        raw[hd] = scale_bytes[0];
        raw[hd + 1] = scale_bytes[1];
        let mut out = vec![bf16::ZERO; bs * nkv * hd];
        dequant_turbo8_block_to_bf16(&raw, bs, nkv, hd, &mut out);
        for i in 0..hd {
            assert!(
                (out[i].to_f32() - 1.0).abs() < 1e-2,
                "elem {i}: expected 1.0, got {}",
                out[i].to_f32(),
            );
        }
    }

    #[test]
    fn multi_head_multi_token_consistency() {
        // 2 tokens × 2 kv_heads × hd=16 → 4 (token,head) groups, each 8 data
        // bytes + 1 scale byte. Deterministically build & verify the indexing.
        let bs = 2;
        let nkv = 2;
        let hd = 16;
        let head_data_bytes = hd / 2;
        let head_scale_bytes = hd / NVFP4_GROUP_SIZE;
        let token_data_stride = nkv * head_data_bytes;
        let token_scale_stride = nkv * head_scale_bytes;
        let data_section_bytes = bs * token_data_stride;
        let scale_section_bytes = bs * token_scale_stride;
        let mut raw = vec![0u8; data_section_bytes + scale_section_bytes];
        // Each (tok, kv_head) gets its own marker pattern.
        for tok in 0..bs {
            for kv_h in 0..nkv {
                let d_off = tok * token_data_stride + kv_h * head_data_bytes;
                let nibble_lo = ((tok * 2 + kv_h) as u8) & 0xF;
                let nibble_hi = (((tok * 2 + kv_h) + 8) as u8) & 0xF;
                for byte_idx in 0..head_data_bytes {
                    raw[d_off + byte_idx] = nibble_lo | (nibble_hi << 4);
                }
                let s_off = data_section_bytes + tok * token_scale_stride + kv_h * head_scale_bytes;
                raw[s_off] = FP8_ONE;
            }
        }
        let mut out = vec![bf16::ZERO; bs * nkv * hd];
        dequant_4bit_block_to_bf16(&raw, bs, nkv, hd, &NVFP4_E2M1_LUT, &mut out);
        for tok in 0..bs {
            for kv_h in 0..nkv {
                let nibble_lo = (tok * 2 + kv_h) & 0xF;
                let nibble_hi = ((tok * 2 + kv_h) + 8) & 0xF;
                let exp_lo = NVFP4_E2M1_LUT[nibble_lo];
                let exp_hi = NVFP4_E2M1_LUT[nibble_hi];
                let base = (tok * nkv + kv_h) * hd;
                for elem in 0..hd {
                    let expected = if elem % 2 == 0 { exp_lo } else { exp_hi };
                    assert!(
                        (out[base + elem].to_f32() - expected).abs() < 1e-2,
                        "tok={tok} kv_h={kv_h} elem={elem}: expected {expected}, got {}",
                        out[base + elem].to_f32(),
                    );
                }
            }
        }
    }
}
