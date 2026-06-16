// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 tensor detection and dequantization to BF16.

use super::ScaleDtype;
use super::lut::{bf16_bytes_to_f32, f32_to_bf16, fp8_e4m3_to_f32};

// ── Detection ──

/// Check if a safetensors dtype string represents an FP8 tensor.
///
/// Recognizes: `"F8_E4M3"`, `"float8_e4m3fn"`, `"float8_e4m3fnuz"`.
pub fn is_fp8_tensor(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "float8_e4m3fn" | "float8_e4m3fnuz")
}

// ── Per-tensor dequant ──

/// Dequantize FP8 E4M3 per-tensor-scaled data to BF16 bytes.
///
/// Each FP8 byte is converted to f32, multiplied by the per-tensor scale,
/// then truncated to BF16. Returns a Vec of BF16 bytes (2 bytes per element).
///
/// This matches the on-disk layout used by `compressed-tensors` FP8 with a
/// single `weight_scale` scalar per weight tensor.
pub fn dequant_fp8_pertensor_to_bf16(fp8_data: &[u8], scale: f32) -> Vec<u8> {
    fp8_data
        .iter()
        .flat_map(|&byte| {
            let val = fp8_e4m3_to_f32(byte) * scale;
            f32_to_bf16(val).to_le_bytes()
        })
        .collect()
}

// ── Block-scaled dequant ──

/// Dequantize FP8 E4M3 block-scaled tensor to BF16 bytes.
///
/// Layout:
///   - `fp8_data`: FP8 E4M3 weight bytes, row-major [N, K] (N*K bytes total).
///   - `scales`: Per-block scale_inv values. Format depends on `scale_dtype`.
///   - `n`, `k`: Logical weight dimensions.
///   - `block_size`: Block size along each dimension (e.g. 128 for [128, 128] blocks).
///   - `scale_dtype`: Precision of scale values (BF16 or FP32).
///
/// Dequantization formula: `bf16[i,j] = fp8[i,j] * scale_inv[i/block, j/block]`
///
/// Returns a Vec of BF16 bytes (2 bytes per element, N*K*2 total).
pub fn dequant_fp8_block_to_bf16(
    fp8_data: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
    block_size: usize,
    scale_dtype: ScaleDtype,
) -> Vec<u8> {
    assert_eq!(
        fp8_data.len(),
        n * k,
        "FP8 data length mismatch: expected {}, got {}",
        n * k,
        fp8_data.len()
    );

    let sn = n.div_ceil(block_size);
    let sk = k.div_ceil(block_size);

    let scale_elem_bytes = match scale_dtype {
        ScaleDtype::Bf16 => 2,
        ScaleDtype::Fp32 => 4,
    };
    let expected_scale_bytes = sn * sk * scale_elem_bytes;
    assert_eq!(
        scales.len(),
        expected_scale_bytes,
        "Scale buffer length mismatch: expected {expected_scale_bytes}, got {}",
        scales.len(),
    );

    let read_scale = |scale_idx: usize| -> f32 {
        let offset = scale_idx * scale_elem_bytes;
        match scale_dtype {
            ScaleDtype::Bf16 => bf16_bytes_to_f32([scales[offset], scales[offset + 1]]),
            ScaleDtype::Fp32 => f32::from_le_bytes([
                scales[offset],
                scales[offset + 1],
                scales[offset + 2],
                scales[offset + 3],
            ]),
        }
    };

    let total = n * k;
    let mut bf16_out = vec![0u8; total * 2];

    for row in 0..n {
        let scale_row = row / block_size;
        for col in 0..k {
            let scale_col = col / block_size;
            let scale_idx = scale_row * sk + scale_col;
            let scale_val = read_scale(scale_idx);

            let fp8_byte = fp8_data[row * k + col];
            let val = fp8_e4m3_to_f32(fp8_byte) * scale_val;
            let bf16_val = f32_to_bf16(val);

            let out_idx = (row * k + col) * 2;
            let [lo, hi] = bf16_val.to_le_bytes();
            bf16_out[out_idx] = lo;
            bf16_out[out_idx + 1] = hi;
        }
    }

    bf16_out
}
