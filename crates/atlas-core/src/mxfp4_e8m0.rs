// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 native MXFP4 host unpack: packed E2M1 nibbles + E8M0 scales.
//!
//! This is the SSOT for the CPU loop previously inlined in
//! `spark_model::weight_map::dequant_nvfp4_e8m0_to_bf16`. K3 official experts
//! (`weight_packed` + `weight_scale`) call this; they do not grow a second
//! dequant stack. CUDA `mx_block_scale<true>` must stay byte-exact with
//! [`fp8_e8m0_to_f32`].
//!
//! TODO(S5 GPU): GB10 GEMM still lives in
//! `kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu`
//! (`moe_w4a16_grouped_gemm_ptrtable_e8m0`). Do not copy it into kimi-k3.

use anyhow::{Result, ensure};

/// E2M1 nibble → f32. Same table as DSV4 `dequant_nvfp4_e8m0_to_bf16`.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Native MXFP4 group size on the DSV4 GPU lander (`quantized_mxfp4_e8m0`).
pub const GROUP_SIZE: usize = 32;

/// FP8 E8M0 → f32 (unsigned exponent, bias 127). exp=0 and exp=255 → 0.0.
const FP8_E8M0_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i: u32 = 0;
    while i < 256 {
        let exp = i as u8;
        table[i as usize] = if exp == 0 || exp == 255 {
            0.0f32
        } else {
            f32::from_bits((exp as u32) << 23)
        };
        i += 1;
    }
    table
};

/// Convert one E8M0 scale byte to f32 (branchless LUT).
#[inline(always)]
pub fn fp8_e8m0_to_f32(bits: u8) -> f32 {
    FP8_E8M0_LUT[bits as usize]
}

/// Unpack packed E2M1 `[n, k/2]` + E8M0 scales to f32 `[n, k]`.
///
/// Block size is `n*k / scales.len()` (DSV4 infers; GPU lander requires 32).
/// Even flat index = low nibble, odd = high nibble.
pub fn dequant_nvfp4_e8m0_to_f32(
    packed: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let total = n.checked_mul(k).expect("mxfp4 n*k");
    ensure!(
        total.is_multiple_of(2),
        "MXFP4 E8M0: n*k={total} is odd (need even nibble count)"
    );
    let packed_bytes = total / 2;
    ensure!(
        packed.len() == packed_bytes,
        "MXFP4 E8M0: packed {} B, expected {packed_bytes} for [{n},{k}]",
        packed.len()
    );
    let num_groups = scales.len();
    ensure!(
        num_groups > 0 && total.is_multiple_of(num_groups),
        "MXFP4 E8M0: weight elems {total} not divisible by E8M0 scale groups {num_groups}"
    );
    let block = total / num_groups;
    let mut out = vec![0.0f32; total];
    for (group, &sb) in scales.iter().enumerate() {
        let block_scale = fp8_e8m0_to_f32(sb);
        for elem in 0..block {
            let flat_idx = group * block + elem;
            let byte_idx = flat_idx / 2;
            let nibble = if flat_idx.is_multiple_of(2) {
                packed[byte_idx] & 0x0F
            } else {
                (packed[byte_idx] >> 4) & 0x0F
            };
            out[flat_idx] = E2M1[nibble as usize] * block_scale;
        }
    }
    Ok(out)
}

/// Same unpack, then `f32_to_bf16` — matches DSV4 host upload.
pub fn dequant_nvfp4_e8m0_to_bf16(
    packed: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<u16>> {
    let f = dequant_nvfp4_e8m0_to_f32(packed, scales, n, k)?;
    Ok(f.into_iter().map(crate::numeric::f32_to_bf16).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_pow2_and_sentinels() {
        assert_eq!(fp8_e8m0_to_f32(0), 0.0);
        assert_eq!(fp8_e8m0_to_f32(255), 0.0);
        assert_eq!(fp8_e8m0_to_f32(127), 1.0);
        assert_eq!(fp8_e8m0_to_f32(128), 2.0);
        assert_eq!(fp8_e8m0_to_f32(126), 0.5);
    }
}
