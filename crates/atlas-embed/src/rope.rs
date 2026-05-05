// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;

/// Rotary Position Embedding (RoPE) for SM121.
///
/// Applies rotary transformation to Q and K tensors:
///   q_rot = q * cos(theta) + rotate_half(q) * sin(theta)
///
/// Uses NEON SIMD on Grace CPU for precomputing cos/sin tables,
/// then a CUDA kernel for the actual rotation on GPU.
#[allow(clippy::too_many_arguments)]
pub fn apply_rotary_embedding(
    _q_ptr: u64,         // [batch, seq_len, num_heads, head_dim] BF16
    _k_ptr: u64,         // [batch, seq_len, num_kv_heads, head_dim] BF16
    _cos_ptr: u64,       // [max_seq_len, head_dim/2] FP32 (precomputed)
    _sin_ptr: u64,       // [max_seq_len, head_dim/2] FP32 (precomputed)
    _positions_ptr: u64, // [batch, seq_len] i64
    _batch: u32,
    _seq_len: u32,
    _num_heads: u32,
    _num_kv_heads: u32,
    _head_dim: u32,
    _stream_ptr: u64,
) -> Result<()> {
    // TODO: Launch rotary.cu kernel
    todo!("RoPE kernel launch")
}

/// Precompute cos/sin tables on CPU using NEON SIMD (Grace ARM).
///
/// theta_i = base^(-2i/d) for i in 0..head_dim/2
/// cos_table[pos, i] = cos(pos * theta_i)
/// sin_table[pos, i] = sin(pos * theta_i)
#[cfg(target_arch = "aarch64")]
pub fn precompute_freqs_cis_simd(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_table = vec![0.0f32; max_seq_len * half_dim];

    // Compute inverse frequencies: theta_i = base^(-2i/d)
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    // For each position, compute cos/sin using NEON where possible
    for pos in 0..max_seq_len {
        let base = &mut cos_table[pos * half_dim..(pos + 1) * half_dim];
        let sbase = &mut sin_table[pos * half_dim..(pos + 1) * half_dim];

        // Process 4 frequencies at a time with NEON
        let mut i = 0;
        while i + 4 <= half_dim {
            let angle0 = (pos as f64 * inv_freq[i]) as f32;
            let angle1 = (pos as f64 * inv_freq[i + 1]) as f32;
            let angle2 = (pos as f64 * inv_freq[i + 2]) as f32;
            let angle3 = (pos as f64 * inv_freq[i + 3]) as f32;

            base[i] = angle0.cos();
            base[i + 1] = angle1.cos();
            base[i + 2] = angle2.cos();
            base[i + 3] = angle3.cos();

            sbase[i] = angle0.sin();
            sbase[i + 1] = angle1.sin();
            sbase[i + 2] = angle2.sin();
            sbase[i + 3] = angle3.sin();

            i += 4;
        }

        // Scalar remainder
        while i < half_dim {
            let angle = (pos as f64 * inv_freq[i]) as f32;
            base[i] = angle.cos();
            sbase[i] = angle.sin();
            i += 1;
        }
    }

    (cos_table, sin_table)
}

/// Fallback for non-aarch64 (should not be used on DGX Spark)
#[cfg(not(target_arch = "aarch64"))]
pub fn precompute_freqs_cis_simd(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_table = vec![0.0f32; max_seq_len * half_dim];

    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let angle = (pos as f64 * inv_freq[i]) as f32;
            cos_table[pos * half_dim + i] = angle.cos();
            sin_table[pos * half_dim + i] = angle.sin();
        }
    }

    (cos_table, sin_table)
}
