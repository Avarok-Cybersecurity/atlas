// SPDX-License-Identifier: AGPL-3.0-only

//! Phase F: precompute YaRN inv_freq table on GPU. Computed once at
//! layer 0, returned by pointer for subsequent layers.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::gpu_alloc_or_managed;

pub(super) fn compute_yarn_inv_freq(
    config: &ModelConfig,
    rope: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    // Reference: HF transformers `_compute_yarn_parameters` in
    // `modeling_rope_utils.py`. The previous Atlas implementation used
    // the Llama-3.1 NTK-by-parts wavelength formula and mis-aliased
    // `llama_4_scaling.beta=0.1` as `low_freq_factor`, which corrupted
    // inv_freq for the lowest-frequency RoPE pairs (j ≈ 25..31) by
    // ~1.2×–2.3×.
    //
    // Correct YaRN: branch in dimension-INDEX space using
    // find_correction_range(beta_fast, beta_slow, dim, base, max_pos)
    // and a linear ramp between the two correction dims.
    //
    // Mistral params (`params.json::yarn`): alpha=1, beta=32, factor=128,
    // original_max_position_embeddings=8192. `alpha` is the low-rotation
    // cutoff (HF `beta_slow`), `beta` is the high-rotation cutoff (HF
    // `beta_fast`).
    let factor = if config.yarn_factor > 0.0 {
        config.yarn_factor
    } else {
        128.0
    };
    let beta_fast = if config.yarn_beta_fast > 0.0 {
        config.yarn_beta_fast
    } else {
        32.0
    };
    let beta_slow = if config.yarn_beta_slow > 0.0 {
        config.yarn_beta_slow
    } else {
        1.0
    };
    let original_max_pos = if config.yarn_original_max_position_embeddings > 0 {
        config.yarn_original_max_position_embeddings as f32
    } else {
        8192.0
    };
    let dim_f = rope as f32;
    let theta_f = config.rope_theta as f32;
    let n_pairs = rope / 2;

    // find_correction_dim(num_rot) = dim * ln(max_pos / (num_rot * 2π))
    //                              / (2 * ln(base))
    let find_correction_dim = |num_rot: f32| -> f32 {
        (dim_f * (original_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * theta_f.ln())
    };
    let low = find_correction_dim(beta_fast).floor().max(0.0);
    let high = find_correction_dim(beta_slow).ceil().min((rope - 1) as f32);
    let ramp_denom = if (high - low).abs() < 1e-6 {
        high - low + 0.001
    } else {
        high - low
    };

    let mut inv_freq_table = vec![0.0f32; n_pairs];
    for j in 0..n_pairs {
        let pos_freq = theta_f.powf((2 * j) as f32 / dim_f);
        let inv_freq_extrap = 1.0 / pos_freq;
        let inv_freq_interp = 1.0 / (factor * pos_freq);

        // Linear ramp in dim-index space: 0 at j=low, 1 at j=high, clamped.
        let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;

        // inv_freq = interpolation*(1-extrap_factor) + extrapolation*extrap_factor
        //          = interpolation*ramp            + extrapolation*(1-ramp)
        inv_freq_table[j] =
            inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
    }
    let bytes: Vec<u8> = inv_freq_table
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ptr = gpu_alloc_or_managed(gpu, bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    tracing::info!(
        "YaRN inv_freq: {} pairs, factor={factor}, beta_fast={beta_fast}, \
         beta_slow={beta_slow}, max_pos={original_max_pos}, low_dim={low:.1}, high_dim={high:.1}",
        n_pairs,
    );
    tracing::info!(
        "YaRN inv_freq sample: [0]={:.6e} [12]={:.6e} [25]={:.6e} [31]={:.6e}",
        inv_freq_table.first().copied().unwrap_or(0.0),
        inv_freq_table.get(12).copied().unwrap_or(0.0),
        inv_freq_table.get(25).copied().unwrap_or(0.0),
        inv_freq_table.get(31).copied().unwrap_or(0.0),
    );
    Ok(ptr)
}
