// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. Source: Lfm3RotaryEmbedding.compute_default_rope_parameters (modeling_lfm3.py).

//! Responsibility: LFM3's two RoPE inverse-frequency tables, computed in f64
//! on the host and narrowed once to f32, and their upload.
//!
//! LFM3 keeps the frequency progression of the WHOLE head for both layer
//! kinds (`1 / theta^(2j / head_dim)`, `j in 0..head_dim/2`). A layer type
//! that is "partially rotated" does not shrink the rotated dimension: the
//! first `int(partial_rotary_factor * head_dim) // 2` frequencies are kept and
//! the rest are EXACTLY zero, which leaves those dimensions unrotated
//! (`cos 0 = 1`, `sin 0 = 0`). That is not the usual `rotary_dim = factor *
//! head_dim` computation (which would use `1 / theta^(2j / rotary_dim)`), so
//! it is a table, not a flag: `rope_yarn_scaled` reads the table with
//! `rotary_dim = head_dim` and `attention_factor = 1`.
//!
//! | layer kind | theta | rotated pairs of 64 |
//! |---|---|---|
//! | sliding_attention | 10 000 | all 64 |
//! | full_attention | 1 000 000 | the first 32, then 32 exact zeros |

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

/// Rotated pairs for a layer kind: `int(factor * head_dim) // 2` (HF).
pub fn rotated_pairs(head_dim: usize, partial_rotary_factor: f64) -> usize {
    ((partial_rotary_factor * head_dim as f64) as usize) / 2
}

/// The table in f64: `head_dim / 2` entries, the first `pairs` following the
/// full-head progression and the rest exactly `0.0`.
pub fn inv_freq_f64(theta: f64, head_dim: usize, pairs: usize) -> Vec<f64> {
    (0..head_dim / 2)
        .map(|j| {
            if j < pairs {
                1.0 / theta.powf((2 * j) as f64 / head_dim as f64)
            } else {
                0.0
            }
        })
        .collect()
}

/// [`inv_freq_f64`] narrowed once to f32 (the upload format).
pub fn inv_freq_f32(theta: f64, head_dim: usize, pairs: usize) -> Vec<f32> {
    inv_freq_f64(theta, head_dim, pairs)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// The sliding-attention layers' table (theta 10 000, every pair rotates).
pub fn sliding_table(theta: f64, head_dim: usize) -> Vec<f32> {
    inv_freq_f32(theta, head_dim, head_dim / 2)
}

/// The full-attention layers' table (zero-frequency tail).
pub fn full_table(theta: f64, head_dim: usize, partial_rotary_factor: f64) -> Vec<f32> {
    inv_freq_f32(
        theta,
        head_dim,
        rotated_pairs(head_dim, partial_rotary_factor),
    )
}

/// Both tables on the device. Each is allocated once and shared by every
/// layer of its kind; the store's derived ledger owns them so teardown frees
/// them (#736).
#[derive(Clone, Copy, Debug)]
pub struct RopeTables {
    /// `[head_dim / 2]` f32, theta 10 000, all pairs rotate.
    pub sliding: DevicePtr,
    /// `[head_dim / 2]` f32, full-attention theta, zero tail.
    pub full: DevicePtr,
    /// Rotary dimension handed to `set_rope_overrides`: the whole head.
    pub rotary_dim: u32,
}

/// Upload one f32 table and register it with the store's derived ledger.
pub fn upload(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    table: &[f32],
    label: &'static str,
) -> Result<DevicePtr> {
    ensure!(!table.is_empty(), "lfm3 {label}: empty RoPE table");
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu
        .alloc(bytes.len())
        .with_context(|| format!("allocate lfm3 {label}"))?;
    gpu.copy_h2d(&bytes, ptr)?;
    store.derived().adopt(label, ptr, bytes.len());
    Ok(ptr)
}

impl RopeTables {
    /// Build and upload both tables.
    pub fn upload(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        head_dim: usize,
        full_theta: f64,
        full_partial_rotary_factor: f64,
        sliding_theta: f64,
    ) -> Result<Self> {
        ensure!(
            head_dim >= 2 && head_dim.is_multiple_of(2),
            "lfm3 head_dim {head_dim} must be even and at least 2"
        );
        Ok(Self {
            sliding: upload(
                gpu,
                store,
                &sliding_table(sliding_theta, head_dim),
                "lfm3 sliding rope table",
            )?,
            full: upload(
                gpu,
                store,
                &full_table(full_theta, head_dim, full_partial_rotary_factor),
                "lfm3 full rope table",
            )?,
            rotary_dim: head_dim as u32,
        })
    }
}
