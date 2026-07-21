// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 CSA/HCA compressor absolute-position encoding (`ape`) — canonical
//! FP32 dtype contract.
//!
//! The checkpoint ships `layers.N.attn.compressor.ape` (the `position_bias` added
//! to the per-window gate before the compressor's per-dim softmax) as F32
//! `[ratio, proj_dim]` on every compressed layer (CSA ratio 4 → `[4, 2*head_dim]`;
//! HCA ratio 128 → `[128, head_dim]`). The `csa_compress` kernel indexes it as
//! `const float*`. This module is the single place that normalizes the loaded
//! buffer to that contract, so the kernel can never index an fp32 buffer as bf16
//! (a 2-byte stride over 4-byte elements reads half of the wrong element and
//! decodes to non-physical magnitudes ≈ ±1e38, corrupting the window softmax on
//! every compressed layer L2–L42). Same defect class as the `attn_sink` #341 fix.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// Load the compressor `ape` as the canonical device **FP32** buffer.
///
/// - **F32 checkpoint** (the DSpark case): pass the store buffer through unchanged
///   (byte no-op vs the raw pointer).
/// - **BF16 checkpoint**: widen once here into a freshly allocated FP32 buffer
///   (process-lifetime, same ownership model as the other derived loader buffers).
/// - **Any other dtype**: fail loudly with the tensor key and dtype.
pub(super) fn load_ape_f32(
    store: &WeightStore,
    key: &str,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let t = store.get(key)?;
    match t.dtype {
        WeightDtype::FP32 => Ok(t.ptr),
        WeightDtype::BF16 => {
            let mut bf16_buf = vec![0u8; t.num_elements() * 2];
            gpu.copy_d2h(t.ptr, &mut bf16_buf)?;
            let f32_buf = super::attn_sink::bf16_bytes_to_f32_bytes(&bf16_buf);
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(ptr)
        }
        other => anyhow::bail!(
            "DeepSeek-V4 compressor.ape '{key}': unexpected dtype {:?} \
             (csa_compress indexes ape as F32; only F32 pass-through or BF16 widening supported)",
            other
        ),
    }
}

#[cfg(test)]
mod csa_ape_dtype_tests {
    //! Regression tests for the DS4F `compressor.ape` FP32 contract. The checkpoint
    //! ships `ape` as F32; `csa_compress` indexes it as `const float*`. The historical
    //! defect (this fix) indexed the same fp32 buffer as bf16 (2-byte stride), reading
    //! the low half of the wrong element and decoding to non-physical magnitudes that
    //! corrupted the window softmax gate. These tests lock the true fp32 read and
    //! reproduce the old bf16-misread's garbage.
    use super::super::attn_sink::bf16_bytes_to_f32_bytes;

    /// A representative F32 ape row (checkpoint-native values, slot0/dim0 = 0.074).
    const APE_TRUE: [f32; 8] = [0.074, 1.5, -2.3, 0.5, 0.031, -1.1, 3.25, -0.008];

    /// bf16 → f32 (bf16 is the high 16 bits of the f32 word).
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// The kernel-arg fix: reading the fp32 buffer as fp32 returns the true values.
    #[test]
    fn fp32_read_returns_true_ape() {
        let mut bytes = Vec::new();
        for &v in &APE_TRUE {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for (i, &t) in APE_TRUE.iter().enumerate() {
            let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            assert!((v - t).abs() < 1e-6, "elem {i}: fp32 read {v} != true {t}");
        }
    }

    /// The bug: indexing that same fp32 byte stream as bf16 (2-byte stride) reads across
    /// the 4-byte element boundaries, so the decoded sequence diverges grossly from the
    /// true F32 values — exactly why an fp32 ape must never be read as bf16. (The runtime
    /// tap saw magnitudes to ±1e38 at the specific straddles; here we only require gross
    /// divergence, which is robust.)
    #[test]
    fn bf16_misread_of_fp32_diverges_grossly() {
        let mut bytes = Vec::new();
        for &v in &APE_TRUE {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // bf16 element i = bytes[2i..2i+2] (little-endian), the kernel's stride.
        let n_bf16 = bytes.len() / 2;
        let misread: Vec<f32> = (0..n_bf16)
            .map(|i| bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
            .collect();
        // Compare the first APE_TRUE.len() bf16 reads (what the kernel indexed as ape[0..8])
        // against the true F32 values — a majority must differ materially.
        let wrong = APE_TRUE
            .iter()
            .zip(&misread)
            .filter(|(t, m)| (**t - **m).abs() > 1e-2)
            .count();
        assert!(
            wrong > APE_TRUE.len() / 2,
            "bf16 misread must corrupt most elements; only {wrong} of {} diverged",
            APE_TRUE.len()
        );
    }

    /// BF16-checkpoint fallback path: widening is the exact high-half embedding.
    #[test]
    fn bf16_widen_roundtrip_exact() {
        for &v in &[0.074f32, -1.5, 0.5, 12.0] {
            let hi = (v.to_bits() >> 16) as u16;
            let widened_bytes = bf16_bytes_to_f32_bytes(&hi.to_le_bytes());
            let widened = f32::from_le_bytes([
                widened_bytes[0],
                widened_bytes[1],
                widened_bytes[2],
                widened_bytes[3],
            ]);
            assert_eq!(widened.to_bits(), (hi as u32) << 16);
        }
    }
}
