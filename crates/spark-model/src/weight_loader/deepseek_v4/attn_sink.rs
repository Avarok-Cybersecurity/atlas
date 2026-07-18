// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 per-head attention sink (`s_aux`) — canonical FP32 dtype contract.
//!
//! The checkpoint ships `layers.N.attn.attn_sink` as F32 `[num_q_heads]`, and every
//! DS4F sink-consuming kernel indexes it as `const float*`. This module is the single
//! place that normalizes the loaded buffer to that contract, so a kernel can never
//! index an fp32 buffer as bf16 (a 2-byte stride over 4-byte elements that reads the
//! low-mantissa half of the wrong element — historically hard-zeroing 7 query heads
//! whose misread decoded large-positive and collapsed the softmax value accumulator).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// Load the per-head attention sink as the canonical device **FP32** buffer.
///
/// - **F32 checkpoint** (the DSpark case): pass the store buffer through unchanged
///   (byte no-op vs the raw pointer).
/// - **BF16 checkpoint**: widen once here into a freshly allocated FP32 buffer
///   (process-lifetime, same ownership model as the other derived loader buffers,
///   e.g. `main_inv_freq`).
/// - **Missing**: `DevicePtr::NULL` (the layer has no sink; kernels skip the branch).
/// - **Any other dtype**: fail loudly with the tensor key and dtype.
pub(super) fn load_attn_sink_f32(
    store: &WeightStore,
    key: &str,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let t = match store.get(key) {
        Err(_) => return Ok(DevicePtr::NULL), // checkpoint has no sink for this layer
        Ok(t) => t,
    };
    match t.dtype {
        WeightDtype::FP32 => Ok(t.ptr),
        WeightDtype::BF16 => {
            let mut bf16_buf = vec![0u8; t.num_elements() * 2];
            gpu.copy_d2h(t.ptr, &mut bf16_buf)?;
            let f32_buf = bf16_bytes_to_f32_bytes(&bf16_buf);
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(ptr)
        }
        other => anyhow::bail!(
            "DeepSeek-V4 attn_sink '{key}': unexpected dtype {:?} \
             (sink kernels require F32; only F32 pass-through or BF16 widening supported)",
            other
        ),
    }
}

/// Widen a bf16 byte buffer to f32 bytes, exactly (no rounding): bf16 is the high
/// 16 bits of the f32 word, so the two bf16 bytes become the high two f32 bytes and
/// the low two are zero. Pure/deterministic → unit-tested.
pub(crate) fn bf16_bytes_to_f32_bytes(bf16: &[u8]) -> Vec<u8> {
    let n = bf16.len() / 2;
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        out[i * 4 + 2] = bf16[i * 2];
        out[i * 4 + 3] = bf16[i * 2 + 1];
    }
    out
}

#[cfg(test)]
mod attn_sink_dtype_tests {
    //! Regression tests for the DS4F `attn_sink` FP32 contract. The checkpoint
    //! ships `attn_sink` as F32 `[num_q_heads]`; the kernels index it as
    //! `const float*`. The historical defect indexed the same fp32 buffer as
    //! bf16 (2-byte stride), so even head `2k` read the low-mantissa half of
    //! fp32 element `k`; 7 such reads were large-positive and collapsed the
    //! softmax value accumulator to exactly zero. These tests lock the true
    //! fp32 read and reproduce the old bf16 misread's exact seven-head set.
    use super::bf16_bytes_to_f32_bytes;

    // First 128 bytes of the banked device `attn_sink` buffer for L0 (q5),
    // captured from the running engine (tap G3_L0_attn_sink_r0.bin, node n3).
    // 128 bytes = 32 fp32 elements (heads 0..31), i.e. what a bf16 index reads
    // for heads 0..63.
    const DEV_SINK_BYTES: [u8; 128] = [
        0x96, 0x80, 0x84, 0x3f, 0x47, 0xf9, 0x01, 0x3f, 0xb3, 0xd3, 0xfb, 0x3e, 0x76, 0x45, 0x4b,
        0x3e, 0x2f, 0x0e, 0xbe, 0x3f, 0x52, 0x76, 0x23, 0x3f, 0x42, 0x5c, 0xbd, 0xbd, 0xbd, 0xb7,
        0xc3, 0x3f, 0x07, 0xc6, 0xd0, 0x3e, 0xd1, 0xd9, 0xb6, 0x3f, 0x9d, 0x7b, 0x70, 0x3e, 0x25,
        0x3c, 0xe2, 0xbd, 0x37, 0xe8, 0x31, 0x3f, 0xaa, 0xb8, 0x6b, 0x3f, 0x51, 0x49, 0x93, 0x3f,
        0xc2, 0xf3, 0x10, 0x3e, 0x32, 0xe5, 0x8a, 0x3f, 0xad, 0x46, 0xe5, 0xbe, 0xf0, 0x33, 0xd0,
        0x3f, 0x57, 0x9f, 0x25, 0x3f, 0x73, 0x33, 0xc1, 0x3f, 0xa2, 0xba, 0xa3, 0xbf, 0x6b, 0xb2,
        0x48, 0x3f, 0x75, 0x3f, 0x94, 0xbe, 0x15, 0x51, 0x6f, 0xbf, 0x72, 0x14, 0xf9, 0x3c, 0x60,
        0x01, 0x63, 0x3f, 0x05, 0xab, 0x8b, 0x3f, 0xbc, 0x94, 0x8a, 0x3f, 0xfb, 0x13, 0x42, 0x3f,
        0x79, 0x17, 0x28, 0x3e, 0x33, 0x08, 0x29, 0x3f,
    ];

    // True fp32 sink logits (checkpoint), heads 0..31.
    const SINK_TRUE_0_31: [f32; 32] = [
        1.035, 0.508, 0.492, 0.199, 1.485, 0.639, -0.092, 1.529, 0.408, 1.429, 0.235, -0.11, 0.695,
        0.921, 1.151, 0.142, 1.085, -0.448, 1.627, 0.647, 1.509, -1.279, 0.784, -0.29, -0.935,
        0.03, 0.887, 1.091, 1.083, 0.758, 0.164, 0.66,
    ];

    fn f32_from_le(b: &[u8]) -> f32 {
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    #[test]
    fn bf16_widen_is_exact() {
        let cases: [(u16, f32); 4] = [
            (0x3F80, 1.0),
            (0x3FC0, 1.5),
            (0xBF80, -1.0),
            (0xBDBC, f32::from_bits(0xBDBC0000)),
        ];
        for (bits, expect) in cases {
            let bf = bits.to_le_bytes();
            let out = bf16_bytes_to_f32_bytes(&bf);
            assert_eq!(out.len(), 4);
            assert_eq!(f32_from_le(&out), expect, "bf16 {bits:#06x} widened");
            assert_eq!(out[0], 0);
            assert_eq!(out[1], 0);
        }
        for &v in &[0.5f32, -3.25, 12.0, 0.235] {
            let hi = (v.to_bits() >> 16) as u16;
            let widened = f32_from_le(&bf16_bytes_to_f32_bytes(&hi.to_le_bytes()));
            assert_eq!(widened.to_bits(), (hi as u32) << 16);
        }
    }

    #[test]
    fn fp32_read_returns_true_sink_values() {
        for h in 0..32usize {
            let v = f32_from_le(&DEV_SINK_BYTES[h * 4..h * 4 + 4]);
            assert!(
                (v - SINK_TRUE_0_31[h]).abs() < 1e-3,
                "head {h}: fp32 read {v} != true {}",
                SINK_TRUE_0_31[h]
            );
        }
    }

    #[test]
    fn bf16_misread_reproduces_seven_zeroed_heads() {
        const MAX_LOGIT: f32 = 6.0;
        let mut zeroed = Vec::new();
        for h in 0..64usize {
            let off = 2 * h;
            let bits = (DEV_SINK_BYTES[off] as u32) | ((DEV_SINK_BYTES[off + 1] as u32) << 8);
            let sg = f32::from_bits(bits << 16);
            let m_new = MAX_LOGIT.max(sg);
            let exp_old = (MAX_LOGIT - m_new).min(0.0).exp();
            if exp_old < 1e-6 {
                zeroed.push(h);
            }
        }
        assert_eq!(zeroed, vec![6, 10, 12, 20, 28, 34, 48], "bug fingerprint");
        assert!(zeroed.iter().all(|h| h % 2 == 0));
    }
}
