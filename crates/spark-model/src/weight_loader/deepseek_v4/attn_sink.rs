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
    use std::collections::HashMap;

    use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::load_attn_sink_f32;

    const KEY: &str = "layers.0.attn.attn_sink";

    fn store_with(ptr: DevicePtr, dtype: WeightDtype, elements: usize) -> WeightStore {
        WeightStore::from_map(HashMap::from([(
            KEY.to_string(),
            WeightTensor {
                ptr,
                shape: vec![elements],
                dtype,
            },
        )]))
    }

    #[test]
    fn fp32_checkpoint_pointer_passes_through_unchanged() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(16).unwrap();
        let store = store_with(ptr, WeightDtype::FP32, 4);

        assert_eq!(load_attn_sink_f32(&store, KEY, &gpu).unwrap(), ptr);
    }

    #[test]
    fn bf16_checkpoint_is_widened_into_a_new_exact_fp32_buffer() {
        let gpu = MockGpuBackend::new();
        let bf16 = [0x3f80u16, 0x3fc0, 0xbf80, 0xbdbc];
        let source: Vec<u8> = bf16.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let source_ptr = gpu.alloc(source.len()).unwrap();
        gpu.copy_h2d(&source, source_ptr).unwrap();
        let store = store_with(source_ptr, WeightDtype::BF16, bf16.len());

        let widened_ptr = load_attn_sink_f32(&store, KEY, &gpu).unwrap();
        assert_ne!(widened_ptr, source_ptr);
        let mut widened = vec![0u8; bf16.len() * 4];
        gpu.copy_d2h(widened_ptr, &mut widened).unwrap();
        let actual: Vec<u32> = widened
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual,
            bf16.iter()
                .map(|bits| (*bits as u32) << 16)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_sink_returns_null_without_allocating() {
        let gpu = MockGpuBackend::new();
        assert_eq!(
            load_attn_sink_f32(&WeightStore::empty(), KEY, &gpu).unwrap(),
            DevicePtr::NULL
        );
    }

    #[test]
    fn unsupported_checkpoint_dtype_names_the_tensor_and_contract() {
        let gpu = MockGpuBackend::new();
        let store = store_with(DevicePtr::NULL, WeightDtype::FP8E4M3, 4);
        let error = load_attn_sink_f32(&store, KEY, &gpu)
            .unwrap_err()
            .to_string();

        assert!(error.contains(KEY), "{error}");
        assert!(error.contains("FP8E4M3"), "{error}");
        assert!(error.contains("require F32"), "{error}");
    }
}
