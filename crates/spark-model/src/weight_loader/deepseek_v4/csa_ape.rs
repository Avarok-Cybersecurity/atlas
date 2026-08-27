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
    use std::collections::HashMap;

    use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::load_ape_f32;

    const KEY: &str = "layers.2.attn.compressor.ape";

    fn store_with(ptr: DevicePtr, dtype: WeightDtype, shape: Vec<usize>) -> WeightStore {
        WeightStore::from_map(HashMap::from([(
            KEY.to_string(),
            WeightTensor { ptr, shape, dtype },
        )]))
    }

    #[test]
    fn fp32_checkpoint_pointer_passes_through_unchanged() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(32).unwrap();
        let store = store_with(ptr, WeightDtype::FP32, vec![2, 4]);

        assert_eq!(load_ape_f32(&store, KEY, &gpu).unwrap(), ptr);
    }

    #[test]
    fn bf16_checkpoint_is_widened_into_a_new_exact_fp32_buffer() {
        let gpu = MockGpuBackend::new();
        let bf16 = [0x3d97u16, 0xbfc0, 0x3f00, 0x4140];
        let source: Vec<u8> = bf16.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let source_ptr = gpu.alloc(source.len()).unwrap();
        gpu.copy_h2d(&source, source_ptr).unwrap();
        let store = store_with(source_ptr, WeightDtype::BF16, vec![2, 2]);

        let widened_ptr = load_ape_f32(&store, KEY, &gpu).unwrap();
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
    fn missing_required_ape_names_the_tensor() {
        let gpu = MockGpuBackend::new();
        let error = load_ape_f32(&WeightStore::empty(), KEY, &gpu)
            .unwrap_err()
            .to_string();
        assert!(error.contains(KEY), "{error}");
    }

    #[test]
    fn unsupported_checkpoint_dtype_names_the_tensor_and_contract() {
        let gpu = MockGpuBackend::new();
        let store = store_with(DevicePtr::NULL, WeightDtype::FP8E4M3, vec![2, 2]);
        let error = load_ape_f32(&store, KEY, &gpu).unwrap_err().to_string();

        assert!(error.contains(KEY), "{error}");
        assert!(error.contains("FP8E4M3"), "{error}");
        assert!(error.contains("indexes ape as F32"), "{error}");
    }
}
