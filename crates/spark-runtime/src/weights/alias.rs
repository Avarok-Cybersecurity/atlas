// SPDX-License-Identifier: AGPL-3.0-only

//! GGUF stacked-MoE expert slices share one `gpu.alloc`. `cuMemFree` accepts
//! only the allocation base; freeing expert 0's pointer (the base) unmaps
//! every sibling, and freeing an offset is `CUDA_ERROR_ILLEGAL_ADDRESS`.

use super::WeightStore;
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Context, Result};

impl WeightStore {
    /// True when another store entry is a same-sized `.offset()` sibling of
    /// `ptr` (GGUF `ExpertStack` fan-out). Owned unique allocs return false:
    /// adjacent same-size tensors are not slices (safetensors per-expert
    /// allocs would otherwise look aliased by pointer stride).
    pub fn has_sliced_sibling(&self, ptr: DevicePtr, slice_bytes: usize) -> bool {
        if slice_bytes == 0 {
            return false;
        }
        let queried_is_sliced = self.weights.values().any(|t| t.ptr == ptr && !t.owned);
        if !queried_is_sliced {
            return false;
        }
        let sz = slice_bytes as u64;
        self.weights.values().any(|t| {
            !t.owned && t.ptr != ptr && t.byte_size() == slice_bytes && {
                let (lo, hi) = if t.ptr.0 < ptr.0 {
                    (t.ptr.0, ptr.0)
                } else {
                    (ptr.0, t.ptr.0)
                };
                hi > lo && (hi - lo).is_multiple_of(sz)
            }
        })
    }

    /// `gpu.free` each stacked-expert family base whose names start with
    /// `prefix` (e.g. `model.layers.0.mlp.experts.`). One free per
    /// `{gate,up,down}_proj` family. Offset views stay in the map as
    /// dangling names; they are `owned: false` so teardown will not
    /// `cuMemFree` them.
    pub fn release_sliced_bf16_stacks(&self, gpu: &dyn GpuBackend, prefix: &str) -> Result<usize> {
        let suffixes = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"];
        let mut freed = 0usize;
        for suffix in suffixes {
            let mut base: Option<DevicePtr> = None;
            for (name, t) in &self.weights {
                if t.owned || !name.starts_with(prefix) || !name.ends_with(suffix) {
                    continue;
                }
                base = Some(match base {
                    None => t.ptr,
                    Some(p) if t.ptr.0 < p.0 => t.ptr,
                    Some(p) => p,
                });
            }
            if let Some(p) = base {
                gpu.free(p)
                    .with_context(|| format!("freeing GGUF expert stack {prefix}*{suffix}"))?;
                freed += 1;
            }
        }
        Ok(freed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;
    use crate::weights::{WeightDtype, WeightStore, WeightTensor};
    use std::collections::HashMap;

    fn bf16_slice(ptr: DevicePtr, elems: usize, owned: bool) -> WeightTensor {
        if owned {
            WeightTensor::new(ptr, vec![elems], WeightDtype::BF16)
        } else {
            WeightTensor::sliced(ptr, vec![elems], WeightDtype::BF16)
        }
    }

    #[test]
    fn sliced_constructor_is_not_owned() {
        let s = WeightTensor::sliced(DevicePtr::NULL, vec![4], WeightDtype::BF16);
        let o = WeightTensor::new(DevicePtr::NULL, vec![4], WeightDtype::BF16);
        assert!(!s.owned);
        assert!(o.owned);
    }

    /// Oracle: two sliced experts of one alloc are siblings; a unique owned
    /// tensor is not. Known-bad: `gpu.free` of the offset view fails on the
    /// mock (it is not an allocation base).
    #[test]
    fn sliced_expert_stack_is_one_alloc() {
        let gpu = MockGpuBackend::new();
        let base = gpu.alloc(8).unwrap(); // 4 BF16 elems
        let e0 = base;
        let e1 = base.offset(4); // 2 elems * 2 bytes
        let mut map = HashMap::new();
        map.insert(
            "model.layers.0.mlp.experts.0.gate_proj.weight".into(),
            bf16_slice(e0, 2, false),
        );
        map.insert(
            "model.layers.0.mlp.experts.1.gate_proj.weight".into(),
            bf16_slice(e1, 2, false),
        );
        map.insert(
            "model.layers.0.mlp.shared_expert.gate_proj.weight".into(),
            bf16_slice(gpu.alloc(4).unwrap(), 2, true),
        );
        let store = WeightStore::from_map(map);
        assert!(store.has_sliced_sibling(e0, 4));
        assert!(store.has_sliced_sibling(e1, 4));
        let shared = store
            .get("model.layers.0.mlp.shared_expert.gate_proj.weight")
            .unwrap();
        assert!(!store.has_sliced_sibling(shared.ptr, shared.byte_size()));
        assert_eq!(gpu.live_alloc_count(), 2);
        assert_eq!(
            store
                .release_sliced_bf16_stacks(&gpu, "model.layers.0.mlp.experts.")
                .unwrap(),
            1
        );
        assert_eq!(gpu.live_alloc_count(), 1, "shared expert alloc still live");
        assert!(
            gpu.free(e1).is_err(),
            "known-bad: offset free must fail (not an alloc base)"
        );
    }
}
