// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time derived weight buffers, and who frees them.
//!
//! **WHY (#736, #915).** A loader that re-encodes a checkpoint tensor — a
//! transposed twin, a row-concatenation of two projections, an NVFP4 requant —
//! allocates a buffer that lives in a *layer struct*. Layer structs have no
//! `Drop` that reaches the GPU and no `ModelResource`, so nothing frees them:
//! `TransformerModel::release_pools` walks buffers, KV cache, SSM pools,
//! `DerivedWeights` and the store, and every one of these falls through to
//! `AtlasCudaBackend::sweep_unreleased`. On 1xH100, 2026-09-11,
//! `Qwen/Qwen3.8-27B-FP8`, that sweep reclaimed **28.01 GB across 1,980
//! allocations** and warned that each was "memory whose owner is unaccounted
//! for". The sweep is a backstop, not an owner: it cannot run before the
//! backend is torn down, so those bytes are unreclaimable for the life of the
//! model even when the layer that holds them has been replaced.
//!
//! This is the owner. The [`WeightStore`](super::WeightStore) is the right home
//! for it because these buffers are *derived from* store tensors, have exactly
//! the store's lifetime (the layers read both until teardown), and the store is
//! already the last `ModelResource` released before the sweep — so adopting a
//! derived buffer here moves it from "swept" to "released", which is the
//! difference the ledger reports.
//!
//! Interior mutability because loaders take `&WeightStore`: threading a `&mut`
//! through `ModelWeightLoader::load_layers` would change the signature of every
//! architecture's loader to fix a bookkeeping gap in one of them.

use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend};

/// One adopted buffer: the pointer to free, its size, and a label for the
/// residency report.
#[derive(Clone, Copy, Debug)]
struct Derived {
    label: &'static str,
    ptr: DevicePtr,
    bytes: usize,
}

/// Device buffers a loader derived from this store's tensors.
#[derive(Default)]
pub struct DerivedStore {
    // parking_lot: no poisoning, so a panic elsewhere cannot block teardown.
    owned: Mutex<Vec<Derived>>,
}

impl DerivedStore {
    /// Take ownership of a derived buffer.
    ///
    /// `label` is a static category ("ssm qkvz fp8 concat", "attn fp8 twin"),
    /// not a per-tensor name: the report aggregates, and a `String` per buffer
    /// would allocate once per layer per twin for text nobody reads per entry.
    pub fn adopt(&self, label: &'static str, ptr: DevicePtr, bytes: usize) {
        if ptr.0 == 0 {
            return;
        }
        self.owned.lock().push(Derived { label, ptr, bytes });
    }

    /// Give up ownership of a buffer the caller is about to free itself.
    ///
    /// The transient half of a fused concat is adopted by the generic loader
    /// that produced it and then freed by the loader that consumed it
    /// (`qwen35_dense.rs`, the GDN `[QKV|Z]` arm). Without this the pointer
    /// would be freed twice — once there, once at teardown. Returns the bytes
    /// that left the ledger, or `None` if this store never held it.
    pub fn disown(&self, ptr: DevicePtr) -> Option<usize> {
        let mut owned = self.owned.lock();
        let i = owned.iter().position(|d| d.ptr == ptr)?;
        Some(owned.swap_remove(i).bytes)
    }

    /// Total adopted bytes still live.
    pub fn bytes(&self) -> usize {
        self.owned.lock().iter().map(|d| d.bytes).sum()
    }

    pub fn len(&self) -> usize {
        self.owned.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `label -> (bytes, count)`, biggest first, for the residency summary.
    pub fn by_label(&self) -> Vec<(&'static str, usize, usize)> {
        let mut rows: Vec<(&'static str, usize, usize)> = Vec::new();
        for d in self.owned.lock().iter() {
            match rows.iter_mut().find(|r| r.0 == d.label) {
                Some(r) => {
                    r.1 += d.bytes;
                    r.2 += 1;
                }
                None => rows.push((d.label, d.bytes, 1)),
            }
        }
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        rows
    }

    /// Free everything adopted. Drains first, so a failure part-way through
    /// cannot leave a freed pointer in the list to be freed again.
    pub fn release(&self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let doomed: Vec<Derived> = self.owned.lock().drain(..).collect();
        let mut first_error = None;
        for d in doomed {
            if let Err(e) = gpu.free(d.ptr)
                && first_error.is_none()
            {
                first_error = Some(e.context(format!("freeing derived weight ({})", d.label)));
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn an_adopted_buffer_is_freed_by_release() {
        let gpu = MockGpuBackend::new();
        let d = DerivedStore::default();
        let a = gpu.alloc(1024).unwrap();
        let b = gpu.alloc(2048).unwrap();
        d.adopt("twin", a, 1024);
        d.adopt("twin", b, 2048);
        assert_eq!(d.len(), 2);
        assert_eq!(d.bytes(), 3072);
        d.release(&gpu).unwrap();
        assert!(d.is_empty(), "release must drain, not merely free");
        assert_eq!(d.bytes(), 0);
    }

    #[test]
    fn release_is_idempotent() {
        let gpu = MockGpuBackend::new();
        let d = DerivedStore::default();
        d.adopt("twin", gpu.alloc(64).unwrap(), 64);
        d.release(&gpu).unwrap();
        // A second release must not double-free: the list is already drained.
        d.release(&gpu).unwrap();
    }

    #[test]
    fn a_null_pointer_is_not_adopted() {
        let d = DerivedStore::default();
        d.adopt("twin", DevicePtr::NULL, 0);
        assert!(d.is_empty(), "a NULL twin is 'not built', not 'owned'");
    }

    #[test]
    fn disown_removes_the_buffer_so_release_cannot_double_free() {
        let gpu = MockGpuBackend::new();
        let d = DerivedStore::default();
        let transient = gpu.alloc(512).unwrap();
        d.adopt("transient", transient, 512);
        assert_eq!(d.disown(transient), Some(512));
        assert!(d.is_empty());
        // A pointer this store never held is not silently "disowned".
        assert_eq!(d.disown(transient), None);
    }

    #[test]
    fn by_label_aggregates_biggest_first() {
        let gpu = MockGpuBackend::new();
        let d = DerivedStore::default();
        d.adopt("small", gpu.alloc(16).unwrap(), 16);
        d.adopt("big", gpu.alloc(1000).unwrap(), 1000);
        d.adopt("big", gpu.alloc(1000).unwrap(), 1000);
        assert_eq!(d.by_label(), vec![("big", 2000, 2), ("small", 16, 1)]);
    }
}
