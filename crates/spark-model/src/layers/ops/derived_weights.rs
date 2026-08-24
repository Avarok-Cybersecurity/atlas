// SPDX-License-Identifier: AGPL-3.0-only

//! GPU allocations derived from a model's weights, memoized for that model's
//! lifetime.
//!
//! Several projection paths need a re-encoded copy of a weight — block-scaled
//! FP8 dequantized to BF16, re-quantized to row-wise FP8, an NVFP4 weight
//! transposed into the CUTLASS byte layout. Producing one costs a kernel launch
//! and an allocation, and the source weights are immutable after load, so the
//! result is memoized by source pointer.
//!
//! **The memo used to live in four `static OnceLock<Mutex<HashMap<u64, …>>>`,
//! and the key is a raw device pointer.** That is the worst possible key for a
//! global. Free a model's weights, load another, and the allocator can hand
//! back the same addresses — at which point a lookup does not miss, it *hits*,
//! and returns a pointer to a re-encoding of a weight that no longer exists.
//! Not a crash: a plausible pointer to the wrong numbers.
//!
//! Owning the cache alongside the weights removes the failure mode rather than
//! guarding it. The map is dropped when the model is, so an entry cannot
//! outlive the allocation it describes, and no key can be recycled into it.

use std::collections::HashMap;
// parking_lot: no poisoning, so teardown cannot be blocked by a panic that
// happened somewhere else entirely.
use parking_lot::Mutex;

/// Per-model memo of derived weight encodings.
///
/// One map per derivation rather than one keyed by `(ptr, kind)`: the value
/// types differ, and a wrong-kind hit would be exactly the class of bug this
/// type exists to remove.
#[derive(Default)]
pub struct DerivedWeights {
    /// FP8 weight ptr → `(row-wise FP8 ptr, per-row scale ptr)`.
    rowwise_fp8: Mutex<HashMap<u64, (u64, u64)>>,
    /// FP8 weight ptr → BF16 ptr.
    bf16: Mutex<HashMap<u64, u64>>,
    /// NVFP4 weight ptr → CUTLASS-layout transposed ptr.
    cutlass_nvfp4_t: Mutex<HashMap<u64, u64>>,
    /// FP8 weight ptr → `(CUTLASS NVFP4 ptr, scale ptr)`.
    cutlass_nvfp4_from_fp8: Mutex<HashMap<u64, (u64, u64)>>,
}

/// Which derivation a lookup is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivation {
    RowwiseFp8,
    Bf16,
    CutlassNvfp4Transposed,
    CutlassNvfp4FromFp8,
}

impl DerivedWeights {
    pub fn new() -> Self {
        Self::default()
    }

    /// Raw memo read. `get_or_build_ptr` is the form to prefer; this pair
    /// exists because the dispatch helpers allocate and launch between the
    /// lookup and the store, and expressing that as a closure would mean
    /// indenting several hundred lines of kernel-launch code for no gain.
    pub fn get_ptr(&self, kind: Derivation, key: u64) -> Option<u64> {
        let map = match kind {
            Derivation::Bf16 => &self.bf16,
            Derivation::CutlassNvfp4Transposed => &self.cutlass_nvfp4_t,
            _ => return None,
        };
        map.lock().get(&key).copied()
    }

    pub fn insert_ptr(&self, kind: Derivation, key: u64, value: u64) {
        let map = match kind {
            Derivation::Bf16 => &self.bf16,
            Derivation::CutlassNvfp4Transposed => &self.cutlass_nvfp4_t,
            _ => return,
        };
        map.lock().entry(key).or_insert(value);
    }

    pub fn get_pair(&self, kind: Derivation, key: u64) -> Option<(u64, u64)> {
        let map = match kind {
            Derivation::RowwiseFp8 => &self.rowwise_fp8,
            Derivation::CutlassNvfp4FromFp8 => &self.cutlass_nvfp4_from_fp8,
            _ => return None,
        };
        map.lock().get(&key).copied()
    }

    pub fn insert_pair(&self, kind: Derivation, key: u64, value: (u64, u64)) {
        let map = match kind {
            Derivation::RowwiseFp8 => &self.rowwise_fp8,
            Derivation::CutlassNvfp4FromFp8 => &self.cutlass_nvfp4_from_fp8,
            _ => return,
        };
        map.lock().entry(key).or_insert(value);
    }

    /// Look up a single-pointer derivation, computing it on a miss.
    ///
    /// `build` runs outside the lock: it launches kernels and allocates, and
    /// holding a `Mutex` across that would serialise every layer's first touch
    /// of a weight behind one another. A duplicate build under a race wastes an
    /// allocation, which is why the loser's value is kept rather than swapped —
    /// the first writer's pointer is the one other threads may already hold.
    pub fn get_or_build_ptr(
        &self,
        kind: Derivation,
        key: u64,
        build: impl FnOnce() -> anyhow::Result<u64>,
    ) -> anyhow::Result<u64> {
        let map = match kind {
            Derivation::Bf16 => &self.bf16,
            Derivation::CutlassNvfp4Transposed => &self.cutlass_nvfp4_t,
            _ => unreachable!("pair-valued derivation routed to get_or_build_ptr"),
        };
        if let Some(&hit) = map.lock().get(&key) {
            return Ok(hit);
        }
        let built = build()?;
        Ok(*map.lock().entry(key).or_insert(built))
    }

    /// Look up a pair-valued derivation, computing it on a miss.
    pub fn get_or_build_pair(
        &self,
        kind: Derivation,
        key: u64,
        build: impl FnOnce() -> anyhow::Result<(u64, u64)>,
    ) -> anyhow::Result<(u64, u64)> {
        let map = match kind {
            Derivation::RowwiseFp8 => &self.rowwise_fp8,
            Derivation::CutlassNvfp4FromFp8 => &self.cutlass_nvfp4_from_fp8,
            _ => unreachable!("single-valued derivation routed to get_or_build_pair"),
        };
        if let Some(&hit) = map.lock().get(&key) {
            return Ok(hit);
        }
        let built = build()?;
        Ok(*map.lock().entry(key).or_insert(built))
    }

    /// Total memoized entries, for diagnostics and for asserting in tests that
    /// a fresh model starts empty.
    pub fn len(&self) -> usize {
        self.rowwise_fp8.lock().len()
            + self.bf16.lock().len()
            + self.cutlass_nvfp4_t.lock().len()
            + self.cutlass_nvfp4_from_fp8.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for DerivedWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DerivedWeights")
            .field("entries", &self.len())
            .finish()
    }
}

/// Release every derived allocation.
///
/// The KEYS are the original weight pointers — owned by the `WeightStore` and
/// freed by it. Only the VALUES are derivations this cache allocated, so only
/// those are freed here. Freeing a key would be a double-free of a weight.
impl atlas_core::scope::ModelResource<dyn spark_runtime::gpu::GpuBackend> for DerivedWeights {
    fn label(&self) -> &'static str {
        "derived weights"
    }

    fn release(&mut self, gpu: &dyn spark_runtime::gpu::GpuBackend) -> anyhow::Result<()> {
        let mut owned: Vec<u64> = Vec::new();
        // Drain so a later lookup cannot hit a pointer into freed memory —
        // the exact failure this cache was restructured to make impossible.
        for (_, (a, b)) in self.rowwise_fp8.lock().drain() {
            owned.push(a);
            owned.push(b);
        }
        owned.extend(self.bf16.lock().drain().map(|(_, v)| v));
        owned.extend(self.cutlass_nvfp4_t.lock().drain().map(|(_, v)| v));
        for (_, (a, b)) in self.cutlass_nvfp4_from_fp8.lock().drain() {
            owned.push(a);
            owned.push(b);
        }
        let mut first_error = None;
        for raw in owned {
            if let Err(e) = gpu.free(spark_runtime::gpu::DevicePtr(raw))
                && first_error.is_none()
            {
                first_error = Some(e);
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
    use atlas_core::scope::ModelResource;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_fresh_cache_is_empty() {
        assert!(DerivedWeights::new().is_empty());
    }

    #[test]
    fn a_derivation_is_built_once_per_key() {
        let d = DerivedWeights::new();
        let builds = AtomicUsize::new(0);
        let build_for = |ptr: u64| {
            d.get_or_build_ptr(Derivation::Bf16, ptr, || {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok(ptr + 1000)
            })
            .unwrap()
        };
        assert_eq!(build_for(10), 1010);
        assert_eq!(build_for(10), 1010);
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(build_for(20), 1020);
        assert_eq!(builds.load(Ordering::Relaxed), 2);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn the_derivations_do_not_share_a_keyspace() {
        // Same source pointer, two different encodings. A single map keyed by
        // pointer alone would return one for the other.
        let d = DerivedWeights::new();
        let bf16 = d
            .get_or_build_ptr(Derivation::Bf16, 0x1000, || Ok(0xB16))
            .unwrap();
        let nvfp4 = d
            .get_or_build_ptr(Derivation::CutlassNvfp4Transposed, 0x1000, || Ok(0x4444))
            .unwrap();
        assert_eq!(bf16, 0xB16);
        assert_eq!(nvfp4, 0x4444);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn a_failed_build_is_not_memoized() {
        let d = DerivedWeights::new();
        assert!(
            d.get_or_build_ptr(Derivation::Bf16, 7, || anyhow::bail!("oom"))
                .is_err()
        );
        assert!(d.is_empty(), "a failure must not poison the key");
        assert_eq!(
            d.get_or_build_ptr(Derivation::Bf16, 7, || Ok(99)).unwrap(),
            99
        );
    }

    #[test]
    fn two_models_memoize_independently_even_on_a_recycled_pointer() {
        // The property the four statics could not have. Model A caches a
        // derivation at 0x7f00_0000; A is released; B's allocator hands back
        // the same address for a DIFFERENT weight.
        let a = DerivedWeights::new();
        let recycled = 0x7f00_0000u64;
        assert_eq!(
            a.get_or_build_ptr(Derivation::Bf16, recycled, || Ok(0xAAAA))
                .unwrap(),
            0xAAAA
        );
        drop(a);

        let b = DerivedWeights::new();
        assert_eq!(
            b.get_or_build_ptr(Derivation::Bf16, recycled, || Ok(0xBBBB))
                .unwrap(),
            0xBBBB,
            "the same address must resolve to the NEW model's derivation"
        );
    }

    #[test]
    fn pair_derivations_round_trip_without_sharing_a_keyspace() {
        let d = DerivedWeights::new();
        let got = d
            .get_or_build_pair(Derivation::RowwiseFp8, 5, || Ok((11, 22)))
            .unwrap();
        assert_eq!(got, (11, 22));
        assert_eq!(
            d.get_or_build_pair(Derivation::CutlassNvfp4FromFp8, 5, || Ok((33, 44)))
                .unwrap(),
            (33, 44)
        );
        assert_eq!(
            d.get_or_build_pair(Derivation::RowwiseFp8, 5, || Ok((99, 99)))
                .unwrap(),
            (11, 22),
            "cached, not rebuilt"
        );
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn release_frees_only_derived_values_and_drains_every_map() {
        let gpu = MockGpuBackend::new();
        let mut d = DerivedWeights::new();
        let keys: Vec<u64> = (0..4).map(|_| gpu.alloc(1).unwrap().0).collect();
        let values: Vec<u64> = (0..6).map(|_| gpu.alloc(1).unwrap().0).collect();

        d.insert_pair(Derivation::RowwiseFp8, keys[0], (values[0], values[1]));
        d.insert_ptr(Derivation::Bf16, keys[1], values[2]);
        d.insert_ptr(Derivation::CutlassNvfp4Transposed, keys[2], values[3]);
        d.insert_pair(
            Derivation::CutlassNvfp4FromFp8,
            keys[3],
            (values[4], values[5]),
        );
        assert_eq!(gpu.alloc_count(), 10);
        assert_eq!(d.len(), 4);

        d.release(&gpu).unwrap();

        assert!(d.is_empty(), "freed pointers must not remain memoized");
        assert_eq!(gpu.alloc_count(), 4, "source-weight keys remain GPU-owned");
        for key in keys {
            gpu.free(spark_runtime::gpu::DevicePtr(key)).unwrap();
        }
    }
}
