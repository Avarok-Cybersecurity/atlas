// SPDX-License-Identifier: AGPL-3.0-only

//! [`PinnedHostStaging`] — a reusable page-locked host blob that a batched
//! device copy can gather into or scatter out of.
//!
//! Two independent users, for the same three reasons: the SSM snapshot tier
//! (spill/fault-in, fixed geometry, [`PinnedHostStaging::acquire`]) and the
//! Marconi aux-state collect (per-checkpoint, geometry grows with context,
//! [`PinnedHostStaging::acquire_at_least`]).
//!
//! # Why page-locked is not an optimisation here
//!
//! `cuMemcpyDtoHAsync` into PAGEABLE memory is asynchronous in name only — the
//! driver must bounce it through its own staging, which serialises it against
//! the host. So batching N drains into "N enqueues + one sync" does not work at
//! all unless the destination is page-locked: without pinning you keep the N
//! stalls and merely stop calling them syncs. Pinning and batching are one
//! change, not two.
//!
//! Why this exists, measured on Holo-3.1-35B-A3B (30 SSM layers, blob =
//! 66,846,720 B) with `ATLAS_SSM_TIER_TIMING=1`:
//!
//! ```text
//! SSM spill: 66846720 B  gather+sync=392936us  store.put=19397us  total=412334us
//! ```
//!
//! ~400 ms per eviction of which the disk write is 19 ms. Two of the three
//! causes are allocation-shaped and are what this type deletes:
//!   1. a fresh `vec![0u8; 66_846_720]` per event — 66 MB of ZEROING plus
//!      ~16,320 first-touch page faults on a brand-new mmap, both of which are
//!      pure waste because the buffer is fully overwritten;
//!   2. a *pageable* buffer, which forces the driver to bounce every
//!      `cuMemcpy*Async` through its own internal staging.
//!
//! (The third — 60 blocking `copy_d2h` calls, one full stream drain each —
//! is fixed in `ssm_snapshot_spill.rs` by `copy_d2h_async` + one sync.)

use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use spark_runtime::gpu::GpuBackend;

/// A host staging blob. `pinned` records whether it came from
/// [`GpuBackend::alloc_host_pinned`] (page-locked, DMA-able) or from the plain
/// heap fallback — the timing log prints it, because whether pinning helps at
/// all on a UMA box (GB10: host and device are the same LPDDR5X) is exactly
/// the question a GPU run has to answer, not one to assume.
struct StagingBlob {
    ptr: *mut u8,
    bytes: usize,
    pinned: bool,
}

// SAFETY: the pointer is either `cuMemAllocHost` memory (process-global, valid
// from any thread) or a plain heap allocation. Ownership is exclusive — it
// lives in a `Mutex` and is only ever reachable through a `StagingGuard`.
unsafe impl Send for StagingBlob {}

/// Lazily-allocated, reusable staging buffer for the SSM spill tier.
///
/// Allocated on FIRST use: the tier is default-off (`ATLAS_SSM_TIER` unset), so
/// a tier-less deployment must not pay 66 MB of pinned host memory for a path
/// it never takes.
///
/// A `Mutex` rather than the `UnsafeCell` idiom used by `PinnedMetaStaging`,
/// because this is a per-EVICTION path, not a per-token one: the lock is free
/// relative to a 66 MB move, and it is what makes handing out `&mut [u8]` from
/// a raw pointer sound.
///
/// INVARIANT: the guard is never held across a call that can itself spill or
/// fault in. `spill_slot` and `fault_in_slot` each acquire it and release it
/// before returning, and the model is single-threaded after construction, so
/// the two never nest — the `Mutex` turns that from a comment into a check.
#[derive(Default)]
pub(crate) struct PinnedHostStaging {
    slot: Mutex<Option<StagingBlob>>,
}

impl PinnedHostStaging {
    /// Borrow the staging blob, allocating (and page-locking) it on first use.
    ///
    /// A size change reallocates — the blob geometry is fixed at model init, so
    /// in practice this happens exactly once. NOT zeroed: every caller
    /// overwrites all `bytes` before reading, and the zero-fill is one of the
    /// costs this type exists to remove.
    pub(crate) fn acquire<'a>(
        &'a self,
        gpu: &dyn GpuBackend,
        bytes: usize,
    ) -> Result<StagingGuard<'a>> {
        let mut slot = self.slot.lock();
        let need_alloc = match slot.as_ref() {
            Some(b) => b.bytes != bytes,
            None => true,
        };
        if need_alloc {
            if let Some(old) = slot.take() {
                free_blob(gpu, old);
            }
            *slot = Some(alloc_blob(gpu, bytes));
        }
        Ok(StagingGuard { slot, len: bytes })
    }

    /// Borrow at least `bytes`, GROWING the blob only when it is too small.
    ///
    /// The difference from [`Self::acquire`] is the whole reason this exists:
    /// `acquire` reallocates whenever the size CHANGES, which is right for the
    /// spill tier (one fixed geometry, so it allocates exactly once) and wrong
    /// for the aux collect, whose blob grows with the sequence's context and
    /// would otherwise re-page-lock on essentially every checkpoint —
    /// `cuMemAllocHost` is a synchronising, whole-device-serialising call, so
    /// that would reintroduce a worse stall than the one being removed.
    ///
    /// The returned slice is exactly `bytes` long, never the capacity: callers
    /// compute offsets from their own plan and must not see trailing slack.
    pub(crate) fn acquire_at_least<'a>(
        &'a self,
        gpu: &dyn GpuBackend,
        bytes: usize,
    ) -> Result<StagingGuard<'a>> {
        let mut slot = self.slot.lock();
        let need_alloc = match slot.as_ref() {
            Some(b) => b.bytes < bytes,
            None => true,
        };
        if need_alloc {
            if let Some(old) = slot.take() {
                free_blob(gpu, old);
            }
            // Grow geometrically. Context grows a token at a time, so exact-fit
            // growth would re-pin on almost every checkpoint; doubling makes the
            // number of page-lock calls logarithmic in the context length.
            let cap = bytes.next_power_of_two().max(64 * 1024);
            *slot = Some(alloc_blob(gpu, cap));
        }
        Ok(StagingGuard { slot, len: bytes })
    }

    /// Release the buffer. Called from `TransformerModel::drop` (the pool holds
    /// no `gpu` handle of its own), mirroring `drop_pinned_staging`.
    pub(crate) fn free(&self, gpu: &dyn GpuBackend) {
        if let Some(b) = self.slot.lock().take() {
            free_blob(gpu, b);
        }
    }
}

/// Page-lock `bytes` of host memory, degrading to the heap on failure.
fn alloc_blob(gpu: &dyn GpuBackend, bytes: usize) -> StagingBlob {
    match gpu.alloc_host_pinned(bytes) {
        Ok(ptr) if !ptr.is_null() => StagingBlob {
            ptr,
            bytes,
            pinned: true,
        },
        other => {
            // Name the degradation: still CORRECT, but the copies fall back to
            // the driver's pageable bounce, i.e. roughly the bandwidth this
            // whole change set exists to escape.
            if let Err(e) = other {
                tracing::warn!(
                    "SSM tier: could not page-lock a {bytes} B staging buffer ({e:#}); \
                     using heap memory — spill/fault-in stay correct but lose the DMA \
                     fast path (expect the old ~165 MB/s D2H bandwidth)"
                );
            }
            let layout = std::alloc::Layout::from_size_align(bytes, 4096)
                .expect("staging layout: bytes > 0, align 4096");
            // SAFETY: non-zero size (blob_bytes > 0 whenever the tier is live);
            // freed by `free_blob` with the identical layout.
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(
                !ptr.is_null(),
                "SSM tier staging heap alloc failed: {bytes} B"
            );
            StagingBlob {
                ptr,
                bytes,
                pinned: false,
            }
        }
    }
}

fn free_blob(gpu: &dyn GpuBackend, b: StagingBlob) {
    if b.pinned {
        if let Err(e) = gpu.free_host_pinned(b.ptr, b.bytes) {
            tracing::warn!("SSM tier: failed to free pinned staging buffer: {e:#}");
        }
        return;
    }
    let layout = std::alloc::Layout::from_size_align(b.bytes, 4096)
        .expect("staging layout: bytes > 0, align 4096");
    // SAFETY: allocated by `alloc_blob`'s heap arm with this exact layout.
    unsafe { std::alloc::dealloc(b.ptr, layout) };
}

/// Exclusive borrow of the staging blob. Dropping it releases the lock; the
/// caller MUST have synchronized the stream first, since in-flight async copies
/// still reference these bytes.
///
/// # Hazard: acquiring twice without dropping DEADLOCKS
///
/// The lock is a plain non-reentrant `Mutex`, so a second `acquire*` reached
/// while a guard is still live blocks forever — it does not error and does not
/// panic. That includes the easy-to-miss form where the guard is a statement
/// TEMPORARY rather than a binding:
///
/// ```ignore
/// staging.acquire_at_least(gpu, n)?.as_mut_slice()[..4].copy_from_slice(&x);
/// let mut g = staging.acquire_at_least(gpu, n)?;  // <-- hangs
/// ```
///
/// Bind the guard and scope it. (The sibling comment on `PinnedHostStaging`
/// says the Mutex "turns that from a comment into a check" — it does not; an
/// unsatisfied invariant here presents as a hang, not a failure.)
pub(crate) struct StagingGuard<'a> {
    slot: MutexGuard<'a, Option<StagingBlob>>,
    /// Bytes the caller asked for. May be less than the blob's capacity under
    /// [`PinnedHostStaging::acquire_at_least`]; the slice is always this long.
    len: usize,
}

impl StagingGuard<'_> {
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        let b = self
            .slot
            .as_mut()
            .expect("acquire() installs the blob before returning the guard");
        debug_assert!(self.len <= b.bytes, "guard len {} > capacity {}", self.len, b.bytes);
        // SAFETY: `b.ptr` owns `b.bytes` of live host memory, `self.len` is
        // <= that capacity by construction in both acquire paths, and this
        // guard holds the only reference to it (the Mutex is the exclusion).
        unsafe { std::slice::from_raw_parts_mut(b.ptr, self.len) }
    }

    /// `"pinned"` / `"heap"` for the `ATLAS_SSM_TIER_TIMING` line.
    pub(crate) fn kind(&self) -> &'static str {
        match self.slot.as_ref() {
            Some(b) if b.pinned => "pinned",
            _ => "heap",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PinnedHostStaging;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// The property the aux collect depends on: a SHRINKING or equal request
    /// must not reallocate. `acquire` (exact-fit) reallocates on any change,
    /// which is right for the spill tier's fixed geometry and would re-pin on
    /// nearly every checkpoint here, since the blob grows with context.
    #[test]
    fn acquire_at_least_grows_only_and_slices_to_the_request() {
        let gpu = MockGpuBackend::new();
        let st = PinnedHostStaging::default();

        let base = {
            let mut g = st.acquire_at_least(&gpu, 1000).unwrap();
            let s = g.as_mut_slice();
            assert_eq!(s.len(), 1000, "slice is the REQUEST, not the capacity");
            s.as_ptr()
        };

        // Smaller request: same allocation, shorter slice.
        {
            let mut g = st.acquire_at_least(&gpu, 500).unwrap();
            let s = g.as_mut_slice();
            assert_eq!(s.len(), 500);
            assert_eq!(s.as_ptr(), base, "shrinking must NOT reallocate");
        }

        // Equal request: still no reallocation.
        {
            let mut g = st.acquire_at_least(&gpu, 1000).unwrap();
            assert_eq!(g.as_mut_slice().as_ptr(), base, "equal must NOT reallocate");
        }

        // Growth past capacity reallocates and the slice tracks the request.
        {
            let mut g = st.acquire_at_least(&gpu, 200_000).unwrap();
            assert_eq!(g.as_mut_slice().len(), 200_000);
        }
        // ...and having grown, it does not shrink back.
        {
            let mut g = st.acquire_at_least(&gpu, 1000).unwrap();
            assert_eq!(g.as_mut_slice().len(), 1000);
        }

        st.free(&gpu);
    }

    /// Bytes written through one guard are visible through the next - the blob
    /// is reused, not reallocated, so the aux collect can enqueue into it and
    /// read back after its single sync.
    #[test]
    fn contents_survive_reacquire() {
        let gpu = MockGpuBackend::new();
        let st = PinnedHostStaging::default();
        // Explicit scopes: the guard holds the mutex, so a second acquire must
        // not be reached until the first one has been dropped.
        {
            let mut g = st.acquire_at_least(&gpu, 64).unwrap();
            g.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        }
        {
            let mut g = st.acquire_at_least(&gpu, 64).unwrap();
            assert_eq!(&g.as_mut_slice()[..4], &[1, 2, 3, 4]);
        }
        st.free(&gpu);
    }
}
