// SPDX-License-Identifier: AGPL-3.0-only

//! The one bounds-checked way to pack the model's pinned metadata staging
//! buffer.
//!
//! Five call sites pack `PinnedMetaStaging` — `prefill_a`, `prefill_c`,
//! `prefill_b::upload_meta`, `prefill_b::stage_batched` and `decode_b` — each
//! with its own hand-rolled sequence of `unsafe { copy_nonoverlapping }` and its
//! own idea of when to check the destination bound. Three checked before the
//! first write. Two checked only AFTER the last one, with an `assert!` that by
//! definition fires on the wreckage: by the time `cursor > stg.bytes` is
//! observed, the writes are already outside the allocation.
//!
//! That asymmetry is the recurring failure in this area — `913f7c1f` hoisted the
//! check in one file and the four siblings kept their own copies of the rule.
//! So the rule lives here now: there is no way to write a byte into the pinned
//! buffer except through [`PinnedPacker::put_at`], and that function checks
//! first. A future sixth call site inherits the check by construction rather
//! than by its author remembering.
//!
//! ## Why `packed()` may span bytes this call never wrote
//!
//! Callers round field offsets up for alignment, which leaves gaps no
//! `copy_nonoverlapping` fills, and then form one `&[u8]` over the whole packed
//! range for a single H2D. A slice over an uninitialised byte is UB regardless
//! of what the device later does with it. That is sound here for exactly one
//! reason: [`spark_runtime::gpu::GpuBackend::alloc_host_pinned`] contractually
//! returns a ZEROED region and never hands the same bytes out twice, so every
//! byte of the buffer is initialised from allocation onward. This module is the
//! single place that reasoning has to be written down.

use std::marker::PhantomData;

use anyhow::{Result, ensure};

/// Types that may be reinterpreted as bytes when packed into pinned staging.
///
/// # Safety
///
/// The implementing type must have no padding and no invalid bit patterns, so
/// that every byte of `[T]` is initialised and reading it as `[u8]` is defined.
/// Stating that once per TYPE is the point — the five call sites used to restate
/// it once per copy, in prose, and prose does not stop a new call site from
/// passing something with padding in it.
pub(crate) unsafe trait PinnedPod: Copy {}

// SAFETY: fixed-width integers — no padding, every bit pattern valid.
unsafe impl PinnedPod for u8 {}
unsafe impl PinnedPod for u32 {}
unsafe impl PinnedPod for i32 {}
unsafe impl PinnedPod for i64 {}
unsafe impl PinnedPod for u64 {}

/// A bounds-checked cursor over the pinned staging allocation.
///
/// Built by [`super::types::PinnedMetaStaging::packer`]. Borrows the staging
/// struct so it cannot outlive it; the bytes it writes are the separate
/// `cuMemAllocHost` region the struct points at, not the struct itself, which is
/// why a shared borrow suffices and the caller can still read its source `Vec`s.
pub(crate) struct PinnedPacker<'a> {
    ptr: *mut u8,
    bytes: usize,
    high_water: usize,
    _staging: PhantomData<&'a ()>,
}

impl<'a> PinnedPacker<'a> {
    /// # Safety
    ///
    /// `ptr` must be the base of a live, zero-initialised host allocation of at
    /// least `bytes` bytes, and the caller must hold exclusive access to it for
    /// `'a` (the scheduler thread is the only user).
    pub(crate) unsafe fn new(ptr: *mut u8, bytes: usize) -> Self {
        Self {
            ptr,
            bytes,
            high_water: 0,
            _staging: PhantomData,
        }
    }

    /// Bytes written so far, i.e. the end of the highest field placed. Callers
    /// use it to compute the next field's aligned offset.
    pub(crate) fn high_water(&self) -> usize {
        self.high_water
    }

    /// Total capacity of the staging allocation.
    pub(crate) fn capacity(&self) -> usize {
        self.bytes
    }

    /// Place `src` at byte offset `at`, refusing BEFORE the write if it would
    /// leave the allocation.
    ///
    /// `what` names the field in the error, because the useful part of an
    /// over-run report is which table was too big, not the byte count.
    pub(crate) fn put_at<T: PinnedPod>(&mut self, what: &str, at: usize, src: &[T]) -> Result<()> {
        let len = std::mem::size_of_val(src);
        let end = at.checked_add(len).ok_or_else(|| {
            anyhow::anyhow!("pinned staging: {what} offset {at} + {len} overflows")
        })?;
        ensure!(
            end <= self.bytes,
            "pinned staging: {what} needs bytes [{at}, {end}) but the buffer is {} B — \
             refusing to pack (a {}-element table at this context length does not fit)",
            self.bytes,
            src.len()
        );
        if len > 0 {
            // SAFETY: `end <= self.bytes` was just checked, so `[at, at + len)`
            // is inside the allocation `new`'s contract guarantees. The source
            // is a live `&[T]` and `len` is `size_of_val` OF THAT SLICE, so it
            // can never exceed it; `T: PinnedPod` makes every one of those bytes
            // initialised. Source and destination cannot overlap — the
            // destination is the pinned allocation, the source is a caller
            // buffer, and nothing packs the pinned buffer into itself.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, self.ptr.add(at), len);
            }
        }
        self.high_water = self.high_water.max(end);
        Ok(())
    }

    /// Place the first `n` elements of `src` at `at`, erroring if `src` is
    /// shorter than `n`.
    ///
    /// The staging `Vec`s are reused across calls and some paths rebuild them
    /// from data-dependent lengths, so "the source is long enough" is a real
    /// question and not a restatement of the destination bound. Both halves of
    /// the rule belong in the same place.
    pub(crate) fn put_prefix_at<T: PinnedPod>(
        &mut self,
        what: &str,
        at: usize,
        src: &[T],
        n: usize,
    ) -> Result<()> {
        let head = src.get(..n).ok_or_else(|| {
            anyhow::anyhow!(
                "pinned staging: {what} has {} elements, needed {n}",
                src.len()
            )
        })?;
        self.put_at(what, at, head)
    }

    /// Extend the packed range to `end` without writing, for a caller whose
    /// device-side layout includes trailing alignment padding.
    ///
    /// Sound for the same reason the interior gaps are: the allocation is zeroed
    /// and never recycled, so those bytes are initialised.
    pub(crate) fn pad_to(&mut self, end: usize) -> Result<()> {
        ensure!(
            end <= self.bytes,
            "pinned staging: padding to {end} exceeds the {} B buffer",
            self.bytes
        );
        self.high_water = self.high_water.max(end);
        Ok(())
    }

    /// The packed bytes, `[0, high_water)`, ready for one H2D copy.
    pub(crate) fn packed(self) -> &'a [u8] {
        // SAFETY: `high_water` only ever advances through `put_at` / `pad_to`,
        // both of which check it against `self.bytes` first, so the range is
        // inside the allocation. Every byte in it is initialised: the ones this
        // call wrote, plus alignment gaps that `alloc_host_pinned` zeroed (see
        // the module docs). The lifetime is the staging borrow, so the region
        // outlives the slice.
        unsafe { std::slice::from_raw_parts(self.ptr, self.high_water) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a packer over a real zeroed heap allocation, standing in for the
    /// zeroed pinned region.
    fn with_buf<R>(bytes: usize, f: impl FnOnce(PinnedPacker<'_>) -> R) -> R {
        let mut backing = vec![0u8; bytes];
        // SAFETY: `backing` is live, zeroed, `bytes` long, and exclusively
        // borrowed for the duration of the call.
        let packer = unsafe { PinnedPacker::new(backing.as_mut_ptr(), bytes) };
        f(packer)
    }

    /// The defect this type exists to remove: an over-long table must be
    /// refused BEFORE any byte lands, not asserted afterwards.
    #[test]
    fn refuses_before_writing_a_single_byte() {
        let mut backing = vec![0u8; 64];
        let over = vec![0xAAu32; 32]; // 128 B into a 64 B buffer
        // SAFETY: see `with_buf`.
        let mut packer = unsafe { PinnedPacker::new(backing.as_mut_ptr(), 64) };
        assert!(packer.put_at("block_table", 0, &over).is_err());
        // The refusal happened first: nothing was written, so the buffer is
        // untouched. Before this type, the equivalent code wrote all 128 B and
        // only then hit its `assert!`.
        assert!(backing.iter().all(|&b| b == 0));
    }

    #[test]
    fn packs_at_offsets_and_reports_the_high_water() {
        with_buf(256, |mut p| {
            p.put_at("positions", 0, &[1u32, 2, 3]).unwrap();
            assert_eq!(p.high_water(), 12);
            // A gap at 12..16 that nothing writes: still inside the packed range.
            p.put_at("slots", 16, &[7i64, 8]).unwrap();
            assert_eq!(p.high_water(), 32);
            let packed = p.packed();
            assert_eq!(packed.len(), 32);
            let positions: Vec<u32> = packed[0..12]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(positions, [1, 2, 3]);
            assert_eq!(&packed[12..16], &[0, 0, 0, 0], "gap stays zeroed");
            let slots: Vec<i64> = packed[16..32]
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(slots, [7, 8]);
        });
    }

    #[test]
    fn the_last_byte_fits_and_one_more_does_not() {
        with_buf(16, |mut p| {
            p.put_at("exact", 8, &[1u32, 2]).unwrap();
            assert_eq!(p.high_water(), 16);
        });
        with_buf(16, |mut p| {
            assert!(p.put_at("one_over", 9, &[1u32, 2]).is_err());
        });
    }

    #[test]
    fn put_prefix_at_checks_the_source_length_too() {
        with_buf(64, |mut p| {
            let src = vec![1u32, 2, 3, 4];
            p.put_prefix_at("positions", 0, &src, 2).unwrap();
            assert_eq!(p.high_water(), 8);
            let e = p.put_prefix_at("positions", 0, &src, 5).unwrap_err();
            assert!(e.to_string().contains("needed 5"), "{e}");
            let packed = p.packed();
            assert_eq!(
                packed,
                [1u32, 2]
                    .into_iter()
                    .flat_map(u32::to_ne_bytes)
                    .collect::<Vec<_>>()
            );
        });
    }

    #[test]
    fn pad_to_extends_the_range_but_is_still_bounded() {
        with_buf(32, |mut p| {
            p.put_at("f", 0, &[1u32]).unwrap();
            p.pad_to(8).unwrap();
            assert_eq!(p.high_water(), 8);
            // pad_to never shrinks.
            p.pad_to(4).unwrap();
            assert_eq!(p.high_water(), 8);
            assert!(p.pad_to(33).is_err());
        });
    }

    /// An empty table is a legitimate shape (no blocks yet) and must not be
    /// turned into a zero-length `copy_nonoverlapping` on a bad pointer.
    #[test]
    fn empty_source_is_a_no_op() {
        with_buf(16, |mut p| {
            p.put_at("empty", 0, &[] as &[u32]).unwrap();
            assert_eq!(p.high_water(), 0);
            assert!(p.packed().is_empty());
        });
    }
}
