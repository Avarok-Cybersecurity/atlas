// SPDX-License-Identifier: AGPL-3.0-only

//! Reused host buffer for the verify-time dequantised logits.
//!
//! Split out of `verify_pipeline_helper.rs`, which is over the 500 LoC cap.

thread_local! {
    /// Reused host buffer for the per-position dequantised logits.
    ///
    /// Mirrors `decode_logits_step::DECODE_LOGITS_HOST_SCRATCH`, which exists on
    /// the twin decode path "to avoid an mmap/munmap + page-fault cycle every
    /// decoded token". The VERIFY path — the one that actually runs under MTP —
    /// never had it, and was allocating a fresh ~1 MB `Vec<f32>` per K position,
    /// i.e. ~4 MB of first-touch pages per verify step.
    ///
    /// Per-thread: the scheduler drives verify on one thread. Residual contents
    /// are irrelevant — every entry is overwritten before any read.
    pub(super) static DEQUANT_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Returns the dequant buffer to [`DEQUANT_SCRATCH`] on drop.
///
/// `verify_pick_with_pipeline` has three exits — the forced-token short
/// circuit, the temp>0 sample, and the argmax — so a guard is used rather than
/// three hand-placed hand-backs: miss one and the reuse is silently lost with
/// no visible failure, which is the same drift that left `step_done` off the
/// K=4 path entirely.
pub(super) struct ScratchGuard(pub(super) Vec<f32>);

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.0);
        DEQUANT_SCRATCH.with(|s| {
            *s.borrow_mut() = buf;
        });
    }
}

impl std::ops::Deref for ScratchGuard {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}

impl std::ops::DerefMut for ScratchGuard {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        &mut self.0
    }
}
