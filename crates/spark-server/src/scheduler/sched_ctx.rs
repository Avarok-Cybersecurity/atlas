// SPDX-License-Identifier: AGPL-3.0-only

//! [`SchedCtx`] — everything one scheduler run needs that is derived from the
//! model rather than from the request.
//!
//! The scheduler had no carrier at all: `run` takes its model-derived values as
//! positional parameters and threads them through the loop body as locals,
//! while everything that would not fit that shape — the vocabulary masks, the
//! `ATLAS_*` levers — ended up in process-global statics instead.
//!
//! This is the carrier those belong on. It is deliberately narrow: state that
//! is fixed for the run and read by the step functions. Per-request state stays
//! on `ActiveSeq`, and values the loop mutates stay locals.

use crate::scheduler::levers::SchedLevers;
use crate::scheduler::limits::SchedLimits;
use crate::scheduler::mtp_timing::RunTiming;
use crate::scheduler::spec_stats::SpecStats;
use crate::scheduler::vocab_masks::VocabMasks;

/// Reusable host buffers for the decode path, sized once and refilled.
///
/// Taking and returning the `Vec` (rather than borrowing across the body)
/// keeps the borrow scoped to the swap, so the decode path can hold the
/// buffer while calling back into the context.
#[derive(Debug, Default)]
pub struct DecodeScratch {
    /// Dequantized FP32 logits for one sequence (~1 MB at a 250k vocab).
    pub seq_f32: std::cell::RefCell<Vec<f32>>,
    /// Raw device-side logits copied to the host, one decode step.
    pub host_bytes: std::cell::RefCell<Vec<u8>>,
}

/// Model-derived state for one scheduler run.
pub struct SchedCtx {
    /// Per-token classification masks for this model's vocabulary.
    pub masks: VocabMasks,
    /// This run's snapshot cell, shared with the dashboard.
    pub snapshot: std::sync::Arc<crate::scheduler::snapshot::SnapshotCell>,
    /// Diagnostic file sinks this run writes to.
    pub dumps: crate::scheduler::dumps::RunDumps,
    /// Reusable host-side decode buffers for this run.
    ///
    /// These were `thread_local!` scratch, which is a process global with a
    /// narrower key: it kept the last model's allocation alive on the
    /// scheduler thread, and a test that ran two runs on one thread shared a
    /// buffer between them. The scheduler drives decode on ONE thread, so a
    /// `RefCell` on the run's own context is the same zero-contention access
    /// with a lifetime that ends when the run does.
    pub scratch: DecodeScratch,
    /// Decode / verify / speculation levers for this run.
    ///
    /// `Arc` because one of them — the loop watchdog — is toggled from the
    /// dashboard thread while the scheduler reads it. That is the whole
    /// reason a process global existed here: two threads needed the same
    /// bool. Sharing the run's levers gives them one that belongs to the run.
    pub levers: std::sync::Arc<SchedLevers>,
    /// Hard stops derived from this model's tokenizer and CLI.
    pub limits: SchedLimits,
    /// Decode-time watchdog tunables from this model's MODEL.toml
    /// `[behavior]` table.
    pub watchdog: crate::scheduler::helpers::WatchdogParams,
    /// Speculation accept/reject telemetry for this run. Mutated through the
    /// shared reference, which is why its counters are atomics.
    pub stats: std::sync::Arc<SpecStats>,
    /// Per-phase verify timing for this run, shared with the grammar state.
    pub timing: std::sync::Arc<RunTiming>,
    /// Trained repetition-onset detection head, when `[behavior].rom_head`
    /// names a loadable artifact. `None` means the F2 heuristic is the
    /// fallback — callers MUST treat it that way.
    ///
    /// Scaffolding until the artifact loader lands. It lives here rather than
    /// in a static because a trained head belongs to the model it was trained
    /// with; putting the seam in the right place now is cheaper than moving it
    /// once something depends on it.
    pub rom_head: Option<std::sync::Arc<dyn crate::scheduler::rollback::RomHead>>,
}

impl SchedCtx {
    pub fn new(
        masks: VocabMasks,
        levers: std::sync::Arc<SchedLevers>,
        snapshot: std::sync::Arc<crate::scheduler::snapshot::SnapshotCell>,
        limits: SchedLimits,
        watchdog: crate::scheduler::helpers::WatchdogParams,
    ) -> Self {
        Self {
            masks,
            snapshot,
            dumps: crate::scheduler::dumps::RunDumps::from_env(),
            scratch: DecodeScratch::default(),
            levers,
            limits,
            watchdog,
            stats: std::sync::Arc::new(SpecStats::new()),
            timing: std::sync::Arc::new(RunTiming::from_env()),
            rom_head: None,
        }
    }

    /// A context with no masks and default levers — for tests, which would
    /// otherwise have to mutate the process environment to exercise a path.
    pub fn for_test() -> Self {
        Self::new(
            VocabMasks::default(),
            std::sync::Arc::new(SchedLevers::defaults()),
            std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
            SchedLimits::NONE,
            crate::scheduler::helpers::WatchdogParams::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_test_context_needs_no_environment() {
        let c = SchedCtx::for_test();
        assert!(c.masks.numeric.is_none());
        assert!(c.levers.fast_masked, "an opt-out lever, on by default");
        assert!(!c.levers.dflash_adaptive);
    }

    #[test]
    fn two_contexts_are_independent() {
        let a = SchedCtx::for_test();
        let b = SchedCtx::for_test();
        a.levers.set_loop_watchdog(true);
        assert!(a.levers.loop_watchdog() && !b.levers.loop_watchdog());
    }
}
