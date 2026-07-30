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
use crate::scheduler::vocab_masks::VocabMasks;

/// Model-derived state for one scheduler run.
pub struct SchedCtx {
    /// Per-token classification masks for this model's vocabulary.
    pub masks: VocabMasks,
    /// Decode / verify / speculation levers for this run.
    pub levers: SchedLevers,
}

impl SchedCtx {
    pub fn new(masks: VocabMasks, levers: SchedLevers) -> Self {
        Self { masks, levers }
    }

    /// A context with no masks and default levers — for tests, which would
    /// otherwise have to mutate the process environment to exercise a path.
    pub fn for_test() -> Self {
        Self::new(VocabMasks::default(), SchedLevers::defaults())
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
