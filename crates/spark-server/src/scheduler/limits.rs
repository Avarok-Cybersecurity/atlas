// SPDX-License-Identifier: AGPL-3.0-only

//! [`SchedLimits`] — the run's hard stops, resolved from this model's
//! tokenizer and CLI rather than installed into process globals.
//!
//! These were three atomics in `helpers.rs`, each with a `set_*` installer
//! called once during serve startup and a getter read on the decode/emit path.
//! Every one of them is derived from something that changes with the model:
//! two are *token ids*, which are meaningless against a different tokenizer,
//! and the third is the served-context ceiling. A stale token id does not
//! fail loudly — it hard-stops generation at whatever token happens to hold
//! that id in the new vocabulary, which reads as a truncation bug far from
//! its cause.
//!
//! `0` was the atomics' "unset" sentinel, so unit tests and any path that ran
//! before the installer silently got the disabled behaviour. That is now
//! [`Option`] and an explicit `0` ceiling, so the disabled case is visible in
//! the type instead of being a magic value.

/// Hard output/length limits for one scheduler run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedLimits {
    /// `<|im_start|>` as a single token id, when this tokenizer has one.
    /// Emitting it means the model has begun a new ChatML turn on its own, so
    /// the sequence is stopped at the role boundary. `None` for non-ChatML
    /// tokenizers → no hard stop.
    pub im_start_hard_stop: Option<u32>,
    /// `<tool_response>` as a single token id, when this tokenizer has one.
    /// Registered but inert unless `SchedLevers::tool_response_stop` is armed.
    pub tool_response_hard_stop: Option<u32>,
    /// Served-context ceiling (`--max-seq-len`), enforced per decode step so a
    /// long think block cannot run past it. `0` = unset → every guard that
    /// reads it becomes a no-op.
    pub max_seq_len: usize,
}

impl SchedLimits {
    /// No hard stops and no ceiling — the shape the pure decision cores are
    /// tested against, and what a run gets when the tokenizer resolves neither
    /// control token.
    pub const NONE: Self = Self {
        im_start_hard_stop: None,
        tool_response_hard_stop: None,
        max_seq_len: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_the_default_and_disables_every_guard() {
        assert_eq!(SchedLimits::default(), SchedLimits::NONE);
        assert_eq!(SchedLimits::NONE.max_seq_len, 0);
        assert!(SchedLimits::NONE.im_start_hard_stop.is_none());
    }

    #[test]
    fn two_runs_can_hold_different_token_ids() {
        // The hazard the atomics carried: `set_im_start_hard_stop` was a store
        // to a process global, so a second model's run would keep the first
        // tokenizer's id for `<|im_start|>` unless it happened to resolve one
        // of its own — and hard-stop on whatever token now holds it.
        let a = SchedLimits {
            im_start_hard_stop: Some(151644),
            ..SchedLimits::NONE
        };
        let b = SchedLimits {
            im_start_hard_stop: Some(200),
            ..SchedLimits::NONE
        };
        assert_ne!(a.im_start_hard_stop, b.im_start_hard_stop);
    }
}
