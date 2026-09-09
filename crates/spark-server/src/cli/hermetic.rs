// SPDX-License-Identifier: AGPL-3.0-only

//! `--hermetic`: measure this server as a known-answer test.
//!
//! A KAT asserts that a given input produces a given output. That assertion is
//! only meaningful if the output is a function of THAT input — so no state
//! produced while serving one request may reach another. Several Atlas
//! subsystems exist precisely to carry state across requests, because carrying
//! it is usually the whole point; under a KAT they are the bug.
//!
//! ## Why one name and not a list of flags
//!
//! The channels could each be closed by its own `--serve-override`, and that
//! is how they were closed while they were being FOUND. It is the wrong way to
//! ship them, for two reasons.
//!
//! First, they drift. Four keys spelled out at four call sites is four chances
//! to close three of them, and a KAT that closed three of four channels is not
//! a KAT — it is a KAT-shaped thing that passes until the fourth channel moves
//! a sample.
//!
//! Second, and worse, the RECORD would not name the regime. A gate record
//! stores its `serve_overrides` map, and a reader comparing two runs has to
//! decide whether they are like-for-like. `hermetic=true` answers that in one
//! token. Four unrelated-looking keys require the reader to know which
//! combination constitutes a KAT — which is to say, to already know the thing
//! the record was supposed to tell them.
//!
//! ## Resolve once
//!
//! Every value here is RESOLVED, and every consumer reads the resolved value.
//! `serve_flags` states the same rule for the kernel flags: "a log that echoes
//! what was asked for rather than what is in force is exactly how a dead knob
//! stays invisible for a campaign." It matters more here, because
//! `enable_prefix_caching` has four independent readers — two of which
//! (`logo`, `preflight`) ANNOUNCE the regime rather than act on it. A
//! `--hermetic` that reached `build` but not `logo` would produce a server
//! that runs a KAT while its own banner says prefix caching is on, and the
//! banner is what an operator reads when a score moves.
//!
//! So the raw fields are not read anywhere outside this module's resolvers.

/// Whether the radix KV prefix cache runs.
///
/// Channel M2: the prefix cache is keyed on token content with no session
/// component at all, so one request's KV blocks are reachable by any later
/// request sharing a prefix. Under `--hermetic` it does not run.
pub(crate) fn prefix_caching_enabled(requested: bool, hermetic: bool) -> bool {
    requested && !hermetic
}

/// What the MTP throughput gate is set to, as `set_mtp_gate_force` wants it.
///
/// `None` means "no flag was given, so `ATLAS_MTP_GATE_FORCE` decides" — the
/// documented fallback, and why this is not a plain `bool`.
///
/// Channel M1: the gate does not only SWITCH arms, it PROBES. `tokens_since_event`
/// is cumulative and is never reset at a request boundary, so crossing
/// `event_interval()` hands the next window to the serial arm — and the serial
/// and batch-K forwards are not byte-equal even at temperature 0. Which
/// request is serving when the counter rolls over is a function of everything
/// served before it. `force` disarms the arbiter, so no probe ever fires.
///
/// Returns `Some(true)` under `--hermetic` rather than deferring to the
/// environment: leaving it as `None` would let `ATLAS_MTP_GATE_FORCE=0`
/// reopen the channel from outside the recorded regime, and the record would
/// still say `hermetic`.
pub(crate) fn mtp_gate_force(requested: Option<&str>, hermetic: bool) -> Option<bool> {
    if hermetic {
        return Some(true);
    }
    requested.map(|gate| gate == "force")
}

impl super::ServeArgs {
    /// The effective prefix-caching setting. Read this, never the raw field.
    pub(crate) fn prefix_caching_enabled(&self) -> bool {
        prefix_caching_enabled(self.enable_prefix_caching, self.hermetic)
    }

    /// The effective MTP gate setting. Read this, never the raw field.
    pub(crate) fn mtp_gate_force(&self) -> Option<bool> {
        mtp_gate_force(self.mtp_gate.as_deref(), self.hermetic)
    }
}

#[cfg(test)]
#[path = "hermetic_tests.rs"]
mod tests;
