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

/// The serve keys `--hermetic` forces, and the values it forces them to.
///
/// ★ WHY THIS EXISTS AS WELL AS THE RESOLVERS BELOW. The resolvers are the
/// ENFORCEMENT — they decide what the server actually does. This table is the
/// DISCLOSURE, and it is needed because a recipe can set one of these keys
/// itself. A gate self-starts from a recipe that turns the prefix cache on, so
/// the rendered command line read `--hermetic --enable-prefix-caching` and
/// `validate_serve_args` refused it — correctly, by its own rule, and the
/// effect was that `--hermetic` could not be used through the only path that
/// self-starts. Measured, not reasoned about: five legs failed in 0s each.
///
/// So a requested `hermetic=true` EXPANDS into these keys before the recipe is
/// rendered. Two things fall out of that, both wanted: the rendered args no
/// longer contradict, and the gate record names the closures explicitly beside
/// the regime, so a reader does not have to know which combination constitutes
/// a KAT.
///
/// It never overwrites a key someone named. A recipe DEFAULT is not intent; an
/// explicit `--serve-override enable_prefix_caching=true` or a baseline pin
/// IS, and beside `--hermetic` it is a real contradiction that must still be
/// refused rather than silently won.
///
/// `hermetic_closures_match_the_resolvers` pins this table against the
/// resolvers so the disclosure cannot come to disagree with the enforcement.
/// Re-exported, NOT redeclared. The table lives in `atlas_plugin::gate::hermetic`
/// because `gate::bench` needs it too — it refuses a BENCH.toml entry that pins
/// `hermetic=true` without these — and atlas-plugin cannot depend on this
/// crate. Two copies would drift, and the failure mode of drift here is a gate
/// that cannot be discharged by the run it asks for.
pub(crate) use atlas_plugin::gate::hermetic::CLOSED_KEYS;

/// Fill in the keys `--hermetic` closes, for any the caller did not name.
pub(crate) fn expand(
    mut requested: std::collections::BTreeMap<String, String>,
) -> std::collections::BTreeMap<String, String> {
    if !atlas_plugin::gate::hermetic::is_requested(&requested) {
        return requested;
    }
    for (k, v) in CLOSED_KEYS {
        requested
            .entry((*k).to_string())
            .or_insert_with(|| (*v).to_string());
    }
    requested
}

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
