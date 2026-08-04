// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler levers, resolved once per run and then carried.
//!
//! The scheduler's counterpart to `spark_model::layers::ops::ModelLevers`.
//! These steered the decode, verify and speculation paths from twenty-odd
//! `OnceLock<bool>` statics reading `ATLAS_*` at first touch; a static outlives
//! the model whose flags it encodes, and it declares nothing in the signature
//! of the function that reads it.
//!
//! One field is genuinely mutable at runtime — the loop watchdog, which the
//! TUI's ops REPL toggles with `/watchdog on|off` while the server is serving.
//! That one is an [`AtomicBool`] inside the carried struct rather than a plain
//! `bool`: the mutation is real, so it is modelled, but it stays *inside* the
//! run's own state instead of being a process global.

use std::sync::atomic::{AtomicBool, Ordering};

/// Decode / verify / speculation levers for one run.
pub struct SchedLevers {
    // ── Grammar & sampling ──
    /// Fast greedy path when a grammar is active. Ships ON;
    /// `ATLAS_DISABLE_FAST_GREEDY=1` opts out.
    pub fast_greedy_grammar: bool,
    /// Fast masked-sampling chat path. Ships ON;
    /// `ATLAS_DISABLE_FAST_MASKED=1` opts out.
    pub fast_masked: bool,
    /// Force temperature 0 regardless of the request. Diagnostic.
    pub force_temp_zero: bool,
    /// Apply min-p during MTP verify. Ships ON; `ATLAS_NO_MTP_MINP=1` opts out.
    pub mtp_minp: bool,
    /// Run the full sample pipeline during MTP verify. Ships ON;
    /// `ATLAS_NO_MTP_VERIFY_SAMPLE=1` opts out.
    pub mtp_verify_sample: bool,

    // ── DFlash speculation ──
    pub dflash_masked_verify: bool,
    pub dflash_seam_serial: bool,
    pub dflash_adaptive: bool,
    pub dflash_serial_append: bool,
    pub dflash_unified_ctx: bool,
    pub dflash_spec_think: bool,
    /// Mean accepted drafts below which adaptive speculation suspends.
    pub dflash_adaptive_min: f32,
    /// Serially-decoded tokens between adaptive re-probes.
    pub dflash_adaptive_reprobe: u32,
    /// `ATLAS_DFLASH_RESUME_GUARD=N` (default 0 = off): keep the first N
    /// post-`</think>` tokens on plain serial decode.
    pub dflash_resume_guard: u32,
    /// `ATLAS_MTP_SHADOW_TOPK` — the verify side of the drafter top-k probe.
    /// Parsed by `spark_model::speculative::shadow_topk`, the SSOT.
    pub shadow_topk: usize,

    // ── Watchdogs ──
    /// Disable every generation watchdog.
    pub disable_watchdogs: bool,
    /// Suppress EOS while inside a thinking block.
    pub eos_suppressed_by_thinking: bool,
    /// Forced-token fast path. Ships ON; `ATLAS_DISABLE_FORCED_TOKEN=1` opts out.
    pub forced_token_fastpath: bool,

    // ── Diagnostics / instrumentation ──
    pub decode_timing: bool,
    pub mtp_timing: bool,
    pub mtp_gate_force: bool,
    pub adadec_diagnostic: bool,

    /// Loop watchdog. **Runtime-mutable** — the TUI ops REPL toggles it while
    /// serving, which is why it is an atomic rather than a plain field.
    loop_watchdog: AtomicBool,
}

/// `ATLAS_FOO=1` enables.
fn opt_in(var: &str) -> bool {
    std::env::var(var).ok().as_deref() == Some("1")
}

/// `ATLAS_FOO=1` DISABLES — the flag names a negative, the field stores the
/// positive, so the inversion happens here instead of at every read site.
fn on_unless(var: &str) -> bool {
    std::env::var(var).ok().as_deref() != Some("1")
}

/// A numeric tunable: the parsed value, or `default` when unset or unparsable.
fn num<T: std::str::FromStr>(var: &str, default: T) -> T {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Presence-gated: ANY value enables, including `0`. Preserved rather than
/// normalised — a diagnostic someone armed with `=0` must stay armed.
fn present(var: &str) -> bool {
    std::env::var(var).is_ok()
}

impl SchedLevers {
    /// Resolve from the environment. Called once, when the run starts.
    pub fn from_env() -> Self {
        Self {
            fast_greedy_grammar: on_unless("ATLAS_DISABLE_FAST_GREEDY"),
            fast_masked: on_unless("ATLAS_DISABLE_FAST_MASKED"),
            force_temp_zero: opt_in("ATLAS_FORCE_TEMP_ZERO"),
            mtp_minp: on_unless("ATLAS_NO_MTP_MINP"),
            mtp_verify_sample: on_unless("ATLAS_NO_MTP_VERIFY_SAMPLE"),

            dflash_masked_verify: opt_in("ATLAS_DFLASH_MASKED_VERIFY"),
            dflash_seam_serial: opt_in("ATLAS_DFLASH_SEAM_SERIAL"),
            dflash_adaptive: opt_in("ATLAS_DFLASH_ADAPTIVE"),
            dflash_serial_append: opt_in("ATLAS_DFLASH_SERIAL_APPEND"),
            dflash_unified_ctx: opt_in("ATLAS_DFLASH_UNIFIED_CTX"),
            dflash_spec_think: opt_in("ATLAS_DFLASH_SPEC_THINK"),
            dflash_adaptive_min: num("ATLAS_DFLASH_ADAPTIVE_MIN", 2.0),
            dflash_adaptive_reprobe: num("ATLAS_DFLASH_ADAPTIVE_REPROBE", 256),
            dflash_resume_guard: num("ATLAS_DFLASH_RESUME_GUARD", 0),
            shadow_topk: spark_model::speculative::shadow_topk(),

            // Reuses the tested parsers in `helpers` rather than re-deriving
            // the rule: both accept "1" OR "true", trimmed, and re-spelling
            // that here as `== "1"` would silently ignore `=true`.
            disable_watchdogs: crate::scheduler::helpers::parse_disable_watchdogs(
                std::env::var("ATLAS_DISABLE_WATCHDOGS").ok().as_deref(),
            ),
            eos_suppressed_by_thinking: opt_in("ATLAS_EOS_SUPPRESS_THINKING"),
            forced_token_fastpath: crate::scheduler::helpers::parse_forced_token_fastpath(
                std::env::var("ATLAS_DISABLE_FORCED_TOKEN").ok().as_deref(),
            ),

            // Presence-gated, not value-gated.
            decode_timing: present("ATLAS_DECODE_TIMING"),
            mtp_timing: opt_in("ATLAS_MTP_TIMING"),
            mtp_gate_force: opt_in("ATLAS_MTP_GATE_FORCE"),
            adadec_diagnostic: present("ATLAS_ADADEC_DIAGNOSTIC"),

            loop_watchdog: AtomicBool::new(false),
        }
    }

    /// Every opt-in off and every opt-out on — what a build with no `ATLAS_*`
    /// set resolves to. Tests use this instead of mutating the environment.
    pub fn defaults() -> Self {
        Self {
            fast_greedy_grammar: true,
            fast_masked: true,
            force_temp_zero: false,
            mtp_minp: true,
            mtp_verify_sample: true,
            dflash_masked_verify: false,
            dflash_seam_serial: false,
            dflash_adaptive: false,
            dflash_serial_append: false,
            dflash_unified_ctx: false,
            dflash_spec_think: false,
            dflash_adaptive_min: 2.0,
            dflash_adaptive_reprobe: 256,
            dflash_resume_guard: 0,
            shadow_topk: 0,
            disable_watchdogs: false,
            eos_suppressed_by_thinking: false,
            forced_token_fastpath: true,
            decode_timing: false,
            mtp_timing: false,
            mtp_gate_force: false,
            adadec_diagnostic: false,
            loop_watchdog: AtomicBool::new(false),
        }
    }

    /// Is the loop watchdog armed?
    pub fn loop_watchdog(&self) -> bool {
        self.loop_watchdog.load(Ordering::Relaxed)
    }

    /// The subset the pre-sample pipeline reads, for `LogitsContext`.
    pub fn sampling(&self) -> crate::scheduler::logit_processors::SamplingLevers {
        crate::scheduler::logit_processors::SamplingLevers {
            force_temp_zero: self.force_temp_zero,
            fast_greedy_grammar: self.fast_greedy_grammar,
            mtp_verify_sample: self.mtp_verify_sample,
            fast_masked: self.fast_masked,
            adadec_diagnostic: self.adadec_diagnostic,
            dflash_masked_verify: self.dflash_masked_verify,
            disable_watchdogs: self.disable_watchdogs,
            forced_token_fastpath: self.forced_token_fastpath,
            mtp_minp: self.mtp_minp,
        }
    }

    /// Arm or disarm the loop watchdog. Called by the TUI ops REPL mid-run.
    pub fn set_loop_watchdog(&self, on: bool) {
        self.loop_watchdog.store(on, Ordering::Relaxed);
    }
}

impl Default for SchedLevers {
    fn default() -> Self {
        Self::defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_opt_out_levers_ship_on() {
        // Each of these is spelled as a NEGATIVE env var. Collapsing them into
        // an opt-in resolver would silently disable five shipped behaviours.
        let d = SchedLevers::defaults();
        assert!(d.fast_greedy_grammar, "ATLAS_DISABLE_FAST_GREEDY");
        assert!(d.fast_masked, "ATLAS_DISABLE_FAST_MASKED");
        assert!(d.mtp_minp, "ATLAS_NO_MTP_MINP");
        assert!(d.mtp_verify_sample, "ATLAS_NO_MTP_VERIFY_SAMPLE");
        assert!(d.forced_token_fastpath, "ATLAS_DISABLE_FORCED_TOKEN");
    }

    #[test]
    fn every_opt_in_lever_ships_off() {
        let d = SchedLevers::defaults();
        assert!(!d.force_temp_zero);
        assert!(!d.dflash_masked_verify && !d.dflash_adaptive && !d.dflash_spec_think);
        assert!(!d.disable_watchdogs);
        assert!(!d.decode_timing && !d.mtp_timing && !d.adadec_diagnostic);
    }

    #[test]
    fn the_loop_watchdog_is_toggleable_at_runtime() {
        // The one lever with real runtime mutation: the TUI ops REPL flips it
        // mid-run. Modelled as an atomic INSIDE the carried struct rather than
        // as a process global with a setter.
        let d = SchedLevers::defaults();
        assert!(!d.loop_watchdog());
        d.set_loop_watchdog(true);
        assert!(d.loop_watchdog());
        d.set_loop_watchdog(false);
        assert!(!d.loop_watchdog());
    }

    #[test]
    fn two_runs_hold_independent_levers() {
        let a = SchedLevers::defaults();
        let b = SchedLevers {
            dflash_adaptive: true,
            ..SchedLevers::defaults()
        };
        assert!(!a.dflash_adaptive && b.dflash_adaptive);
        a.set_loop_watchdog(true);
        assert!(!b.loop_watchdog(), "and independent runtime state");
    }
}
