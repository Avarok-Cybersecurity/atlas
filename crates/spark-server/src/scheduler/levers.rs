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
    /// GRAMMARLESS verify fast-greedy — the chat sibling of the #237 grammar
    /// arm. Ships ON; `ATLAS_NO_FAST_GREEDY_CHAT=1` restores the per-seq
    /// `[K,vocab]`-D2H slow path (the byte-invariant tie-breaking arm).
    pub fast_greedy_chat: bool,
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
    /// Pin the MTP throughput gate to the VERIFY arm for DFlash at
    /// `active.len() <= 2` (`ATLAS_DFLASH_GATE_PIN_C2=0` restores
    /// arbitration). Measured 2026-08-19 (qwen3.8-27B+DFlash2, C=2):
    /// arbitration was par on tok/s (25.4 vs 24.4) but its serial↔batch-K
    /// forward flips FORK the temp-0 token stream mid-answer and the bad
    /// attractor degenerates into repetition (content-loop watchdog kills,
    /// 300-cap rambles); pinned verify holds C1-parity accept (75% vs 38%)
    /// and completions EOS normally. C>=3 keeps arbitration — per-seq serial
    /// verify genuinely loses there (22.0 vs 28.1 tok/s at C=4).
    pub dflash_gate_pin_c2: bool,
    /// Cross-sequence batched DFlash K=γ verify (`ATLAS_DFLASH_BATCH_VERIFY=0`
    /// forces the per-sequence loop). One R=n*(γ+1)-row forward replaces n
    /// full weight sweeps; the GDN body still runs per sequence, so accepts
    /// are unchanged and only the wall moves.
    pub dflash_batch_verify: bool,
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

/// Ships ON; `=0` opts out. For levers that graduated from opt-in after
/// validation — the variable keeps its historical name and `=1` stays a
/// harmless no-op, so every recipe that set it remains correct.
fn on_unless_zero(var: &str) -> bool {
    std::env::var(var).ok().as_deref() != Some("0")
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

/// `--mtp-gate`, published before the scheduler's levers resolve.
static MTP_GATE_FORCE_CLI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Publish the command line's `--mtp-gate`. Call once, at serve time.
///
/// `None` means the flag was NOT given. Publishing the `auto` default instead
/// sealed this cell on every `spark serve` and left `ATLAS_MTP_GATE_FORCE=1`
/// documented but dead — the fallback below could never be reached. An absent
/// flag now publishes nothing, so the variable works again for the scripts it
/// exists for, and an explicit `--mtp-gate auto` still overrides it.
pub fn set_mtp_gate_force(force: Option<bool>) {
    if let Some(force) = force {
        let _ = MTP_GATE_FORCE_CLI.set(force);
    }
}

/// The `--mtp-gate force` decision IN FORCE: the flag when it was given, the
/// legacy variable otherwise.
///
/// The SSOT for the resolution — `SchedLevers::from_env` reads it, and so does
/// the startup log, which must print what is in force rather than what was
/// asked for. Two spellings of this rule is how the log came to report `auto`
/// on a run that was forcing.
pub fn mtp_gate_force() -> bool {
    MTP_GATE_FORCE_CLI
        .get()
        .copied()
        .unwrap_or_else(|| opt_in("ATLAS_MTP_GATE_FORCE"))
}

impl SchedLevers {
    /// Resolve from the environment. Called once, when the run starts.
    pub fn from_env() -> Self {
        Self {
            fast_greedy_grammar: on_unless("ATLAS_DISABLE_FAST_GREEDY"),
            fast_masked: on_unless("ATLAS_DISABLE_FAST_MASKED"),
            fast_greedy_chat: on_unless("ATLAS_NO_FAST_GREEDY_CHAT"),
            force_temp_zero: opt_in("ATLAS_FORCE_TEMP_ZERO"),
            mtp_minp: on_unless("ATLAS_NO_MTP_MINP"),
            mtp_verify_sample: on_unless("ATLAS_NO_MTP_VERIFY_SAMPLE"),

            dflash_masked_verify: opt_in("ATLAS_DFLASH_MASKED_VERIFY"),
            dflash_seam_serial: opt_in("ATLAS_DFLASH_SEAM_SERIAL"),
            dflash_adaptive: opt_in("ATLAS_DFLASH_ADAPTIVE"),
            dflash_serial_append: opt_in("ATLAS_DFLASH_SERIAL_APPEND"),
            // DEFAULT-ON since 2026-08-19: without the unified ctx commit the
            // drafter conditions on a starved/poisoned hidden accumulator
            // (only row k-1 — an almost-always-rejected draft — captured, one
            // slot per step regardless of num_accepted, serial stretches
            // dropped entirely). Validated on qwen3.8-27B+DFlash2: code-leg
            // 10.5 -> 36.8 tok/s (+250%, accept 41% -> 84%), prose +130%,
            // count +31% (PR #604). `ATLAS_DFLASH_UNIFIED_CTX=0` restores the
            // legacy append for A/B.
            dflash_unified_ctx: on_unless_zero("ATLAS_DFLASH_UNIFIED_CTX"),
            dflash_spec_think: opt_in("ATLAS_DFLASH_SPEC_THINK"),
            dflash_gate_pin_c2: on_unless_zero("ATLAS_DFLASH_GATE_PIN_C2"),
            dflash_batch_verify: on_unless_zero("ATLAS_DFLASH_BATCH_VERIFY"),
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
            // `--mtp-gate force` is the configured spelling; the env var is
            // the fallback for scripts that predate the flag.
            mtp_gate_force: mtp_gate_force(),
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
            fast_greedy_chat: true,
            force_temp_zero: false,
            mtp_minp: true,
            mtp_verify_sample: true,
            dflash_masked_verify: false,
            dflash_seam_serial: false,
            dflash_adaptive: false,
            dflash_serial_append: false,
            dflash_unified_ctx: true,
            dflash_spec_think: false,
            dflash_gate_pin_c2: true,
            dflash_batch_verify: true,
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
            fast_greedy_chat: self.fast_greedy_chat,
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
    fn an_absent_mtp_gate_flag_leaves_the_legacy_variable_reachable() {
        // The whole of the fix: publishing the clap default sealed
        // `MTP_GATE_FORCE_CLI` on every `spark serve`, so the
        // `ATLAS_MTP_GATE_FORCE` fallback in `mtp_gate_force` could never run
        // even though `--help` documents it. `None` must not seal.
        //
        // ★ The cell is process-global with no reset, so this is the only test
        // in this binary that may write it — a second writer would make both
        // order-dependent.
        for _ in 0..3 {
            set_mtp_gate_force(None);
        }
        set_mtp_gate_force(Some(true));
        assert!(
            mtp_gate_force(),
            "an absent flag must leave the cell open for the next writer"
        );
        assert!(
            SchedLevers::from_env().mtp_gate_force,
            "and the carried levers read the same resolution — one rule, not two"
        );
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
