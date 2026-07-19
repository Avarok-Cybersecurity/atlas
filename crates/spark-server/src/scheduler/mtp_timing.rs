// SPDX-License-Identifier: AGPL-3.0-only

//! Env-gated phase timing for the MTP K=2 verify path (#237 fixed-overhead hunt).
//!
//! `ATLAS_MTP_TIMING=1` arms per-phase accumulators across the verify step:
//! sync/EP/forward, the per-position host pipeline (D2H, dequant, processor
//! stages, penalties, argmax), grammar mask fills (both `fill_bitmask` and the
//! `forced_token` path, which computes a full mask of its own), SSM/proposer
//! state bookkeeping, MTP propose, and the Marconi checkpoint. A single
//! `info!` summary is emitted every [`SUMMARY_PERIOD`] completed verify steps
//! (no per-token spam), then the accumulators reset.
//!
//! Purely diagnostic: zero behavioral effect, and near-zero cost when the env
//! is unset (`enabled()` is a cached bool; `record` returns immediately).
//!
//! `ATLAS_MTP_GATE_FORCE=1` (diagnostic companion, wired in `scheduler::mod`)
//! disarms the throughput gate so verify steps keep flowing even in a regime
//! the gate would call net-negative — required to collect ~100 verify samples
//! for attribution. Never set in production.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Verify steps per summary line.
const SUMMARY_PERIOD: u64 = 25;

/// Timed phases. `StepTotal` must stay last (it sizes the arrays).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub(crate) enum Phase {
    SyncSecondary = 0,
    EpBroadcast,
    VerifyForward,
    FastGreedy,
    D2h,
    Dequant,
    PipelineProc,
    GrammarFill,
    ForcedTok,
    Penalties,
    Argmax,
    Commit,
    SaveHidden,
    TrimProposer,
    ProposeMask,
    Propose,
    MarconiCkpt,
    StepTotal,
}

const NUM_PHASES: usize = Phase::StepTotal as usize + 1;

const NAMES: [&str; NUM_PHASES] = [
    "sync",
    "ep",
    "fwd",
    "fast_greedy",
    "d2h",
    "dequant",
    "pipeline",
    "grammar_fill",
    "forced_tok",
    "penalties",
    "argmax",
    "commit",
    "save_hidden",
    "trim",
    "propose_mask",
    "propose",
    "marconi",
    "TOTAL",
];

static SUM_US: [AtomicU64; NUM_PHASES] = [const { AtomicU64::new(0) }; NUM_PHASES];
static COUNT: [AtomicU64; NUM_PHASES] = [const { AtomicU64::new(0) }; NUM_PHASES];
static STEPS: AtomicU64 = AtomicU64::new(0);

/// Whether `ATLAS_MTP_TIMING=1` armed the accumulators (cached once).
pub(crate) fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_MTP_TIMING").ok().as_deref() == Some("1"))
}

/// Diagnostic: `ATLAS_MTP_GATE_FORCE=1` keeps MTP verify running regardless
/// of the throughput gate (measurement companion; see module docs).
pub(crate) fn gate_forced() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_MTP_GATE_FORCE").ok().as_deref() == Some("1"))
}

/// Record the elapsed time since `since` under `phase`. No-op when disarmed.
pub(crate) fn record(phase: Phase, since: Instant) {
    if !enabled() {
        return;
    }
    let us = u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX);
    SUM_US[phase as usize].fetch_add(us, Ordering::Relaxed);
    COUNT[phase as usize].fetch_add(1, Ordering::Relaxed);
}

/// Mark one verify step complete (records `StepTotal` from `step_start`) and
/// emit the periodic summary. Call once per `step_verify_k2` invocation.
pub(crate) fn step_done(step_start: Instant, seq_len: usize) {
    if !enabled() {
        return;
    }
    record(Phase::StepTotal, step_start);
    let steps = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
    if !steps.is_multiple_of(SUMMARY_PERIOD) {
        return;
    }
    use std::fmt::Write as _;
    let mut line = String::with_capacity(NUM_PHASES * 32);
    for i in 0..NUM_PHASES {
        let sum = SUM_US[i].swap(0, Ordering::Relaxed);
        let cnt = COUNT[i].swap(0, Ordering::Relaxed);
        if cnt == 0 {
            continue;
        }
        // Per-VERIFY-STEP average (a phase can fire >1x per step, e.g. one
        // dequant per verify position); `xN.N` is the avg fires per step.
        let per_step_ms = sum as f64 / 1000.0 / SUMMARY_PERIOD as f64;
        let fires = cnt as f64 / SUMMARY_PERIOD as f64;
        let _ = write!(line, " {}={per_step_ms:.2}ms(x{fires:.1})", NAMES[i]);
    }
    tracing::info!("MTP K2 timing [{SUMMARY_PERIOD} steps, seq_len={seq_len}]:{line}");
}
