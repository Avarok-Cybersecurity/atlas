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
//! is unset (`RunTiming::armed` is a plain bool; `record` returns immediately).
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
    /// Whole `step_mtp` invocation (outer bracket). `step_mtp − Σ StepTotal`
    /// is the host prep/tail INSIDE the step driver but OUTSIDE the per-chunk
    /// verify guard: classification, D-Cut planning, chunk sort, bootstrap.
    StepOuter,
    /// Out-of-step wall: elapsed from the previous verify step's guard drop
    /// to the next guard construction. This is the "~14% between steps" the
    /// wave-10 ledger left unattributed; the `loop_*` phases below name its
    /// scheduler-tick components (whatever GAP holds beyond them is emit /
    /// gate / verify-ctx / misc glue).
    Gap,
    LoopDrain,
    LoopSnapshot,
    LoopAdmit,
    LoopPrefill,
    LoopRetire,
    LoopSwap,
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
    "step_mtp",
    "GAP",
    "loop_drain",
    "loop_snapshot",
    "loop_admit",
    "loop_prefill",
    "loop_retire",
    "loop_swap",
    "TOTAL",
];

/// Per-phase microsecond accumulators for ONE run.
///
/// These were three statics plus an `enabled` `OnceLock`. A timing histogram
/// that spans a model swap averages two models together and describes neither,
/// so the sink belongs to the run — reached as `SchedCtx::timing`, and cloned
/// into `GrammarState` (which records two phases from a subsystem that has no
/// scheduler context and should not grow one).
#[derive(Debug)]
pub struct RunTiming {
    /// Armed by `ATLAS_MTP_TIMING=1`. When false every `record` is a
    /// predictable branch and nothing else.
    pub armed: bool,
    sum_us: [AtomicU64; NUM_PHASES],
    count: [AtomicU64; NUM_PHASES],
    steps: AtomicU64,
    /// Anchor-micros of the last verify step's guard drop (0 = no step yet).
    /// Written by [`StepTimer::drop`], read by [`StepTimer::new`] to record
    /// [`Phase::Gap`] — the out-of-step wall between consecutive verify steps.
    /// A per-run field, not a static, for the same reason the accumulators are:
    /// a gap measured across a model swap describes neither model.
    last_step_end_us: AtomicU64,
}

impl RunTiming {
    pub fn from_env() -> Self {
        Self::new(std::env::var("ATLAS_MTP_TIMING").ok().as_deref() == Some("1"))
    }

    pub fn new(armed: bool) -> Self {
        Self {
            armed,
            sum_us: [const { AtomicU64::new(0) }; NUM_PHASES],
            count: [const { AtomicU64::new(0) }; NUM_PHASES],
            steps: AtomicU64::new(0),
            last_step_end_us: AtomicU64::new(0),
        }
    }

    /// Record the elapsed time since `since` under `phase`. No-op when disarmed.
    pub(crate) fn record(&self, phase: Phase, since: Instant) {
        if !self.armed {
            return;
        }
        let us = u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.sum_us[phase as usize].fetch_add(us, Ordering::Relaxed);
        self.count[phase as usize].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn sum_us(&self, phase: Phase) -> u64 {
        self.sum_us[phase as usize].load(Ordering::Relaxed)
    }

    pub(crate) fn count(&self, phase: Phase) -> u64 {
        self.count[phase as usize].load(Ordering::Relaxed)
    }

    pub fn bump_steps(&self) -> u64 {
        self.steps.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }
}

impl Default for RunTiming {
    fn default() -> Self {
        Self::new(false)
    }
}

// `ATLAS_MTP_GATE_FORCE` is now `SchedLevers::mtp_gate_force`, read from
// `SchedCtx` where the gate is armed.

/// Mark one verify step complete (records `StepTotal` from `step_start`) and
/// emit the periodic summary. Call once per `step_verify_k2` invocation.
pub(crate) fn step_done(timing: &RunTiming, step_start: Instant, seq_len: usize) {
    if !timing.armed {
        return;
    }
    timing.record(Phase::StepTotal, step_start);
    let steps = timing.bump_steps();
    if !steps.is_multiple_of(SUMMARY_PERIOD) {
        return;
    }
    use std::fmt::Write as _;
    let mut line = String::with_capacity(NUM_PHASES * 32);
    for i in 0..NUM_PHASES {
        let sum = timing.sum_us[i].swap(0, Ordering::Relaxed);
        let cnt = timing.count[i].swap(0, Ordering::Relaxed);
        if cnt == 0 {
            continue;
        }
        // Per-VERIFY-STEP average (a phase can fire >1x per step, e.g. one
        // dequant per verify position); `xN.N` is the avg fires per step.
        let per_step_ms = sum as f64 / 1000.0 / SUMMARY_PERIOD as f64;
        let fires = cnt as f64 / SUMMARY_PERIOD as f64;
        let _ = write!(line, " {}={per_step_ms:.2}ms(x{fires:.1})", NAMES[i]);
    }
    tracing::info!("MTP verify timing [{SUMMARY_PERIOD} steps, seq_len={seq_len}]:{line}");
}

/// Drop guard that emits the [`step_done`] summary on EVERY exit path of a
/// verify step.
///
/// Exists because `verify_k4_step` — the SHIPPED config (`--num-drafts 3`) —
/// has four accept branches and several early error returns. The per-phase
/// `record()` calls already fired there (picks route through
/// `verify_pipeline_helper`), but nothing called `step_done`, which lived only
/// in `verify_k2_step`. The accumulators filled and the summary was never
/// emitted: a probe generating ~1800 tokens at K=4 produced zero timing lines.
/// A guard cannot drift out of date the way a per-tail call can.
///
/// `seq_len` is captured at construction and is only the log's label — the
/// verify path advances it mid-step. The measurement is elapsed step time.
/// Error returns are counted too; they set `a.finished` and are rare, but an
/// unusually low `total` beside a high step count implies they fired.
///
/// Costs one `Instant::now()` when `ATLAS_MTP_TIMING` is unset, since
/// `step_done` returns immediately when the sink is disarmed.
pub(crate) struct StepTimer<'a> {
    start: Instant,
    seq_len: usize,
    /// The run's sink. Borrowed rather than reached globally, so the summary
    /// covers one model's steps.
    timing: &'a RunTiming,
}

/// Monotonic anchor for the inter-step GAP clock. `Instant` cannot live in an
/// atomic, so GAP timestamps are micros since this per-process anchor. The
/// anchor is only an origin — the DIFFERENCE of two readings is what is
/// recorded, so it is process-scoped without averaging anything across runs.
fn anchor_us() -> u64 {
    static ANCHOR: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    u64::try_from(ANCHOR.get_or_init(Instant::now).elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl<'a> StepTimer<'a> {
    pub(crate) fn new(timing: &'a RunTiming, seq_len: usize) -> Self {
        if timing.armed {
            let prev = timing.last_step_end_us.load(Ordering::Relaxed);
            if prev != 0 {
                let gap = anchor_us().saturating_sub(prev);
                timing.sum_us[Phase::Gap as usize].fetch_add(gap, Ordering::Relaxed);
                timing.count[Phase::Gap as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
        Self {
            start: Instant::now(),
            seq_len,
            timing,
        }
    }
}

impl Drop for StepTimer<'_> {
    fn drop(&mut self) {
        step_done(self.timing, self.start, self.seq_len);
        if self.timing.armed {
            self.timing
                .last_step_end_us
                .store(anchor_us().max(1), Ordering::Relaxed);
        }
    }
}
