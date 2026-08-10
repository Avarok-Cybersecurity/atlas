// SPDX-License-Identifier: AGPL-3.0-only

//! Accept-rate-aware n=16 ladder rung (wave 28).
//!
//! # Why the static rung cannot win
//!
//! Wave 27 measured the SAME two serve boots produce OPPOSITE signs at the
//! n=16 width depending only on the shape of the traffic:
//!
//! | workload | p1 | p2_cond | token ratio | cost ratio | winner |
//! |---|---|---|---|---|---|
//! | decode_short L=64 | 0.840 | 0.542 | 1.186 | 1.170 | tie |
//! | decode_short L=128 | 0.773 | 0.539 | 1.218 | 1.234 | `16:1` +1.2% |
//! | decode_short L=1024 | 0.732 | 0.567 | 1.229 | 1.258 | `16:1` +2.4% |
//! | tool-shaped, natural EOS | 0.917 | **0.877** | **1.424** | 1.223 | `16:2` **+7.9%** |
//!
//! Depth pays iff the token ratio clears the step-cost ratio. The COST ratio
//! is nearly constant (1.17-1.26 measured; wave 19's ladder comment claimed
//! 1.306, which is WRONG and is corrected in `ladder.rs`). The TOKEN ratio is
//! what moves, and it is driven by the SECOND-token conditional accept
//! `p2_cond`, which is bimodal: ~0.54 on prose, 0.877 on structured
//! function-call text. `p1` barely separates the regimes by comparison
//! (0.72-0.92) and is nearly identical between the two arms of the same
//! workload, i.e. the accept rate is a property of the TRAFFIC, not the rung.
//!
//! Output length is NOT the regime variable — the wave-27 length sweep is
//! flat and signless — so a length-aware rung would be the wrong fix.
//!
//! # The rule
//!
//! `token_ratio = (1 + p1 + p1*p2_cond) / (1 + p1) = 1 + p1*p2_cond/(1 + p1)`
//!
//! and `k=2` wins iff that clears the cost ratio. Evaluated on the four
//! wave-27 points the formula reproduces the measurement: 1.247 / 1.235 /
//! 1.240 on prose against **1.420** on tool-shaped (measured 1.424).
//!
//! # Where the band sits, and why it is ABOVE the arithmetic break-even
//!
//! The arithmetic break-even is the measured cost ratio, 1.17-1.26. The band
//! is placed HIGHER, at [`LEAVE`] = 1.30 / [`ENTER`] = 1.32, for two reasons,
//! both empirical:
//!
//! 1. At the margin the two-term model is optimistic. Prose measures a token
//!    ratio of 1.218-1.229 and a LOSS for `k=2` — the model omits prefill
//!    dilution, the host verdict walk over the extra rows, and graph
//!    switching, all of which favour the cheaper state. Break-even in
//!    PRACTICE is above break-even in ARITHMETIC.
//! 2. The band must separate two measured regimes, not sit inside one. The
//!    highest prose observation is 1.247 and the tool observation is 1.420;
//!    1.30/1.32 sits ~4.3% above the former and ~7% below the latter, so
//!    neither regime lands in the dead band and neither is one noisy flush
//!    away from flipping.
//!
//! The band is ASYMMETRIC (enter high, leave lower) because the two states
//! are not symmetric: `k=1` is the cheaper, lower-variance state and the one
//! that clears the campaign's same-box vLLM C=16 bar, so depth must be
//! EARNED on strong evidence and is surrendered on weaker evidence.
//!
//! # Hysteresis and how `p2_cond` is observed at all
//!
//! ★ `p2_cond` is UNOBSERVABLE while running `k=1`: with one draft in flight
//! nothing ever proposes a second token, so no second-position match exists
//! to score. The controller must PROBE — spend a telemetry flush at `k=2`
//! purely to re-measure.
//!
//! ★★ AND A PROBE IS EXPENSIVE, which is the single most important measured
//! fact in this module. It is NOT the ~1-2% that `k=2`'s own step cost would
//! suggest. Measured on dgx1 2026-08-01 at C=16 from the flush timestamps: a
//! steady flush takes 1.16 s, the flush that ENTERS a probe takes 2.90 s, and
//! the flush that returns to `k=1` takes 1.18 s — so a probe costs
//! **+1.74 s, all of it on the way IN, and none on the way out**. That
//! asymmetry names the cause: the batched-verify graph for the rarely-used
//! `k` is LRU-EVICTED between probes (the cache is keyed on the ssm-slot
//! vector as well as `k`, and slot vectors churn), so every probe pays a
//! fresh n=16 graph CAPTURE, while the constantly-used `k=1` graph never
//! leaves the cache. A naive fixed 12.5% duty cycle therefore cost
//! **-11.2%** at C=16 on prose (159.76 against a same-binary static `16:1`
//! control of 179.96) — an order of magnitude worse than the arithmetic
//! predicted, and enough on its own to sink the rung.
//!
//! The fix is NOT to keep both graphs resident (that means growing
//! `VERIFY_BATCHED_GRAPH_CAP` and its memory against a churning key space,
//! for a state we enter rarely). It is to make probes RARE by spending them
//! only when there is reason to. `p1` is observed for FREE at every flush in
//! every state; `p2_cond` is what costs 1.74 s. And the two co-move strongly
//! across the regimes that matter (prose p1 0.72-0.77, tool-shaped 0.79-0.98
//! — a 0.17 gap against a per-flush EWMA noise of ~0.012, i.e. ~14 sigma).
//! So the controller uses the free signal to decide when to spend on the
//! expensive one: it probes when `p1` has RISEN by [`P1_TRIGGER`] above the
//! `p1` at which the current state was last confirmed, plus a long
//! [`PROBE_TICKS`] backstop to catch a `p2_cond` drift at constant `p1`.
//!
//! Two details, both paid for in measurements:
//!
//! * The trigger uses its OWN slow EWMA ([`ALPHA_SLOW`] = 0.15, window ~7
//!   flushes ~ 8 s), NOT the fast decision EWMA. Measured per-flush `p1`
//!   noise at n=16 on prose is sigma = 0.053 — far higher than the decision
//!   loop cares about — which through the fast alpha=0.5 EWMA leaves ~0.031
//!   of wander. A trigger read off THAT signal fires on jitter: at 0.06 it
//!   spent 16 probes in 236 flushes and still cost -4.8% on prose. The slow
//!   EWMA cuts the noise to ~0.015, which makes [`P1_TRIGGER`] = 0.08 a
//!   ~4-sigma event on steady traffic and a ~0.17 (>10-sigma) certainty on a
//!   prose-to-tool change.
//! * The trigger is ONE-SIDED — only a RISE in `p1` buys a probe. The token
//!   ratio is monotone increasing in `p1` at fixed `p2_cond`, so a fall can
//!   never make depth newly attractive, and ignoring it halves the false-fire
//!   rate for free.
//!
//! In steady traffic that is ~0 probes and ~0 cost; on a regime change it is
//! one probe within ~5 flushes (~6 s), which is inside a single agentic
//! burst.
//!
//! In the `k=2` state no probe is ever needed — every flush is a depth flush,
//! so `p2_cond` is observed continuously and for free.
//!
//! Estimates are EWMAs with `alpha` = [`ALPHA`] = 0.5, i.e. an effective
//! window of `1/alpha` = 2 samples. A sample is one `mtp_accept_debug` flush
//! = `PERIOD` (128) verifies = 8 steps at n=16 ~ 1 s, so the window is ~2 s
//! of traffic and the probe interval ([`PROBE_TICKS`] = 8 flushes ~ 8 s)
//! dominates convergence. Sized against the WORKLOAD, not against noise: a
//! tool-regime rep is ~15 s, so a rung that needed 3+ probes to latch would
//! spend most of a short agentic burst in the wrong state. Noise is handled
//! by the dead band instead — 128 verifies give p1 a standard error of
//! ~0.037 and the token ratio ~0.015, against a 0.17 gap between the two
//! measured regimes (~11 sigma), so a single noisy flush cannot flip the
//! rung and flipping every step (which would thrash the verify CUDA graphs)
//! cannot happen. Estimates are seeded on the first sample rather than from
//! zero, so the first probe is acted on immediately instead of after a ramp.
//!
//! ★ Graph residency: the batched-verify graph cache is LRU keyed on
//! `(ssm-slot vector, k)` with `VERIFY_BATCHED_GRAPH_CAP` = 32 entries, so
//! BOTH `k` graphs stay resident across a flip at a stable slot vector — a
//! switch costs one capture the first time each `k` is seen at that vector
//! and replays thereafter. No re-capture churn, no cap change, no extra
//! reserve budget.
//!
//! # Scope
//!
//! Adaptation applies ONLY to the 9..=16 width band (the n=16 rung). n<=8 and
//! n>16 take the static ladder unchanged.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Widths the controller owns: the n=16 rung. n<=8 (the 8:3 rung) and n>16
/// (the 32:1 rung) are out of scope for wave 28 and unchanged.
const BAND: std::ops::RangeInclusive<usize> = 9..=16;

/// Enter `k=2` above this token ratio. See the module doc for the placement
/// argument; override with `ATLAS_MTP_RUNG_ENTER`.
const ENTER: f64 = 1.32;
/// Leave `k=2` below this token ratio (dead band 0.02 wide). Override with
/// `ATLAS_MTP_RUNG_LEAVE`.
const LEAVE: f64 = 1.30;
/// EWMA weight for `p1` and `p2_cond` in the DECISION (effective window
/// `1/ALPHA` = 2 flushes). Override with `ATLAS_MTP_RUNG_ALPHA`.
const ALPHA: f64 = 0.5;
/// EWMA weight for the `p1` the probe TRIGGER reads (window ~7 flushes).
/// Deliberately slower than [`ALPHA`]: the decision wants to react, the
/// trigger wants to not be fooled. Override with `ATLAS_MTP_RUNG_ALPHA_SLOW`.
const ALPHA_SLOW: f64 = 0.15;
/// BACKSTOP probe interval in flushes, for a `p2_cond` drift at constant
/// `p1` that the trigger below would never see. Long on purpose: at 1.74 s a
/// probe and ~1.16 s a flush, 2048 flushes is one probe per ~40 min of n=16
/// traffic, i.e. ~0.07% — small enough to be invisible on the prose bar.
/// Override with `ATLAS_MTP_RUNG_PROBE_TICKS`.
const PROBE_TICKS: u64 = 2048;
/// Probe when the SLOW `p1` EWMA has RISEN this far above the `p1` at which
/// the current state was last confirmed. 0.08 is ~4 sigma of the slow EWMA's
/// noise (~0.015, from a measured per-flush sigma of 0.053) and half the
/// measured prose-to-tool gap (0.17). Override with
/// `ATLAS_MTP_RUNG_P1_TRIGGER`.
const P1_TRIGGER: f64 = 0.08;

/// PRESENCE check for `ATLAS_MTP_STATIC_RUNG` (house convention — `=0` is NOT
/// off): pins the static ladder, restoring the pre-wave-28 behaviour.
/// An explicit `ATLAS_MTP_K_LADDER` ALSO disables adaptation: an operator who
/// spells out the rungs is asking for exactly those rungs.
pub fn adaptation_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("ATLAS_MTP_STATIC_RUNG").is_some()
            || std::env::var_os("ATLAS_MTP_K_LADDER").is_some()
    })
}

/// Named, documented, overridable threshold (PCND — no implicit magic).
fn tunable(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

fn enter_at() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| tunable("ATLAS_MTP_RUNG_ENTER", ENTER))
}
fn leave_at() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| tunable("ATLAS_MTP_RUNG_LEAVE", LEAVE))
}
fn alpha() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| tunable("ATLAS_MTP_RUNG_ALPHA", ALPHA).min(1.0))
}
fn alpha_slow() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| tunable("ATLAS_MTP_RUNG_ALPHA_SLOW", ALPHA_SLOW).min(1.0))
}
fn probe_ticks() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| tunable("ATLAS_MTP_RUNG_PROBE_TICKS", PROBE_TICKS as f64) as u64)
}
fn p1_trigger() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| tunable("ATLAS_MTP_RUNG_P1_TRIGGER", P1_TRIGGER))
}

/// Expected tokens per verify step at `k=2` relative to `k=1`:
/// `(1 + p1 + p1*p2_cond) / (1 + p1)`. Pure — the unit tests are the SSOT
/// for the rule, and the wave-27 rows in the module doc are its anchor.
pub fn token_ratio(p1: f64, p2_cond: f64) -> f64 {
    if p1 <= 0.0 || !p1.is_finite() || !p2_cond.is_finite() {
        return 1.0;
    }
    1.0 + p1 * p2_cond / (1.0 + p1)
}

/// The second-token CONDITIONAL accept implied by a `k>=2` flush:
/// `mean_na = p1 + p1*p2_cond` (+ deeper terms, which the n=16 rung never
/// runs), so `p2_cond = (mean_na - p1) / p1`.
pub fn p2_cond_from(p1: f64, mean_na: f64) -> Option<f64> {
    (p1 > 0.0).then(|| ((mean_na - p1) / p1).clamp(0.0, 1.0))
}

/// Decide the next state from the smoothed token ratio and the current one.
/// `true` = `k=2`. Pure, so hysteresis is unit-testable without the GPU.
pub fn next_state(at_depth: bool, tr: f64) -> bool {
    if at_depth {
        tr >= leave_at()
    } else {
        tr >= enter_at()
    }
}

struct Ctl {
    p1: AtomicU64,
    /// Slow-EWMA `p1`, read ONLY by the probe trigger.
    p1_slow: AtomicU64,
    p2: AtomicU64,
    seeded_p1: AtomicBool,
    seeded_p1_slow: AtomicBool,
    seeded_p2: AtomicBool,
    at_depth: AtomicBool,
    tick: AtomicU64,
    last_probe: AtomicU64,
    /// Smoothed `p1` at the last flush that actually OBSERVED `p2_cond`, i.e.
    /// the `p1` the current state was confirmed at. Movement away from it is
    /// the free evidence that a fresh (expensive) probe is worth buying.
    p1_at_decision: AtomicU64,
    flips: AtomicU64,
}

static CTL: Ctl = Ctl {
    p1: AtomicU64::new(0),
    p1_slow: AtomicU64::new(0),
    p2: AtomicU64::new(0),
    seeded_p1: AtomicBool::new(false),
    seeded_p1_slow: AtomicBool::new(false),
    seeded_p2: AtomicBool::new(false),
    at_depth: AtomicBool::new(false),
    tick: AtomicU64::new(0),
    last_probe: AtomicU64::new(0),
    p1_at_decision: AtomicU64::new(0),
    flips: AtomicU64::new(0),
};

fn ewma_a(cell: &AtomicU64, seeded: &AtomicBool, sample: f64, a: f64) -> f64 {
    let prev = f64::from_bits(cell.load(Ordering::Relaxed));
    let next = if seeded.swap(true, Ordering::Relaxed) {
        a * sample + (1.0 - a) * prev
    } else {
        sample
    };
    cell.store(next.to_bits(), Ordering::Relaxed);
    next
}

/// One telemetry flush at batch width `n`. Called from
/// [`super::mtp_accept_debug`] — the SSOT for accept statistics; this module
/// adds NO parallel accounting path.
pub(super) fn observe(n: usize, k_drafts: usize, p1: f64, mean_na: f64) {
    if adaptation_disabled() || !BAND.contains(&n) {
        return;
    }
    let p1_e = ewma_a(&CTL.p1, &CTL.seeded_p1, p1, alpha());
    let p1_slow = ewma_a(&CTL.p1_slow, &CTL.seeded_p1_slow, p1, alpha_slow());
    let tick = CTL.tick.fetch_add(1, Ordering::Relaxed) + 1;
    if k_drafts >= 2 {
        // A depth flush: this is where p2_cond becomes observable, and it
        // also ENDS any probe in flight (self-terminating — the probe runs
        // exactly until one full flush has been scored at depth).
        CTL.last_probe.store(tick, Ordering::Relaxed);
        CTL.p1_at_decision
            .store(p1_slow.to_bits(), Ordering::Relaxed);
        if let Some(p2) = p2_cond_from(p1, mean_na) {
            let p2_e = ewma_a(&CTL.p2, &CTL.seeded_p2, p2, alpha());
            let tr = token_ratio(p1_e, p2_e);
            let was = CTL.at_depth.load(Ordering::Relaxed);
            let now = next_state(was, tr);
            if now != was {
                CTL.at_depth.store(now, Ordering::Relaxed);
                let flips = CTL.flips.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::info!(
                    "MTP rung n={n} -> k_drafts={} (token_ratio={tr:.4} p1={p1_e:.3} \
                     p2_cond={p2_e:.3} enter={:.3} leave={:.3} tick={tick} flips={flips})",
                    if now { 2 } else { 1 },
                    enter_at(),
                    leave_at(),
                );
            }
        }
    }
}

/// The per-step draft count for `n_active`, accept-rate-aware at the n=16
/// rung and the static ladder everywhere else.
pub fn drafts_for(n_active: usize, num_drafts: usize) -> usize {
    let base = spark_model::speculative::mtp_ladder_drafts(n_active, num_drafts);
    if adaptation_disabled()
        || spark_model::speculative::mtp_ladder_disabled()
        || num_drafts < 2
        || !BAND.contains(&n_active)
    {
        return base;
    }
    // Probe while in `k=1`. A probe costs a measured 1.74 s of graph
    // re-capture, so it is bought on evidence, never on a timer alone:
    //   1. ALWAYS until the first depth flush has been scored, so a fresh
    //      serve measures `p2_cond` immediately instead of guessing.
    //   2. When the FREE signal `p1` (slow EWMA) has RISEN `p1_trigger()`
    //      above the value the current state was confirmed at — a regime
    //      change. One-sided: a FALL in p1 cannot make depth attractive.
    //   3. A long `probe_ticks()` backstop for `p2_cond` drift at constant
    //      `p1`.
    let tick = CTL.tick.load(Ordering::Relaxed);
    let p1_moved = f64::from_bits(CTL.p1_slow.load(Ordering::Relaxed))
        - f64::from_bits(CTL.p1_at_decision.load(Ordering::Relaxed))
        >= p1_trigger();
    let probing = !CTL.seeded_p2.load(Ordering::Relaxed)
        || p1_moved
        || tick.saturating_sub(CTL.last_probe.load(Ordering::Relaxed)) >= probe_ticks();
    if CTL.at_depth.load(Ordering::Relaxed) || probing {
        return 2.min(num_drafts).max(base);
    }
    base
}

/// Runtime engagement state of speculation, by batch WIDTH. `true` = the
/// current width is inside the dispatch cap and speculation is live.
static WIDTH_ENGAGED: AtomicBool = AtomicBool::new(true);
/// Monotone count of width-regime transitions since boot.
static WIDTH_FLIPS: AtomicU64 = AtomicU64::new(0);

/// ★ THE OTHER HALF OF THE RUNTIME SPECULATION REGIME (wave 47).
///
/// [`drafts_for`] picks the DEPTH once speculation is engaged. Whether it is
/// engaged AT ALL is a WIDTH decision, taken at the scheduler's dispatch site
/// as `active.len() <= speculative::mtp_max_seqs()` (default 32). That
/// predicate has been in the tree since the ladder shipped, and wave 47
/// measured that it is already a complete regime selector: ONE serve carrying
/// `--speculative --num-drafts 3` at `--max-batch-size 128` speculates at
/// C<=32 and plain-decodes at C=64/128, landing every rung of the
/// concurrency ladder on its own best-configuration value. No second
/// controller was needed — the campaign's two serve configurations were one
/// configuration observed at two widths.
///
/// What was missing is any way to SEE it. A batch that grows past the cap
/// stops speculating silently, so throughput alone cannot distinguish an
/// engaged rung from a disengaged one, and `ATLAS_MTP_ACCEPT_DEBUG`'s flushes
/// simply STOP — an absence, which is indistinguishable from telemetry being
/// off. This records the transition in the same shape wave 28 used for the
/// depth decision: atomics, one INFO on CHANGE only (never per step), and a
/// monotone flip counter so oscillation is countable rather than inferred.
///
/// SSOT: the caller passes the predicate it already evaluated and dispatches
/// on. This adds no second decision and no parallel accounting path.
pub(super) fn note_width_regime(n_active: usize, engaged: bool) {
    if WIDTH_ENGAGED.swap(engaged, Ordering::Relaxed) == engaged {
        return;
    }
    let flips = WIDTH_FLIPS.fetch_add(1, Ordering::Relaxed) + 1;
    let cap = spark_model::speculative::mtp_max_seqs();
    if engaged {
        tracing::info!(
            "speculation ENGAGED at width n={n_active} (dispatch cap {cap}) — flips={flips}"
        );
    } else {
        tracing::info!(
            "speculation DISENGAGED at width n={n_active} > dispatch cap {cap}: this width \
             plain-decodes (ATLAS_MTP_MAX_SEQS raises the cap; the verify pools grow with it) \
             — flips={flips}"
        );
    }
}

#[cfg(test)]
#[path = "adaptive_rung_tests.rs"]
mod tests;
