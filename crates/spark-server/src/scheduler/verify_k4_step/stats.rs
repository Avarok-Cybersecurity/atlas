// SPDX-License-Identifier: AGPL-3.0-only

//! K=4 acceptance telemetry: the two recorders `step_verify_k4` calls, and
//! the periodic summary they emit.
//!
//! Split from the verify step itself for the file-size cap. The seam is a
//! real one — nothing here influences a pick; it observes what the step
//! decided and reports it every `SUMMARY_PERIOD` steps.

// The AtomicU64 counters that lived here are now `SchedCtx::stats`
// (`scheduler::spec_stats::SpecStats`), so a run's acceptance rate describes
// the model that produced it rather than blending two across a swap.

// Periodic accept-distribution summary (P4, 2026-05-24). Mirrors K=3.
const K4_SUMMARY_PERIOD: u64 = 100;

// UNCONDITIONAL per-position draft-match counters (2026-07-21). See the
// matching block in verify_k3_step.rs for the rationale: the accept chain
// short-circuits, so it can only ever report CONDITIONAL rates for positions
// >= 2, and a conditional rate at position 3 selects contexts where BOTH
// earlier positions succeeded (survivorship). The verify pass computes the
// target argmax at every position in one batch, so every position is
// observable on every step.

#[inline]
pub(in crate::scheduler) fn k4_record_positional(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    d1: bool,
    d2: bool,
    d3: bool,
    seq_len: usize,
) {
    sched
        .stats
        .k4_steps
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if d1 {
        sched
            .stats
            .k4_d1
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if d2 {
            sched
                .stats
                .k4_d2_cond
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if d3 {
                sched
                    .stats
                    .k4_d3_cond
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    if d2 {
        sched
            .stats
            .k4_d2_uncond
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if d3 {
        sched
            .stats
            .k4_d3_uncond
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if sched
        .stats
        .k4_steps
        .load(std::sync::atomic::Ordering::Relaxed)
        >= K4_SUMMARY_PERIOD
    {
        let n = sched
            .stats
            .k4_steps
            .swap(0, std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let d1c = sched
            .stats
            .k4_d1
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let d2u = sched
            .stats
            .k4_d2_uncond
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let d3u = sched
            .stats
            .k4_d3_uncond
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let d2c = sched
            .stats
            .k4_d2_cond
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let d3c = sched
            .stats
            .k4_d3_cond
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            "K4 positional: steps={n} p1={:.3} p2_uncond={:.3} p3_uncond={:.3} \
             p2_cond={:.3} p3_cond={:.3} (d1={d1c} d2u={d2u} d3u={d3u} d2c={d2c} d3c={d3c}) \
             seq_len={seq_len}",
            (d1c as f64) / (n as f64),
            (d2u as f64) / (n as f64),
            (d3u as f64) / (n as f64),
            if d1c > 0 {
                (d2c as f64) / (d1c as f64)
            } else {
                f64::NAN
            },
            if d2c > 0 {
                (d3c as f64) / (d2c as f64)
            } else {
                f64::NAN
            },
        );
    }
}

#[inline]
pub(in crate::scheduler) fn k4_record_outcome(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_accepted: usize,
    seq_len: usize,
) {
    let counter = match num_accepted {
        3 => &sched.stats.k4_accept[3],
        2 => &sched.stats.k4_accept[2],
        1 => &sched.stats.k4_accept[1],
        _ => &sched.stats.k4_accept[0],
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = sched.stats.k4_accept[3].load(std::sync::atomic::Ordering::Relaxed)
        + sched.stats.k4_accept[2].load(std::sync::atomic::Ordering::Relaxed)
        + sched.stats.k4_accept[1].load(std::sync::atomic::Ordering::Relaxed)
        + sched.stats.k4_accept[0].load(std::sync::atomic::Ordering::Relaxed);
    if total >= K4_SUMMARY_PERIOD {
        let a3 = sched.stats.k4_accept[3].swap(0, std::sync::atomic::Ordering::Relaxed);
        let a2 = sched.stats.k4_accept[2].swap(0, std::sync::atomic::Ordering::Relaxed);
        let a1 = sched.stats.k4_accept[1].swap(0, std::sync::atomic::Ordering::Relaxed);
        let a0 = sched.stats.k4_accept[0].swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (a3 + a2 + a1 + a0).max(1);
        let mean = (3 * a3 + 2 * a2 + a1) as f64 / total as f64;
        tracing::info!(
            "K4 summary: {a3} accept-3 / {a2} accept-2 / {a1} accept-1 / {a0} reject in last {total} steps (mean accepted={mean:.2}) seq_len={seq_len}"
        );
    }
}
