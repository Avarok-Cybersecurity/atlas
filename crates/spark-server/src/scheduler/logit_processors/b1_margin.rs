// SPDX-License-Identifier: AGPL-3.0-only

//! B1 (2026-05-26) margin-ratio drift detector — observability only.
//!
//! Moved here (STEP 5 of the logit-pipeline unification) from the inline
//! `decode_logits_seq::process_seq_logits` block so the single unified
//! `super::process_position_logits` fn owns it. B1 NEVER mutates logits
//! and NEVER alters which token wins — it scans `logits` for top-1/top-2,
//! records the gap, and (periodically) emits a WARN gauge. It is gated to
//! the FINAL decode position only (`PositionKind::FinalDecode`, risk R6):
//! the MTP verify path picks K positions speculatively and most are rolled
//! back, so counting them would inflate the gauge with non-emitted tokens.
//!
//! In long-context FP8 decode 23.7% of decode positions have gap<1.5
//! logprobs (see bench/fp8_dgx2_drift/research_C1_results.md) — the regime
//! where FP8 numerical noise flips a low-margin argmax. The detector logs
//! those positions without taking action.

use crate::scheduler::ActiveSeq;

/// Periodic summary stride: emit one WARN line every N low-margin firings.
const B1_SUMMARY_PERIOD: u64 = 100;
/// Below this top1−top2 gap (logprobs) a position is "low margin".
const LOW_MARGIN_THRESHOLD: f32 = 1.5;

static B1_LOW_MARGIN_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn record_low_margin(margin: f32, top1: u32, top2: u32) {
    let n = B1_LOW_MARGIN_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    // Per-event trace — TRACE level by design (23.7% of long-ctx positions
    // trip this; INFO would spam). Power-user diagnostic:
    // `RUST_LOG=spark::scheduler::logit_processors::b1_margin=trace`.
    tracing::trace!("B1 low margin: gap={margin:.3} top1={top1} top2={top2}");
    if n.is_multiple_of(B1_SUMMARY_PERIOD) {
        tracing::warn!(
            "B1 drift gauge: {n} low-margin (<1.5 logprobs) decode positions \
             observed inside parameter bodies. \
             FP8 numerical noise is in the argmax-flip regime — consider \
             reviewing tool-arg outputs for whitespace / digit-collapse drift."
        );
    }
}

/// Observe the top-1/top-2 margin of the (post-grammar-mask) distribution
/// and record a low-margin firing when inside a parameter body with chars
/// already emitted. Pure observability — `logits` is read, never written.
pub(super) fn observe(logits: &[f32], a: &ActiveSeq) {
    // Single O(V) scan for top-1 and top-2 (pre-penalty, pre-bias) — same
    // scan the inline B1 block ran.
    let (top1_idx, top1_val, top2_idx, top2_val) = {
        let mut t1_idx = 0u32;
        let mut t1_val = f32::NEG_INFINITY;
        let mut t2_idx = 0u32;
        let mut t2_val = f32::NEG_INFINITY;
        for (idx, &v) in logits.iter().enumerate() {
            if v > t1_val {
                t2_val = t1_val;
                t2_idx = t1_idx;
                t1_val = v;
                t1_idx = idx as u32;
            } else if v > t2_val {
                t2_val = v;
                t2_idx = idx as u32;
            }
        }
        (t1_idx, t1_val, t2_idx, t2_val)
    };
    let margin = top1_val - top2_val;
    let low_margin_in_body =
        a.inside_parameter_body && a.param_body_chars_emitted > 0 && margin < LOW_MARGIN_THRESHOLD;
    if low_margin_in_body {
        record_low_margin(margin, top1_idx, top2_idx);
    }
}
