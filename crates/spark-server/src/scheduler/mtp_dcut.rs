// SPDX-License-Identifier: AGPL-3.0-only

//! D-Cut: adaptive verification-depth pruning (arXiv 2607.14647).
//!
//! # What it does
//!
//! The drafter emits `k_drafts` tokens per sequence and each one costs a VERIFY
//! ROW. Rows are the verify step's price: the batched forward reads every weight
//! once for `R = Σ rows_i` rows, and R is capped at 96 by the row buffers
//! (`VERIFY_ROW_BUDGET`; 64 before the wave-11 depth-at-width widening, 32
//! before the 32:1 ladder rung). D-Cut spends that budget where it will be
//! accepted instead of spreading it evenly — but only in the <= 8-sequence
//! regime it was measured to win ([`dcut_width_cap`]).
//!
//! The drafter reports a per-position top-1 log-probability `ln c_{i,t}`
//! (`argmax_bf16_batch_lp`). A draft at depth `j` is only reachable if every
//! draft before it was accepted, so its SURVIVAL score is the prefix product
//! `s_{i,j} = Π_{t<=j} c_{i,t}` — in log space the prefix SUM. All prunable
//! positions across the WHOLE batch are ranked by that score and the top
//! `ratio` fraction is retained.
//!
//! ★ Because log-probabilities are <= 0, `s_{i,j}` is non-increasing in `j`, so
//! the retained set is automatically a per-sequence contiguous PREFIX — no tree,
//! no gaps, just a per-sequence draft count. That is the whole reason this needs
//! ragged row counts and nothing else.
//!
//! # v1 scope (deliberate)
//!
//! * Every sequence keeps AT LEAST ONE draft. That holds `rows_i` in 2..=4 —
//!   the exact envelope `can_batch_verify`, the `gdn_decode_wy{2,3,4}` handles
//!   and the SSM intermediates pools were built and audited for. Dropping a
//!   sequence to zero drafts (rows_i = 1) is a separate, untested regime.
//! * The budget is a FIXED ratio from the discrete bucket set, not a profiled
//!   cost table. `ATLAS_MTP_DCUT_RATIO` picks it; values snap to the nearest
//!   bucket so the search space stays the paper's four points.
//! * Pruning changes only the VERIFY width. The propose already ran at full
//!   width when this is called, so v1 banks the row saving, not a drafter
//!   saving.

use super::types::ActiveSeq;

/// The paper's discrete retention buckets.
const BUCKETS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// Verify row-buffer capacity — the exact bound `can_batch_verify` enforces
/// as `Σ rows_i <= 96` (logits rows / meta gaps / bt staging, `sizes.rs`;
/// model twin: `VERIFY_ROW_CAP`, verify_e2.rs — keep in lock-step). 96 since
/// wave 11 (depth at width: 32:2 = n=32 × k=3 rows hits 96 dead on, 24:2 =
/// 72); previously 64 (the 32:1 rung, n=32 × k=2), 32 before that. Raising
/// the budget is behavior-neutral for every default-reachable shape: the
/// default ladder's widest row totals (16:2 = 48, 32:1 = 64, 8:3 = 32) all
/// fit the OLD 64-row bound in one chunk, so chunking and pruning are
/// unchanged — only explicit `ATLAS_MTP_K_LADDER` depth-at-width overrides
/// (24:2 / 32:2) reach rows 65..=96.
// Mirrors `VERIFY_ROW_CAP` (spark-model verify_e2.rs) — keep in
// lock-step. 96 -> 160 with the DFlash n=20 × k=8 widening.
pub(super) const VERIFY_ROW_BUDGET: usize = 160;

/// Widest verify batch in SEQUENCES that the model can accept — the batched
/// verify's hidden stash slot count, which `can_batch_verify` enforces as
/// `(2..=VERIFY_WY_TABLE_SEQS).contains(&n)`. Read from the model crate, not
/// restated, so the chunker tracks the stash if it is ever resized.
pub(super) const WIDTH_CAP: usize = spark_model::layer::VERIFY_WY_TABLE_SEQS;

/// Widest verify batch (SEQUENCES, not rows) D-Cut may prune —
/// the D-Cut-at-depth policy. Value-parsed from `ATLAS_MTP_DCUT_MAX_SEQS`
/// once per process (0 disables pruning entirely; `ATLAS_NO_MTP_DCUT` also
/// does).
///
/// Default 8, anchored to two measurements on the same binary class:
/// * D-Cut's win is a C=8 result — ratio 0.75 pooled 108.57 vs 105.56 off
///   (+2.6%, binary `296b9674`), measured at n<=8 where `ladder_nd = 3`.
/// * At depth-at-width (the 16:2 rung, n=16 × nd=2) pruning is NEGATIVE:
///   fixer r2 leg D read 176.6-179.4 at C=16 vs 194.4-196.0 for the same
///   ladder with pruning off (-9%) — ragged nd=2 pruning fragments the
///   contiguous GDN depth runs and sheds winning drafts. The wave-11 grid
///   confirmed the winner (195.0) with pruning off at n=16.
/// So pruning engages only at batch width <= 8 — exactly the regime it was
/// measured to win — and the 16:2 default rung always verifies the uniform
/// single-chunk `[3; n]` shape that measured +5.7%.
pub(super) fn dcut_width_cap() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_MTP_DCUT_MAX_SEQS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Default ON, kill switch `ATLAS_NO_MTP_DCUT` — PRESENCE check (house
/// convention: `=0` is NOT off).
///
/// Measured at C=8 on binary `296b9674` (one fresh serve per leg, warmup
/// discarded, 5 scored reps): D-Cut off 105.56, ratio 1.0 (wiring live, zero
/// rows pruned) 105.80 — statistically identical, so the plumbing is inert
/// when it prunes nothing — and ratio 0.75 pools to 108.57 over two serves,
/// **+2.6%**. Pruning is additionally a no-op at `ladder_nd < 2` and above
/// [`dcut_width_cap`] sequences (the D-Cut-at-depth policy — see that fn:
/// pruning at the 16:2 rung's n=16 measured -9%).
pub(super) fn dcut_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_DCUT").is_none())
}

/// Retention ratio, VALUE-parsed once and snapped to the nearest
/// [`BUCKETS`] entry.
///
/// Default 0.75 — the ONLY bucket that wins. The whole bucket set was swept at
/// C=8 on binary `296b9674`, one fresh serve each: 1.0 (control) 105.80 ·
/// **0.75 108.57** · 0.5 107.43 · 0.25 101.56. Telemetry shows exactly why —
/// tok_step degrades monotonically as rows are pruned (2.52 / 2.54 / 2.43 /
/// 2.16) while kept_frac falls (1.000 / 0.876 / 0.750 / 0.626), and 0.75 is
/// the one point where the row saving outruns the token loss.
/// 1.0 remains the natural A/B control (wiring live, zero rows pruned).
pub(super) fn dcut_ratio() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let raw = std::env::var("ATLAS_MTP_DCUT_RATIO")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.75)
            .clamp(0.0, 1.0);
        *BUCKETS
            .iter()
            .min_by(|a, b| {
                (*a - raw)
                    .abs()
                    .partial_cmp(&(*b - raw).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("BUCKETS is non-empty")
    })
}

/// Retained draft count per sequence for one verify batch.
///
/// `confs[i]` is sequence i's per-draft top-1 LOG-probability, in draft order.
/// A short or empty row means "not measured": those positions score 0.0 (=
/// certainty) and therefore survive, so a drafter that cannot report confidence
/// is never pruned on a number nobody produced.
///
/// Returns `retained[i]` in `1..=k_drafts`. `row_budget` caps `Σ (retained+1)`
/// so the result can never exceed the verify row buffer.
pub(super) fn select(
    confs: &[&[f32]],
    k_drafts: usize,
    row_budget: usize,
    ratio: f32,
) -> Vec<usize> {
    let n = confs.len();
    // Depth 1 is mandatory (see module docs), so only depths 2..=k_drafts are
    // rankable. Nothing to do at k_drafts <= 1.
    let mut retained = vec![1usize.min(k_drafts); n];
    if k_drafts <= 1 || n == 0 {
        return retained;
    }
    let prunable = n * (k_drafts - 1);

    // Score every prunable position by its log survival (prefix sum).
    let mut ranked: Vec<(f32, usize, usize)> = Vec::with_capacity(prunable);
    for (i, c) in confs.iter().enumerate() {
        let mut acc = 0.0f32;
        for j in 0..k_drafts {
            // Missing measurement -> 0.0 (certain), which sorts to the top.
            acc += c.get(j).copied().unwrap_or(0.0);
            if j >= 1 {
                ranked.push((acc, i, j));
            }
        }
    }
    // Descending by score; ties break on (sequence, depth) so the selection is
    // a deterministic function of the batch — a graph key derived from the
    // resulting shape must not depend on sort instability.
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let by_ratio = ((prunable as f32) * ratio).round() as usize;
    // Rows already committed: one base row + one mandatory draft per sequence.
    let committed = 2 * n;
    let by_budget = row_budget.saturating_sub(committed);
    let keep = by_ratio.min(by_budget).min(prunable);

    for &(_, i, j) in ranked.iter().take(keep) {
        // Scores are non-increasing in depth, so the top-`keep` set is already
        // prefix-closed; `max` records the deepest retained position.
        retained[i] = retained[i].max(j + 1);
    }
    retained
}

/// Plan one verify batch: choose the retained draft-count MULTISET from the
/// drafter's confidences, assign it to the batch's ssm slots in the canonical
/// order, truncate each sequence's drafts to its assigned prefix, and return
/// the resulting per-sequence ROW count (`retained + 1`) in dispatch order.
///
/// ★ The depth→slot ASSIGNMENT is canonical (`verify_key::verify_batch_order`
/// — depths descending paired with slots ascending), not confidence-ordered,
/// AT BATCH WIDTHS >= `verify_key::CANONICAL_KEY_MIN_WIDTH` (8). D-Cut's row
/// saving comes from the multiset, which stays confidence-chosen; WHO gets
/// which depth was the half that multiplied the batched-verify CUDA-graph key
/// space (266 arrangements at n=8 against a 32-entry cache → 89% of steps
/// re-capturing, 23.2 ms/step). It also RECONCILES the batch's two ordering
/// demands — contiguous equal-depth runs and ascending ssm slots — which the
/// confidence-ordered arrangement put in direct conflict. Runs are physically
/// consecutive only when the selected pool slots have no gaps; the model
/// checks their pointers and declines the batched fast path otherwise.
///
/// Below that width the key space is 2 (n=2) or 10 (n=4) keys against a
/// 32-entry cache — nothing to collapse — and the forced assignment measured
/// NET NEGATIVE there (-2.4% at C=2, -3.7% at C=4), so this falls back to the
/// pre-canonical assignment byte for byte. `verify_key::canonical_assignment`
/// is the single gate (threshold + `ATLAS_CANONICAL_KEY_MIN_WIDTH` override +
/// the `ATLAS_NO_CANONICAL_VERIFY_KEY` kill switch); see `verify_key`'s module
/// docs and `CANONICAL_KEY_MIN_WIDTH` for the A/B table.
///
/// With `ATLAS_NO_MTP_DCUT` set — or the batch wider than [`dcut_width_cap`]
/// sequences (the D-Cut-at-depth policy: pruning at the 16:2 rung's n=16
/// measured -9%, so depth-at-width always verifies the uniform shape that
/// won) — this is the uniform ladder shape and the batch order is untouched:
/// the caller's downstream path is then byte-identical to the pre-D-Cut one.
pub(super) fn plan(
    active: &mut [ActiveSeq],
    batchable: &mut Vec<usize>,
    ladder_nd: usize,
    rows: usize,
) -> Vec<usize> {
    let mut ks: Vec<usize> = vec![rows; batchable.len()];
    if !dcut_enabled()
        || ladder_nd < 2
        || batchable.is_empty()
        || batchable.len() > dcut_width_cap()
    {
        return ks;
    }
    let confs: Vec<&[f32]> = batchable
        .iter()
        .map(|&i| {
            let a = &active[i];
            // Length-matched or nothing: a stale or absent confidence vector
            // must read as "not measured" (full depth), never as a score.
            if a.pending_draft_conf.len() == a.pending_drafts.len() {
                a.pending_draft_conf.as_slice()
            } else {
                &[]
            }
        })
        .collect();
    let retained = select(&confs, ladder_nd, VERIFY_ROW_BUDGET, dcut_ratio());
    for (pos, r) in retained.iter().enumerate() {
        ks[pos] = (*r).clamp(1, ladder_nd) + 1;
    }
    // Dispatch order + depth assignment, from the ONE ordering rule shared
    // with the graph key. Canonical: depths descending onto slots ascending,
    // so `ks_out[p]` is the multiset's p-th deepest row count, NOT
    // necessarily the confidence-chosen depth of the sequence placed there.
    // Below `CANONICAL_KEY_MIN_WIDTH` (or with the kill switch set): each
    // sequence keeps its own depth, deepest-first. This is the ONE place the
    // assignment is decided, so it is the ONE place the width gate is asked —
    // `mtp_step` re-applies the ORDER for the same `batchable.len()`.
    let slots: Vec<usize> = batchable
        .iter()
        .map(|&idx| active[idx].seq.ssm_slot_idx().unwrap_or(usize::MAX))
        .collect();
    let (order, ks_out) = spark_model::speculative::verify_key::verify_batch_order(
        &slots,
        &ks,
        spark_model::speculative::verify_key::canonical_assignment(batchable.len()),
    );
    // Truncate to the ASSIGNED depth. Every batchable sequence entered the
    // step with exactly `ladder_nd` drafts (`mtp_step` truncates the surplus)
    // and `ks_out[p] - 1` is in `1..=ladder_nd`, so this is always a prefix
    // of what the drafter produced — deepening a sequence past its own drafts
    // is unrepresentable, not merely unlikely.
    let reordered: Vec<usize> = order.iter().map(|&p| batchable[p]).collect();
    for (idx, &k) in reordered.iter().zip(&ks_out) {
        let a = &mut active[*idx];
        debug_assert!(
            k >= 2 && k - 1 <= a.pending_drafts.len(),
            "assigned depth {k} exceeds the {} drafts proposed",
            a.pending_drafts.len()
        );
        a.pending_drafts.truncate(k - 1);
        a.pending_draft_conf.truncate(k - 1);
    }
    record(batchable.len() * rows, ks_out.iter().sum(), &ks_out);
    *batchable = reordered;
    ks_out
}

/// Split a batch into verify chunks: `[lo, hi)` index ranges over `ks`.
///
/// TWO caps, because `can_batch_verify` enforces two:
/// * the row-buffer bound (`VERIFY_ROW_BUDGET` = 160) — the audited verify
///   envelope (meta gaps / logits rows / bt staging, sizes.rs), from which a
///   per-chunk sequence cap is DERIVED (`budget / widest rows`: rows=4 → 40
///   seqs, rows=3 → 53, rows=2 → 80), no longer a hardcoded 8 for the deep
///   widths;
/// * the WIDTH bound `VERIFY_WY_TABLE_SEQS` = 32 — the batched verify's
///   hidden stash has exactly 32 slots, so `can_batch_verify` admits only
///   `(2..=32).contains(&n)`.
///
/// ★ The width bound used to be omitted here, on the reasoning that "n is
/// separately bounded at 32" by the dispatch cap. It is — by
/// `speculative::mtp_max_seqs()`, whose DEFAULT is 32 but which is an
/// operator value (`ATLAS_MTP_MAX_SEQS`), and by nothing else. Raise it and
/// the row-derived cap lets a chunk reach 40/53/80 sequences; every such
/// chunk is refused WHOLESALE by the `can_batch_verify` gate in `mtp_step`
/// and falls to the per-seq verify loop — one weight sweep PER SEQUENCE.
/// That is the same "stale cap silently serializes the batch" artifact this
/// function was written to close (the hardcoded 8), closed for ROWS and left
/// open for WIDTH. Deriving BOTH caps here means a chunk this function emits
/// is always one the model will actually batch. The old 8 was stale from the
/// 32-row budget era and SILENTLY SERIALIZED any depth shape above n=8 into
/// 8-wide verify chunks (double weight reads per step): a 2026-07-30 fixer-r2
/// leg with `ATLAS_MTP_K_LADDER=..,16:2,..` measured 127-135 tok/s at C=16 vs
/// a 184-185 same-session 16:1 control, and its accept telemetry read
/// `n=8 k_drafts=2` — the chunk cap, not depth economics (the exact artifact
/// class the ladder history documents for "8:3 collapses" / "16:2 → 94.1").
/// Every default-ladder shape (`4:3,8:3,16:2,32:1` — widest totals 32/48/64
/// rows) is a SINGLE chunk under both the old 64 and this 96 budget, so the
/// widening only opens the explicit 24:2/32:2 env rungs (72/96 rows). With
/// uniform `ks` this still reproduces `chunks(budget/rows)` exactly, so
/// D-Cut-off stays byte-identical per chunk. `ks` is deepest-first, so the
/// widest row count is the chunk's first element and the cap never changes
/// mid-chunk.
pub(super) fn chunk_ranges(ks: &[usize]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut lo = 0usize;
    while lo < ks.len() {
        // Derived, not hardcoded: rows <= 4 is ensured by the ladder clamp,
        // so the division is well-defined and >= 40. Clamped by the verify
        // stash width so the chunk is one `can_batch_verify` will accept.
        let seq_cap = (VERIFY_ROW_BUDGET / ks[lo].max(1)).min(WIDTH_CAP);
        let mut hi = lo;
        let mut r = 0usize;
        while hi < ks.len() && hi - lo < seq_cap && r + ks[hi] <= VERIFY_ROW_BUDGET {
            r += ks[hi];
            hi += 1;
        }
        // A single sequence wider than the whole budget cannot happen (rows <=
        // 4), but never emit an empty range.
        if hi == lo {
            hi = lo + 1;
        }
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// Per-step retained-rows telemetry, under the existing
/// `ATLAS_MTP_ACCEPT_DEBUG` gate. Counters only; one line per `PERIOD` steps.
fn record(rows_full: usize, rows_kept: usize, ks: &[usize]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    const PERIOD: u64 = 200;
    static STEPS: AtomicU64 = AtomicU64::new(0);
    static FULL: AtomicU64 = AtomicU64::new(0);
    static KEPT: AtomicU64 = AtomicU64::new(0);
    if !spark_model::speculative::mtp_accept_debug() {
        return;
    }
    FULL.fetch_add(rows_full as u64, Ordering::Relaxed);
    KEPT.fetch_add(rows_kept as u64, Ordering::Relaxed);
    if STEPS.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        let steps = STEPS.swap(0, Ordering::Relaxed).max(1);
        let full = FULL.swap(0, Ordering::Relaxed).max(1);
        let kept = KEPT.swap(0, Ordering::Relaxed);
        tracing::info!(
            "MTP D-Cut ratio={:.2} steps={steps} rows_full={full} rows_kept={kept} \
             kept_frac={:.3} last_ks={ks:?}",
            dcut_ratio(),
            kept as f64 / full as f64,
        );
    }
}
#[cfg(test)]
#[path = "mtp_dcut_tests.rs"]
mod tests;
