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
pub(super) const VERIFY_ROW_BUDGET: usize = 96;

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

/// Plan one verify batch: choose each batchable sequence's retained draft
/// count, truncate its drafts to that prefix, reorder the batch deepest-first,
/// and return the resulting per-sequence ROW count (`retained + 1`).
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
    for (slot, &idx) in batchable.iter().enumerate() {
        let keep = retained[slot].clamp(1, ladder_nd);
        active[idx].pending_drafts.truncate(keep);
        active[idx].pending_draft_conf.truncate(keep);
        ks[slot] = keep + 1;
    }
    record(batchable.len() * rows, ks.iter().sum(), &ks);
    // Deepest first so equal-depth sequences form CONTIGUOUS row runs — the
    // cross-sequence batched GDN conv+WY fast path launches once per run, so
    // fragmenting the depths would trade verify rows for kernel launches.
    // Secondary key stays the ssm slot (canonical graph key + the
    // consecutive-slot precondition).
    let mut order: Vec<(usize, usize)> = batchable.iter().copied().zip(ks).collect();
    order.sort_by_key(|&(idx, k)| {
        (
            std::cmp::Reverse(k),
            active[idx].seq.ssm_slot_idx().unwrap_or(usize::MAX),
        )
    });
    *batchable = order.iter().map(|&(i, _)| i).collect();
    order.iter().map(|&(_, k)| k).collect()
}

/// Split a batch into verify chunks: `[lo, hi)` index ranges over `ks`.
///
/// ONE cap: the row-buffer bound (`VERIFY_ROW_BUDGET` = 96) — the audited
/// verify envelope (meta gaps / logits rows / bt staging, sizes.rs). The
/// sequence-count cap is DERIVED from it per chunk (`budget / widest rows`:
/// rows=4 → 24 seqs, rows=3 → 32 = the 32:2 shape in one chunk, rows=2 → 48;
/// `can_batch_verify` separately bounds n at 32 = `VERIFY_WY_TABLE_SEQS`), no
/// longer a hardcoded 8 for the deep widths. The old 8 was stale from the
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
        // so the division is well-defined and >= 24.
        let seq_cap = VERIFY_ROW_BUDGET / ks[lo].max(1);
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
mod tests {
    use super::*;

    #[test]
    fn ratio_one_retains_full_depth() {
        let c: Vec<Vec<f32>> = vec![vec![-0.1, -2.0, -3.0]; 4];
        let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
        assert_eq!(select(&refs, 3, 32, 1.0), vec![3, 3, 3, 3]);
    }

    #[test]
    fn zero_ratio_keeps_the_mandatory_first_draft() {
        let c: Vec<Vec<f32>> = vec![vec![-0.1, -0.2, -0.3]; 4];
        let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
        assert_eq!(select(&refs, 3, 32, 0.0), vec![1, 1, 1, 1]);
    }

    #[test]
    fn budget_spent_on_the_confident_sequence() {
        // seq 0 is confident throughout, seq 1 collapses after its first draft.
        let c = [vec![-0.01f32, -0.02, -0.03], vec![-3.0f32, -4.0, -5.0]];
        let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
        // 2 sequences x 2 prunable depths = 4 candidates; ratio 0.5 keeps 2,
        // both of which belong to seq 0.
        assert_eq!(select(&refs, 3, 32, 0.5), vec![3, 1]);
    }

    #[test]
    fn retained_set_is_always_a_prefix() {
        let c = [vec![-0.1f32, -9.0, -0.001], vec![-0.2f32, -0.2, -0.2]];
        let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
        let r = select(&refs, 3, 32, 0.5);
        // Depth-3 of seq 0 has a WORSE survival score than depth-2 despite its
        // own high confidence, because survival is the prefix product.
        assert!(r[0] <= 2, "prefix product must dominate the local value");
        assert!(r.iter().all(|&k| (1..=3).contains(&k)));
    }

    #[test]
    fn row_budget_is_never_exceeded() {
        let c: Vec<Vec<f32>> = vec![vec![-0.001, -0.001, -0.001]; 8];
        let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
        // 8 sequences, budget 24 rows: 16 committed, 8 spare -> 8 extra depths.
        let r = select(&refs, 3, 24, 1.0);
        let rows: usize = r.iter().map(|k| k + 1).sum();
        assert!(rows <= 24, "rows={rows}");
    }

    #[test]
    fn missing_confidences_are_never_pruned() {
        let empty: Vec<f32> = Vec::new();
        let c = [vec![-5.0f32, -5.0, -5.0], empty];
        let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
        assert_eq!(select(&refs, 3, 32, 0.5), vec![1, 3]);
    }

    #[test]
    fn chunk_ranges_reproduce_the_uniform_caps() {
        // Every default-ladder-reachable shape is a SINGLE chunk — true under
        // the old 64-row budget too, so the 96 widening is default-inert.
        assert_eq!(chunk_ranges(&[4; 8]), vec![(0, 8)]);
        assert_eq!(chunk_ranges(&[3; 8]), vec![(0, 8)]);
        // The 16:2 default rung: [3; 16] = 48 rows, one chunk.
        assert_eq!(chunk_ranges(&[3; 16]), vec![(0, 16)]);
        assert_eq!(chunk_ranges(&[2; 16]), vec![(0, 16)]);
        // The 32:1 rung: one chunk up to n=32 (R = 64).
        assert_eq!(chunk_ranges(&[2; 17]), vec![(0, 17)]);
        assert_eq!(chunk_ranges(&[2; 32]), vec![(0, 32)]);
    }

    #[test]
    fn chunk_ranges_seq_cap_derives_from_the_row_budget() {
        // Depth above n=8 (env-ladder / ragged-D-Cut shapes) is no longer
        // serialized into 8-wide chunks: the row budget is the only bound.
        // rows=3: 96/3 = 32 seqs — 16:2, 24:2 AND 32:2 are each ONE chunk.
        assert_eq!(chunk_ranges(&[3; 21]), vec![(0, 21)]);
        assert_eq!(chunk_ranges(&[3; 24]), vec![(0, 24)]);
        assert_eq!(chunk_ranges(&[3; 32]), vec![(0, 32)]);
        assert_eq!(chunk_ranges(&[3; 33]), vec![(0, 32), (32, 33)]);
        // rows=4: 96/4 = 24 seqs.
        assert_eq!(chunk_ranges(&[4; 9]), vec![(0, 9)]);
        assert_eq!(chunk_ranges(&[4; 24]), vec![(0, 24)]);
        assert_eq!(chunk_ranges(&[4; 25]), vec![(0, 24), (24, 25)]);
        // rows=2: 96/2 = 48 seqs.
        assert_eq!(chunk_ranges(&[2; 48]), vec![(0, 48)]);
        assert_eq!(chunk_ranges(&[2; 49]), vec![(0, 48), (48, 49)]);
        // rows=3 with 10 seqs (the old (0,8),(8,10) split): one chunk now.
        assert_eq!(chunk_ranges(&[3; 10]), vec![(0, 10)]);
    }

    #[test]
    fn chunk_ranges_respect_the_row_budget_when_ragged() {
        // Deepest-first, mixed depths: rows must never exceed the budget per
        // chunk.
        let ks = vec![4, 4, 4, 4, 4, 3, 3, 2, 2, 2];
        for (lo, hi) in chunk_ranges(&ks) {
            let rows: usize = ks[lo..hi].iter().sum();
            assert!(rows <= VERIFY_ROW_BUDGET, "rows={rows}");
            assert!(hi > lo);
        }
    }

    // Env-independent as long as the test process does not set
    // ATLAS_MTP_DCUT_MAX_SEQS (CI does not) — same pattern as the ladder
    // default-shape test.
    #[test]
    fn dcut_width_cap_default_is_the_measured_win_regime() {
        // 8 = the C=8 regime where ratio 0.75 measured +2.6%; pruning at the
        // 16:2 rung's n=16 measured -9% (fixer r2 leg D), so `plan` must
        // return the uniform shape for any wider batch.
        assert_eq!(dcut_width_cap(), 8);
    }

    #[test]
    fn ratio_snaps_to_a_bucket() {
        // Pure snapping arithmetic, no env: 0.6 is closest to 0.5.
        let nearest = |raw: f32| {
            *BUCKETS
                .iter()
                .min_by(|a, b| (*a - raw).abs().partial_cmp(&(*b - raw).abs()).unwrap())
                .unwrap()
        };
        assert_eq!(nearest(0.6), 0.5);
        assert_eq!(nearest(0.9), 1.0);
        assert_eq!(nearest(0.1), 0.25);
    }
}
