// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-verify graph key: the canonical depth→slot assignment and the key
//! bytes derived from it. ONE ordering rule, shared by the scheduler that
//! dispatches the batch (`mtp_dcut::plan`, `mtp_step`) and the model that
//! builds the CUDA-graph cache key (`verify_e2::verify_batched_graph_key`).
//!
//! # The measured defect (nsys + A/B on dgx2, binary `b508679e4`)
//!
//! The batched-verify graph cache keys on the per-row `(ssm slot, depth k)`
//! pairs in batch order, because a capture bakes each row's pool state
//! addresses AND the depth-run launch structure. D-Cut re-ranks WHICH
//! sequence gets WHICH depth every step, so the key space was the set of
//! ARRANGEMENTS of the step's depth multiset over the batch's slots:
//!
//! ```text
//! n=8, the three multisets actually observed:
//!   8!/(5!·2!·1!) + 8!/(4!·4!) + 8!/(6!·2!) = 168 + 70 + 28 = 266 keys
//! ```
//!
//! against `VERIFY_BATCHED_GRAPH_CAP` = 32. Measured key counts: n=2 → 2,
//! n=4 → 10, **n=8 → 160-253**, n=16 → 1 (D-Cut is off above width 8). nsys
//! at C=8: 149 captures in 167 steps (89% of steps), `cuGraphInstantiate` +
//! `cuGraphExecDestroy` + `cuGraphDestroy` = 23.2 ms/step ≈ 20% of the step;
//! GPU busy 96.3% → 77.2%. A/B at C=8: control 78.89 tok/s vs 84.35 with
//! D-Cut off (+6.9%, key count 253 → 1) — and that leg also LOSES the row
//! pruning, so the thrash alone costs more than 6.9%.
//!
//! # The fix: canonical depth→slot assignment
//!
//! D-Cut's ranking chooses HOW MANY drafts survive at each depth (the
//! multiset) — that is where its row saving comes from. It also chooses WHO
//! gets them, and that half is what multiplies the key space. So the multiset
//! stays confidence-chosen and the ARRANGEMENT becomes a pure function of the
//! batch: depths descending are paired with slots ascending. The key is then
//! determined by (slot set, depth multiset) alone — at n=8 the 266 observed
//! arrangements collapse to the 3 multisets that produced them (worst case
//! over all reachable shapes: multisets of size 8 over depths {2,3,4} =
//! C(10,2) = 45, versus 3^8 = 6561 arrangements).
//!
//! ★ The two orderings RECONCILE instead of fighting. The dispatch needs
//! depths descending (equal depths must form contiguous runs — the batched
//! conv+WY fast path launches once per run,
//! `trait_decode_batched_conv_gdn_multi.rs`) and the SSM batched arms need
//! slots ascending in batch order (`ssm_batched_recurrent.rs`,
//! `decode_step.rs`, `mtp_step.rs`). Under the confidence-chosen arrangement
//! those two demands are in direct conflict: a ragged batch sorted
//! deepest-first scrambles the slot order, so each depth run gets an
//! arbitrary SUBSET of the pool slots and the consecutive-slot precondition
//! fails. Pairing depths-descending with slots-ascending makes the two orders
//! THE SAME order. A depth run owns a consecutive slot block only when the
//! selected pool slots are themselves consecutive; the model checks actual
//! pointers and declines the batched fast path when fragmentation leaves gaps.
//!
//! Correctness: which sequence gets which depth is a pure PERFORMANCE choice.
//! Every batchable sequence enters the step with exactly `ladder_nd` drafts
//! (`mtp_step` truncates the surplus), each assigned depth is in
//! `1..=ladder_nd` drafts, and a verify of a shorter draft prefix is the same
//! math on fewer rows. Σ rows is unchanged, so the row budget and chunking
//! are unchanged. What is NOT free to change is the pairing between a batch
//! POSITION and the slot whose pointers the graph baked there — hence one
//! ordering rule, used by both the dispatch and the key.
//!
//! Kill switch `ATLAS_NO_CANONICAL_VERIFY_KEY` (PRESENCE — house convention,
//! `=0` is NOT off) restores the pre-canonical behaviour: each sequence keeps
//! its own confidence-chosen depth and the batch is sorted deepest-first,
//! ssm-slot second.
//!
//! # The width gate: it only pays where the key space explodes
//!
//! Collapsing the key space is not free — forcing the assignment overrides
//! D-Cut's confidence pairing and re-shapes the depth runs — and the key
//! space only explodes at the TOP of D-Cut's width range. Measured key
//! counts against `VERIFY_BATCHED_GRAPH_CAP` = 32: n=2 → 2, n=4 → 10,
//! **n=8 → 160-253**, n=16 → 1. At n=2 and n=4 there is essentially nothing
//! to collapse, and the A/B says so — see [`CANONICAL_KEY_MIN_WIDTH`], which
//! is the ONE threshold and carries the table. Below it the pre-canonical
//! assignment is restored BYTE-IDENTICALLY; at/above it the canonical
//! assignment applies. [`canonical_assignment`] is the single gate; call
//! sites never re-derive it.

/// Canonical assignment ON unless `ATLAS_NO_CANONICAL_VERIFY_KEY` is present.
/// Read once per process.
pub fn canonical_verify_key_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_CANONICAL_VERIFY_KEY").is_none())
}

/// Batch WIDTH (sequences) at or above which the canonical depth→slot
/// assignment is applied. Below it [`verify_batch_order`] /
/// [`verify_batch_permutation`] take their `canonical = false` arm, which is
/// the pre-canonical (pre-PR-#552) behaviour byte for byte: each sequence
/// keeps its own confidence-chosen depth and the batch sorts deepest-first,
/// ssm-slot second, ties on input index (a stable
/// `sort_by_key(|(a, k)| (Reverse(k), slot))`, exactly what both call sites
/// used before).
///
/// Default **8**, from a same-binary same-session A/B on dgx2 (ladder-38
/// round 7, tip `e0b845f11`) with ONE variable — the kill switch
/// `ATLAS_NO_CANONICAL_VERIFY_KEY=1`. tok/s, higher is better:
///
/// ```text
///  C  | canonical ON            | canonical OFF          | verdict
/// ----+-------------------------+------------------------+------------------
///   2 | 30.09                   | 30.83                  | costs -2.4%
///   4 | 64.00 (round 7: 65.56)  | 68.11 (round 6, no it) | costs ~-3.7%
///   8 | 110.63                  | 106.48                 | GAINS +3.9%
///  16 | 203.50                  | 203.44                 | no effect
///  32 | 291.50                  | 291.52                 | no effect
///  64 | 387.62                  | 386.99                 | no effect
/// 128 | 477.55                  | 477.69                 | no effect
/// ```
///
/// The shape of that table follows the key counts (module docs): the
/// collapse pays exactly where the arrangement space is large. Above width 8
/// the gate is inert in either direction — D-Cut is off there
/// (`dcut_width_cap`), `ks` is uniform, and both arms reduce to "sort by
/// slot" (pinned by `uniform_depths_are_identical_under_both_arms`), which
/// is why the C>=16 rungs move by <= 0.2% either way.
///
/// Hypothesis for the cost below 8, recorded but NOT load-bearing for this
/// threshold (the threshold is the measurement, not the mechanism): forcing
/// the assignment makes the two-launch batched GDN conv+WY fast path decline
/// more often, i.e. `n*(2k-1)` launches per layer instead of 2 — 768 vs 96
/// per step at n=2, k=4 over 48 GDN layers. PR #553's rate telemetry under
/// `ATLAS_MTP_ACCEPT_DEBUG` reports that decline rate directly.
pub const CANONICAL_KEY_MIN_WIDTH: usize = 8;

/// Sweep the threshold without a rebuild: `ATLAS_CANONICAL_KEY_MIN_WIDTH=<n>`
/// (VALUE-parsed; 0 = canonical at every width, a value above the widest
/// batch = never). Unset or unparseable ⇒ [`CANONICAL_KEY_MIN_WIDTH`].
/// Parsed once per process, like `dcut_width_cap`.
pub fn canonical_key_min_width() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| min_width_from_env(std::env::var_os(ENV_MIN_WIDTH)))
}

/// The env var name, named once so the parser and its tests cannot drift.
const ENV_MIN_WIDTH: &str = "ATLAS_CANONICAL_KEY_MIN_WIDTH";

/// Pure parse of [`ENV_MIN_WIDTH`] — the I/O lives in
/// [`canonical_key_min_width`] so the policy is testable without touching
/// process env (which a `OnceLock` would latch anyway).
fn min_width_from_env(raw: Option<std::ffi::OsString>) -> usize {
    raw.and_then(|v| v.into_string().ok())
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(CANONICAL_KEY_MIN_WIDTH)
}

/// **THE GATE** — the one decision "does this batch get the canonical
/// depth→slot assignment?". Both seams ask this and nothing else:
/// `mtp_dcut::plan` (which decides order AND assignment) and `mtp_step`
/// (permutation only), each passing the FULL batch width so the two can
/// never disagree — `plan` gates on `batchable.len()`, and a chunked
/// dispatch must use that same width, not the chunk's.
///
/// `n` is the batch width in SEQUENCES. All logic lives in
/// `canonical_assignment_at`; this is only the env binding (SBIO).
pub fn canonical_assignment(n: usize) -> bool {
    canonical_assignment_at(n, canonical_key_min_width(), canonical_verify_key_enabled())
}

/// The gate policy, with its two environment inputs INJECTED (SBIO): the
/// resolved threshold and whether the kill switch is CLEAR. The kill switch
/// dominates — once `ATLAS_NO_CANONICAL_VERIFY_KEY` is set, no width and no
/// `ATLAS_CANONICAL_KEY_MIN_WIDTH` value can turn the assignment back on.
///
/// Split out because `OnceLock`-latched env cannot be moved from a test, and
/// a policy nobody can exercise is a policy nobody has checked.
fn canonical_assignment_at(n: usize, min_width: usize, kill_switch_clear: bool) -> bool {
    kill_switch_clear && n >= min_width
}

/// Dispatch ORDER for one verify batch — the permutation only.
///
/// `slots[i]` / `ks[i]` describe batch member `i` in the caller's arbitrary
/// order (`ks[i]` = that member's ROW count, drafts+1). `order[p]` is the
/// input index dispatched at position `p`.
///
/// * `canonical = true` — sort by ssm slot ASCENDING. `ks` is unread: under
///   the canonical assignment the depths are already descending along that
///   order, so slot order IS depth order and there is nothing to trade off.
/// * `canonical = false` — the kill-switch path: deepest first, ssm slot
///   second (today's behaviour, where the two demands genuinely conflict).
///
/// Ties break on input index, so the result is a deterministic function of
/// the inputs — a graph key must never depend on sort instability. Callers
/// that build the batch in ascending active-sequence index therefore agree
/// on the order of slot-less (`usize::MAX`) members.
///
/// Idempotent under `canonical = true`, so a chunked caller may re-apply it
/// to a contiguous sub-range of an already-ordered batch.
pub fn verify_batch_permutation(slots: &[usize], ks: &[usize], canonical: bool) -> Vec<usize> {
    assert_eq!(
        slots.len(),
        ks.len(),
        "verify_batch_permutation: slots/ks mismatch"
    );
    let n = slots.len().min(ks.len());
    let mut order: Vec<usize> = (0..n).collect();
    if canonical {
        order.sort_by_key(|&i| (slots[i], i));
    } else {
        order.sort_by_key(|&i| (std::cmp::Reverse(ks[i]), slots[i], i));
    }
    order
}

/// Order one verify batch AND assign its depths — the planner's entry point
/// (`mtp_dcut::plan`), the one place a sequence's verify depth is decided.
///
/// Returns `(order, depths)` where `order` is [`verify_batch_permutation`]
/// and `depths[p]` is the row count position `p` verifies.
///
/// * `canonical = true` — the depth MULTISET is re-paired onto the ordered
///   batch, deepest onto the lowest slot. `depths[p]` is therefore NOT
///   generally `ks[order[p]]`; the multiset is preserved exactly, only the
///   pairing is re-made. Both dispatch invariants then hold by construction:
///   slots non-decreasing in `p` (the SSM consecutive-slot precondition) and
///   depths non-increasing in `p` (the contiguous depth-run precondition).
/// * `canonical = false` — each member keeps its own depth.
///
/// Because a caller must TRUNCATE each sequence's drafts to the depth it was
/// assigned, this must be called exactly once per batch; downstream stages
/// that only need the batch in dispatch order use
/// [`verify_batch_permutation`], which cannot disturb an assignment.
pub fn verify_batch_order(
    slots: &[usize],
    ks: &[usize],
    canonical: bool,
) -> (Vec<usize>, Vec<usize>) {
    let order = verify_batch_permutation(slots, ks, canonical);
    let depths: Vec<usize> = if canonical {
        let mut d: Vec<usize> = ks[..order.len()].to_vec();
        d.sort_unstable_by(|a, b| b.cmp(a));
        d
    } else {
        order.iter().map(|&i| ks[i]).collect()
    };
    (order, depths)
}

/// The batched-verify CUDA-graph cache key for one batch: the `(ssm slot,
/// row count)` pairs in DISPATCH order, then a wy-tables-present sentinel.
///
/// Every SSM pointer a capture bakes (h/conv state, rollback intermediates,
/// WY table contents) is a pure function of the pair at that batch position,
/// and the depth-run launch structure is a pure function of the depth
/// sequence — so the key must carry both, in order. All other captured
/// addresses (hidden/logits/scratch/meta) are fixed buffers refreshed
/// pre-replay. The sentinel keeps a table-less capture from ever replaying a
/// table-full step or vice versa.
///
/// Pairs arrive in the order [`verify_batch_order`] produced, so with
/// canonicalization on the key is a pure function of (slot set, depth
/// multiset, sentinel) — the whole point of this module.
pub fn verify_graph_key(pairs: &[(u32, u32)], wy_tables_null: bool) -> Vec<u32> {
    let mut key: Vec<u32> = Vec::with_capacity(2 * pairs.len() + 1);
    for &(slot, k) in pairs {
        key.push(slot);
        key.push(k);
    }
    key.push(u32::MAX - u32::from(wy_tables_null));
    key
}

#[cfg(test)]
#[path = "verify_key_tests.rs"]
mod tests;
