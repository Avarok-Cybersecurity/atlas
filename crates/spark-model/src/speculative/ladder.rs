// SPDX-License-Identifier: AGPL-3.0-only

//! K-vs-batch ladder (task #35): per-step draft count as a function of the
//! number of active sequences.
//!
//! Fixed K=4 (3 drafts) over n >= 8 sequences MEASURED as a collapse to a
//! ~55 tok/s plateau at every C (cap=16 sweep, 2026-07-28): n*(K+1) verify
//! rows of SUPERLINEAR per-sequence GDN plus graph-key churn. The ladder
//! shrinks the per-sequence draft count as concurrency grows so the verify
//! row total stays small while the weight-read amortization of the batched
//! verify keeps growing with n:
//! n <= 4 -> 3 drafts (4 rows/seq, today's proven regime, bit-for-bit),
//! n <= 8 -> 1 draft (2 rows/seq, R <= 16).
//! The ladder ENDS at n=8: the finalizer matrix (2026-07-28) measured spec
//! at n=16 as a LOSS at even the minimum depth — C=16 114.6-117.3 vs 131.5
//! MTP-off in BOTH the 16:1 and 8:1,16:1 configs (the K=1 verify step costs
//! ~1.9x a plain batch-16 decode step vs the <1.72x break-even at p1~0.72;
//! suspects: per-seq GDN conv/WY loop at k<4, chunked batched propose).
//! At n<=8 the 8:1 step measured BEST: C=8 82.4 (+12% over 73.5 MTP-off)
//! vs 81.1 for the 8:2 variant. n>8 is MTP-off via the cap (default 8).
//!
//! Overrides:
//! * `ATLAS_MTP_K_LADDER="4:3,8:2,16:1"` — comma-separated `n_max:drafts`
//!   steps, VALUE-parsed once per process. Draft counts clamp to
//!   `[1, num_drafts]` (the CLI `--num-drafts` remains the ceiling, so
//!   `"4:4,..."` parses to the full configured draft count).
//! * `ATLAS_NO_MTP_K_LADDER` — PRESENCE check (house convention, `=0` is
//!   NOT off): disables the ladder entirely (fixed `num_drafts` at every n)
//!   AND drops the [`super::mtp_max_seqs`] default back to 4, restoring the
//!   pre-ladder adaptive policy (batched K=4 MTP at C<=4, MTP-off above).

/// PRESENCE check for `ATLAS_NO_MTP_K_LADDER`. Read once per process.
pub fn mtp_ladder_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_K_LADDER").is_some())
}

/// Parsed ladder steps `(n_max, drafts)`, ascending by `n_max`. Falls back
/// to the default ladder when `ATLAS_MTP_K_LADDER` is unset or unparseable
/// (a malformed value must not silently disable speculation).
fn mtp_ladder_steps() -> &'static [(usize, usize)] {
    static STEPS: std::sync::OnceLock<Vec<(usize, usize)>> = std::sync::OnceLock::new();
    STEPS.get_or_init(|| {
        let parsed = std::env::var("ATLAS_MTP_K_LADDER").ok().and_then(|v| {
            let mut steps: Vec<(usize, usize)> = Vec::new();
            for part in v.split(',') {
                let (n, k) = part.trim().split_once(':')?;
                steps.push((n.trim().parse().ok()?, k.trim().parse().ok()?));
            }
            (!steps.is_empty()).then_some(steps)
        });
        // Default ladder (finalizer matrix 2026-07-28): 3 drafts at n<=4,
        // 1 draft at n<=8, NO step beyond 8 — spec at n=16 measured as a
        // regression at every depth (see module docs). The cap (default 8)
        // makes n>8 MTP-off by construction.
        let mut steps = parsed.unwrap_or_else(|| vec![(4, 3), (8, 1)]);
        steps.sort_by_key(|&(n, _)| n);
        steps
    })
}

/// The per-step draft count for `n_active` concurrent sequences.
///
/// `num_drafts` is the configured ceiling (CLI `--num-drafts`); the return
/// value is always in `[1, num_drafts]` (or 0 when `num_drafts` is 0, i.e.
/// speculation off). Ladder disabled -> fixed `num_drafts` (pre-ladder
/// behavior). `n_active` beyond the last ladder step uses the last step's
/// draft count (the cap gates dispatch anyway).
pub fn mtp_ladder_drafts(n_active: usize, num_drafts: usize) -> usize {
    if num_drafts == 0 {
        return 0;
    }
    if mtp_ladder_disabled() {
        return num_drafts;
    }
    let steps = mtp_ladder_steps();
    steps
        .iter()
        .find(|&&(n_max, _)| n_active <= n_max)
        .or(steps.last())
        .map(|&(_, k)| k.clamp(1, num_drafts))
        .unwrap_or(num_drafts)
}

/// SSOT for the multi-sequence MTP cap (`ATLAS_MTP_MAX_SEQS`; default 8
/// with the K-vs-batch ladder, 4 under `ATLAS_NO_MTP_K_LADDER`).
/// Value-parsed, not presence-checked. Lives beside the ladder (moved from
/// `speculative.rs`, originally `scheduler/mod.rs`) because the two are one
/// policy: the model-side single-sequence MTP structures (catchup ring,
/// refeed labels, carry slot) gate on the same value the scheduler gates
/// dispatch on.
///
/// The cap IS the adaptive per-concurrency policy: the scheduler gates
/// dispatch on `active.len() <= mtp_max_seqs()`. Per-step K comes from
/// [`mtp_ladder_drafts`] (task #35): 3 drafts at n<=4 (the proven K=4
/// regime, bit-for-bit), 1 draft at n<=8 (finalizer matrix 2026-07-28:
/// C=8 82.4 vs 73.5 MTP-off, +12%). Cap 8 (NOT 16): spec at n=16 measured
/// as a 11-13% REGRESSION vs MTP-off at BOTH remaining depths (114.6-117.3
/// vs 131.5), so n>8 falls back to the plain multi-seq decode path.
/// `ATLAS_NO_MTP_K_LADDER` (presence) restores fixed K=4 + cap 4 — the
/// dafd990d adaptive policy. Set `ATLAS_MTP_MAX_SEQS=1` to restore
/// single-sequence-only.
pub fn mtp_max_seqs() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_MTP_MAX_SEQS")
            .ok()
            .and_then(|v| v.parse().ok())
            // Default 8, NOT 16 (finalizer matrix 2026-07-28, binary
            // 4b92a774): with the ladder, C=8 at 1 draft = 82.4 tok/s
            // (+12% over the 73.5 MTP-off floor), but C=16 speculation
            // LOSES at every depth (114.6 at 16:1 defaults, 117.3 at
            // 8:1,16:1 — vs 131.5 MTP-off): the K=1 verify step costs
            // ~1.9x a plain batch-16 decode step, above the ~1.72x
            // break-even at p1~0.72. Cap 8 makes n>8 MTP-off by
            // construction, preserving the 131.0 C=16 floor. Raising the
            // cap requires first cutting the n=16 verify-step cost (k<4
            // GDN table-form, wider batched propose). Pre-ladder baseline
            // (cap=4, binary 472ed410): C=1 25.55 (1.80x vLLM) · C=2
            // 35.35 (1.27x) · C=4 54.1 (1.01x) · C=8/16 MTP-off 73.5/131.0.
            .unwrap_or(if mtp_ladder_disabled() { 4 } else { 8 })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default-ladder shape (env-independent as long as the test process
    // does not set ATLAS_MTP_K_LADDER / ATLAS_NO_MTP_K_LADDER — CI does not).
    #[test]
    fn default_ladder_steps_down_with_n() {
        assert_eq!(mtp_ladder_drafts(1, 3), 3);
        assert_eq!(mtp_ladder_drafts(4, 3), 3);
        assert_eq!(mtp_ladder_drafts(5, 3), 1);
        assert_eq!(mtp_ladder_drafts(8, 3), 1);
        // Beyond the last step: last step's value (the cap — default 8 —
        // gates dispatch, so n>8 never speculates at defaults).
        assert_eq!(mtp_ladder_drafts(16, 3), 1);
        assert_eq!(mtp_ladder_drafts(32, 3), 1);
    }

    #[test]
    fn ladder_clamps_to_configured_ceiling() {
        // num_drafts=1 caps every step at 1; num_drafts=0 means spec off.
        assert_eq!(mtp_ladder_drafts(2, 1), 1);
        assert_eq!(mtp_ladder_drafts(2, 0), 0);
    }
}
