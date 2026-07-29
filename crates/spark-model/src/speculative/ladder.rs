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
//! n <= 8 -> 3 drafts (4 rows/seq, R = 32 = the exact row-buffer bound),
//! n <= 16 -> 1 draft (2 rows/seq, R = 32 again at n=16).
//!
//! ★ The depth step-down that used to sit at n>4 was an artifact of the
//! `mtp_step` chunk cap, NOT of GDN depth cost: `rows=4` was capped at 4
//! sequences, so an 8-wide batch ran TWO serialized 4-wide verify forwards
//! (2x the weight reads per step). Every "8:3 collapses" measurement
//! (57.9 on 2026-07-28, and 62.6 when re-measured this session) recorded
//! that chunking, not depth-3 at width 8. Raising the cap to the row-buffer
//! bound makes the true 8-wide K=4 step the BEST measured point at C=8.
//! ★ The n=16 rung took three rounds to earn its place, and its history is
//! the record of a COST curve, not of a depth curve. It measured a loss
//! (16:1 -> 128.4 vs a 131.9 MTP-off control) after the three eager-cost
//! fixes (`b93982d9` k-parameterized cross-seq GDN conv/WY, `a83627a2`
//! propose widened to n=16, `fa373bf4` batched Phase-A bootstrap), then
//! exact PARITY (131.93 vs 131.42) after the accept lift (`36d340a0`
//! per-sequence drafter prefill lifted p1 at n=16 to 0.797, making
//! break-even 1.797x against a measured ~1.79x). `296b9674`'s three
//! per-row verify cuts took the implied cost to ~1.55x, and the SAME 16:1
//! shape then measured **152.01 tok/s over two serves against a
//! same-session MTP-off control of 131.40 (+15.7%)**, p1 0.78-0.86,
//! tok_step ~1.83. Depth at n=16 is still dead: 16:2 -> 94.1 and 16:3 ->
//! 120.76. Clearing vLLM's 168.9 needs the cost under ~1.40x.
//!
//! At n<=8, measured C=8 on binary 9bef3b49 (this ladder + the raised
//! chunk cap), one fresh serve per config: 8:3 95.84 (range 94.9-96.6,
//! 8 reps) > 8:2 93.30 (92.5-94.0) — disjoint, reproduced on a second
//! serve at 95.68. Accept telemetry at n=8: p1 0.793, tok_step 2.606
//! (vs 0.780 / 2.301 at 8:2).
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
        // Default ladder (finalizer matrix 2026-07-29): 3 drafts up to the
        // n=8 rung, then ONE draft at n<=16. The 16:1 rung was dead weight
        // while the cap was 8; with the cap raised to 16 it is the shape
        // that measures 152.01 tok/s at C=16 against a 131.40 MTP-off
        // control (+15.7%). It is inert at n<=8 by construction (the n<=8
        // rung matches first). Depth at n=16 stays dead: 16:3 measures
        // 120.76 on the same binary, and D-Cut cannot rescue it (16 seqs x
        // its mandatory 2 rows = the whole 32-row budget, so it is forced
        // back to k=1 while the propose still pays for 3 drafts).
        //
        // The 8:2 step-down was an ARTIFACT of the `mtp_step` chunk cap,
        // not of depth: with `rows=4` capped at 4 sequences, an 8-wide
        // batch ran TWO serialized 4-wide verify forwards, which is what
        // the "8:3 collapses" numbers (57.9, and 62.6 measured this
        // session) recorded. With the cap raised to the row-buffer bound
        // (R = n*k <= 32, so 8 seqs x 4 rows fits exactly) a true 8-wide
        // K=4 verify MEASURES 95.84 tok/s at C=8 vs 93.30 for 8:2 on the
        // same binary — disjoint ranges (94.9-96.6 vs 92.5-94.0), two
        // independent serves. tok_step 2.606 vs 2.301 (+13.3%) for a
        // verify step ~11% more expensive.
        let mut steps = parsed.unwrap_or_else(|| vec![(4, 3), (8, 3), (16, 1)]);
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

/// SSOT for the multi-sequence MTP cap (`ATLAS_MTP_MAX_SEQS`; default 16
/// with the K-vs-batch ladder, 4 under `ATLAS_NO_MTP_K_LADDER`).
/// Value-parsed, not presence-checked. Lives beside the ladder (moved from
/// `speculative.rs`, originally `scheduler/mod.rs`) because the two are one
/// policy: the model-side single-sequence MTP structures (catchup ring,
/// refeed labels, carry slot) gate on the same value the scheduler gates
/// dispatch on.
///
/// The cap IS the adaptive per-concurrency policy: the scheduler gates
/// dispatch on `active.len() <= mtp_max_seqs()`. Per-step K comes from
/// [`mtp_ladder_drafts`] (task #35): `4:3,8:3,16:1` — 3 drafts through n=8
/// (matrix 2026-07-28: C=8 95.84 at 8:3 vs 93.30 at 8:2 on the same binary,
/// and 73.5 MTP-off), then 1 draft through n=16 (matrix 2026-07-29: C=16
/// 152.01 vs a 131.40 MTP-off control).
/// `ATLAS_NO_MTP_K_LADDER` (presence) restores fixed K=4 + cap 4 — the
/// dafd990d adaptive policy. Set `ATLAS_MTP_MAX_SEQS=1` to restore
/// single-sequence-only.
pub fn mtp_max_seqs() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_MTP_MAX_SEQS")
            .ok()
            .and_then(|v| v.parse().ok())
            // Default 16 (finalizer matrix 2026-07-29). It was 8 for three
            // rounds because spec at n=16 measured a LOSS (128.4 at 16:1)
            // and then a PARITY (131.93 vs a 131.42 MTP-off control): the
            // verify step cost ~1.79x a plain batch-16 decode step against
            // a break-even of 1.797x at p1 0.797. `296b9674`'s three
            // per-row verify cuts (wide LM-head arm, one-launch gated RMS
            // norm, fused BA+gates) bought ~0.20x of that cost — implied
            // cost is now ~1.55x — and the same 16:1 shape MEASURES 152.01
            // tok/s over two serves against a same-session MTP-off control
            // of 131.40 (+15.7%), with p1 0.78-0.86 and tok_step ~1.83.
            // ★ The raise is inert at n<=8: the cap only gates dispatch
            // above 8 and the `16:1` ladder rung only matches above 8, so
            // the C=1/2/4/8 code paths are unchanged. Set
            // `ATLAS_MTP_MAX_SEQS=8` to restore the round-3 cap.
            // Pre-ladder baseline (cap=4, binary 472ed410): C=1 25.55
            // (1.80x vLLM) · C=2 35.35 (1.27x) · C=4 54.1 (1.01x) ·
            // C=8/16 MTP-off 73.5/131.0.
            .unwrap_or(if mtp_ladder_disabled() { 4 } else { 16 })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default-ladder shape (env-independent as long as the test process
    // does not set ATLAS_MTP_K_LADDER / ATLAS_NO_MTP_K_LADDER — CI does not).
    #[test]
    fn default_ladder_holds_depth_to_the_cap() {
        assert_eq!(mtp_ladder_drafts(1, 3), 3);
        assert_eq!(mtp_ladder_drafts(4, 3), 3);
        assert_eq!(mtp_ladder_drafts(5, 3), 3);
        assert_eq!(mtp_ladder_drafts(8, 3), 3);
        // The 16:1 rung: depth at n=16 is dead (16:2 -> 94.1, 16:3 ->
        // 120.76 vs 152.01 at 16:1), so above the n=8 rung K drops to 2.
        assert_eq!(mtp_ladder_drafts(9, 3), 1);
        assert_eq!(mtp_ladder_drafts(16, 3), 1);
        // Beyond the last step: last step's value (the cap — default 16 —
        // gates dispatch, so n>16 never speculates at defaults).
        assert_eq!(mtp_ladder_drafts(32, 3), 1);
    }

    // A step-down ladder must still be honored when asked for explicitly
    // (the 8:2 shape stays reachable via ATLAS_MTP_K_LADDER).
    #[test]
    fn explicit_steps_are_honored() {
        let steps = [(4usize, 3usize), (8, 2)];
        let drafts = |n: usize| {
            steps
                .iter()
                .find(|&&(n_max, _)| n <= n_max)
                .or(steps.last())
                .map(|&(_, k)| k.clamp(1, 3))
                .unwrap()
        };
        assert_eq!(drafts(4), 3);
        assert_eq!(drafts(8), 2);
    }

    #[test]
    fn ladder_clamps_to_configured_ceiling() {
        // num_drafts=1 caps every step at 1; num_drafts=0 means spec off.
        assert_eq!(mtp_ladder_drafts(2, 1), 1);
        assert_eq!(mtp_ladder_drafts(2, 0), 0);
    }
}
