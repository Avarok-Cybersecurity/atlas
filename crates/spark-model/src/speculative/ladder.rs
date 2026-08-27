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
//! n <= 8 -> 3 drafts (4 rows/seq, R = 32),
//! n <= 16 -> 1 draft (2 rows/seq, R = 32 at n=16 — single chunk; wave 19,
//!            taken back from 16:2 when the measured p1 fell below the
//!            rung's break-even — see the default-ladder comment),
//! n <= 32 -> 1 draft (2 rows/seq, R = 64 at n=32).
//!
//! ★ This module is the STATIC ladder and the floor. Since wave 28 the n=16
//! rung is chosen at RUNTIME from the observed accept statistics by
//! `spark_server::scheduler::adaptive_rung` — a static value cannot be right
//! for both traffic regimes, because the second-token conditional accept is
//! bimodal (~0.54 prose / 0.877 tool-shaped) and moves the break-even across
//! the rung. `ATLAS_MTP_STATIC_RUNG` (PRESENCE) pins the static value here;
//! so does an explicit `ATLAS_MTP_K_LADDER`.
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
//! tok_step ~1.83.
//! ★ The historical "depth at n=16 is dead" numbers (16:2 -> 94.1, 16:3 ->
//! 120.76) recorded the CHUNK CAP, not depth — the same artifact class as
//! the 8:3 story below (rows=3/4 chunks were hardcoded to 8 seqs, so a
//! 16-wide depth batch ran TWO serialized 8-wide verifies). With the cap
//! derived from the row budget (fixer r2 2026-07-30), TRUE single-chunk
//! 16:2 measures **194-196 tok/s at C=16 vs a 184-185 same-session 16:1
//! control (+6%)** with D-Cut OFF (tok_step 2.44-2.50, p1 0.81-0.85) — but
//! **176-179 (-4%) with D-Cut pruning AT DEPTH** (ragged pruning at nd=2
//! fragments the GDN runs and sheds winning drafts). The wave-11
//! implementer grid re-measured the full depth set at n=16 on one binary:
//! 16:2 = 195.0/187.6 > 16:3 = 190.9/187.9 > 16:1 = 184.5/184.1 (K=4's
//! extra 16 rows cost more than the +0.46 tok/step they buy; p3_cond
//! unstable 0.51-0.75). **16:2 is therefore the DEFAULT rung**, paired with
//! the D-Cut-at-depth policy that makes it safe: `mtp_dcut::dcut_width_cap`
//! holds pruning to batches of <= 8 sequences, so the n=16 verify runs the
//! exact uniform single-chunk [3; 16] shape that measured the win, while
//! the C=8 D-Cut win (+2.6%) is untouched.
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
fn parse_ladder(value: &str) -> Option<Vec<(usize, usize)>> {
    let mut steps = Vec::new();
    for part in value.split(',') {
        let (n, k) = part.trim().split_once(':')?;
        steps.push((n.trim().parse().ok()?, k.trim().parse().ok()?));
    }
    if steps.is_empty() {
        return None;
    }
    steps.sort_by_key(|&(n, _)| n);
    Some(steps)
}

fn mtp_ladder_steps() -> &'static [(usize, usize)] {
    static STEPS: std::sync::OnceLock<Vec<(usize, usize)>> = std::sync::OnceLock::new();
    STEPS.get_or_init(|| {
        let parsed = std::env::var("ATLAS_MTP_K_LADDER")
            .ok()
            .and_then(|value| parse_ladder(&value));
        // Default ladder: 3 drafts up to the n=8 rung, then TWO drafts at
        // n<=16 (the 16:2 rung), then one at n<=32.
        //
        // 16:2 (2026-07-30, wave-11 implementer grid + fixer r2, one binary,
        // one serve boot per ladder): the depth grid at n=16, truly measured
        // for the first time after the chunk-cap fix (88539cf7), is ordered
        // 16:2 (195.0/187.6) > 16:3 (190.9/187.9) > 16:1 (184.5/184.1) at
        // C=16 — +5.7% best-vs-best for 16:2, reproducing fixer r2's
        // 194.4-196.0 band; tok_step 2.483 at p1 0.859, single-chunk
        // `n=16 k_drafts=2` telemetry. The historical "depth at n=16 is
        // dead" numbers (16:2 -> 94.1, 16:3 -> 120.76) were the 8-seq chunk
        // cap, not depth (module doc ★). The rung REQUIRES the
        // D-Cut-at-depth policy (`mtp_dcut::dcut_width_cap`, prune only at
        // <= 8 sequences): with pruning active at n=16 the same shape LOSES
        // -4% (176.6-179.4). It is inert at n<=8 by construction (the n<=8
        // rung matches first).
        //
        // 24:2 / 32:2 (wave 11): the 96-row verify envelope makes depth at
        // n<=32 a SINGLE chunk (24 x 3 = 72 rows, 32 x 3 = 96 rows), so
        // `ATLAS_MTP_K_LADDER="4:3,8:3,16:2,24:2,32:2"` is now a measurable
        // shape. NOT default: 32:2 is a PROJECTION so far (~277 tok/s at
        // C=32 from the measured n=16 K=3 verify cost — +50% rows for
        // +26.5% step time — vs the 269.3 bar); the default flips only on a
        // measured win, like every rung before it.
        //
        // 32:1 (2026-07-30, spec at n=32): the wave-9 native-bs32 profile
        // shows the n=32 plain step is 97.6% GPU-busy at 160 ms with FFN /
        // projections FLAT in rows and the GDN state term per-STEP, so K=2's
        // ~1.8 tok/step (p1~0.8) amortizes the whole step — the one level
        // with NO MTP multiplier while every C<=16 level enjoys one. R =
        // 32 x 2 = 64 rows = the widened VERIFY_ROW_CAP/meta/logits/stash
        // envelope (verify_e). Explicit rung (not last-step fallthrough) so
        // the shape is visible in `ATLAS_MTP_K_LADDER` terms; dispatch above
        // 16 additionally needs the `mtp_max_seqs` default raised to 32
        // (below).
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
        //
        // 16:1 (wave 19, 2026-07-31, dgx1, tip 19b365c2, ONE binary,
        // env-toggled arms, one fresh serve per arm, 3 scored reps each, every
        // rep token-matched at 16384 completion tokens): the full depth grid
        // at n=16 is MONOTONE DECREASING in depth, so the 16:2 rung lost the
        // lead it held since wave 11 —
        //   16:1  181.63  (182.50 / 181.27 / 181.13)  tok_step 1.715
        //   16:2  172.70  (173.23 / 171.50 / 173.36)  tok_step 2.13
        //   16:3  152.97  (152.50 / 152.76 / 153.65)  tok_step 2.27
        // 16:1 beats 16:2 by +5.2% with DISJOINT ranges, and is the first
        // C=16 number in the campaign to CLEAR the vLLM bar (178.72, same
        // box, wave 17). 16:4 is not a shape: `can_batch_verify` admits only
        // rows in 2..=4, so 3 drafts is the depth ceiling at any width.
        //
        // ★ Nothing regressed — the rung is a function of accept rate, and
        // the accept rate MOVED. The wave-11 grid picked 16:2 at p1 0.859 /
        // tok_step 2.483; the same shape on this tip and this workload
        // measures p1 0.70-0.79 / tok_step 2.13. The rung's break-even is
        // exactly the tok_step ratio against the step-cost ratio.
        //
        // ★★ CORRECTION (wave 28, 2026-08-01). Wave 19 fitted
        // `step = F + c*R` over the depth grid and derived F = 65.8 ms /
        // c = 2.70 ms per row-sequence, i.e. a 16:2-over-16:1 COST RATIO of
        // **1.306**. That number is WRONG. Wave 27 measured the step cost
        // directly from `mtp_accept_debug` tok_step and decode-only
        // throughput on one binary across four workloads and reads
        // **1.17-1.26** — depth is CHEAPER than the fit claimed, drifting up
        // slowly with sequence length, which is why this rung sits far
        // closer to break-even than the fit implied. Do not re-use the
        // 65.8/2.70 model to justify a rung; re-measure.
        //
        // ★★ SUPERSEDED AS A STATIC RUNG (wave 28). The break-even is a
        // function of the SECOND-token conditional accept `p2_cond`, and
        // `p2_cond` is BIMODAL BY TRAFFIC: ~0.54 on prose vs 0.877 on
        // tool-shaped function-call text, measured on the SAME two serve
        // boots. Token ratio 1.19-1.23 (prose, below cost -> 16:1 wins by
        // 1.2-2.4%) vs 1.424 (tool-shaped, far above cost -> 16:2 wins by
        // 7.9%). Output length is NOT the regime variable — the length
        // sweep is flat. So NO static value of this rung is right for both
        // regimes, and the shipped decision is made at RUNTIME from the
        // observed accept statistics by
        // `spark_server::scheduler::adaptive_rung`, which reads this rung as
        // its floor and may lift n in 9..=16 to 2 drafts. This entry stays
        // the static default (and the value under
        // `ATLAS_MTP_STATIC_RUNG`). Restore the old static rung with
        // `ATLAS_MTP_K_LADDER="4:3,8:3,16:2,32:1"` — an explicit ladder also
        // pins adaptation off — and re-run the grid after any accept lift.
        //
        // ★ The per-row term DOMINATES at n=16 — 129.5 of the 197.3 ms at
        // 16:2, and still 86.3 of 151.1 ms at 16:1 — so the wave-15 step
        // model's "fixed 82.7 ms is about half the n=16 step" is WRONG here:
        // the measured fixed cost is 65.8 ms, only 33% of the 16:2 step.
        // Cutting c is the standing lever, and it is what would let a deeper
        // rung pay again.
        parsed.unwrap_or_else(|| vec![(4, 3), (8, 3), (16, 1), (32, 1)])
    })
}

fn ladder_drafts_from_steps(steps: &[(usize, usize)], n_active: usize, num_drafts: usize) -> usize {
    if num_drafts == 0 {
        return 0;
    }
    steps
        .iter()
        .find(|&&(n_max, _)| n_active <= n_max)
        .or(steps.last())
        .map(|&(_, k)| k.clamp(1, num_drafts))
        .unwrap_or(num_drafts)
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
    ladder_drafts_from_steps(mtp_ladder_steps(), n_active, num_drafts)
}

/// SSOT for the multi-sequence MTP cap (`ATLAS_MTP_MAX_SEQS`; default 32
/// with the K-vs-batch ladder, 4 under `ATLAS_NO_MTP_K_LADDER`).
/// Value-parsed, not presence-checked. Lives beside the ladder (moved from
/// `speculative.rs`, originally `scheduler/mod.rs`) because the two are one
/// policy: the model-side single-sequence MTP structures (catchup ring,
/// refeed labels, carry slot) gate on the same value the scheduler gates
/// dispatch on.
///
/// The cap IS the adaptive per-concurrency policy: the scheduler gates
/// dispatch on `active.len() <= mtp_max_seqs()`. Per-step K comes from
/// [`mtp_ladder_drafts`] (task #35): `4:3,8:3,16:1,32:1` — 3 drafts through
/// n=8 (matrix 2026-07-28: C=8 95.84 at 8:3 vs 93.30 at 8:2 on the same
/// binary, and 73.5 MTP-off), then 1 draft through n=16 (wave 19
/// 2026-07-31: C=16 181.9 at 16:1 vs 172.70 at 16:2 on one binary, disjoint
/// ranges — the wave-11 grid's 16:2 lead does not survive the drop in p1
/// from 0.859 to ~0.72), then 1 draft through n=32 (2026-07-30, the
/// native-bs32 rung — R = 64 verify rows).
/// `ATLAS_NO_MTP_K_LADDER` (presence) restores fixed K=4 + cap 4 — the
/// dafd990d adaptive policy. Set `ATLAS_MTP_MAX_SEQS=1` to restore
/// single-sequence-only.
pub fn mtp_max_seqs() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_MTP_MAX_SEQS")
            .ok()
            .and_then(|v| v.parse().ok())
            // Default 32 (2026-07-30, spec at n=32 — the 32:1 ladder rung).
            // The raise is inert at n<=16: the cap only gates dispatch
            // above 16 and the `32:1` rung only matches above 16, so every
            // measured C<=16 code path is unchanged. Set
            // `ATLAS_MTP_MAX_SEQS=16` to restore the wave-9 cap (spec off
            // above n=16).
            // History: default 16 (finalizer matrix 2026-07-29) — it was 8
            // for three rounds because spec at n=16 measured a LOSS (128.4
            // at 16:1) and then a PARITY (131.93 vs a 131.42 MTP-off
            // control): the verify step cost ~1.79x a plain batch-16 decode
            // step against a break-even of 1.797x at p1 0.797. `296b9674`'s
            // three per-row verify cuts (wide LM-head arm, one-launch gated
            // RMS norm, fused BA+gates) bought ~0.20x of that cost — and
            // the same 16:1 shape MEASURES 152.01 tok/s over two serves
            // against a same-session MTP-off control of 131.40 (+15.7%),
            // with p1 0.78-0.86 and tok_step ~1.83.
            // Pre-ladder baseline (cap=4, binary 472ed410): C=1 25.55
            // (1.80x vLLM) · C=2 35.35 (1.27x) · C=4 54.1 (1.01x) ·
            // C=8/16 MTP-off 73.5/131.0.
            .unwrap_or(if mtp_ladder_disabled() { 4 } else { 32 })
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
        // The 16:1 rung (wave 19): ONE draft (K=2, 2 rows/seq) through n=16.
        // 16:2 held this rung from wave 11 until its accept rate fell:
        // re-measured on one binary with env-toggled arms, 16:1 reads
        // 181.27-182.50 against 16:2's 171.50-173.36 (+5.4%, disjoint), and
        // clears the 178.72 vLLM bar. Restore 16:2 with
        // ATLAS_MTP_K_LADDER="4:3,8:3,16:2,32:1".
        assert_eq!(mtp_ladder_drafts(9, 3), 1);
        assert_eq!(mtp_ladder_drafts(16, 3), 1);
        // The 32:1 rung (2026-07-30): ONE draft (K=2) through n=32 — the
        // native-bs32 regime, R = 64 verify rows.
        assert_eq!(mtp_ladder_drafts(17, 3), 1);
        assert_eq!(mtp_ladder_drafts(32, 3), 1);
        // Beyond the last step: last step's value (the cap — default 32 —
        // gates dispatch, so n>32 never speculates at defaults).
        assert_eq!(mtp_ladder_drafts(64, 3), 1);
        // --num-drafts stays the ceiling: 16:2 clamps to 1 at num_drafts=1.
        assert_eq!(mtp_ladder_drafts(16, 1), 1);
    }

    #[test]
    fn depth_at_width_env_rungs_parse_shape() {
        let steps = parse_ladder("32:2, 4:3,8:3, 16:2,24:2").unwrap();
        assert_eq!(steps, [(4, 3), (8, 3), (16, 2), (24, 2), (32, 2)]);
        // 24:2 = 24 x 3 = 72 rows; 32:2 = 32 x 3 = 96 rows — both a single
        // chunk under VERIFY_ROW_BUDGET = 96 (mtp_dcut::chunk_ranges).
        assert_eq!(ladder_drafts_from_steps(&steps, 17, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 24, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 25, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 32, 3), 2);
    }

    // A step-down ladder must still be honored when asked for explicitly
    // (the 8:2 shape stays reachable via ATLAS_MTP_K_LADDER).
    #[test]
    fn explicit_steps_are_honored() {
        let steps = parse_ladder("4:3,8:2").unwrap();
        assert_eq!(ladder_drafts_from_steps(&steps, 4, 3), 3);
        assert_eq!(ladder_drafts_from_steps(&steps, 5, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 8, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 9, 3), 2);
    }

    #[test]
    fn malformed_ladder_is_rejected_as_a_unit() {
        assert_eq!(parse_ladder(""), None);
        assert_eq!(parse_ladder("4:3,broken,8:2"), None);
        assert_eq!(parse_ladder("4:three"), None);
    }

    #[test]
    fn ladder_clamps_to_configured_ceiling() {
        // num_drafts=1 caps every step at 1; num_drafts=0 means spec off.
        assert_eq!(mtp_ladder_drafts(2, 1), 1);
        assert_eq!(mtp_ladder_drafts(2, 0), 0);
    }
}
