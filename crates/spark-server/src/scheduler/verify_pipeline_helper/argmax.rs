// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-time argmax with the sampler's first-index-wins tie-break.
//!
//! Split out of `verify_pipeline_helper.rs`, which is over the 500 LoC cap.

/// Argmax with the sampler's exact tie-break: strict `>` scanning forward, so
/// the FIRST index holding the maximum wins.
///
/// Replaces the obvious single loop
/// `for (i, &v) { if v > best_val { best_val = v; best_id = i } }`,
/// which carries a loop dependency through BOTH the running value and the
/// running index. That blocks vectorisation and leaves one long dependency
/// chain: measured 1.19 ms per verify step for 4 x 248k f32 (~840 MB/s, far
/// under memory bandwidth) and 70% of the whole host logits pipeline
/// (`AVAROK_MTP_TIMING`, K=4, 150 steps).
///
/// Two passes instead:
///   1. max only, over 8 independent accumulators — no index dependency, so
///      the chains are independent and the compiler is free to widen them;
///   2. the first index equal to that max, which typically exits early.
///
/// ## Why this is bit-identical, including the awkward cases
/// The original loop's final `best_val` IS the maximum, and because the test is
/// strictly `>`, `best_id` is the FIRST index attaining it. So "max, then first
/// index equal to max" is the same answer, provided the edge cases agree:
///
/// * **NaN** — `v > best` is false for NaN in both passes, so a NaN never
///   becomes the max in either form. If EVERY entry is NaN, the original leaves
///   `best_val = -inf` / `best_id = 0`; here pass 1 also leaves `-inf`, pass 2
///   finds no `v == -inf` (NaN compares unequal) and falls back to 0. Same.
/// * **-0.0 vs +0.0** — IEEE says they are equal, so neither `>` nor `==`
///   distinguishes them. Whichever zero appears FIRST is the one the original
///   latches, and pass 2's `==` matches at that same first index. Same.
///
/// Deliberately NOT `f32::max`: that returns the non-NaN operand, which would
/// let a NaN-adjacent value win where the original `>` ignored it.
pub(super) fn argmax_first_wins(logits: &[f32]) -> u32 {
    // Delegates to the runtime SSOT (`spark_runtime::sampler::argmax_first_wins_f32`)
    // so the verify path and the sampler share ONE first-index-wins argmax.
    // The equivalence tests below remain the harness proving it against the
    // exact loop this replaced.
    spark_runtime::sampler::argmax_first_wins_f32(logits)
}

/// A144b (2026-09-25): the verify slow path's FINAL pick — the pick decode's
/// own host pipeline would make at this position — must use decode's tie
/// rule, which is LAST-index-wins (`greedy_pick_last_wins` in
/// `spark_runtime::sampler::sample_impl`), NOT `argmax_first_wins` above.
///
/// Root cause of A144b (54/60 divergent TEB transcripts, temp=0): decode's
/// single-row host greedy path (`sample_with_params_seeded` →
/// `greedy_pick_last_wins`) and MTP-K3 verify's slow path both process the
/// SAME penalty/bias/grammar-masked logits to the SAME FP32 precision (both
/// dequant BF16→F32 via `bf16_to_f32`; no extra rounding either side — see
/// `decode_logits_seq::process_seq_logits` vs `verify_pick_with_pipeline`
/// above), so an exact tie is common on quantised checkpoints. Verify was
/// picking the FIRST tied index while decode picks the LAST — a real
/// tie-break divergence, not a precision or masking bug. This wrapper is the
/// fix: verify's argmax call site now delegates to the exact function decode
/// uses, instead of duplicating a different tie-break.
pub(super) fn greedy_pick_last_wins(logits: &[f32]) -> u32 {
    spark_runtime::sampler::greedy_pick_last_wins(logits)
}

#[cfg(test)]
mod argmax_tests {
    use super::argmax_first_wins;

    /// Reference: the exact loop this replaced.
    fn reference(logits: &[f32]) -> u32 {
        let mut best_id: u32 = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as u32;
            }
        }
        best_id
    }

    fn agree(v: &[f32]) {
        assert_eq!(
            argmax_first_wins(v),
            reference(v),
            "diverged from the original loop on {v:?}"
        );
    }

    #[test]
    fn matches_reference_on_edge_cases() {
        agree(&[]);
        agree(&[1.0]);
        agree(&[1.0, 2.0, 3.0]);
        agree(&[3.0, 2.0, 1.0]);
        // Ties MUST resolve to the FIRST index.
        agree(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        // Tie straddling the 8-lane chunk boundary and the remainder tail.
        agree(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0]);
        agree(&[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]);
        // All-negative (the max is NOT -inf-adjacent by accident).
        agree(&[-5.0, -1.0, -3.0]);
        // Signed zeros both orders.
        agree(&[-0.0, 0.0]);
        agree(&[0.0, -0.0]);
        agree(&[-1.0, -0.0, 0.0, -1.0]);
        // NaN must never win, in any position.
        agree(&[f32::NAN, 1.0, 2.0]);
        agree(&[1.0, f32::NAN, 2.0]);
        agree(&[1.0, 2.0, f32::NAN]);
        agree(&[f32::NAN, f32::NAN]);
        // Infinities.
        agree(&[f32::NEG_INFINITY, -1.0]);
        agree(&[f32::INFINITY, 1.0]);
        agree(&[1.0, f32::INFINITY, f32::INFINITY]);
        agree(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn matches_reference_on_vocab_sized_input() {
        // Deterministic pseudo-random, vocab-scale, with the max deliberately
        // placed past several chunk boundaries and duplicated.
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[123_457] = 999.0;
        v[200_003] = 999.0; // duplicate max later — first must win
        assert_eq!(argmax_first_wins(&v), reference(&v));
        assert_eq!(argmax_first_wins(&v), 123_457);
    }
}

#[cfg(test)]
mod greedy_pick_last_wins_tests {
    use super::{argmax_first_wins, greedy_pick_last_wins};

    /// The tie rule decode's host path (`sample_impl::greedy_pick_last_wins`)
    /// has always used: `max_by(partial_cmp.unwrap_or(Equal))`, last-wins.
    fn decode_reference(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// A144b regression test: on an exact tie, verify's final-pick function
    /// must agree with decode's host greedy pick — the whole point of this
    /// wrapper existing.
    #[test]
    fn exact_tie_matches_decodes_host_pick() {
        let v = [1.0f32, 5.0, 5.0, 5.0, 2.0];
        assert_eq!(greedy_pick_last_wins(&v), decode_reference(&v));
        assert_eq!(greedy_pick_last_wins(&v), 3, "LAST of the tied indices (1,2,3) must win");
    }

    /// The bug this fixes: `argmax_first_wins` and `greedy_pick_last_wins`
    /// MUST disagree on a real tie (otherwise the whole A144b fix would be a
    /// no-op). If this ever starts failing because the two functions agree,
    /// that is not progress — it means one of them silently changed its
    /// tie-break and the divergence this test guards is gone for the wrong
    /// reason.
    #[test]
    fn diverges_from_first_wins_on_a_real_tie() {
        let v = [1.0f32, 5.0, 5.0, 5.0, 2.0];
        assert_eq!(argmax_first_wins(&v), 1, "first tied index");
        assert_eq!(greedy_pick_last_wins(&v), 3, "last tied index");
        assert_ne!(argmax_first_wins(&v), greedy_pick_last_wins(&v));
    }

    #[test]
    fn vocab_sized_last_wins_matches_decode() {
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[100_000] = 999.0;
        v[200_000] = 999.0; // duplicate max — LAST must win, matching decode
        assert_eq!(greedy_pick_last_wins(&v), decode_reference(&v));
        assert_eq!(greedy_pick_last_wins(&v), 200_000);
    }
}
