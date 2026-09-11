// SPDX-License-Identifier: AGPL-3.0-only

//! The split policy, graded as a pure function — no GPU, no environment, no
//! baked constant. Every number here is sourced: 132 SMs is
//! `kernels/hopper/HARDWARE.toml`, 48 is `kernels/gb10`, 24 q heads and
//! `--max-batch-size 16` are `kernels/hopper/qwen3.8-27b/MODEL.toml` and the
//! campaign's serve recipe.

use super::*;

/// The Qwen3.8-27B full-attention head count.
const NQ: u32 = 24;
/// H100 SXM5.
const HOPPER_SMS: u32 = 132;
/// DGX Spark GB10.
const GB10_SMS: u32 = 48;
/// `--max-batch-size 16`.
const PIN: u32 = 16;

/// ★ THE DEFECT, pinned. The shipped rule reaches `num_splits = 1` at C=1 on
/// an H100 — 24 CTAs on 132 SMs — and it does so for TWO independent reasons,
/// so a fix that addresses only one is not a fix.
///
/// Oracle: nsys, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8, round 13 cell T1N —
/// `paged_decode_attn_fp8` `grid=(24,1,1)`, 231.51 us/launch, 9.93 MB,
/// 42.9 GB/s = 1.28% of HBM (`ATTN-DECODE-SPLITK-ATTRIBUTION.md` §C.5).
#[test]
fn the_legacy_rule_leaves_an_h100_at_twenty_four_ctas() {
    // Reason one: the GB10 SM count was compiled in, so the rule thought the
    // device was full at 48 CTAs.
    assert_eq!(legacy_splits(GB10_SMS, NQ, PIN), 1);
    // Reason two: even with the RIGHT SM count it still says 1, because the
    // reference batch it multiplies by is the PINNED max batch (16), not the
    // one sequence actually in flight.
    assert_eq!(legacy_splits(HOPPER_SMS, NQ, PIN), 1);
    // …and 24 CTAs is what that produces.
    assert!(NQ * legacy_splits(HOPPER_SMS, NQ, PIN) < HOPPER_SMS);
}

/// The policy formula, stated as an assertion rather than only in prose:
/// `clamp(ceil(2 * sm_count / num_q_heads), 1, 16)`.
#[test]
fn auto_fills_two_waves_at_the_single_stream_shape() {
    // Hopper: ceil(264 / 24) = 11 -> 264 CTAs, exactly two waves of 132.
    assert_eq!(auto_splits(HOPPER_SMS, NQ), 11);
    assert!(NQ * auto_splits(HOPPER_SMS, NQ) >= SPLITK_TARGET_WAVES * HOPPER_SMS);
    // GB10: ceil(96 / 24) = 4. Not what GB10 serves (it declares `legacy`),
    // but the formula must be one formula.
    assert_eq!(auto_splits(GB10_SMS, NQ), 4);
    // B200's declared 148 -> ceil(296/24) = 13, still under the cap.
    assert_eq!(auto_splits(148, NQ), 13);
    // A wide-head model already fills the device with one split.
    assert_eq!(auto_splits(HOPPER_SMS, 512), 1);
    // A one-head decode head would ask for 264 and is clamped.
    assert_eq!(auto_splits(HOPPER_SMS, 1), MAX_DECODE_SPLITS);
}

/// ★ THE INVARIANT THE WHOLE MODULE EXISTS FOR. The split count — and so the
/// non-associative online-softmax reduction tree — must not move with the
/// runtime co-batched count, or one sequence decodes to different bytes alone
/// than beside fifteen others.
///
/// Oracle: `tasks/determinism_investigation.md` and
/// `qwen3_attention::split_ref_seqs`, which pinned the OLD rule to the
/// configured max batch for exactly this reason. `auto` does not need pinning
/// because it never reads a batch at all.
#[test]
fn the_auto_split_count_does_not_move_with_the_co_batched_count() {
    let at = |n: u32| num_splits(SplitkPolicy::Auto, HOPPER_SMS, NQ, n);
    for n in [1u32, 4, 15, 16] {
        assert_eq!(
            at(n),
            at(1),
            "num_splits moved at num_seqs={n}: the reduction tree is not fixed \
             per serve, which is the determinism defect this policy must not \
             re-create"
        );
    }
    assert_eq!(at(1), 11);
    // A pinned count is equally invariant.
    let pinned = |n: u32| num_splits(SplitkPolicy::Pinned(6), HOPPER_SMS, NQ, n);
    for n in [1u32, 4, 15, 16] {
        assert_eq!(pinned(n), 6, "a pinned split count is a constant");
    }
}

/// …and `legacy` is DELIBERATELY not invariant: it is the old rule preserved
/// verbatim, whose co-batch dependence `split_ref_seqs` neutralises by feeding
/// it `max(num_seqs, max_decode_seqs)` instead of `num_seqs`. Asserted so the
/// difference between the two arms reads as a decision.
#[test]
fn legacy_is_the_old_rule_including_its_reference_batch() {
    // 8 heads, 48 SMs, one reference sequence -> 6 splits; the pin is what
    // stops that from becoming 1 when five more sequences arrive.
    assert_eq!(legacy_splits(GB10_SMS, 8, 1), 6);
    assert_eq!(legacy_splits(GB10_SMS, 8, 6), 1);
    // Which is precisely why `split_ref_seqs` hands it the PINNED batch.
    assert_eq!(
        num_splits(SplitkPolicy::Legacy, GB10_SMS, 8, 6),
        legacy_splits(GB10_SMS, 8, 6)
    );
}

/// The grammar, including the two spellings that must not become `auto` by
/// accident.
#[test]
fn the_grammar_parses_every_documented_spelling() {
    assert_eq!(parse("legacy"), Some(SplitkPolicy::Legacy));
    assert_eq!(parse(" AUTO "), Some(SplitkPolicy::Auto));
    assert_eq!(parse("4"), Some(SplitkPolicy::Pinned(4)));
    // `0`/`off` is "no split-K", which is one split — the geometry an H100
    // served before this lever existed.
    assert_eq!(parse("0"), Some(SplitkPolicy::Pinned(1)));
    assert_eq!(parse("off"), Some(SplitkPolicy::Pinned(1)));
    // Clamped, not rejected: an over-large pin is an operator asking for more
    // than the workspace holds, and the workspace is sized to the same cap.
    assert_eq!(parse("999"), Some(SplitkPolicy::Pinned(MAX_DECODE_SPLITS)));
    // A typo keeps the target's declaration rather than arming a geometry
    // change on a card with no receipt for it.
    assert_eq!(parse("aut0"), None);
    assert_eq!(parse(""), None);
    for p in [
        SplitkPolicy::Legacy,
        SplitkPolicy::Auto,
        SplitkPolicy::Pinned(6),
    ] {
        assert_eq!(parse(&p.label()), Some(p), "label must round-trip");
    }
}

/// Baked default first, environment second — the same rung order every other
/// lever in `TargetDefaults` follows.
#[test]
fn the_environment_overrides_the_declaration_and_says_so() {
    assert_eq!(resolve_policy("auto", None), (SplitkPolicy::Auto, false));
    assert_eq!(
        resolve_policy("auto", Some("0")),
        (SplitkPolicy::Pinned(1), true),
        "the A/B control an H100 round runs against the new default"
    );
    assert_eq!(
        resolve_policy("legacy", Some("auto")),
        (SplitkPolicy::Auto, true)
    );
    // An unparseable override is not an override.
    assert_eq!(
        resolve_policy("auto", Some("yes-please")),
        (SplitkPolicy::Auto, false)
    );
    // An unparseable DECLARATION falls to the conservative arm, never `auto`.
    assert_eq!(resolve_policy("", None), (SplitkPolicy::Legacy, false));
}

/// ★ The silent-corruption guard. The kernel addresses
/// `((seq * heads) + head) * num_splits + split`, so an arena sized for fewer
/// slots than the grid can reach is an out-of-bounds DEVICE WRITE with no
/// error. Every policy's worst case must fit what `sizes.rs` allocates.
#[test]
fn the_workspace_covers_every_slot_the_grid_can_address() {
    // `DecodeMetaLayout::rows()` is max(32, max_batch_size) — the widest batch
    // the metadata upload accepts, which is the real bound on `num_seqs`.
    let rows = 32u32;
    for policy in [
        SplitkPolicy::Legacy,
        SplitkPolicy::Auto,
        SplitkPolicy::Pinned(MAX_DECODE_SPLITS),
    ] {
        let slots = workspace_slots(policy, HOPPER_SMS, NQ, rows, PIN);
        for num_seqs in [1u32, 4, 16, rows] {
            let ref_seqs = num_seqs.max(PIN);
            let splits = num_splits(policy, HOPPER_SMS, NQ, ref_seqs);
            // One split is NO split-K: the dispatch takes the non-split kernel,
            // which writes the BF16 output directly and never touches the
            // workspace. Only a split launch can index it.
            if splits == 1 {
                continue;
            }
            let used = num_seqs * NQ * splits;
            assert!(
                used <= slots,
                "{policy:?} at num_seqs={num_seqs}: grid addresses {used} slots, \
                 arena holds {slots}"
            );
        }
    }
    // Legacy keeps the pre-#928 arena exactly: `NUM_SMS` slots.
    assert_eq!(
        workspace_slots(SplitkPolicy::Legacy, GB10_SMS, NQ, rows, PIN),
        GB10_SMS
    );
    // Hopper's `auto` arena: 32 rows x 24 heads x 11 splits.
    assert_eq!(
        workspace_slots(SplitkPolicy::Auto, HOPPER_SMS, NQ, rows, PIN),
        32 * 24 * 11
    );
}
