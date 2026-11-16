// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-integer pins for the preflight ring term and its #915 auto-fit. No
//! GPU, no model, no checkpoint — which is the point: this decision is made
//! before the weights exist.
//!
//! The numbers are the 2026-09-05 rental-H100 boot from issue #915
//! (Qwen/Qwen3.8-27B-FP8, one 80 GB H100, hopper recipe, evidence cell
//! `qwen38.avarok.a.lat.c1`): 48 GDN layers, 151.5 MiB of SSM state per
//! sequence, `--max-batch-size 32`, inference reserve 45,823 MiB of which
//! 37.88 GiB was ring, 57.2 GiB of weights inside a 71.3 GiB budget.
use super::*;
use clap::Parser as _;

/// These cases pin the PRE-LOAD arm — the behaviour every route without a
/// predictable post-load residency still takes. The post-load arm (#915
/// second pass) has its own file, `headroom_tests.rs`.
const PRE_LOAD: Yardstick = Yardstick::PreLoadFree("no residency prediction in this test");

/// 48 GDN layers x (h 48*128*128*4 + conv (16*128*2 + 48*128)*4*4) =
/// 158,859,264 B = exactly 151.5 MiB.
const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));
/// The 45,823 MiB inference reserve minus its 38,784 MiB 8-slot ring.
const RESERVE_WITHOUT_RING: usize = 7_039 * 1024 * 1024;

fn args() -> cli::ServeArgs {
    cli::ServeArgs::parse_from(["spark", "Qwen/Qwen3.8-27B-FP8", "--max-batch-size", "32"])
}

/// The formula issue #915's third bullet asks preflight to print: an operator
/// staring at 45.8 GiB could not see it was 8 snapshots x batch 32.
#[test]
fn the_formula_spells_out_snapshots_times_batch_times_per_seq_state() {
    assert_eq!(
        formula(8, 32, PER_SEQ_BLOB),
        "ring: 8 slots x 32 seqs x 151.5 MB/seq = 37.88 GB"
    );
    // The same arithmetic at the depth the fit chose — one function, so the
    // INFO line, the shrink WARN and the refusal cannot quote three answers.
    assert_eq!(
        formula(1, 32, PER_SEQ_BLOB),
        "ring: 1 slots x 32 seqs x 151.5 MB/seq = 4.73 GB"
    );
    assert_eq!(
        formula(0, 32, PER_SEQ_BLOB),
        "ring: 0 slots x 32 seqs x 151.5 MB/seq = 0.00 GB"
    );
}

/// The #915 boot itself: 14.1 GiB free (71.3 GiB budget less 57.2 GiB of
/// weights) against a reserve whose ring alone wants 37.88 GiB. It must
/// shrink to depth 1 and boot, not refuse.
#[test]
fn the_915_boot_shrinks_to_the_largest_fitting_depth_instead_of_refusing() {
    let args = args();
    let slot = slot_bytes(&args, PER_SEQ_BLOB);
    assert_eq!(slot, 32 * PER_SEQ_BLOB);
    let free = (14.1 * 1024.0 * 1024.0 * 1024.0) as usize;

    let fit = fit_ring(
        &args,
        8,
        slot,
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        free,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 1, "8 -> 4 -> 2 -> 1 is the first rung that fits");
    assert!(
        RESERVE_WITHOUT_RING + fit.slots * slot <= free,
        "the depth it kept must actually fit"
    );

    let warning = fit.warning.expect("a shrink must be logged, never silent");
    assert!(
        warning.contains("ring: 1 slots x 32 seqs x 151.5 MB/seq = 4.73 GB"),
        "{warning}"
    );
    assert!(warning.contains("(was 37.88 GB)"), "{warning}");
    assert!(warning.contains("reserve 11.61 of 14.10 GB"), "{warning}");
    assert!(
        warning.contains("Sized from pre-load free memory"),
        "the WARN must name the yardstick it used: {warning}"
    );
    assert!(warning.contains("#915"), "{warning}");
    assert!(
        warning.contains("--ssm-decode-ring-slots"),
        "the warning must name the flag that pins a depth: {warning}"
    );
}

/// A reserve that already fits is left exactly as the flags asked for, and
/// says nothing — the WARN is reserved for the case where the serve is
/// running at a depth its recipe does not record.
#[test]
fn a_reserve_that_fits_is_not_touched() {
    let args = args();
    let slot = slot_bytes(&args, PER_SEQ_BLOB);
    let free = 64 * 1024 * 1024 * 1024;
    let fit = fit_ring(
        &args,
        8,
        slot,
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        free,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
    // Silent is not the same as absent: the decision is logged either way.
    assert!(
        fit.decision.contains("pre-load free memory"),
        "{}",
        fit.decision
    );
}

/// Nothing to shrink is not a shrink: a ringless serve (`--speculative`,
/// watchdogs off, `--ssm-decode-ring-slots 0`, or a pure-attention model)
/// goes straight to the refusal with no warning of its own.
#[test]
fn a_ringless_serve_is_left_to_the_refusal() {
    let args = args();
    let free = RESERVE_WITHOUT_RING / 2;
    let fit = fit_ring(
        &args,
        0,
        slot_bytes(&args, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        free,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 0);
    assert!(fit.warning.is_none());
    // Same when the model has no SSM state at all: a zero-byte depth unit.
    let fit = fit_ring(&args, 8, 0, 0, RESERVE_WITHOUT_RING, free, &PRE_LOAD, false);
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
}

/// When even depth 0 is over budget the ring is not what is wrong. Shrinking
/// it behind the operator's back would swap a clear refusal for a serve with
/// no rollback depth that still cannot boot, so the depth is returned
/// UNCHANGED and the caller refuses quoting the full formula.
#[test]
fn an_unfittable_reserve_keeps_the_requested_depth_for_the_refusal() {
    let args = args();
    let slot = slot_bytes(&args, PER_SEQ_BLOB);
    let fit = fit_ring(
        &args,
        8,
        slot,
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        RESERVE_WITHOUT_RING - 1,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 8, "the refusal must quote what was asked for");
    assert!(fit.warning.is_none());
}

/// `--ssm-decode-ring-slots 0` is a real value the CLI accepts, and the
/// requested-depth path must honour a published depth rather than the
/// constant 8 — the allocation side reads the same cell.
#[test]
fn the_flag_parses_through_the_model_side_ssot() {
    use spark_model::ssm_reserve::parse_decode_ring_slots;
    assert_eq!(parse_decode_ring_slots("auto"), Ok(None));
    assert_eq!(parse_decode_ring_slots("2"), Ok(Some(2)));
    assert!(parse_decode_ring_slots("12").is_err());
    // The clap default is the `auto` this module's fit depends on.
    assert_eq!(args().ssm_decode_ring_slots, "auto");
}

// ── The memory-budget cap (`ssm_reserve/budget.rs`) ───────────────────────────
//
// Two machines again: GB10 (the default part, whose configurations must not
// move) and a 16 GB Ada card serving a 4B hybrid model in BF16, where the
// default 8 x 8 ring was measured at 3,264 MiB.

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;
/// 4B per-sequence SSM state blob: 3,264 MiB / (8 slots x 8 seqs).
const BLOB_4B: usize = 51 * MIB;

fn args_with_batch(batch: &str) -> cli::ServeArgs {
    cli::ServeArgs::parse_from(["spark", "Qwen/Qwen3.5-4B", "--max-batch-size", batch])
}

/// The 4060 Ti case: 15.3 GiB free, an 8.3 GiB checkpoint (clamped to half of
/// free, see `memory_beside_weights`), 2 GiB of reserve that does not scale
/// with the ring. Pre-load free memory would wave the 3,264 MiB ring through
/// (`2 + 3.19 <= 15.3`) and leave the KV cache to find out.
fn ada_fit(args: &cli::ServeArgs, explicit: bool) -> RingFit {
    let free = 15 * GIB + 300 * MIB;
    let beside = spark_model::ssm_reserve::memory_beside_weights(free, 8 * GIB + 300 * MIB);
    fit_ring_budgeted(
        args,
        8,
        slot_bytes(args, BLOB_4B),
        BLOB_4B,
        2 * GIB,
        free,
        beside,
        &PRE_LOAD,
        explicit,
    )
}

#[test]
fn the_ada_card_halves_the_ring_at_the_default_batch_and_says_so() {
    let args = args_with_batch("8");
    let fit = ada_fit(&args, false);
    assert_eq!(fit.slots, 4, "8 x 8 x 51 MiB = 3,264 MiB -> 1,632 MiB");
    let warning = fit.warning.expect("a cap must be logged, never silent");
    assert!(
        warning.contains("ring: 4 slots x 8 seqs x 51.0 MB/seq = 1.59 GB (was 3.19 GB)"),
        "{warning}"
    );
    assert!(
        warning.contains("capped at 1/3 of the 7.65 GB"),
        "{warning}"
    );
    assert!(
        warning.contains("--ssm-decode-ring-slots"),
        "the warning must name the flag that pins a depth: {warning}"
    );
}

#[test]
fn a_single_stream_serve_keeps_the_full_ring_on_the_ada_card() {
    let args = args_with_batch("1");
    let fit = ada_fit(&args, false);
    assert_eq!(
        fit.slots, 8,
        "8 x 1 x 51 MiB = 408 MiB is well inside budget"
    );
    assert!(fit.warning.is_none());
}

#[test]
fn a_pinned_depth_is_never_capped() {
    // `--ssm-decode-ring-slots 8` is honored on both sides or refused: the
    // budget must not become the "silent third answer".
    let args = args_with_batch("8");
    let fit = ada_fit(&args, true);
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
}

#[test]
fn gb10_defaults_are_untouched() {
    // ~115 GiB free, a 17 GiB checkpoint, the 27B's 151.5 MiB blob, default
    // batch 8 and the largest batch the budget leaves alone (27).
    let free = 115 * GIB;
    let beside = spark_model::ssm_reserve::memory_beside_weights(free, 17 * GIB);
    for batch in ["8", "16", "27"] {
        let args =
            cli::ServeArgs::parse_from(["spark", "Qwen/Qwen3.6-27B", "--max-batch-size", batch]);
        let fit = fit_ring_budgeted(
            &args,
            8,
            slot_bytes(&args, PER_SEQ_BLOB),
            PER_SEQ_BLOB,
            RESERVE_WITHOUT_RING,
            free,
            beside,
            &PRE_LOAD,
            false,
        );
        assert_eq!(fit.slots, 8, "batch {batch}");
        assert!(fit.warning.is_none(), "batch {batch}: {:?}", fit.warning);
    }
}

#[test]
fn the_cap_and_the_yardstick_can_both_bite_and_both_are_reported() {
    // Free memory so tight that after the budget cap (8 -> 4) the reserve
    // still does not fit: 2 GiB + 4 x 408 MiB = 3.6 GiB against 3.0 GiB.
    let args = args_with_batch("8");
    let free = 3 * GIB;
    let beside = spark_model::ssm_reserve::memory_beside_weights(15 * GIB + 300 * MIB, 8 * GIB);
    let fit = fit_ring_budgeted(
        &args,
        8,
        slot_bytes(&args, BLOB_4B),
        BLOB_4B,
        2 * GIB,
        free,
        beside,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 2, "4 does not fit 3.0 GiB; 2 does (2.8 GiB)");
    let warning = fit.warning.expect("both stages must be logged");
    assert!(warning.contains("capped at 1/3"), "{warning}");
    assert!(warning.contains("Then:"), "{warning}");
    assert!(
        warning.contains("Sized from pre-load free memory"),
        "{warning}"
    );
}
