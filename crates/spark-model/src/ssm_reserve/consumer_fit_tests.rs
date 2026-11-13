// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the consumer verify-slot fit (port lane L24) reproduced on
//! the 27B's real state sizes and the 4060 Ti's budget (report 77 section
//! 12.1), and the pin that an unpublished cap leaves every target's slot count
//! exactly as before. The process-wide cell is never written here: a test that
//! published it would change `mtp_state_slots` for every other test.

use super::super::{SsmRollbackMode, mtp_state_slots_with, ssm_pool_reserve_bytes};
use super::*;

const MIB: usize = 1024 * 1024;

/// Qwen3.8-27B GDN state, per sequence across its 48 linear-attention layers:
/// h = 48 value heads x 128 x 128 x FP32 per layer (144 MiB in all), conv =
/// conv_dim 10,240 x 4 taps x FP32 per layer. Together the 151.5 MiB blob the
/// preflight log prints for this model.
const H_27B: usize = 48 * 48 * 128 * 128 * 4;
const CONV_27B: usize = 48 * 10_240 * 4 * 4;

/// Verify bytes of `n` slots at `drafts`, uniform width (env-free), snapshot
/// rollback, FP32 h: the SSOT term preflight and the pool allocate through.
fn verify_bytes(n: usize, drafts: usize) -> usize {
    ssm_pool_reserve_bytes(
        0,
        H_27B,
        CONV_27B,
        true,
        drafts,
        n,
        true,
        false,
        SsmRollbackMode::Snapshot,
    )
}

#[test]
fn the_27b_state_blob_is_the_logged_151_5_mib() {
    assert_eq!(H_27B, 144 * MIB);
    assert_eq!((H_27B + CONV_27B) as f64 / MIB as f64, 151.5);
    // One slot at --num-drafts 3: three h snapshots, four conv snapshots and
    // the pre-verify checkpoint blob, about 613.5 MiB.
    assert_eq!(
        verify_bytes(1, 3),
        3 * H_27B + 4 * CONV_27B + H_27B + CONV_27B
    );
    assert_eq!(verify_bytes(1, 3), 643_301_376);
}

/// The 4060 Ti budget of 77 section 12.1: 15,660 MiB free (the guide's
/// measured figure), the UD-IQ4_XS file minus the 2.37 GiB BF16 embedding the
/// `embed_host_offload` row moves to the host, the FP8 KV floor (4,096 tokens
/// x 32 KiB a token over 16 attention layers) and the Qwen MTP headroom at
/// 8,192 tokens (512 + 80 + 512 MiB).
fn budget_27b(batch: usize) -> usize {
    let free = 15_660 * MIB;
    let resident = 14_252_845_984usize - 2_542_796_800;
    let base = batch * (H_27B + CONV_27B);
    let kv_floor = batch * 4096 * 32 * 1024;
    let headroom = 512 * MIB + 8192 * 5120 * 2 + 512 * MIB;
    verify_budget_bytes(free, resident, base, kv_floor, headroom)
}

#[test]
fn the_27b_on_16_gb_keeps_drafts_for_one_stream_at_batch_8() {
    let budget = budget_27b(8);
    assert_eq!(budget, 604_203_568);
    // Eight slots want 4.79 GiB; not even one fits its half of what is left,
    // so the floor keeps the first stream speculating.
    let cap = fit_verify_slots(8, budget, |n| verify_bytes(n, 3));
    assert_eq!(cap, 1);
}

#[test]
fn a_smaller_batch_leaves_room_for_two_speculating_streams() {
    let budget = budget_27b(2);
    let cap = fit_verify_slots(2, budget, |n| verify_bytes(n, 3));
    assert_eq!(cap, 2, "budget {budget}");
    let cap4 = fit_verify_slots(4, budget_27b(4), |n| verify_bytes(n, 3));
    assert!(cap4 < 4, "four slots cannot all verify on 16 GB: {cap4}");
}

/// The 4B on the same card (24 GDN layers of 32 x 128 x 128 FP32 h, the
/// 51 MiB blob of the guide) keeps all eight slots at the default batch: the
/// fit publishes nothing and the measured serves are unchanged.
#[test]
fn the_4b_keeps_every_slot_on_the_4060_ti() {
    let h = 24 * 32 * 128 * 128 * 4;
    let conv = 24 * 8_192 * 4 * 4;
    let free = 15_660 * MIB;
    for resident in [6_570 * MIB, 8_680 * MIB] {
        let budget = verify_budget_bytes(
            free,
            resident,
            8 * (h + conv),
            8 * 4096 * 32 * 1024,
            1024 * MIB + 8192 * 2560 * 2,
        );
        let bytes = |n| {
            ssm_pool_reserve_bytes(
                0,
                h,
                conv,
                true,
                3,
                n,
                true,
                false,
                SsmRollbackMode::Snapshot,
            )
        };
        assert_eq!(fit_verify_slots(8, budget, bytes), 8, "resident {resident}");
    }
}

#[test]
fn the_fit_is_monotone_and_never_drops_below_one_slot() {
    let per = 100usize;
    let bytes = |n: usize| n * per;
    assert_eq!(fit_verify_slots(8, 0, bytes), 1);
    assert_eq!(fit_verify_slots(8, 350, bytes), 3);
    assert_eq!(fit_verify_slots(8, 800, bytes), 8);
    assert_eq!(fit_verify_slots(8, usize::MAX, bytes), 8);
    assert_eq!(fit_verify_slots(0, usize::MAX, bytes), 0);
    let mut last = 0;
    for budget in (0..1000).step_by(50) {
        let n = fit_verify_slots(8, budget, bytes);
        assert!(n >= last);
        last = n;
    }
}

#[test]
fn the_budget_saturates_instead_of_wrapping() {
    assert_eq!(verify_budget_bytes(10, 20, 0, 0, 0), 0);
    // (100 - 10 - 10 - 10 - 10) / VERIFY_BUDGET_DIVISOR: all four terms are subtracted.
    assert_eq!(verify_budget_bytes(100, 10, 10, 10, 10), 30);
}

/// The pin the plan asks for (B8): with no cap published, which is every
/// target but ada, the slot count is the pre-sprint pure core, and the kill
/// switch ignores a cap.
#[test]
fn an_unpublished_cap_leaves_every_slot_count_as_before() {
    for (bs, cap, full) in [
        (1, 32, false),
        (8, 32, false),
        (64, 32, false),
        (64, 32, true),
    ] {
        let before = mtp_state_slots_with(bs, cap, full);
        assert_eq!(apply_consumer_cap(before, None, full), before);
    }
    // gb10 numbers: bs=8 keeps 8 slots, bs=64 keeps the 32-slot diet.
    assert_eq!(
        apply_consumer_cap(mtp_state_slots_with(8, 32, false), None, false),
        8
    );
    assert_eq!(
        apply_consumer_cap(mtp_state_slots_with(64, 32, false), None, false),
        32
    );
    assert_eq!(apply_consumer_cap(8, Some(1), false), 1);
    assert_eq!(apply_consumer_cap(8, Some(1), true), 8);
    assert_eq!(
        apply_consumer_cap(1, Some(4), false),
        1,
        "a cap never widens"
    );
}
