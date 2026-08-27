// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`ssm_reserve`). A sibling file via `#[path]` — the
//! `validate.rs`/`validate_tests.rs` idiom — so `ssm_reserve.rs` stays under
//! the 500-line cap; module position (child of `ssm_reserve`) is unchanged,
//! so `super::*` paths are untouched.
use super::*;

/// Reserve-diet ledger constants, Qwen3.6-27B (config.json of
/// centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf), `--max-seq-len 4096
/// --num-drafts 3 --speculative`, kv bf16.
///
/// Per-seq SSM blob: 48 GDN layers × (h 48·128·128·4 B + conv
/// (16·128·2 + 48·128)·4·4 B) = 48 × 3,309,568 = 158,859,264 B —
/// the "158.9 MB" (151.5 MiB) blob every campaign doc quotes. The
/// verify tiers split it: H is 95% of the blob, conv the other 5%.
const H_BLOB: usize = 48 * (48 * 128 * 128 * 4);
const CONV_BLOB: usize = 48 * ((16 * 128 * 2 + 48 * 128) * 4 * 4);
const BLOB: usize = H_BLOB + CONV_BLOB;
const ND: usize = 3; // --num-drafts 3 (K=4 ceiling)

/// The historical formula the pre-tier sizing reproduced at bs<=32:
/// `max_batch × blob × (1 + (nd+1) + 1)`.
fn legacy_pool_bytes(bs: usize, spec_on: bool) -> usize {
    let mult = if spec_on { 1 + (ND + 1) + 1 } else { 1 };
    bs * BLOB * mult
}

/// The DEFAULT ladder shape (`4:3,8:3,16:1,32:1`), spelled out so these
/// tests do not depend on process env (CI sets neither
/// ATLAS_MTP_K_LADDER nor ATLAS_NO_MTP_K_LADDER; the env-reading
/// wrappers are covered by the ladder's own tests).
fn default_ladder(n: usize) -> usize {
    if n <= 8 { 3 } else { 1 }
}

fn tiered_pool_bytes(bs: usize, spec_on: bool) -> usize {
    ssm_pool_reserve_bytes(
        bs,
        H_BLOB,
        CONV_BLOB,
        spec_on,
        ND,
        mtp_state_slots_with(bs, 32, false),
        false,
        false,
        SsmRollbackMode::Snapshot,
    )
}

/// Stage-3 twin of `tiered_pool_bytes`: the f16-SIZED pool.
fn tiered_pool_bytes_f16(bs: usize, spec_on: bool) -> usize {
    ssm_pool_reserve_bytes(
        bs,
        H_BLOB,
        CONV_BLOB,
        spec_on,
        ND,
        mtp_state_slots_with(bs, 32, false),
        false,
        true,
        SsmRollbackMode::Snapshot,
    )
}

#[test]
fn cap_identity_at_or_below_32_every_config() {
    // bs<=32 slot COUNT must be identical to the legacy sizing for every
    // dispatch-cap value (incl. ATLAS_NO_MTP_K_LADDER's 4) because the
    // floor is VERIFY_WY_TABLE_SEQS = 32.
    for bs in 1..=32 {
        for cap in [1, 4, 16, 32, 64] {
            assert_eq!(
                mtp_state_slots_with(bs, cap, false),
                bs,
                "bs={bs} cap={cap}"
            );
        }
    }
}

#[test]
fn tier_capacity_default_ladder_shape() {
    // Slots 0..8 keep the full K=4 depth; slots 8.. are sized for the
    // ladder's deepest possible draft count at widths that can reach
    // them — 1 (K=2) under the default ladder.
    for slot in 0..8 {
        assert_eq!(verify_slot_drafts_with(slot, 32, 3, default_ladder), 3);
    }
    for slot in 8..32 {
        assert_eq!(verify_slot_drafts_with(slot, 32, 3, default_ladder), 1);
    }
    // Beyond the dispatch cap (transient churn can still park a covered
    // sequence there): last-rung depth, never zero.
    assert_eq!(verify_slot_drafts_with(40, 32, 3, default_ladder), 1);
    // --num-drafts remains the ceiling and the floor collapses with it.
    for slot in 0..32 {
        assert_eq!(verify_slot_drafts_with(slot, 32, 1, default_ladder), 1);
        assert_eq!(verify_slot_drafts_with(slot, 32, 0, default_ladder), 0);
    }
    // An explicit deeper ladder (e.g. "4:3,8:3,16:2,24:2,32:2") widens
    // the low tier with it — capacity follows the POLICY, not a magic 8.
    let deep = |n: usize| if n <= 8 { 3 } else { 2 };
    assert_eq!(verify_slot_drafts_with(8, 32, 3, deep), 2);
    assert_eq!(verify_slot_drafts_with(31, 32, 3, deep), 2);
}

#[test]
fn k_minus_1_shrink_and_kill_switch_shape() {
    // The K-1 h-intermediate shrink applies EVERYWHERE (it removes a
    // slot that is never written or read, not a policy tier): every
    // sizing differs from the historical formula by exactly one h blob
    // per verify slot, plus the tier savings where the tiers bite.
    //
    // bs<=8: tiers cannot bite (all slots full-K), so the delta is the
    // dead-slot removal alone.
    for bs in 1..=8 {
        assert_eq!(
            legacy_pool_bytes(bs, true) - tiered_pool_bytes(bs, true),
            bs * H_BLOB,
            "bs={bs}: exactly one dead h blob per slot"
        );
        // Spec off: base only, unchanged from the historical formula.
        assert_eq!(tiered_pool_bytes(bs, false), legacy_pool_bytes(bs, false));
    }
    // uniform_verify (DFlash-γ pools / ATLAS_MTP_POOL_FULL_WIDTH /
    // ladder disabled): same dead-slot removal, no tiers, at every bs.
    for bs in 1..=32 {
        for spec_on in [false, true] {
            let expect = legacy_pool_bytes(bs, spec_on) - if spec_on { bs * H_BLOB } else { 0 };
            assert_eq!(
                ssm_pool_reserve_bytes(
                    bs,
                    H_BLOB,
                    CONV_BLOB,
                    spec_on,
                    ND,
                    bs,
                    true,
                    false,
                    SsmRollbackMode::Snapshot,
                ),
                expect,
                "bs={bs} spec={spec_on}: uniform = legacy minus the dead h blob/slot"
            );
        }
    }
}

#[test]
fn cap_bites_above_32_and_kill_switch_restores() {
    // Default dispatch cap 32 ⇒ 64-slot pool covers 32 verify slots.
    assert_eq!(mtp_state_slots_with(64, 32, false), 32);
    // ATLAS_MTP_MAX_SEQS=48 widens the pools with the dispatch cap.
    assert_eq!(mtp_state_slots_with(64, 48, false), 48);
    // ATLAS_NO_MTP_K_LADDER (cap 4) still floors at 32 — defense in depth.
    assert_eq!(mtp_state_slots_with(64, 4, false), 32);
    // Kill switch / EP-v2: full width.
    assert_eq!(mtp_state_slots_with(64, 32, true), 64);
}

#[test]
fn tiered_totals_pinned() {
    // Verify-pool bytes per covered slot: H tier (capacity h blobs —
    // K-1 per K-row verify) + uniform conv (ND+1) + one checkpoint
    // blob. Aggregates pinned EXACTLY so any future drift in the
    // formula is a test edit, not an accident.
    //
    // bs=16: 8 low-tier slots × 2 h blobs (tier) + 16 dead h blobs.
    assert_eq!(legacy_pool_bytes(16, true), 15_250_489_344);
    assert_eq!(tiered_pool_bytes(16, true), 10_418_651_136);
    assert_eq!(
        legacy_pool_bytes(16, true) - tiered_pool_bytes(16, true),
        (16 + 16) * H_BLOB // tier 2.25 GiB + dead-slot 2.25 GiB
    );
    // bs=32: tier saves 48 h blobs (6.75 GiB — the task-#1 estimate of
    // 7.1 GiB counted full blobs; conv stays uniform for the
    // batched-conv stride precondition) and the K-1 shrink another 32
    // (4.5 GiB): 80 h blobs = 11.25 GiB total.
    assert_eq!(
        legacy_pool_bytes(32, true) - tiered_pool_bytes(32, true),
        80 * H_BLOB // 12_079_595_520
    );
    assert_eq!(tiered_pool_bytes(32, true) - 32 * BLOB, 13_337_886_720);
    // Spec off: base only, at any bs.
    assert_eq!(tiered_pool_bytes(64, false), 64 * BLOB);
}

#[test]
fn bs64_ledger_before_after_and_fit() {
    // ── Pool terms: the diet rungs ──
    let full_width = legacy_pool_bytes(64, true);
    assert_eq!(full_width, 61_001_957_376); // 56.81 GiB (pre-diet)
    // Historical slot-count-capped rung (wave 10, PRE the K-1 shrink):
    // kept as the formula the refusal/fit logs were cut against.
    let slot_capped = 64 * BLOB + 32 * (ND + 2) * BLOB;
    assert_eq!(slot_capped, 35_584_475_136); // 33.14 GiB (slot-count cap)
    assert_eq!(full_width - slot_capped, 25_417_482_240); // 23.67 GiB
    // Today's uniform mode = slot-capped minus the dead h blob/slot.
    assert_eq!(
        ssm_pool_reserve_bytes(
            64,
            H_BLOB,
            CONV_BLOB,
            true,
            ND,
            32,
            true,
            false,
            SsmRollbackMode::Snapshot,
        ),
        slot_capped - 32 * H_BLOB // 30_752_636_928
    );
    let tiered = tiered_pool_bytes(64, true);
    assert_eq!(tiered, 23_504_879_616); // 21.89 GiB (tiers + K-1 shrink)
    assert_eq!(slot_capped - tiered, 80 * H_BLOB); // 11.25 GiB more

    // ── Full inference reserve (mirrors preflight_reserve term-by-term) ──
    // snapshot: --ssm-cache-slots 32 × blob (decode ring skipped: spec on)
    let snapshot = 32 * BLOB; // 5_083_496_448
    // GDN two-phase chunked-prefill scratch: 4096 tokens ×
    // (conv_dim 10240×2 + nv 48×2×4 + value_dim 6144×2 + 6144×2) B/tok
    let gdn = 4096 * (10240 * 2 + 48 * 2 * 4 + 6144 * 2 + 6144 * 2);
    assert_eq!(gdn, 186_122_240);
    // CUDA headroom under spec
    let headroom = 4usize * 1024 * 1024 * 1024;

    let full_reserve = full_width + snapshot + gdn + headroom;
    // = the EXACT 67297 MiB the wave-10 bs=64 refusal logged.
    assert_eq!(full_reserve, 70_566_543_360);
    assert_eq!(full_reserve / (1024 * 1024), 67_297);

    let capped_reserve = slot_capped + snapshot + gdn + headroom;
    assert_eq!(capped_reserve, 45_149_061_120); // 42.05 GiB
    let tiered_reserve = tiered + snapshot + gdn + headroom;
    assert_eq!(tiered_reserve, 33_069_465_600); // 30.80 GiB

    // ── Fit at util 0.70 (values from the wave-9/10 refusal logs) ──
    // total_budget: "budget 85.2 GB (util 0.70)" ⇒ 85.2 GiB.
    let budget = (85.2f64 * 1024.0 * 1024.0 * 1024.0) as usize;
    // pre-KV consumed (weights + arena + twins), worst logged: 38.5 GiB
    // (wave-9 bs=64 scout; wave-10 leg read 37.6 GiB).
    let pre_kv = (38.5f64 * 1024.0 * 1024.0 * 1024.0) as usize;
    // KV floor: the C=64 synthetic decode_short peak, dense worst case —
    // 64 seqs × (128 ISL + 1024 OSL) tok × 64 KiB/tok (16 attn layers ×
    // 2 × 4 kv_heads × 256 head_dim × 2 B bf16).
    let kv_floor = 64 * (128 + 1024) * (16 * 2 * 4 * 256 * 2);
    assert_eq!(kv_floor, 4_831_838_208); // 4.50 GiB

    // Full-width reserve: refused with ~19 GiB overshoot before any KV.
    assert!(pre_kv + full_reserve > budget);
    // Slot-capped reserve: boots, KV budget clears the workload floor
    // (the wave-10 claim, preserved byte-for-byte).
    let kv_left = budget - pre_kv - capped_reserve;
    assert!(
        kv_left >= kv_floor,
        "bs=64 KV budget {kv_left} must cover the decode_short peak {kv_floor}"
    );
    assert!(kv_left - kv_floor >= 150 * 1024 * 1024);
    // Tiered reserve: strictly better — 11.25 GiB rejoins the KV pool.
    assert!(budget - pre_kv - tiered_reserve - kv_floor >= 150 * 1024 * 1024);
}

#[test]
fn h_stored_bytes_is_identity_off_and_half_on() {
    // Flag off: EXACT identity at any width — stage 1/2 keep the pool
    // FP32-sized, and every currently-serveable config takes this arm.
    for b in [4usize, 128, H_BLOB, CONV_BLOB] {
        assert_eq!(ssm_h_stored_bytes(b, false), b);
    }
    // Stage 3: half. h blobs are FP32-element sized, so /2 is exact.
    assert_eq!(ssm_h_stored_bytes(H_BLOB, true), H_BLOB / 2);
    assert_eq!(ssm_h_stored_bytes(4, true), 2);
}

#[test]
fn h_stored_bytes_rejects_non_fp32_width() {
    for f16_pool in [false, true] {
        let result = std::panic::catch_unwind(|| ssm_h_stored_bytes(3, f16_pool));
        assert!(result.is_err(), "f16_pool={f16_pool}");
    }
}

#[test]
fn f16_pool_sizing_pinned_and_flag_off_untouched() {
    // The h_f16_pool=false arm must not move a BYTE — re-pin the exact
    // totals the flag-off tests above already pin, through the new
    // parameter position (a transposed argument would show up here).
    assert_eq!(tiered_pool_bytes(32, true) - 32 * BLOB, 13_337_886_720);
    assert_eq!(tiered_pool_bytes(128, true), 33_671_872_512);

    // Stage 3 (`--ssm-h-dtype f16-pool`, refused at serve until prefill
    // narrowing lands — these pin the ALLOCATOR-side arithmetic): every
    // h term (base, tiered intermediates, checkpoints) halves; conv is
    // untouched. bs=128/K=4, the reference shape:
    //   base 128 × (H/2 + CONV)               = 10_670_309_376
    //   slots 0..8:  8 × (3·H/2 + 4·CONV + (H/2 + CONV)) = 2_730_491_904
    //   slots 8..32: 24 × (1·H/2 + 4·CONV + (H/2 + CONV)) = 4_567_597_056
    let f16 = tiered_pool_bytes_f16(128, true);
    assert_eq!(f16, 17_968_398_336); // 16.73 GiB
    // vs the FP32-sized stage-1/2 pool: 14.62 GiB rejoins the KV budget.
    assert_eq!(tiered_pool_bytes(128, true) - f16, 15_703_474_176);
    // Spec off: only the base h blobs narrow.
    assert_eq!(
        tiered_pool_bytes_f16(64, false),
        64 * (H_BLOB / 2 + CONV_BLOB)
    );
}

/// ONE layer's FP32 h blob — what the prefill staging arena is sized by.
/// `H_BLOB` is the across-layers per-seq total, so dividing by the 48 GDN
/// layers is the only place these two widths may be related.
const H_LAYER: usize = H_BLOB / 48;

#[test]
fn prefill_staging_costs_one_fp32_layer_blob_per_slot() {
    // Flag off: ZERO, at every batch size. Nothing is allocated and nothing
    // is reserved, which is what makes the FP32-sized pool byte-identical.
    for bs in [1usize, 32, 64, 128] {
        assert_eq!(ssm_h_prefill_stage_bytes(bs, H_LAYER, false), 0);
    }
    // Stage 3: one FP32 layer blob per slot. NOT per slot per layer — that
    // would be `48 ×` this and would consume three quarters of the win.
    assert_eq!(H_LAYER, 3_145_728);
    assert_eq!(ssm_h_prefill_stage_bytes(128, H_LAYER, true), 402_653_184); // 384 MiB
    assert_eq!(ssm_h_prefill_stage_bytes(32, H_LAYER, true), 100_663_296);
}

/// Replay-mode helper at the reference shape.
fn replay_pool_bytes(bs: usize, spec_on: bool) -> usize {
    ssm_pool_reserve_bytes(
        bs,
        H_BLOB,
        CONV_BLOB,
        spec_on,
        ND,
        mtp_state_slots_with(bs, 32, false),
        false,
        false,
        SsmRollbackMode::Replay,
    )
}

#[test]
fn replay_mode_keeps_one_checkpoint_blob_per_slot() {
    // Replay drops EVERY per-token intermediate (h AND conv): the verify
    // term collapses to one checkpoint blob per covered slot.
    assert_eq!(replay_pool_bytes(128, true), (128 + 32) * BLOB);
    assert_eq!(replay_pool_bytes(8, true), (8 + 8) * BLOB);
    // Spec off: identical to snapshot mode (there is nothing to roll back).
    for bs in [1, 32, 128] {
        assert_eq!(replay_pool_bytes(bs, false), tiered_pool_bytes(bs, false));
    }
}

#[test]
fn replay_ring_bytes_pinned_27b() {
    // One cached row per token per layer: qkvz (16384 BF16 = 32768 B) +
    // gate/beta (48 x 2 FP32 = 384 B) = 33152 B.
    let row = ssm_replay_row_bytes(16384, 48);
    assert_eq!(row, 33_152);
    // K=4 window caches K-1 = 3 rows per covered slot across 48 layers.
    let ring = ssm_replay_ring_bytes(48, row, 4, 32);
    assert_eq!(ring, 152_764_416); // 145.7 MiB
    // No slots / K<=1 / snapshot-mode callers pass 0 slots: all zero.
    assert_eq!(ssm_replay_ring_bytes(48, row, 4, 0), 0);
    assert_eq!(ssm_replay_ring_bytes(48, row, 1, 32), 0);
}

#[test]
fn rollback_mode_parses_and_rejects() {
    use std::str::FromStr;
    assert_eq!(
        SsmRollbackMode::from_str("snapshot").unwrap(),
        SsmRollbackMode::Snapshot
    );
    assert_eq!(
        SsmRollbackMode::from_str("replay").unwrap(),
        SsmRollbackMode::Replay
    );
    // Fail fast on anything else — the CLI relies on THIS parse (SSOT).
    assert!(SsmRollbackMode::from_str("Replay").is_err());
    assert!(SsmRollbackMode::from_str("").is_err());
}

#[test]
fn decode_ring_decision_matrix() {
    let decide = |layers, spec, override_value, watchdogs| {
        let decision = decode_rollback_ring_slots_with(layers, spec, override_value, watchdogs);
        (decision.slots, decision.skip_reason)
    };
    let ring = atlas_kernels::DECODE_ROLLBACK_RING_SLOTS;

    for value in ["1", "true", " TRUE "] {
        assert!(
            watchdogs_disabled_from_value(Some(value)),
            "value={value:?}"
        );
    }
    for value in [None, Some(""), Some("0"), Some("false"), Some("yes")] {
        assert!(!watchdogs_disabled_from_value(value), "value={value:?}");
    }

    assert_eq!(decide(0, false, Some("1"), false), (0, None));
    assert_eq!(decide(48, true, Some("1"), true), (ring, None));
    assert_eq!(decide(48, false, Some("0"), false), (0, None));
    assert_eq!(
        decide(48, true, None, false),
        (0, Some("speculative decode active"))
    );
    assert_eq!(
        decide(48, false, None, true),
        (0, Some("watchdogs disabled"))
    );
    assert_eq!(
        decide(48, true, Some("invalid"), true),
        (0, Some("speculative decode active"))
    );
    assert_eq!(decide(48, false, None, false), (ring, None));
}
