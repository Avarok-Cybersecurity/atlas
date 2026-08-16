// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT for the Phase-C decode-rollback ring depth.
//!
//! Two call sites MUST agree on this number or a serve either
//! under-reserves (runtime CUDA alloc failure after weights load) or
//! over-reserves (preflight refuses batch sizes the runtime could fund):
//!
//! * `spark-server` `preflight_reserve` — sizes the SSM-snapshot GPU
//!   reservation before weights load;
//! * `TransformerModel::new` (`impl_a1.rs`) — allocates the actual ring.
//!
//! The ring's ONLY writer (scheduler `snapshot_boundary_if_ssm`) and reader
//! (content-loop `rollback_to_boundary`) live on the PLAIN decode path — the
//! speculative path does its rejection rollback through the verify snapshot,
//! never this ring. Under `--speculative` the ring is unreachable, and it is
//! NOT cheap: 8 slots × max_batch × the full SSM blob (27B: 158.9 MB) is
//! ~19 GB at batch 16 and ~38 GB at batch 32. Reserving it unconditionally
//! while the runtime skipped it capped the native batch at ~20 on GB10
//! (SSM reserve 75.2 GB vs an 85.2 GB budget at util 0.70).
//!
//! Env contract (read HERE and nowhere else):
//!
//! * `ATLAS_SSM_DECODE_RING=1` force-allocates the ring even under spec
//!   (mixed workloads whose grammar-bound sequences fall to plain decode and
//!   should keep loop re-steer); `=0` force-disables it even without spec.
//! * `ATLAS_DISABLE_WATCHDOGS=1|true` (trimmed, case-insensitive — mirrors
//!   spark-server's `parse_disable_watchdogs`): the ring's only reader can
//!   never fire, so the ring is skipped.

/// Outcome of the ring-depth decision.
///
/// `skip_reason` is `Some` only for the IMPLICIT skip (speculative decode /
/// watchdogs off) — never for an explicit `ATLAS_SSM_DECODE_RING=0`
/// override — so the allocating call site can log the savings once.
pub struct DecodeRingDecision {
    pub slots: usize,
    pub skip_reason: Option<&'static str>,
}

/// Number of SSM-pool slots the MTP/DFlash VERIFY state pools (per-token
/// intermediates + pre-verify checkpoints) must cover.
///
/// Three call sites MUST agree on this number (same contract as the decode
/// ring above):
///
/// * `spark-server` `preflight_reserve` — sizes the pre-load GPU reserve;
/// * `SsmStatePool::new` — allocates the intermediate/checkpoint pools;
/// * the scheduler's spec dispatch — gates every speculative step on
///   `slot_idx < mtp_state_slots(..)` so an uncovered slot can never be
///   verified (uncovered slots plain-decode until retirement-time
///   compaction migrates them under the cap).
///
/// WHY a cap exists: the verify pools were sized `max_batch_size × K` even
/// though spec dispatch is bounded by `speculative::mtp_max_seqs()`
/// (default 32 — the widest batched-verify chunk,
/// `layer::VERIFY_WY_TABLE_SEQS`). On the 27B at `--max-batch-size 64`
/// with `--num-drafts 3` that is 32 dead slots × 5 SSM blobs × 158.9 MB =
/// 25.4 GB of reserve for states no code path can ever touch — the
/// difference between bs=64 refusing at preflight (util 0.70) and booting.
///
/// The cap NEVER bites at `max_batch_size <= 32`: the floor is
/// `VERIFY_WY_TABLE_SEQS` (32), so bs<=32 sizing and behavior are
/// byte-identical in every env combination (slots are always `< bs`).
///
/// Env contract (read HERE and nowhere else):
///
/// * `ATLAS_MTP_POOL_FULL_WIDTH` (presence, house convention — `=0` is NOT
///   off): restore full-width pools (`max_batch_size` slots) and make the
///   scheduler guard vacuous. Kill switch for the bs>32 reserve diet.
/// * `ATLAS_EP_PROTOCOL=v2` implies full width: v2 pins slots in place for
///   the worker mirror (no compaction — see `retire_finished_sequences`),
///   so a high slot may legitimately speculate forever.
/// * `ATLAS_MTP_MAX_SEQS` participates via [`crate::speculative::mtp_max_seqs`]:
///   raising the dispatch cap above 32 widens the pools with it.
///
/// ★ WHAT THE DIET COSTS, AND THE UTILISATION FLOOR IT SETS (wave 47,
/// dgx3, 27B W4A4). The diet is what makes a single serve able to cover the
/// whole concurrency ladder — speculation is dispatch-capped at 32, so one
/// serve at `--max-batch-size 128 --speculative --num-drafts 3` speculates
/// at C<=32 and plain-decodes above it. But the verify pools it keeps are
/// still sized by `--num-drafts`, and at bs=128 that is not free. Measured
/// preflight reserve, `--max-seq-len 4096`, blob 151.5 MB:
///
/// | config | base | verify pools | snapshot/misc | reserve |
/// |---|---|---|---|---|
/// | bs=128, spec OFF | 18.9 GB (128 blobs) | — | 5.5 GB | **24.3 GB** |
/// | bs=128, spec ON, 3 drafts | 18.9 GB | **23.7 GB** (32 slots x 5 blobs) | 8.9 GB | **51.5 GB** |
///
/// With 39.8 GB already consumed before KV, that reserve REFUSES at
/// `--gpu-memory-utilization 0.70` (39.8 + 51.5 = 91.3 GB committed against
/// an 85.2 GB budget) and boots at 0.85 (103.4 GB budget, 13.3 GB left for
/// KV = 217k tokens). The floor for the one-serve ladder is therefore
/// **util ~0.82**, and it is set HERE, by the verify pools — not by the KV
/// dtype, which moves the answer by well under a GB at these widths. A
/// cheaper diet (row-budget-sized intermediates rather than slot-major)
/// would recover ~9 GB and still not reach 0.70; the reserve, not the
/// speculation regime, is what makes the low-util single config impossible.
pub fn mtp_state_slots(max_batch_size: usize) -> usize {
    mtp_state_slots_with(
        max_batch_size,
        crate::speculative::mtp_max_seqs(),
        mtp_pool_full_width(),
    )
}

/// The `ATLAS_MTP_POOL_FULL_WIDTH` kill switch (PRESENCE, house convention —
/// `=0` is NOT off), plus the EP-v2 implication (v2 pins slots in place for
/// the worker mirror, so a high slot may legitimately speculate forever).
/// SSOT for BOTH pool diets it disables: the bs>32 slot-count cap
/// ([`mtp_state_slots`]) and the tiered per-slot verify capacity
/// ([`verify_slot_drafts`]) — one switch restores the full-width,
/// uniform-K sizing everywhere (pool, preflight, scheduler clamp).
pub fn mtp_pool_full_width() -> bool {
    std::env::var_os("ATLAS_MTP_POOL_FULL_WIDTH").is_some()
        || matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2"))
}

/// Pure core of [`mtp_state_slots`] (env-free, unit-testable).
///
/// `spec_dispatch_cap` is `speculative::mtp_max_seqs()` — the scheduler
/// never dispatches a speculative step wider than this. The floor
/// `VERIFY_WY_TABLE_SEQS` (32) guarantees bs<=32 configs are untouched even
/// under `ATLAS_NO_MTP_K_LADDER` (which drops the dispatch cap to 4).
pub fn mtp_state_slots_with(
    max_batch_size: usize,
    spec_dispatch_cap: usize,
    full_width: bool,
) -> usize {
    if full_width {
        return max_batch_size;
    }
    max_batch_size.min(spec_dispatch_cap.max(crate::layer::VERIFY_WY_TABLE_SEQS))
}

/// Per-slot verify DRAFT capacity — the tiered half of the verify-pool
/// diet (2026-08-16). Pure core; `drafts_at(n)` is the ladder policy
/// (`speculative::mtp_ladder_drafts`).
///
/// A sequence occupying pool slot `slot_idx` can only be co-active with at
/// least `slot_idx + 1` sequences UNDER the contiguity invariant ("active
/// sequences occupy contiguous slots [0..n)"), so the deepest draft count
/// the ladder can ever hand it is the max over widths `n > slot_idx`. The
/// invariant is TRANSIENTLY breakable (LIFO free-list claim after churn),
/// which is why this number is also ENFORCED at dispatch: the scheduler
/// clamps the step's draft count to the minimum capacity across the active
/// slots (`step_mtp`), so a high-slotted straggler shrinks K for its step
/// instead of overflowing its slot's pools.
///
/// Default ladder (`4:3,8:3,16:1,32:1`, `--num-drafts 3`): slots 0..8 keep
/// capacity 3 (K=4), slots 8.. get capacity 1 (K=2). NOTE the runtime
/// `adaptive_rung` lift (n in 9..=16 to 2 drafts on tool-shaped accept
/// stats) EXCEEDS the static ladder this sizing derives from; under the
/// tiered default it is clamped back to K=2 whenever any active sequence
/// sits in a capacity-1 slot — i.e. at every n >= 9 under contiguity.
/// `ATLAS_MTP_POOL_FULL_WIDTH` restores uniform full-K pools and re-enables
/// the lift.
pub fn verify_slot_drafts_with(
    slot_idx: usize,
    dispatch_cap: usize,
    num_drafts: usize,
    drafts_at: impl Fn(usize) -> usize,
) -> usize {
    if num_drafts == 0 {
        return 0;
    }
    let hi = dispatch_cap.max(slot_idx + 1);
    ((slot_idx + 1)..=hi)
        .map(&drafts_at)
        .max()
        .unwrap_or(num_drafts)
        .clamp(1, num_drafts)
}

/// Env-reading wrapper of [`verify_slot_drafts_with`]: the ladder policy
/// (with its `ATLAS_MTP_K_LADDER` / `ATLAS_NO_MTP_K_LADDER` overrides — a
/// disabled ladder returns `num_drafts` at every width, making the tiers
/// vacuous) plus the [`mtp_pool_full_width`] kill switch.
pub fn verify_slot_drafts(slot_idx: usize, num_drafts: usize) -> usize {
    if mtp_pool_full_width() {
        return num_drafts;
    }
    verify_slot_drafts_with(
        slot_idx,
        crate::speculative::mtp_max_seqs(),
        num_drafts,
        |n| crate::speculative::mtp_ladder_drafts(n, num_drafts),
    )
}

/// Number of per-token H-state intermediates the verify pools allocate for
/// pool slot `slot_idx`: the slot's draft capacity + 1 (one snapshot per
/// verify row). `uniform_verify` (DFlash-γ pools, whose verify width does
/// not follow the MTP ladder) sizes every slot at the full `num_drafts + 1`.
///
/// Only the H side tiers. The CONV intermediates stay UNIFORM at
/// `num_drafts + 1` per slot: the batched conv verify kernel
/// (`gdn_verify_fused_conv_kn_batched`) requires a uniform cross-sequence
/// snapshot stride (checked against the actual pointers in
/// `trait_decode_batched_conv_gdn_multi.rs`) and writes all K snapshots —
/// tiering conv would silently decline the two-launch fast path for every
/// spec batch spanning the tier boundary (all n >= 9). Conv is ~5% of the
/// blob, so the forgone saving is ~0.35 GiB at 32 slots while the H side
/// carries the other 6.75 GiB.
pub fn verify_slot_h_intermediates(
    slot_idx: usize,
    num_drafts: usize,
    uniform_verify: bool,
) -> usize {
    if uniform_verify {
        return num_drafts + 1;
    }
    verify_slot_drafts(slot_idx, num_drafts) + 1
}

/// SSM state-pool reserve bytes for the pre-load preflight — MUST mirror
/// what `SsmStatePool::new` allocates (modulo the +1 dummy slot per pool,
/// which preflight has never counted; the CUDA headroom term absorbs it):
///
/// * base: `max_batch_size` live per-seq blobs (h_state + conv_state across
///   all SSM layers);
/// * spec, per verify slot (`mtp_state_slots` of them):
///   - H intermediates: [`verify_slot_h_intermediates`] × h blob (TIERED);
///   - conv intermediates: `num_drafts + 1` × conv blob (uniform — see
///     [`verify_slot_h_intermediates`] for why conv does not tier);
///   - 1 pre-verify checkpoint blob (h + conv).
///
/// `h_blob_bytes` / `conv_blob_bytes` are the per-seq totals across all SSM
/// layers (`num_ssm_layers × ssm_h_state_bytes/ssm_conv_state_bytes`).
/// With `uniform_verify` (DFlash, `ATLAS_MTP_POOL_FULL_WIDTH`, ladder
/// disabled) and `mtp_state_slots == max_batch_size` this reproduces the
/// historical `max_batch × blob × (1 + (num_drafts+1) + 1)` byte-for-byte.
pub fn ssm_pool_reserve_bytes(
    max_batch_size: usize,
    h_blob_bytes: usize,
    conv_blob_bytes: usize,
    spec_on: bool,
    num_drafts: usize,
    mtp_state_slots: usize,
    uniform_verify: bool,
) -> usize {
    let blob = h_blob_bytes + conv_blob_bytes;
    let base = max_batch_size * blob;
    if !spec_on {
        return base;
    }
    let verify: usize = (0..mtp_state_slots)
        .map(|slot| {
            verify_slot_h_intermediates(slot, num_drafts, uniform_verify) * h_blob_bytes
                + (num_drafts + 1) * conv_blob_bytes
                + blob
        })
        .sum();
    base + verify
}

/// Decide the per-sequence decode-rollback ring depth.
///
/// `use_speculative` MUST be the same flag `factory::build_model` receives
/// (`--speculative || --dflash` as plumbed by spark-server) at every call
/// site, or preflight and allocation diverge.
pub fn decode_rollback_ring_slots(
    num_ssm_layers: usize,
    use_speculative: bool,
) -> DecodeRingDecision {
    if num_ssm_layers == 0 {
        return DecodeRingDecision {
            slots: 0,
            skip_reason: None,
        };
    }
    let watchdogs_disabled = std::env::var("ATLAS_DISABLE_WATCHDOGS")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true"
        })
        .unwrap_or(false);
    match std::env::var("ATLAS_SSM_DECODE_RING").ok().as_deref() {
        Some("1") => DecodeRingDecision {
            slots: atlas_kernels::DECODE_ROLLBACK_RING_SLOTS,
            skip_reason: None,
        },
        Some("0") => DecodeRingDecision {
            slots: 0,
            skip_reason: None,
        },
        _ if use_speculative || watchdogs_disabled => DecodeRingDecision {
            slots: 0,
            skip_reason: Some(if use_speculative {
                "speculative decode active"
            } else {
                "watchdogs disabled"
            }),
        },
        _ => DecodeRingDecision {
            slots: atlas_kernels::DECODE_ROLLBACK_RING_SLOTS,
            skip_reason: None,
        },
    }
}
#[cfg(test)]
mod mtp_state_slot_tests {
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
        )
    }

    #[test]
    fn blob_matches_campaign_constant() {
        assert_eq!(H_BLOB, 150_994_944);
        assert_eq!(CONV_BLOB, 7_864_320);
        assert_eq!(BLOB, 158_859_264);
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
    fn tiering_cannot_bite_at_or_below_8_and_kill_switch_restores() {
        // Slots 0..8 are full-K under the default ladder, so bs<=8 sizing is
        // byte-identical to the legacy formula in every mode.
        for bs in 1..=8 {
            for spec_on in [false, true] {
                assert_eq!(
                    tiered_pool_bytes(bs, spec_on),
                    legacy_pool_bytes(bs, spec_on),
                    "bs={bs} spec={spec_on}: bs<=8 ledger must not move by a byte"
                );
            }
        }
        // uniform_verify (DFlash-γ pools / ATLAS_MTP_POOL_FULL_WIDTH /
        // ladder disabled) restores the legacy bytes at every bs<=32.
        for bs in 1..=32 {
            for spec_on in [false, true] {
                assert_eq!(
                    ssm_pool_reserve_bytes(bs, H_BLOB, CONV_BLOB, spec_on, ND, bs, true),
                    legacy_pool_bytes(bs, spec_on),
                    "bs={bs} spec={spec_on}: uniform sizing must reproduce legacy"
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
        // Verify-pool bytes per covered slot: H tier (capacity+1 h blobs) +
        // uniform conv (ND+1) + one checkpoint blob. Aggregates pinned
        // EXACTLY so any future drift in the formula is a test edit, not an
        // accident.
        //
        // bs=16 (slots 8..16 on the low tier): saves 8 slots × 2 h blobs.
        assert_eq!(legacy_pool_bytes(16, true), 15_250_489_344);
        assert_eq!(tiered_pool_bytes(16, true), 12_834_570_240);
        assert_eq!(
            legacy_pool_bytes(16, true) - tiered_pool_bytes(16, true),
            16 * H_BLOB // 2.25 GiB
        );
        // bs=32: 24 low-tier slots × 2 h blobs = 48 h blobs = 6.75 GiB.
        // (The task-#1 estimate of 7.1 GiB counted 48 FULL blobs; conv
        // stays uniform for the batched-conv stride precondition, so the H
        // side carries 6.75 GiB and conv's 0.35 GiB is deliberately kept.)
        assert_eq!(
            legacy_pool_bytes(32, true) - tiered_pool_bytes(32, true),
            48 * H_BLOB // 7_247_757_312
        );
        assert_eq!(tiered_pool_bytes(32, true) - 32 * BLOB, 18_169_724_928);
        // Spec off: base only, at any bs.
        assert_eq!(tiered_pool_bytes(64, false), 64 * BLOB);
    }

    #[test]
    fn bs64_ledger_before_after_and_fit() {
        // ── Pool terms: the three diet rungs ──
        let full_width = legacy_pool_bytes(64, true);
        assert_eq!(full_width, 61_001_957_376); // 56.81 GiB (pre-diet)
        let slot_capped = ssm_pool_reserve_bytes(64, H_BLOB, CONV_BLOB, true, ND, 32, true);
        assert_eq!(slot_capped, 35_584_475_136); // 33.14 GiB (slot-count cap)
        assert_eq!(full_width - slot_capped, 25_417_482_240); // 23.67 GiB
        let tiered = tiered_pool_bytes(64, true);
        assert_eq!(tiered, 28_336_717_824); // 26.39 GiB (+ tiered slots)
        assert_eq!(slot_capped - tiered, 48 * H_BLOB); // 6.75 GiB more

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
        assert_eq!(tiered_reserve, 37_901_303_808); // 35.30 GiB

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
        // Tiered reserve: strictly better — the 6.75 GiB rejoins the KV pool.
        assert!(budget - pre_kv - tiered_reserve - kv_floor >= 150 * 1024 * 1024);
    }

    #[test]
    fn bs128_ledger_matches_campaign_reference() {
        // The wave-47 measured bs=128 reserve the campaign docs quote as
        // "51.5 GiB": base 128 blobs + 32 verify slots × 5 blobs + 32
        // Marconi slots + GDN scratch + 4 GiB spec headroom.
        let gdn = 186_122_240usize;
        let headroom = 4usize * 1024 * 1024 * 1024;
        let old_pool = ssm_pool_reserve_bytes(128, H_BLOB, CONV_BLOB, true, ND, 32, true);
        assert_eq!(old_pool, 45_751_468_032);
        let old_reserve = old_pool + 32 * BLOB + gdn + headroom;
        assert_eq!(old_reserve, 55_316_054_016); // 51.52 GiB — the reference
        // Tiered slots: −6.75 GiB from the H intermediates.
        let new_pool = tiered_pool_bytes(128, true);
        assert_eq!(new_pool, 38_503_710_720);
        let new_reserve = new_pool + 32 * BLOB + gdn + headroom;
        assert_eq!(new_reserve, 48_068_296_704); // 44.77 GiB
        // With the concurrency profile's Marconi 32→8 (#0): another 3.55 GiB.
        assert_eq!(new_pool + 8 * BLOB + gdn + headroom, 44_255_674_368); // 41.22 GiB
    }
}
