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
    let full_width = std::env::var_os("ATLAS_MTP_POOL_FULL_WIDTH").is_some()
        || matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2"));
    mtp_state_slots_with(
        max_batch_size,
        crate::speculative::mtp_max_seqs(),
        full_width,
    )
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

/// SSM state-pool reserve bytes for the pre-load preflight — MUST mirror
/// what `SsmStatePool::new` allocates (modulo the +1 dummy slot per pool,
/// which preflight has never counted; the CUDA headroom term absorbs it):
///
/// * base: `max_batch_size` live per-seq blobs (h_state + conv_state across
///   all SSM layers);
/// * spec: `mtp_state_slots` × (`num_drafts`+1 per-token intermediates
///   + 1 pre-verify checkpoint) blobs.
///
/// At `mtp_state_slots == max_batch_size` this reproduces the historical
/// `max_batch × blob × (1 + (num_drafts+1) + 1)` byte-for-byte.
pub fn ssm_pool_reserve_bytes(
    max_batch_size: usize,
    per_seq_blob_bytes: usize,
    spec_on: bool,
    num_drafts: usize,
    mtp_state_slots: usize,
) -> usize {
    let base = max_batch_size * per_seq_blob_bytes;
    if !spec_on {
        return base;
    }
    base + mtp_state_slots * per_seq_blob_bytes * (num_drafts + 2)
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

    /// bs=64 reserve-diet ledger, Qwen3.6-27B (config.json of
    /// centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf), `--max-seq-len 4096
    /// --num-drafts 3 --ssm-cache-slots 32 --speculative`, kv bf16.
    ///
    /// Per-seq SSM blob: 48 GDN layers × (h 48·128·128·4 B + conv
    /// (16·128·2 + 48·128)·4·4 B) = 48 × 3,309,568 = 158,859,264 B —
    /// the "158.9 MB" blob every campaign doc quotes.
    const BLOB: usize = 48 * (48 * 128 * 128 * 4 + (16 * 128 * 2 + 48 * 128) * 4 * 4);
    const ND: usize = 3; // --num-drafts 3 (K=4 ceiling)

    /// The historical formula this diet must reproduce at bs<=32:
    /// `max_batch × blob × (1 + (nd+1) + 1)`.
    fn legacy_pool_bytes(bs: usize, spec_on: bool) -> usize {
        let mult = if spec_on { 1 + (ND + 1) + 1 } else { 1 };
        bs * BLOB * mult
    }

    #[test]
    fn blob_matches_campaign_constant() {
        assert_eq!(BLOB, 158_859_264);
    }

    #[test]
    fn cap_identity_at_or_below_32_every_config() {
        // bs<=32 must be BYTE-IDENTICAL to the legacy sizing for every
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
            for spec_on in [false, true] {
                let slots = mtp_state_slots_with(bs, 32, false);
                assert_eq!(
                    ssm_pool_reserve_bytes(bs, BLOB, spec_on, ND, slots),
                    legacy_pool_bytes(bs, spec_on),
                    "bs={bs} spec={spec_on}: bs<=32 ledger must not move by a byte"
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
    fn bs64_ledger_before_after_and_fit() {
        // ── Pool term ──
        let old_pool = legacy_pool_bytes(64, true);
        assert_eq!(old_pool, 61_001_957_376); // 56.81 GiB
        let new_pool = ssm_pool_reserve_bytes(64, BLOB, true, ND, 32);
        assert_eq!(new_pool, 35_584_475_136); // 33.14 GiB (64 base + 32×5 spec blobs)
        assert_eq!(old_pool - new_pool, 25_417_482_240); // the diet: 23.67 GiB

        // ── Full inference reserve (mirrors preflight_reserve term-by-term) ──
        // snapshot: --ssm-cache-slots 32 × blob (decode ring skipped: spec on)
        let snapshot = 32 * BLOB; // 5_083_496_448
        // GDN two-phase chunked-prefill scratch: 4096 tokens ×
        // (conv_dim 10240×2 + nv 48×2×4 + value_dim 6144×2 + 6144×2) B/tok
        let gdn = 4096 * (10240 * 2 + 48 * 2 * 4 + 6144 * 2 + 6144 * 2);
        assert_eq!(gdn, 186_122_240);
        // CUDA headroom under spec
        let headroom = 4usize * 1024 * 1024 * 1024;

        let old_reserve = old_pool + snapshot + gdn + headroom;
        // = the EXACT 67297 MiB the wave-10 bs=64 refusal logged.
        assert_eq!(old_reserve, 70_566_543_360);
        assert_eq!(old_reserve / (1024 * 1024), 67_297);

        let new_reserve = new_pool + snapshot + gdn + headroom;
        assert_eq!(new_reserve, 45_149_061_120); // 42.05 GiB

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

        // Old reserve: refused with ~19 GiB overshoot before any KV.
        assert!(pre_kv + old_reserve > budget);
        // New reserve: boots, and the KV budget clears the workload floor.
        let kv_left = budget - pre_kv - new_reserve;
        assert!(
            kv_left >= kv_floor,
            "bs=64 KV budget {kv_left} must cover the decode_short peak {kv_floor}"
        );
        // Documented margin: ≥150 MiB over the dense worst case on the
        // worst logged box state (1.05 GiB on the wave-10 state); paged-KV
        // overcommit (default) back-pressures anything beyond it.
        assert!(kv_left - kv_floor >= 150 * 1024 * 1024);
    }
}
