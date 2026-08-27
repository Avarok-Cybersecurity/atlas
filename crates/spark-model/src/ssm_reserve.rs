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
/// pool slot `slot_idx`: exactly the slot's draft capacity (K-1 snapshots
/// for a K-row verify). `uniform_verify` (DFlash-γ pools, whose verify
/// width does not follow the MTP ladder) sizes every slot at the full
/// `num_drafts`.
///
/// WHY K-1 and not K (2026-08-16 audit): no verify arm ever writes OR
/// reads H intermediate index K-1. The fused WY kernels write
/// Hi_0..Hi_{K-2} plus the final H in place (`gdn_decode_wy{2,3,4}`,
/// `wyn`/`wy17`, the strided `_snap` twins NULL-skip index K-1), the
/// single-seq K=2/3/4 arms and the exact arm skip the dead snapshot
/// explicitly, and the sequential fallback now skips t = K-1 too. Every
/// reader is bounded at index K-2: `commit_accepted_prefix` pins the
/// reachable index to [0, k-2], `rollback_ssm_states` validates against
/// the vec length with callers guaranteeing a rejected draft, and
/// `start_rollback_and_checkpoint_async` is only called with 1..=K-1
/// (index ≤ K-2). See the reader enumeration in
/// `trait_decode_batched_conv_gdn.rs`.
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
        return num_drafts;
    }
    verify_slot_drafts(slot_idx, num_drafts)
}

/// Storage width of one h-state blob in the SSM state pools (stage 3 of
/// `--ssm-h-dtype f16`): 2 bytes per element under the f16-SIZED pool, the
/// FP32 4 bytes otherwise. SSOT — `SsmStatePool::new` (allocation strides),
/// `preflight_reserve` (the pre-load reserve) and every byte-copier that
/// moves h-state between pool regions derive their width from THIS, so
/// sizing and copies cannot disagree.
///
/// `f16_pool` is `gdn_flags::ssm_h_f16_pool_enabled()` at the production
/// call sites (`--ssm-h-dtype f16-pool`), passed as a parameter so pool
/// construction and sizing stay testable without the process-global flag
/// cell. NOTE stage 1/2 (`--ssm-h-dtype f16`) deliberately keep the pool
/// FP32-SIZED (`f16_pool = false`): the state bits are FP16 during decode
/// but prefill still writes FP32 in place, so the slot must stay wide.
pub fn ssm_h_stored_bytes(h_f32_bytes: usize, f16_pool: bool) -> usize {
    assert!(
        h_f32_bytes.is_multiple_of(4),
        "h-state blobs are FP32-element sized"
    );
    if f16_pool {
        h_f32_bytes / 2
    } else {
        h_f32_bytes
    }
}

/// FP32 h-state PREFILL STAGING bytes (stage 3 of `--ssm-h-dtype f16`).
///
/// Under the f16-SIZED pool a slot's h region is 2 bytes/element, but every
/// GDN prefill kernel family reads and writes the running h-state as FP32 in
/// place — over a 2-byte slot that is an overrun into the neighbouring slot.
/// Stage 3 therefore gives each pool slot ONE FP32 staging blob, and the
/// layer widens the slot into it before its FP32 kernels run and narrows it
/// back after (`ssm_h_fp16::prefill_h_begin` / `prefill_h_end`).
///
/// ★ ONE blob per SLOT, **not** per slot per layer. The staging blob is live
/// only for the duration of one SSM layer's prefill call: the layers of a
/// pass are issued in order on a single stream, each narrowing back before
/// the next widens, so layer L+1 reuses layer L's blob. Sizing it per slot
/// (rather than per concurrently-prefilling sequence) is what makes that
/// safe without knowing the co-dispatch width: a sequence owns exactly one
/// slot for its whole life, so two sequences can never share a blob.
///
/// `h_layer_f32_bytes` is ONE layer's FP32 h blob (`ssm_h_state_bytes()`) —
/// NOT the across-layers per-seq total the pool-reserve terms use. Zero when
/// the pool is FP32-sized: prefill then writes the slot in place as it
/// always has, and no staging exists to reserve.
///
/// SSOT for both `SsmStatePool::new` (which allocates it, passing
/// `max_slots + 1` for the dummy slot) and the preflight reserve (which
/// passes `max_batch_size`, matching its standing convention of not
/// counting the dummy — the CUDA headroom term absorbs it).
pub fn ssm_h_prefill_stage_bytes(slots: usize, h_layer_f32_bytes: usize, f16_pool: bool) -> usize {
    if f16_pool {
        slots * h_layer_f32_bytes
    } else {
        0
    }
}

/// SSM state-pool reserve bytes for the pre-load preflight — MUST mirror
/// what `SsmStatePool::new` allocates (modulo the +1 dummy slot per pool,
/// which preflight has never counted; the CUDA headroom term absorbs it):
///
/// * base: `max_batch_size` live per-seq blobs (h_state + conv_state across
///   all SSM layers);
/// * spec, per verify slot (`mtp_state_slots` of them):
///   - H intermediates: [`verify_slot_h_intermediates`] × h blob (TIERED,
///     and K-1 per K-row verify — index K-1 is never written or read);
///   - conv intermediates: `num_drafts + 1` × conv blob (uniform AND still
///     K — the fused conv kernels write all K snapshots on-device; see
///     [`verify_slot_h_intermediates`] for why conv does not tier);
///   - 1 pre-verify checkpoint blob (h + conv).
///
/// `h_blob_bytes` / `conv_blob_bytes` are the per-seq totals across all SSM
/// layers (`num_ssm_layers × ssm_h_state_bytes/ssm_conv_state_bytes`),
/// ALWAYS at the FP32 width — `h_f16_pool` narrows every h term through
/// [`ssm_h_stored_bytes`] inside, so preflight and `SsmStatePool::new`
/// cannot narrow differently.
/// The historical sizing was `max_batch × blob × (1 + (num_drafts+1) + 1)`;
/// today's uniform mode differs from it by exactly one h blob per slot
/// (the dead K-1 intermediate).
pub fn ssm_pool_reserve_bytes(
    max_batch_size: usize,
    h_blob_bytes: usize,
    conv_blob_bytes: usize,
    spec_on: bool,
    num_drafts: usize,
    mtp_state_slots: usize,
    uniform_verify: bool,
    h_f16_pool: bool,
    rollback: SsmRollbackMode,
) -> usize {
    let h_blob_bytes = ssm_h_stored_bytes(h_blob_bytes, h_f16_pool);
    let blob = h_blob_bytes + conv_blob_bytes;
    let base = max_batch_size * blob;
    if !spec_on {
        return base;
    }
    let verify: usize = (0..mtp_state_slots)
        .map(|slot| match rollback {
            SsmRollbackMode::Snapshot => {
                verify_slot_h_intermediates(slot, num_drafts, uniform_verify) * h_blob_bytes
                    + (num_drafts + 1) * conv_blob_bytes
                    + blob
            }
            // Replay keeps ONLY the pre-verify checkpoint blob per slot —
            // partial accepts are reconstructed by replaying the accepted
            // tokens from it, so no per-token h/conv snapshots exist. The
            // verify-window input ring is a SEPARATE term
            // ([`ssm_replay_ring_bytes`]) because it is sized by activation
            // rows, not state blobs.
            SsmRollbackMode::Replay => blob,
        })
        .sum();
    base + verify
}

/// SSM verify-rollback mode (`--ssm-rollback-mode`, EXPERIMENTAL scaffold).
///
/// * `Snapshot` (the serve default, explicit in the CLI): every verify arm
///   writes per-token h/conv state snapshots; a partial accept restores from
///   `intermediates[num_accepted - 1]`. This is the only mode whose device
///   path is wired — its sizing and behavior are pinned byte-for-byte.
/// * `Replay`: keep ONLY the pre-verify checkpoint blob per verify slot and
///   cache the verify window's per-token GDN INPUTS (the deinterleaved qkvz
///   row each conv1d consumes plus the gate/beta row — the tensors the WY
///   verify kernels read) in a small ring; a partial accept re-runs the
///   accepted tokens from the checkpoint through the existing sequential
///   recurrent path. Device wiring (capture + replay) is NOT implemented:
///   a serve in this mode boots — the reserve shows the capacity win — and
///   every speculative verify entry refuses loudly
///   (`SsmStatePool::require_verify_rollback_supported`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsmRollbackMode {
    Snapshot,
    Replay,
}

impl std::str::FromStr for SsmRollbackMode {
    type Err = String;
    /// SSOT parse for the `--ssm-rollback-mode` value (CLI validation and
    /// the serve publication both go through this).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "snapshot" => Ok(Self::Snapshot),
            "replay" => Ok(Self::Replay),
            other => Err(format!(
                "unknown ssm-rollback-mode '{other}' (valid: snapshot, replay)"
            )),
        }
    }
}

/// The published rollback mode. Written once from the serve command line
/// (which carries an EXPLICIT `default_value = "snapshot"`), read by pool
/// construction and preflight. Same first-write-wins cell pattern as
/// `gdn_flags`.
static ROLLBACK_MODE: std::sync::OnceLock<SsmRollbackMode> = std::sync::OnceLock::new();

/// Publish the command line's mode. Returns the value in force (first
/// write wins, matching `gdn_flags::set_from_cli`).
pub fn set_ssm_rollback_mode(mode: SsmRollbackMode) -> SsmRollbackMode {
    let _ = ROLLBACK_MODE.set(mode);
    *ROLLBACK_MODE.get().expect("just set")
}

/// The mode in force. `Snapshot` when nothing was published — mirroring the
/// CLI's explicit default for non-serve contexts (tests, examples), which
/// never carry the flag. Production sizing/pool call sites take the mode as
/// a PARAMETER and read this only at the outermost boundary, so unit tests
/// never depend on the process-global cell.
pub fn ssm_rollback_mode() -> SsmRollbackMode {
    *ROLLBACK_MODE.get_or_init(|| SsmRollbackMode::Snapshot)
}

/// One cached verify-row of GDN inputs for replay, per SSM layer: the
/// deinterleaved qkvz row (`qkvz_elems` BF16 — what conv1d consumes; Z
/// included, the gated norm needs it) + the gate/beta row (`nv * 2` FP32).
/// These are exactly the per-token tensors the WY verify kernels read
/// (`ConvGdnArgs::deinterleaved` / `gates_buf` rows), and re-running them
/// through the sequential conv+GDN path from the checkpoint reproduces the
/// snapshot the dropped intermediates used to hold.
pub fn ssm_replay_row_bytes(qkvz_elems: usize, nv: usize) -> usize {
    qkvz_elems * 2 + nv * 2 * 4
}

/// Replay-mode verify-window input ring: `k_ceiling - 1` cached rows per
/// covered slot per SSM layer (a partial accept replays at most K-1 tokens
/// — rows 0..K-2; a full accept replays nothing). Reserved by preflight and
/// allocated by `SsmStatePool::new` through THIS function so the two cannot
/// disagree. Zero when speculation is off or the mode is `Snapshot`.
pub fn ssm_replay_ring_bytes(
    num_ssm_layers: usize,
    row_bytes: usize,
    k_ceiling: usize,
    mtp_state_slots: usize,
) -> usize {
    mtp_state_slots * k_ceiling.saturating_sub(1) * num_ssm_layers * row_bytes
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
    let watchdogs_value = std::env::var("ATLAS_DISABLE_WATCHDOGS").ok();
    let watchdogs_disabled = watchdogs_disabled_from_value(watchdogs_value.as_deref());
    let ring_override = std::env::var("ATLAS_SSM_DECODE_RING").ok();
    decode_rollback_ring_slots_with(
        num_ssm_layers,
        use_speculative,
        ring_override.as_deref(),
        watchdogs_disabled,
    )
}

fn watchdogs_disabled_from_value(value: Option<&str>) -> bool {
    value
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true"
        })
        .unwrap_or(false)
}

fn decode_rollback_ring_slots_with(
    num_ssm_layers: usize,
    use_speculative: bool,
    ring_override: Option<&str>,
    watchdogs_disabled: bool,
) -> DecodeRingDecision {
    if num_ssm_layers == 0 {
        return DecodeRingDecision {
            slots: 0,
            skip_reason: None,
        };
    }
    match ring_override {
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
#[path = "ssm_reserve_tests.rs"]
mod mtp_state_slot_tests;
