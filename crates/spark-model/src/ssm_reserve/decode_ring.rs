// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT for the Phase-C decode-rollback ring DEPTH.
//!
//! A sibling of `ssm_reserve.rs` (which sits over the 500-line cap),
//! following the `per_sequence_state.rs` precedent: the depth decision, its
//! publication cell and the #915 auto-fit live here.
//!
//! Two call sites MUST agree on this number or a serve either under-reserves
//! (runtime CUDA alloc failure after weights load) or over-reserves
//! (preflight refuses batch sizes the runtime could fund):
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
//! ## Why the depth is now sized from free memory (#915)
//!
//! The depth was a CONSTANT (8) multiplied by `--max-batch-size`, so it grew
//! with a flag that says nothing about what the card can hold. Serving
//! Qwen3.8-27B-FP8 on one 80 GB H100 with the hopper recipe
//! (`--max-batch-size 32`, 48 GDN layers, 151.5 MiB per-seq state blob) asked
//! for 8 × 32 × 151.5 MiB = 37.88 GiB of ring alone — an inference reserve of
//! 45,823 MiB against a 71.3 GiB budget with 57.2 GiB of weights — and the
//! serve REFUSED to boot (rental H100, 2026-09-05). The recipe workaround was
//! `--max-batch-size 4`, i.e. paying for rollback depth with four fifths of
//! the serve's concurrency.
//!
//! The ring degrades gracefully with depth (fewer retained boundaries = fewer
//! reachable re-steer anchors, never wrong output), so preflight now SHRINKS
//! it down the [`DECODE_RING_FIT_LADDER`] until the reserve fits and publishes
//! the chosen depth through [`set_decode_ring_slots`], instead of refusing.
//! Refusal is kept only for the case where even depth 0 does not fit.
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
/// override, a published `--ssm-decode-ring-slots N` or the #915 auto-fit —
/// so the allocating call site can log the savings once.
pub struct DecodeRingDecision {
    pub slots: usize,
    pub skip_reason: Option<&'static str>,
}

/// Depths preflight's #915 auto-fit may choose from, LARGEST FIRST.
///
/// Halving rather than decrementing: each rung halves the ring's share of
/// the reserve, so at most four steps separate "the default" from "no ring",
/// and every rung is a power of two — the ring's slot assignment is
/// `(next_slot + 1) % capacity` (`SsmDecodeRing::record`), which is exact at
/// any capacity but wraps most evenly at these.
///
/// 8 is [`atlas_kernels::DECODE_ROLLBACK_RING_SLOTS`] and covers the
/// 3-repeat fuzzy loop detector with margin; 1 still anchors a single clean
/// boundary; 0 means the sequence hard-stops instead of re-steering
/// (`RollbackFallback::NoSsmSnapshot` — an honest decline, never a partial
/// rewind).
pub const DECODE_RING_FIT_LADDER: [usize; 5] = [8, 4, 2, 1, 0];

/// The depth published by the serve command line (`--ssm-decode-ring-slots
/// N`) or by preflight's auto-fit. Written once, read by BOTH sizing call
/// sites — same first-write-wins cell pattern as
/// [`super::set_ssm_rollback_mode`].
///
/// Absent is NOT a value: `--ssm-decode-ring-slots auto` publishes nothing,
/// so the documented `ATLAS_SSM_DECODE_RING` fallback stays reachable and
/// preflight can publish the auto-fit depth later in the same boot.
static DECODE_RING_SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Publish the decode-ring depth. Returns the value in force (first write
/// wins, matching `gdn_flags::set_from_cli`): an explicit
/// `--ssm-decode-ring-slots N` is published before preflight runs, so the
/// auto-fit's later write is a no-op and an operator's explicit depth is
/// never silently shrunk.
pub fn set_decode_ring_slots(slots: usize) -> usize {
    let _ = DECODE_RING_SLOTS.set(slots);
    *DECODE_RING_SLOTS.get().expect("just set")
}

/// The published depth, or `None` when nothing has been published.
///
/// Does NOT initialize the cell — preflight asks this to tell an EXPLICIT
/// depth (auto-fit disabled, refuse as before) from `auto` (auto-fit
/// allowed), and asking must not itself seal the decision.
pub fn published_decode_ring_slots() -> Option<usize> {
    DECODE_RING_SLOTS.get().copied()
}

/// SSOT parse for the `--ssm-decode-ring-slots` value: `auto` (`None` — size
/// it from free memory at preflight) or an explicit `0..=8`.
///
/// `validate_serve_args` and `publish_kernel_flags` both go through THIS, so
/// what the CLI accepts and what it publishes cannot drift.
pub fn parse_decode_ring_slots(s: &str) -> Result<Option<usize>, String> {
    if s == "auto" {
        return Ok(None);
    }
    let n: usize = s
        .parse()
        .map_err(|_| format!("unknown ssm-decode-ring-slots '{s}' (valid: auto, 0..=8)"))?;
    if n > atlas_kernels::DECODE_ROLLBACK_RING_SLOTS {
        return Err(format!(
            "ssm-decode-ring-slots {n} exceeds the {} the ring is sized for",
            atlas_kernels::DECODE_ROLLBACK_RING_SLOTS
        ));
    }
    Ok(Some(n))
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
        published_decode_ring_slots(),
        ring_override.as_deref(),
        watchdogs_disabled,
    )
}

/// `ATLAS_DISABLE_WATCHDOGS` truthiness — trimmed, case-insensitive, `1` or
/// `true` only (mirrors spark-server's `parse_disable_watchdogs`).
pub fn watchdogs_disabled_from_value(value: Option<&str>) -> bool {
    value
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true"
        })
        .unwrap_or(false)
}

/// Pure core of [`decode_rollback_ring_slots`] (env-free, unit-testable).
///
/// Precedence, highest first:
///
/// 1. no SSM layers — there is no recurrent state to snapshot;
/// 2. `published` — `--ssm-decode-ring-slots N`, or the depth preflight's
///    auto-fit chose. It outranks the env override AND the implicit skips for
///    the same reason `ATLAS_SSM_DECODE_RING=1` always has: an operator (or a
///    reserve that has already been sized against free memory) asking for a
///    specific depth must get that depth on BOTH sides, or the two diverge;
/// 3. `ATLAS_SSM_DECODE_RING=1|0` — the legacy spelling of "8" and "0";
/// 4. the implicit skips (speculative decode, watchdogs off);
/// 5. the default depth.
pub fn decode_rollback_ring_slots_with(
    num_ssm_layers: usize,
    use_speculative: bool,
    published: Option<usize>,
    ring_override: Option<&str>,
    watchdogs_disabled: bool,
) -> DecodeRingDecision {
    if num_ssm_layers == 0 {
        return DecodeRingDecision {
            slots: 0,
            skip_reason: None,
        };
    }
    if let Some(slots) = published {
        return DecodeRingDecision {
            slots,
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

/// Largest [`DECODE_RING_FIT_LADDER`] depth `<= start_slots` whose ring term
/// still fits in `free_mem` alongside everything else the reserve needs
/// (#915). Pure — preflight owns the bytes, this owns the ladder.
///
/// `bytes_per_ring_slot` is ONE depth unit: `max_batch_size × the per-seq SSM
/// state blob`, the same product `ssm_snapshot_bytes` multiplies the depth by.
///
/// Returns 0 when nothing fits — including the case where the rest of the
/// reserve ALONE exceeds `free_mem`, which is the caller's cue to refuse with
/// the existing suggested-batch text rather than to boot a ringless serve.
pub fn fit_decode_ring_slots(
    start_slots: usize,
    reserve_without_ring: usize,
    bytes_per_ring_slot: usize,
    free_mem: usize,
) -> usize {
    DECODE_RING_FIT_LADDER
        .iter()
        .copied()
        .filter(|&slots| slots <= start_slots)
        .find(|&slots| {
            reserve_without_ring.saturating_add(slots.saturating_mul(bytes_per_ring_slot))
                <= free_mem
        })
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "decode_ring_tests.rs"]
mod decode_ring_tests;
