// SPDX-License-Identifier: AGPL-3.0-only

//! The decode-rollback ring reserve term and its #915 auto-fit.
//!
//! A sibling of `preflight.rs` (which sits at the 500-line cap), following
//! the `per_sequence_state.rs` precedent: the term, the formula it prints and
//! the shrink decision live here; `preflight.rs` gains only the calls.
//!
//! # Why this term needed a fit and not just a number
//!
//! `ssm_snapshot_bytes` multiplies a CONSTANT ring depth (8) by
//! `--max-batch-size`, a flag that says nothing about what the card can hold.
//! Serving Qwen/Qwen3.8-27B-FP8 on one 80 GB H100 with the hopper recipe
//! (`--max-batch-size 32`, 48 GDN layers, 151.5 MiB per-seq state blob) that
//! is 8 x 32 x 151.5 MiB = 37.88 GiB of ring inside a 45,823 MiB inference
//! reserve, against a 71.3 GiB budget already carrying 57.2 GiB of weights —
//! and the serve REFUSED to boot (rental H100, 2026-09-05, evidence cell
//! `qwen38.atlas.a.lat.c1`). The shipped workaround was `--max-batch-size 4`
//! (cell `qwen38.atlas.c.lat.c1`, reserve 8.54 GiB, GO in 22 s): paying for
//! rollback depth with four fifths of the serve's concurrency.
//!
//! Ring depth degrades GRACEFULLY (fewer retained boundaries = fewer
//! reachable re-steer anchors; a sequence that finds none hard-stops through
//! `RollbackFallback::NoSsmSnapshot`, which is an honest decline, never a
//! partial SSM rewind). Batch does not: it is the serve's concurrency. So the
//! term that yields is the ring, and it yields down
//! `ssm_reserve::DECODE_RING_FIT_LADDER` until the whole reserve fits.

use atlas_core::config::ModelConfig;

use crate::cli;

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The ring depth preflight will reserve for, and the warning the operator
/// must see when it is not the depth the flags asked for.
pub(super) struct RingFit {
    pub(super) slots: usize,
    /// `Some` only when the auto-fit SHRANK the ring — logged at WARN by the
    /// caller, because a serve that quietly kept less rollback depth than its
    /// recipe records is a serve whose re-steer behaviour cannot be
    /// reproduced from the recipe.
    pub(super) warning: Option<String>,
}

/// The ring depth the flags and environment ask for, before any fit.
///
/// SSOT: `spark_model::ssm_reserve::decode_rollback_ring_slots` makes the
/// SAME decision (same published cell, same env vars, same constant) the
/// runtime allocation in `TransformerModel::new` makes — including the skip
/// under `--speculative`/`--dflash` (the ring's save/rollback path only runs
/// on plain decode; the spec path rolls back through the verify snapshot).
/// Reserving the ring unconditionally while the runtime skipped it stranded
/// ~38 GB at bs32 on the 27B and capped the native batch at ~20.
/// `use_speculative` here MUST mirror what `build_model` passes:
/// `args.speculative || args.dflash`.
///
/// Kill switch: `ATLAS_SSM_RESERVE_RING_FULL` present => restore the old
/// unconditional reservation (accounting-only, safe over-reserve;
/// presence-style — `=0` is NOT "off").
pub(super) fn requested_slots(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    if std::env::var("ATLAS_SSM_RESERVE_RING_FULL").is_ok() {
        return if config.num_ssm_layers() > 0 {
            atlas_kernels::DECODE_ROLLBACK_RING_SLOTS
        } else {
            0
        };
    }
    spark_model::ssm_reserve::decode_rollback_ring_slots(
        config.num_ssm_layers(),
        args.speculative || args.dflash,
    )
    .slots
}

/// Bytes ONE unit of ring depth costs: `--max-batch-size` x the per-sequence
/// SSM state blob. The same product `ssm_snapshot_bytes` multiplies the depth
/// by, factored out so the fit and the reserve cannot use different arithmetic.
pub(super) fn slot_bytes(args: &cli::ServeArgs, per_seq_blob: usize) -> usize {
    args.max_batch_size * per_seq_blob
}

/// `snapshots x batch x per-seq state bytes`, spelled out (issue #915's third
/// bullet: preflight must print the formula so an operator can see why the
/// reserve asked for what it asked for). SSOT for the INFO line, the shrink
/// WARN and the refusal text, so all three quote the same arithmetic.
pub(super) fn formula(slots: usize, max_batch: usize, per_seq_blob: usize) -> String {
    format!(
        "ring: {slots} slots x {max_batch} seqs x {:.1} MB/seq = {:.2} GB",
        per_seq_blob as f64 / MIB,
        (slots * max_batch * per_seq_blob) as f64 / GIB,
    )
}

/// Shrink the ring until the reserve fits, or leave it alone.
///
/// `reserve_without_ring` is the WHOLE reserve minus the ring term
/// (`inference_reserve + buffer_arena_bytes`), so the caller adds exactly
/// `slots * slot_bytes` back.
///
/// Leaves the depth untouched — returning no warning — when the reserve
/// already fits, when there is no ring to shrink, or when the depth is
/// EXPLICIT (`--ssm-decode-ring-slots N`, published before preflight runs):
/// an operator who named a depth gets that depth or a refusal, never a
/// silent third answer.
pub(super) fn autofit(
    args: &cli::ServeArgs,
    requested: usize,
    slot_bytes: usize,
    per_seq_blob: usize,
    reserve_without_ring: usize,
    free_mem: usize,
) -> RingFit {
    let asked = reserve_without_ring.saturating_add(requested.saturating_mul(slot_bytes));
    let explicit = spark_model::ssm_reserve::published_decode_ring_slots().is_some();
    if asked <= free_mem || requested == 0 || slot_bytes == 0 || explicit {
        return RingFit {
            slots: requested,
            warning: None,
        };
    }
    let fitted = spark_model::ssm_reserve::fit_decode_ring_slots(
        requested,
        reserve_without_ring,
        slot_bytes,
        free_mem,
    );
    let fitted_total = reserve_without_ring.saturating_add(fitted.saturating_mul(slot_bytes));
    if fitted_total > free_mem {
        // Even a ringless serve does not fit: the ring is not the problem, so
        // do not shrink it behind the operator's back — the caller refuses
        // and quotes the formula for the depth that was actually asked for.
        return RingFit {
            slots: requested,
            warning: None,
        };
    }
    // Published so `TransformerModel::new` allocates the SAME depth this
    // reserve was sized for. Both sides read the one cell; without this the
    // runtime would allocate 8 slots against a reserve that funded fewer.
    spark_model::ssm_reserve::set_decode_ring_slots(fitted);
    RingFit {
        slots: fitted,
        warning: Some(format!(
            "{} (was {:.2} GB); reserve {:.2} of {:.2} GB. Sized from free memory, not from \
             --max-batch-size {} (#915): rollback depth yields, concurrency does not. Pass \
             --ssm-decode-ring-slots N to pin a depth (and be refused rather than shrunk).",
            formula(fitted, args.max_batch_size, per_seq_blob),
            (requested * slot_bytes) as f64 / GIB,
            fitted_total as f64 / GIB,
            free_mem as f64 / GIB,
            args.max_batch_size,
        )),
    }
}

#[cfg(test)]
#[path = "decode_ring_tests.rs"]
mod tests;
