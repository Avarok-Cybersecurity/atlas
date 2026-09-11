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

use super::headroom::Yardstick;
use crate::cli;

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The ring depth preflight will reserve for, why, and the warning the
/// operator must see when it is not the depth the flags asked for.
pub(super) struct RingFit {
    pub(super) slots: usize,
    /// The fit decision with every term spelled out, logged at INFO on EVERY
    /// boot — including the boots where nothing was shrunk. #915's first pass
    /// was silent whenever it did not fire, which is precisely why two H100
    /// rounds could not tell "the fitter approved this" from "the fitter
    /// never looked" (`h100-round2-report.md`, stages 3 and 5).
    pub(super) decision: String,
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

/// Shrink the ring until it fits its yardstick, or leave it alone.
///
/// `reserve_without_ring` is the WHOLE reserve minus the ring term
/// (`inference_reserve + buffer_arena_bytes`), so the caller adds exactly
/// `slots * slot_bytes` back. `free_mem` is pre-load free memory. Which of
/// the two the ladder is actually measured against is
/// [`Yardstick::ladder_basis`]'s decision, and the second pass's whole point:
/// on the native-FP8 dense route the ring competes with the KV cache inside
/// the PREDICTED post-load headroom, not with the rest of the reserve inside
/// pre-load free memory (see `headroom.rs`).
///
/// Leaves the depth untouched — returning no warning — when it already fits,
/// when there is no ring to shrink, or when the depth is EXPLICIT
/// (`--ssm-decode-ring-slots N`, published before preflight runs): an
/// operator who named a depth gets that depth or a refusal, never a silent
/// third answer.
pub(super) fn autofit(
    args: &cli::ServeArgs,
    requested: usize,
    slot_bytes: usize,
    per_seq_blob: usize,
    reserve_without_ring: usize,
    free_mem: usize,
    yardstick: &Yardstick,
) -> RingFit {
    // The two impure halves, kept OUT of `fit_ring` so the ladder is testable
    // without a process-global `OnceLock` that one test would seal for the
    // whole binary.
    let explicit = spark_model::ssm_reserve::published_decode_ring_slots().is_some();
    let fit = fit_ring(
        args,
        requested,
        slot_bytes,
        per_seq_blob,
        reserve_without_ring,
        free_mem,
        yardstick,
        explicit,
    );
    if fit.slots != requested {
        // Published so `TransformerModel::new` allocates the SAME depth this
        // reserve was sized for. Both sides read the one cell; without this
        // the runtime would allocate 8 slots against a reserve that funded
        // fewer.
        spark_model::ssm_reserve::set_decode_ring_slots(fit.slots);
    }
    fit
}

/// Pure core of [`autofit`]: the ladder, the wording, and nothing that reads
/// or writes global state. `explicit` is
/// `published_decode_ring_slots().is_some()`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_ring(
    args: &cli::ServeArgs,
    requested: usize,
    slot_bytes: usize,
    per_seq_blob: usize,
    reserve_without_ring: usize,
    free_mem: usize,
    yardstick: &Yardstick,
    explicit: bool,
) -> RingFit {
    let (beside, limit) = yardstick.ladder_basis(reserve_without_ring, free_mem);
    let asked = beside.saturating_add(requested.saturating_mul(slot_bytes));
    let decide = |slots: usize| describe(args, yardstick, slots, slot_bytes, beside, limit);
    if requested == 0 || slot_bytes == 0 || explicit || asked <= limit {
        return RingFit {
            slots: requested,
            decision: decide(requested),
            warning: None,
        };
    }
    let fitted =
        spark_model::ssm_reserve::fit_decode_ring_slots(requested, beside, slot_bytes, limit);
    let fitted_total = beside.saturating_add(fitted.saturating_mul(slot_bytes));
    if fitted_total > limit && matches!(yardstick, Yardstick::PreLoadFree(_)) {
        // Even a ringless serve does not fit in FREE MEMORY: the ring is not
        // the problem, so do not shrink it behind the operator's back — the
        // caller refuses and quotes the formula for the depth actually asked
        // for. Not symmetric with the post-load arm on purpose: free memory
        // is measured and the refusal is sound, whereas the post-load
        // headroom is an ESTIMATE and must never be the thing that refuses a
        // boot. There, dropping to depth 0 hands the decision to the KV
        // budget stage, which measures.
        return RingFit {
            slots: requested,
            decision: decide(requested),
            warning: None,
        };
    }
    RingFit {
        slots: fitted,
        decision: decide(fitted),
        warning: Some(format!(
            "{} (was {:.2} GB); {}. Sized from {}, not from --max-batch-size {} (#915): \
             rollback depth yields, concurrency does not. Pass --ssm-decode-ring-slots N to \
             pin a depth (and be refused rather than shrunk).",
            formula(fitted, args.max_batch_size, per_seq_blob),
            (requested * slot_bytes) as f64 / GIB,
            if fitted_total > limit {
                format!(
                    "even 0 slots leaves {:.2} GB under the {:.2} GB it needs beside the KV \
                     floor — the KV budget stage will decide",
                    fitted_total as f64 / GIB,
                    limit as f64 / GIB,
                )
            } else {
                format!(
                    "{} {:.2} of {:.2} GB",
                    yardstick.total_label(),
                    fitted_total as f64 / GIB,
                    limit as f64 / GIB,
                )
            },
            yardstick.name(),
            args.max_batch_size,
        )),
    }
}

/// The INFO line: which yardstick, every term behind it, and what the ladder
/// concluded. #915's third bullet applied to the FIT rather than the reserve.
fn describe(
    args: &cli::ServeArgs,
    yardstick: &Yardstick,
    slots: usize,
    slot_bytes: usize,
    beside: usize,
    limit: usize,
) -> String {
    let ring_bytes = slots.saturating_mul(slot_bytes);
    match yardstick {
        Yardstick::PostLoad(h) => format!(
            "SSM decode-ring fit against the predicted post-load KV headroom: {}",
            h.describe(args.gpu_memory_utilization, ring_bytes, slots),
        ),
        Yardstick::PreLoadFree(why) => format!(
            "SSM decode-ring fit against pre-load free memory ({why}): ring({slots}) {:.2} + \
             rest of reserve {:.2} = {:.2} of {:.2} GB free",
            ring_bytes as f64 / GIB,
            beside as f64 / GIB,
            (beside + ring_bytes) as f64 / GIB,
            limit as f64 / GIB,
        ),
    }
}

#[cfg(test)]
#[path = "decode_ring_tests.rs"]
mod tests;
