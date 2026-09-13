// SPDX-License-Identifier: AGPL-3.0-only

//! Startup check that every rank agrees on the scalars that shape the COLLECTIVE SCHEDULE.
//!
//! 🔴 Some tuning levers are not perf knobs. `ATLAS_GLM_PREFILL_ROWS` is read independently on
//! each rank by [`crate::layers::glm5next_layer::prefill_rows`], and it decides how many
//! sub-chunks a prefill chunk is split into — i.e. **how many `reduce_partial` all-reduces the
//! chunk issues and how wide each one is**. A mismatch between ranks is therefore not a perf
//! skew: it is a mismatched collective schedule — a hang, or a reduce over the wrong extent.
//!
//! Until now the hazard was masked by the launch harness passing one env block to every rank.
//! That is the harness being careful, not the engine being safe; `ATLAS_EP_PROTOCOL` carried the
//! same exposure with only a doc comment ("both ranks must agree") behind it. This module makes
//! it the engine's problem: rank 0 broadcasts its values, every rank compares against its own,
//! and a disagreement bails at startup naming the offending lever.
//!
//! Cheap by construction — one broadcast of a handful of `u64`s, once, before the first token.
//!
//! 🪤 Levers that do **not** change the collective schedule do not belong here.
//! `ATLAS_GLM_MOE_ROW_BATCH_MAX` is included anyway because a rank skew there is still a
//! confusing perf asymmetry, and the check costs nothing — but it is a *correctness* gate only
//! for the schedule-shaping entries.

use anyhow::{Result, bail};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

/// Broadcast rank 0's `items` and bail if this rank's own values differ.
///
/// `items` is `(name, value)`; the name appears only in the error message. Single-rank runs and
/// an empty list are no-ops, so callers need no guard of their own.
///
/// 🪤 This is a **collective**: every rank must call it, with the same `items.len()`, at the same
/// point in construction. Both ranks run one `TransformerModel::new`, which is why the call sits
/// there and not behind a model-specific branch.
pub(crate) fn assert_ranks_agree(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    items: &[(&str, u64)],
) -> Result<()> {
    if comm.world_size() < 2 || items.is_empty() {
        return Ok(());
    }
    let bytes = items.len() * 8;
    let buf = gpu.alloc(bytes)?;

    let gathered = (|| -> Result<Vec<u64>> {
        let mut host: Vec<u8> = items.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
        gpu.copy_h2d(&host, buf)?;
        comm.broadcast(buf.0, bytes, 0)?;
        gpu.copy_d2h(buf, &mut host)?;
        Ok(host
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect())
    })();

    // Free on the error path too — the scratch is 8 bytes per item, but leaking it on a bail
    // would be exactly the D7 shape this campaign just wrote up.
    if let Err(e) = gpu.free(buf) {
        tracing::warn!("rank-agreement scratch free failed (non-fatal): {e}");
    }

    let root = gathered?;
    let rank = comm.rank();
    let bad: Vec<String> = items
        .iter()
        .zip(&root)
        .filter(|((_, mine), theirs)| mine != *theirs)
        .map(|((name, mine), theirs)| {
            format!("{name}: rank {rank} has {mine}, rank 0 has {theirs}")
        })
        .collect();
    if !bad.is_empty() {
        bail!(
            "ranks disagree on collective-shaping config — this is a hang or a wrong-extent \
             reduce, not a perf skew. Set the same value on EVERY rank: {}",
            bad.join("; ")
        );
    }
    tracing::info!(
        "rank-agreement OK on {} collective-shaping scalar(s): {}",
        items.len(),
        items
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The collective itself needs 2 ranks, so the unit tests cover the pure part: the
    //! comparison that turns a gathered root vector into an error. Kept as a mirror of the
    //! filter above so a change to the message shape breaks here first.

    fn mismatches(items: &[(&str, u64)], root: &[u64], rank: usize) -> Vec<String> {
        items
            .iter()
            .zip(root)
            .filter(|((_, mine), theirs)| mine != *theirs)
            .map(|((name, mine), theirs)| {
                format!("{name}: rank {rank} has {mine}, rank 0 has {theirs}")
            })
            .collect()
    }

    #[test]
    fn agreement_is_silent() {
        let items = [("ATLAS_GLM_PREFILL_ROWS", 8u64), ("ep_protocol_v2", 0)];
        assert!(mismatches(&items, &[8, 0], 1).is_empty());
    }

    #[test]
    fn a_single_disagreement_names_the_lever_and_both_values() {
        let items = [("ATLAS_GLM_PREFILL_ROWS", 4u64), ("ep_protocol_v2", 0)];
        let bad = mismatches(&items, &[8, 0], 1);
        assert_eq!(
            bad,
            vec!["ATLAS_GLM_PREFILL_ROWS: rank 1 has 4, rank 0 has 8"]
        );
    }

    #[test]
    fn every_disagreement_is_reported_not_just_the_first() {
        let items = [("ATLAS_GLM_PREFILL_ROWS", 4u64), ("ep_protocol_v2", 1)];
        assert_eq!(mismatches(&items, &[8, 0], 1).len(), 2);
    }
}
