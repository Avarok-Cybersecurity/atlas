// SPDX-License-Identifier: AGPL-3.0-only

//! A100 (2026-09-09): the Marconi SSM-snapshot restore decision must be
//! **rank-agreed**, not rank-local.
//!
//! F83 makes every rank agree on `matched_tokens`. Everything that came after
//! it in `prefix_lookup` was still decided per rank: whether the local
//! snapshot pool holds a checkpoint for the agreed prefix (pools evict
//! independently; the F83-capped rank re-looks-up a shorter prefix), the
//! checkpoint's depth, the `has_hidden` / session / aux gates. Under TP=2/EP=2
//! rank 1 restored a checkpoint at token 16336 (replay 20) while rank 0 had
//! none and recomputed from 0. Different `kv_write_start` → different
//! `proc_count` → the MoE collectives no longer pair up → both GPUs spin in
//! NCCL and the request never returns (observed twice, 4-slot campaign).
//!
//! The fix: every rank evaluates ALL of its local gates first and turns them
//! into a single **proposal** — the token depth `T` it can restore, or `0` for
//! RECOMPUTE. Proposals are exchanged with the same rooted-broadcast loop F83
//! uses, and the decision is all-or-nothing: RESTORE `T` iff every rank
//! proposed the very same nonzero `T`, else every rank recomputes. A naive
//! `min` would pick a token some rank does not own.

use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::super::types::TransformerModel;

/// Everything after F83 that can legitimately differ between ranks, folded
/// into plain values so the decision is testable without a GPU.
#[derive(Debug, Clone, Copy)]
pub(in crate::model) struct LocalGates {
    /// Effective depth of the local candidate snapshot; `0` when this rank
    /// holds no snapshot for the agreed prefix.
    pub snap_tok: usize,
    /// F83-agreed matched prefix length.
    pub matched: usize,
    /// Prompt length.
    pub total: usize,
    /// `marconi_min_tokens()` — below it a restore costs more than it saves.
    pub min_tokens: usize,
    /// The candidate slot carries a last-token hidden (leaf snapshot).
    pub has_hidden: bool,
    /// `ATLAS_MARCONI_EXACT=1` (exact full-prompt shortcut re-enabled).
    pub exact_enabled: bool,
    /// The candidate is a tail snapshot (state bleeds past the prefix).
    pub is_tail: bool,
    /// The candidate's session tag matches this sequence.
    pub session_ok: bool,
    /// The model carries per-sequence aux state (DSA indexer) …
    pub needs_aux: bool,
    /// … and the candidate slot stored aux blobs.
    pub has_aux: bool,
}

/// This rank's vote: the snapshot token it is able to restore, or `0`.
///
/// Mirrors the gate chain that used to guard `restore()` directly. Every
/// clause is evaluated here, BEFORE the collective, so no rank can decline
/// after the agreement point.
pub(in crate::model) fn local_proposal(g: &LocalGates) -> u32 {
    if g.snap_tok == 0 {
        return 0;
    }
    let exact = g.snap_tok == g.matched && g.matched == g.total;
    // Exact hit on a hiddenless snapshot cannot produce the first token.
    let exact_without_hidden = exact && !g.has_hidden;
    // The exact-leaf shortcut is unsound by construction (see prefix_lookup);
    // bypassed unless explicitly re-enabled for A/B.
    let bypass_exact = exact && !g.exact_enabled;
    let ok = g.snap_tok >= g.min_tokens
        && g.matched <= g.total
        && !exact_without_hidden
        && !bypass_exact
        && (!g.is_tail || g.session_ok)
        && (!g.needs_aux || g.has_aux);
    if ok { g.snap_tok as u32 } else { 0 }
}

/// All-or-nothing agreement over every rank's proposal (own vote included):
/// `Some(T)` iff every rank proposed the same nonzero `T`.
pub(in crate::model) fn agree(proposals: &[u32]) -> Option<u32> {
    let first = *proposals.first()?;
    (first != 0 && proposals.iter().all(|&p| p == first)).then_some(first)
}

/// The skip point the suffix prefill resumes from, given the agreed decision.
/// Pure function of rank-agreed inputs, so the replay length
/// (`total - skip_point`) is identical on every rank by construction.
pub(in crate::model) fn skip_point(
    skip: bool,
    snap_tok: usize,
    matched: usize,
    total: usize,
    has_ssm: bool,
) -> usize {
    if skip && !has_ssm {
        matched
    } else if skip && matched == total && snap_tok == matched {
        matched
    } else if skip {
        snap_tok
    } else {
        0
    }
}

/// Collect one `u32` from every rank of `comm` via `world` rooted
/// broadcasts (root `r` contributes element `r`). Every rank returns the same
/// vector, so any pure function of it is a rank-agreed decision. This is the
/// F83 wire schedule — `ep_min_u32` is its `min`.
///
/// `buf` is a rank-local 4-byte device buffer the broadcasts go through.
pub(in crate::model) fn gather_u32_via_broadcast(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    buf: DevicePtr,
    world: usize,
    val: u32,
) -> Result<Vec<u32>> {
    let stream = gpu.default_stream();
    let mut out = Vec::with_capacity(world);
    for root in 0..world {
        let v = if comm.rank() == root {
            gpu.copy_h2d(&val.to_le_bytes(), buf)?;
            comm.broadcast(buf.0, 4, root)?;
            val
        } else {
            comm.broadcast(buf.0, 4, root)?;
            gpu.synchronize(stream)?;
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(buf, &mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        out.push(v);
    }
    Ok(out)
}

impl TransformerModel {
    /// Every rank's `val`, indexed by rank. Single-rank: `vec![val]`.
    pub(in crate::model) fn ep_gather_u32(&self, val: u32) -> Result<Vec<u32>> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(vec![val]);
        };
        // Loop over the ranks of the ACTUAL communicator (pure TP has
        // `ep_world_size == 1` but a `tp_world_size`-wide comm).
        let world = self.config.ep_world_size.max(self.config.tp_world_size);
        gather_u32_via_broadcast(
            self.gpu.as_ref(),
            comm.as_ref(),
            self.ep_cmd_buf,
            world,
            val,
        )
    }
}
