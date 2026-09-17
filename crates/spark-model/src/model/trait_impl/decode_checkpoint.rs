// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use avarok_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// A109 (2026-09-16): EP worker command — "save the decode-time Marconi
/// checkpoint rank 0 just saved", at the same `(slot, token, session)`.
///
/// **Invariant: every rank saves the same `(slot, token, session)` decode
/// checkpoint, or none does.** Before this command the head saved decode
/// checkpoints from a bare local call in scheduler-only code and the EP wire
/// protocol had no counterpart, so under TP=2/EP=2 rank 1 never held them
/// (HANDOFF-30 §6 — accidental, scheduler-sited). The A100 rank-agreed
/// restore then correctly refused every restore only one rank could serve,
/// and the head's extra snapshots evicted its own prefill checkpoints.
///
/// Wire shape: the `(seq_id, cmd)` preamble, then ONE bulk broadcast of
/// [`EP_CKPT_WORDS`] u32 — see [`encode_ckpt_payload`].
///
/// A113 (2026-09-16): moved from `0xFFFF_FFF6` to `0xFFFF_FFF8` — that value
/// was independently assigned on the DFlash lane (`EP_CMD_VERIFY_KGAMMA`,
/// `speculative.rs`), and the two lanes never built against each other until
/// campaign integration surfaced the collision as an `unreachable_patterns`
/// compile error, not a textual merge conflict.
pub(in crate::model) const EP_CMD_DECODE_CKPT: u32 = 0xFFFF_FFF8;

// Opcode band, checked at compile time. Worker commands must sit ABOVE the
// token range (`0..=0xFFFF_FFEF` is dispatched as a decode token id) and must
// not collide with a code already on the wire: F0 prefill chunk, F1
// alloc-slot, F2/F3/F4 verify K=2/3/4, F5 MTP propose, F6/F7 reserved for the
// DFlash lane, FF shutdown, E0 batched decode (matched before the token
// fallthrough).
const _: () = assert!(
    EP_CMD_DECODE_CKPT > 0xFFFF_FFEF,
    "EP_CMD_DECODE_CKPT would be dispatched as a decode token id"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFE0,
    "collides with batched decode"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF0,
    "collides with prefill chunk"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF1,
    "collides with alloc-slot"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF2,
    "collides with verify K=2"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF3,
    "collides with verify K=3"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF4,
    "collides with verify K=4"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != crate::speculative::EP_CMD_MTP_PROPOSE,
    "collides with MTP propose"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF6,
    "reserved: DFlash EP_CMD_VERIFY_KGAMMA (A113)"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF7,
    "reserved: DFlash ctx-commit (A113)"
);
const _: () = assert!(EP_CMD_DECODE_CKPT != 0xFFFF_FFFF, "collides with shutdown");

/// Payload width of [`EP_CMD_DECODE_CKPT`], in u32 words.
pub(in crate::model) const EP_CKPT_WORDS: usize = 6;

/// What a decode checkpoint covers: the registered token count and the
/// block-table prefix it was taken over. Rank-agreed by construction — the
/// head computes it, every worker receives it verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model) struct CkptPlan {
    /// TRUE coverage of the saved state (`seq.tokens.len()`), which under MTP
    /// can exceed `end_block * block_size` by 1..=bs+2 — see the note in
    /// `decode_ckpt_save_and_register`.
    pub snap_tokens: usize,
    /// Complete KV blocks the checkpoint spans.
    pub end_block: usize,
}

/// Everything the fire/skip decision reads, as plain values — so the decision
/// (and therefore "was an EP command emitted at all?") is testable with no
/// GPU and no container.
#[derive(Debug, Clone, Copy)]
pub(in crate::model) struct CkptInputs {
    /// `ssm_snapshots.is_enabled() && prefix_cache.is_active()` — the
    /// prefix-cache enable predicate (`AVAROK_GLM53_PREFIX_CACHE_UNPROVEN` +
    /// `--enable-prefix-caching`). FALSE ⇒ no save, and no EP command.
    pub enabled: bool,
    pub num_ssm_layers: usize,
    pub hss_window_start: usize,
    pub slot_idx: usize,
    pub tokens_len: usize,
    pub block_size: usize,
    pub block_table_len: usize,
    pub last_ckpt_block: usize,
    /// `AVAROK_DECODE_CKPT_BLOCKS`, already defaulted.
    pub interval: usize,
}

/// The cheap half of [`decode_ckpt_plan`], split out so the hot decode path
/// can bail before reading the env var or taking the KV lock — and so the two
/// cannot drift apart.
pub(in crate::model) fn ckpt_preconditions(
    enabled: bool,
    num_ssm_layers: usize,
    hss_window_start: usize,
    slot_idx: usize,
) -> bool {
    enabled && num_ssm_layers != 0 && hss_window_start == 0 && slot_idx != usize::MAX
}

/// The rank-0 fire/skip decision, as a pure function. `None` ⇒ nothing is
/// saved and no [`EP_CMD_DECODE_CKPT`] goes on the wire.
///
/// The vision-pad veto is deliberately NOT here: it scans the whole token
/// slice, so the caller applies it after this (cheap) decision says "fire".
pub(in crate::model) fn decode_ckpt_plan(i: &CkptInputs) -> Option<CkptPlan> {
    if !ckpt_preconditions(i.enabled, i.num_ssm_layers, i.hss_window_start, i.slot_idx) {
        return None;
    }
    if i.block_size == 0 || i.interval == 0 {
        return None;
    }
    // Derive the block count from tokens.len() (what we slice + cache), NOT
    // seq_len: under MTP seq_len can transiently exceed tokens.len() (verify
    // bonus position), which would overrun the token slice.
    let end_block = i.tokens_len / i.block_size;
    if end_block == 0
        || !end_block.is_multiple_of(i.interval)
        || end_block == i.last_ckpt_block
        // Only checkpoint blocks that physically exist. NOTE: the prefill-era
        // `kv_valid_tokens` guard does NOT apply here — that field tracks the
        // contiguous KV-written prefix during PREFILL and is never advanced by
        // decode, so it would wrongly veto every decode checkpoint past the
        // prompt length. During decode each token writes its KV inline, so all
        // `end_block` complete blocks are fully written.
        || end_block > i.block_table_len
    {
        return None;
    }
    Some(CkptPlan {
        snap_tokens: i.tokens_len,
        end_block,
    })
}

/// Encode the [`EP_CMD_DECODE_CKPT`] payload: plan + the head's session and
/// adapter identity, little-end word first for each u64.
pub(in crate::model) fn encode_ckpt_payload(
    plan: CkptPlan,
    session_hash: u64,
    adapter_id: u64,
) -> [u32; EP_CKPT_WORDS] {
    [
        plan.snap_tokens as u32,
        plan.end_block as u32,
        session_hash as u32,
        (session_hash >> 32) as u32,
        adapter_id as u32,
        (adapter_id >> 32) as u32,
    ]
}

/// Inverse of [`encode_ckpt_payload`]. Returns `(plan, session_hash,
/// adapter_id)` exactly as rank 0 sent them.
pub(in crate::model) fn decode_ckpt_payload(w: &[u32]) -> Result<(CkptPlan, u64, u64)> {
    if w.len() != EP_CKPT_WORDS {
        bail!(
            "EP_CMD_DECODE_CKPT: payload is {} words, expected {EP_CKPT_WORDS}",
            w.len()
        );
    }
    Ok((
        CkptPlan {
            snap_tokens: w[0] as usize,
            end_block: w[1] as usize,
        },
        w[2] as u64 | ((w[3] as u64) << 32),
        w[4] as u64 | ((w[5] as u64) << 32),
    ))
}

impl TransformerModel {
    /// #155 iter3: save a block-aligned Marconi SSM snapshot DURING decode so
    /// the next turn's warm hit restores from decode-produced state near the
    /// conversation end (no prefill-replay of decode tokens). Mirrors
    /// `prefill_b_save_checkpoint` but keyed on `seq.seq_len` / generated
    /// tokens. Fires once per `interval`-block boundary; the 16-slot pool's
    /// LRU keeps the most-recent window, so the deepest snapshot ≤ next-turn
    /// matched is within `interval` blocks → tiny replay tail. Live SSM state
    /// must be canonical at call time (post-commit on the MTP path).
    pub(super) fn decode_marconi_checkpoint_dispatch(&self, seq: &mut SequenceState) {
        let enabled = self.ssm_snapshots.is_enabled() && self.prefix_cache.is_active();
        // Cheap half first: this runs on EVERY decode step, and neither the
        // env read nor the KV lock below should be paid by a model that can
        // never checkpoint.
        if !ckpt_preconditions(
            enabled,
            self.config.num_ssm_layers(),
            seq.hss_window_start(),
            seq.slot_idx,
        ) {
            return;
        }
        // Block-count between decode checkpoints. Env-tunable (no rebuild) so
        // the cadence/drift tradeoff can be swept; default 4 blocks = 64 tok.
        let interval = std::env::var("AVAROK_DECODE_CKPT_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);
        let block_size = self.kv_cache.lock().block_size();
        let Some(plan) = decode_ckpt_plan(&CkptInputs {
            enabled,
            num_ssm_layers: self.config.num_ssm_layers(),
            hss_window_start: seq.hss_window_start(),
            slot_idx: seq.slot_idx,
            tokens_len: seq.tokens.len(),
            block_size,
            block_table_len: seq.block_table.len(),
            last_ckpt_block: seq.last_decode_ckpt_block,
            interval,
        }) else {
            return;
        };
        // The registered prefix is the FULL token slice, so vision-pad must be
        // checked over the full slice too.
        if self.tokens_have_vision_pad(&seq.tokens) {
            return;
        }
        let session_hash = seq.session_hash;
        let adapter_id = seq.adapter_id;
        if !self.decode_ckpt_save_and_register(seq, plan, session_hash, adapter_id, "decode-ckpt") {
            // Nothing was saved locally — emit NO command, so the worker does
            // not end up holding a checkpoint rank 0 lacks. (Either asymmetry
            // is answered correctly by the A100 vote; this is the cheaper one.)
            return;
        }
        // A109: tell every worker rank to save the SAME (slot, token,
        // session) checkpoint. Broadcast AFTER the local save succeeded so the
        // command is emitted only for checkpoints rank 0 actually holds. The
        // three call sites (verify_k2_step accept/reject, decode_logits_step)
        // are all quiescent points in the command stream — the worker is
        // parked in `ep_recv_seq_and_cmd`, exactly where the next decode or
        // prefill command would land.
        if let Err(e) = self.ep_broadcast_decode_ckpt(plan, session_hash, adapter_id, seq.slot_idx)
        {
            tracing::warn!("A109 decode-ckpt broadcast failed (ranks may diverge): {e:#}");
        }
    }

    /// Head side of [`EP_CMD_DECODE_CKPT`]. No-op on a single-GPU build and on
    /// any rank that is not driving the command stream.
    fn ep_broadcast_decode_ckpt(
        &self,
        plan: CkptPlan,
        session_hash: u64,
        adapter_id: u64,
        slot_idx: usize,
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        // Only rank 0 writes the command stream; a worker reaching here would
        // pair its send against the head's send and desynchronise the wire.
        if self.comm.as_ref().map(|c| c.rank()) != Some(0) {
            return Ok(());
        }
        // Under v2 the preamble routes the command to the worker's matching
        // slot; `alloc-slot` already bails if the worker's SSM-pool slot ever
        // differs from the head's seq_id, so `slot_idx` IS the seq_id. Under
        // v1 the preamble is skipped and everything targets slot 0.
        self.ep_broadcast_seq_and_cmd(slot_idx as u32, EP_CMD_DECODE_CKPT, self.ep_protocol_v2)?;
        self.ep_broadcast_tokens(&encode_ckpt_payload(plan, session_hash, adapter_id))?;
        Ok(())
    }

    /// Worker side of [`EP_CMD_DECODE_CKPT`] (A109): save the checkpoint rank
    /// 0 just saved, at the position rank 0 chose.
    ///
    /// The worker does NOT re-derive the decision — cadence, the thinking-gate
    /// at the `decode_logits_step` call site and `session_hash` are all
    /// head-side state, and any independent re-derivation is exactly the
    /// divergence A109 is about. It obeys the broadcast or fails loudly.
    ///
    /// `session_hash` is adopted from the head (it is stored metadata only;
    /// `session_matches` short-circuits to `true` for this rank's own lookups,
    /// whose `seq.session_hash` is the worker-local default). `adapter_id`
    /// stays RANK-LOCAL on purpose: it feeds `hash_token_prefix`, so
    /// registering under the head's value would key the entry where this
    /// rank's own lookup will never search for it.
    pub(in crate::model) fn decode_marconi_checkpoint_worker(
        &self,
        seq: &mut SequenceState,
        words: &[u32],
    ) -> Result<()> {
        let (plan, head_session, head_adapter) = decode_ckpt_payload(words)?;
        if seq.slot_idx == usize::MAX {
            bail!("A109 decode-ckpt: rank 0 checkpointed a slot this rank has no SSM state for");
        }
        if plan.snap_tokens > seq.tokens.len() || plan.end_block > seq.block_table.len() {
            bail!(
                "A109 decode-ckpt: head asked for snap_tokens={} end_block={} but this rank has \
                 tokens={} blocks={} — head and worker have diverged",
                plan.snap_tokens,
                plan.end_block,
                seq.tokens.len(),
                seq.block_table.len(),
            );
        }
        if head_adapter != seq.adapter_id {
            // Pre-existing A109 residual (HANDOFF-30 §6): worker ranks never
            // set adapter_id/session_hash. Keying stays rank-local, so the
            // rank still finds its own entry; logged, not fixed here.
            tracing::debug!(
                "A109 decode-ckpt: head adapter_id={head_adapter} != local {} — registering \
                 under the local id so this rank's own lookup can find it",
                seq.adapter_id,
            );
        }
        let adapter_id = seq.adapter_id;
        if !self.decode_ckpt_save_and_register(seq, plan, head_session, adapter_id, "decode-ckpt/w")
        {
            tracing::warn!(
                "A109 decode-ckpt: rank 0 saved (slot={} tok={}) but this rank could not — the \
                 A100 vote will refuse the restore (correct, slower)",
                seq.slot_idx,
                plan.snap_tokens,
            );
        }
        Ok(())
    }

    /// The save itself, shared by the head and the worker so neither can drift
    /// from the other. Returns whether a checkpoint was registered.
    ///
    /// `session_hash` / `adapter_id` are parameters rather than reads of `seq`
    /// precisely because the worker must register the HEAD's session tag.
    fn decode_ckpt_save_and_register(
        &self,
        seq: &mut SequenceState,
        plan: CkptPlan,
        session_hash: u64,
        adapter_id: u64,
        who: &str,
    ) -> bool {
        // Order the default stream after any in-flight secondary-stream commit
        // (MTP path writes the canonical live SSM state there) so the snapshot
        // reads the committed state, not a racing partial. No-op on the
        // non-MTP path (no pending secondary work). GPU-side, ~free.
        let _ = self.sync_secondary_dispatch();
        let stream = self.gpu.default_stream();
        let mut kv = self.kv_cache.lock();
        let bs = kv.block_size();
        let snap_id = match self.ssm_snapshots.save(
            seq.slot_idx,
            session_hash,
            self.seq_ssm_h_is_f16(seq),
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => id,
            Ok(None) => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    &mut kv,
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    match self.ssm_snapshots.save(
                        seq.slot_idx,
                        session_hash,
                        self.seq_ssm_h_is_f16(seq),
                        &self.ssm_pool,
                        self.gpu.as_ref(),
                        stream,
                    ) {
                        Ok(Some(id)) => id,
                        _ => return false,
                    }
                } else {
                    return false;
                }
            }
            Err(e) => {
                tracing::warn!("{who} Marconi checkpoint save error: {e}");
                return false;
            }
        };
        // Order any later warm restore (prefill stream) after this save's D2D.
        if let Err(e) = self.record_snapshot_save_dispatch(stream) {
            tracing::warn!("{who} Marconi checkpoint: record snapshot event: {e}");
        }
        drop(kv);
        // Aux (PLE + QSA lexical state) is canonical at exactly
        // seq.tokens.len() here (same post-commit point the SSM save reads),
        // matching the snap_tokens registration below. A collection failure
        // leaves the snapshot aux-less: the restore gate then declines it —
        // slower, never stale.
        match self.collect_aux_states(seq, stream) {
            Ok(aux) => {
                if !aux.is_empty() {
                    self.ssm_snapshots.set_aux(snap_id, aux);
                }
            }
            Err(e) => tracing::warn!("{who} Marconi checkpoint: aux collect failed: {e:#}"),
        }
        // #155 MTP×cache root cause: the live state just saved (post
        // sync_secondary, post-commit) is canonical at exactly
        // seq.tokens.len() tokens — under MTP K=2 the verify stride (+2 on
        // accept) can step OVER the interval boundary, so tokens.len() may
        // exceed end_token by 1..=bs+2. Registering at the floored end_token
        // mislabeled state@(N+k) as @N; a warm-turn restore then replayed the
        // already-incorporated token(s) through the non-idempotent GDN delta
        // rule, corrupting h_state. Register at the TRUE coverage instead:
        // the snapshot index and the warm-restore replay path support
        // arbitrary non-block-aligned token counts (leaf snapshots already
        // are). On the non-MTP +1 stride tokens.len() == end_token at every
        // fire, so this is bit-identical to the old block-floored slice.
        let CkptPlan {
            snap_tokens,
            end_block,
        } = plan;
        let end_token = end_block * bs;
        let boundary_tokens = &seq.tokens[..snap_tokens];
        let boundary_blocks = &seq.block_table[..end_block];
        let boundary_disk: &[u32] = if seq.disk_block_ids.len() >= end_block {
            &seq.disk_block_ids[..end_block]
        } else {
            &[]
        };
        let displaced = self.prefix_cache.insert_intermediate_snapshot(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            snap_id,
            session_hash,
            snap_tokens,
            adapter_id,
        );
        if let Some(old) = displaced {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "{who} SAVE: snap_tokens={snap_tokens} end_block={end_block} snap_id={snap_id} \
             block_table_len={} straddle={} slot={} session={session_hash:#x}",
            seq.block_table.len(),
            snap_tokens.saturating_sub(end_token),
            seq.slot_idx,
        );
        if std::env::var("AVAROK_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("decode_ckpt_save snap={snap_id} tok={snap_tokens}"),
            );
        }
        seq.last_decode_ckpt_block = end_block;
        true
    }

    /// #155: save the finish-leaf SSM snapshot at sequence retire (called by
    /// `cache_sequence_dispatch` before the radix insert). End-of-prefill-only
    /// leaves made every warm turn replay this turn's decode tokens through
    /// the prefill recurrence (different kernel) — drift ratcheted into FP8
    /// argmax flips. No hidden stashed; the exact-hit shortcut skips
    /// hiddenless snapshots (prefix_lookup.rs).
    pub(super) fn finish_leaf_snapshot(&self, seq: &SequenceState) -> Option<usize> {
        if self.config.num_ssm_layers() == 0 || seq.slot_idx == usize::MAX {
            return None;
        }
        // #155 ROOT CAUSE of the MTP×warm-restore token-stutter: on a turn
        // ending in a K2 REJECT, the canonical-state restore (intermediate[0]
        // → live h/conv) is still in flight on the SECONDARY stream — the
        // commit records an event instead of waiting (async_chkpt.rs; a
        // commit-side wait costs ~25% decode wall). Without this ordering,
        // the default-stream snapshot copies below raced that commit and
        // could capture the pre-commit live state — the GDN recurrent memory
        // still holding the REJECTED draft token — poisoning the leaf the
        // next warm turn restores. Same guard the decode checkpoint uses.
        let _ = self.sync_secondary_dispatch();
        let stream = self.gpu.default_stream();
        let saved = match self.ssm_snapshots.save(
            seq.slot_idx,
            seq.session_hash,
            self.seq_ssm_h_is_f16(seq),
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => Some(id),
            Ok(None) => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    &mut self.kv_cache.lock(),
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    let retry = self.ssm_snapshots.save(
                        seq.slot_idx,
                        seq.session_hash,
                        self.seq_ssm_h_is_f16(seq),
                        &self.ssm_pool,
                        self.gpu.as_ref(),
                        stream,
                    );
                    retry.ok().flatten()
                } else {
                    None
                }
            }
            Err(e) => {
                tracing::warn!("finish-leaf SSM snapshot save error: {e}");
                None
            }
        };
        if let Some(id) = saved {
            // Order any later warm restore (prefill stream) after this save.
            if let Err(e) = self.record_snapshot_save_dispatch(stream) {
                tracing::warn!("finish-leaf snapshot: record snapshot event: {e}");
            }
            // Aux (PLE + QSA lexical state) at retire covers prompt +
            // generated tokens — the deepest anchor a chat continuation can
            // match. Failure => aux-less snapshot, declined on restore.
            match self.collect_aux_states(seq, stream) {
                Ok(aux) => {
                    if !aux.is_empty() {
                        self.ssm_snapshots.set_aux(id, aux);
                    }
                }
                Err(e) => tracing::warn!("finish-leaf snapshot: aux collect failed: {e:#}"),
            }
            tracing::info!(
                "Saved finish-leaf SSM snapshot {} for {} tokens",
                id,
                seq.tokens.len(),
            );
            if std::env::var("AVAROK_SSM_SAVE_DUMP").is_ok() {
                self.ssm_pool.debug_state_checksum(
                    seq.slot_idx,
                    self.gpu.as_ref(),
                    stream,
                    &format!("finish_leaf_save snap={id} tok={}", seq.tokens.len()),
                );
            }
        }
        saved
    }
}
