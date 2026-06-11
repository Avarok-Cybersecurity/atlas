// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
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
        if !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || self.config.num_ssm_layers() == 0
            || seq.hss_window_start() != 0
            || seq.slot_idx == usize::MAX
        {
            return;
        }
        // Block-count between decode checkpoints. Env-tunable (no rebuild) so
        // the cadence/drift tradeoff can be swept; default 4 blocks = 64 tok.
        let interval = std::env::var("ATLAS_DECODE_CKPT_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);
        let mut kv = self.kv_cache.lock();
        let bs = kv.block_size();
        // Derive the block count from tokens.len() (what we slice + cache),
        // NOT seq_len: under MTP seq_len can transiently exceed tokens.len()
        // (verify bonus position), which would overrun the token slice.
        let complete_blocks = seq.tokens.len() / bs;
        let end_block = complete_blocks;
        if end_block == 0
            || !end_block.is_multiple_of(interval)
            || end_block == seq.last_decode_ckpt_block
        {
            return;
        }
        // Only checkpoint blocks that physically exist. NOTE: the prefill-era
        // `kv_valid_tokens` guard does NOT apply here — that field tracks the
        // contiguous KV-written prefix during PREFILL and is never advanced by
        // decode, so it would wrongly veto every decode checkpoint past the
        // prompt length. During decode each token writes its KV inline, so all
        // `end_block` complete blocks (= tokens.len()/bs) are fully written.
        if end_block > seq.block_table.len() {
            return;
        }
        // Cadence/log only — NOT the registered coverage (see snap_tokens below).
        let end_token = end_block * bs;
        // The registered prefix is the FULL token slice, so vision-pad must be
        // checked over the full slice too.
        if self.tokens_have_vision_pad(&seq.tokens) {
            return;
        }
        // Order the default stream after any in-flight secondary-stream commit
        // (MTP path writes the canonical live SSM state there) so the snapshot
        // reads the committed state, not a racing partial. No-op on the
        // non-MTP path (no pending secondary work). GPU-side, ~free.
        let _ = self.sync_secondary_dispatch();
        let stream = self.gpu.default_stream();
        let snap_id = match self.ssm_snapshots.save(
            seq.slot_idx,
            seq.session_hash,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => id,
            Ok(None) => {
                if self
                    .ssm_snapshots
                    .reclaim_from_cache(self.prefix_cache.as_ref(), &mut kv)
                {
                    match self.ssm_snapshots.save(
                        seq.slot_idx,
                        seq.session_hash,
                        &self.ssm_pool,
                        self.gpu.as_ref(),
                        stream,
                    ) {
                        Ok(Some(id)) => id,
                        _ => return,
                    }
                } else {
                    return;
                }
            }
            Err(e) => {
                tracing::warn!("decode Marconi checkpoint save error: {e}");
                return;
            }
        };
        drop(kv);
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
        let snap_tokens = seq.tokens.len();
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
            seq.session_hash,
            snap_tokens,
        );
        if let Some(old) = displaced {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "decode-ckpt SAVE: snap_tokens={snap_tokens} end_block={end_block} snap_id={snap_id} \
             block_table_len={} straddle={}",
            seq.block_table.len(),
            snap_tokens - end_token,
        );
        if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("decode_ckpt_save snap={snap_id} tok={snap_tokens}"),
            );
        }
        seq.last_decode_ckpt_block = end_block;
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
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => Some(id),
            Ok(None) => {
                if self
                    .ssm_snapshots
                    .reclaim_from_cache(self.prefix_cache.as_ref(), &mut self.kv_cache.lock())
                {
                    let retry = self.ssm_snapshots.save(
                        seq.slot_idx,
                        seq.session_hash,
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
            tracing::info!(
                "Saved finish-leaf SSM snapshot {} for {} tokens",
                id,
                seq.tokens.len(),
            );
            if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
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
