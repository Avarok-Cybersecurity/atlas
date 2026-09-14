// SPDX-License-Identifier: AGPL-3.0-only
//! Phase 2, batched: reserve every stream's prefix match up front so a
//! kernel-batched wave admits or declines as a whole, and release what was
//! reserved when it declines.
#![allow(unused_imports, dead_code, clippy::too_many_arguments)]
use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;
use crate::traits::{PrefillSlice, SequenceState};

impl TransformerModel {
    /// Acquire all cache matches before the batched path mutates any sequence
    /// or KV state. On rejection, roll back exactly the references acquired by
    /// this pass; a later cache insertion cannot make that rollback touch a
    /// deeper node.
    pub(in crate::model) fn prefill_b_reserve_batched_prefix_matches(
        &self,
        streams: &[PrefillSlice<'_>],
        block_size: usize,
    ) -> Option<Vec<PrefixMatch>> {
        if !self.prefix_cache.is_active() || streams.first()?.chunk_start != 0 {
            return Some(Vec::new());
        }
        // A multi-rank prefix match needs the normal EP min-reduction, which
        // is not safe inside this transactional admission.
        if self.multi_rank_protocol_active() {
            tracing::info!(
                target: "atlas::q12",
                "batched prefix reservation declined: multi-rank world needs \
                 the EP min-reduction — falling back to per-stream"
            );
            return None;
        }

        let mut matches = Vec::with_capacity(streams.len());
        for slice in streams {
            let seq = &*slice.seq;
            if self.tokens_have_vision_pad(slice.prompt_tokens)
                || seq.collect_prompt_logprobs.is_some()
            {
                tracing::info!(
                    target: "atlas::q12",
                    "batched prefix reservation declined: vision pads or \
                     prompt-logprob collection — falling back to per-stream"
                );
                self.release_batched_prefix_reservations(streams, &matches, block_size);
                return None;
            }
            matches.push(self.prefix_cache.lookup(
                slice.prompt_tokens,
                block_size,
                seq.session_hash,
                seq.adapter_id,
            ));
        }

        // Hybrid-SSM models: a WARM prefix match implies a KV/Marconi skip
        // whose recurrent-state interplay this transactional admission does
        // not handle (the v1 rule). But a model-level blanket veto rejected
        // COLD batches too, which serialized every chunk-0 wave on hybrid
        // checkpoints — the entire measured C=32/C=128 prefill ramp
        // (2026-08-16 stackval: every wave logged "cache plan not admitted").
        // An all-cold reservation (matched_tokens == 0 everywhere) acquires
        // no blocks, restores no snapshot and skips nothing — provably the
        // same state as the cache-inactive admission above, which has always
        // admitted hybrid models. Warm hybrid batches keep falling back to
        // the per-stream path, whose restore logic is established.
        if !super::batch_kernel::batched_reserve_hybrid_ssm_ok(
            &matches,
            self.config.num_ssm_layers() != 0,
        ) {
            tracing::info!(
                target: "atlas::q12",
                "batched prefix reservation declined: hybrid-SSM model with a \
                 warm prefix match — falling back to per-stream"
            );
            self.release_batched_prefix_reservations(streams, &matches, block_size);
            return None;
        }

        if !super::batch_kernel::cache_batch_matches_compatible(&matches, streams[0].chunk_len) {
            tracing::info!(
                target: "atlas::q12",
                "batched prefix reservation declined: prefix matches not \
                 batch-compatible — falling back to per-stream"
            );
            self.release_batched_prefix_reservations(streams, &matches, block_size);
            return None;
        }
        Some(matches)
    }

    fn release_batched_prefix_reservations(
        &self,
        streams: &[PrefillSlice<'_>],
        matches: &[PrefixMatch],
        block_size: usize,
    ) {
        for (slice, prefix_match) in streams.iter().zip(matches) {
            if prefix_match.matched_tokens > 0 {
                self.prefix_cache.release_matched(
                    slice.prompt_tokens,
                    block_size,
                    prefix_match.matched_tokens,
                    slice.seq.adapter_id,
                );
            }
        }
    }
}
