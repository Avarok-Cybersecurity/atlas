// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: prefix-cache lookup + EP-sync of matched count + Marconi
//! SSM snapshot restore. Returns (kv_write_start, marconi_skip).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::block_mgmt::reuse_prefix_match_disk_ids;
use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn prefill_b_prefix_lookup(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        total: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<(usize, bool)> {
        let bs = kv_cache.block_size();
        if chunk_start == 0 {
            let mut prefix_match = if self.tokens_have_vision_pad(tokens) {
                spark_runtime::prefix_cache::PrefixMatch::empty()
            } else {
                self.prefix_cache.lookup(tokens, bs, seq.session_hash)
            };
            // F83 (2026-04-30): on EP>1, head and worker have
            // independent local prefix caches whose match counts can
            // diverge (eviction order differences, async insert
            // timing). If we proceed with different `matched` per
            // rank, the chunked prefill computes different proc_count
            // values → MoE allreduce sizes mismatch → collective
            // deadlock. Sync via 2 rooted broadcasts (one per rank,
            // accumulating min): both ranks agree on the min match.
            // If `matched_min < local_matched`, release the extra
            // matched blocks (the lookup inc_ref'd them — undo so
            // they're not leaked) and re-walk for the agreed count.
            // F83 (2026-04-30): UNCONDITIONAL on EP-active, even if
            // local matched_tokens == 0. Both ranks must call
            // ep_min_u32 so the rooted broadcasts on each rank find a
            // matching receiver. Earlier (fix53) the call was gated by
            // `matched_tokens > 0`, which deadlocked when head matched
            // but worker didn't: head broadcast had no receiver.
            // Calling unconditionally on EP active fixes that — when
            // either side has matched=0 the agreed value is 0 and we
            // simply fall through to the no-cache path on both sides.
            let ep_active = self.comm.is_some() && self.config.ep_world_size > 1;
            if ep_active {
                let local = prefix_match.matched_tokens as u32;
                let agreed = self.ep_min_u32(local)? as usize;
                if agreed < prefix_match.matched_tokens {
                    self.prefix_cache.release(tokens, bs);
                    if agreed > 0 {
                        prefix_match =
                            self.prefix_cache
                                .lookup(&tokens[..agreed], bs, seq.session_hash);
                    } else {
                        prefix_match = spark_runtime::prefix_cache::PrefixMatch::empty();
                    }
                    tracing::info!(
                        "F83 EP-cache-sync: local_matched={local} agreed_matched={agreed} \
                         (cap to min across ranks)"
                    );
                } else if local > 0 || agreed > 0 {
                    tracing::debug!(
                        "F83 EP-cache-sync: local_matched={local} agreed_matched={agreed} (no cap)"
                    );
                }
            }
            let matched = prefix_match.matched_tokens;
            seq.cached_prefix_tokens = matched;
            seq.prompt_len = total;
            for &block_idx in &prefix_match.matched_blocks {
                kv_cache.inc_ref(block_idx);
                seq.block_table.push(block_idx);
            }
            reuse_prefix_match_disk_ids(
                &prefix_match.matched_disk_block_ids,
                &mut seq.disk_block_ids,
            );
            // Marconi: restore SSM snapshot if available.
            // With intermediate checkpoints, ssm_snapshot_tokens may be less than
            // matched_tokens. We skip SSM computation only up to ssm_snapshot_tokens
            // and recompute SSM (+ overwrite KV) for tokens between the checkpoint
            // and matched_tokens. This trades some redundant KV writes for correct
            // SSM state propagation.
            let mut skip = if let Some(snap_id) = prefix_match.ssm_snapshot {
                let snap_tok = prefix_match.ssm_snapshot_tokens;
                if snap_tok > 0
                    && matched <= total
                    && self
                        .ssm_snapshots
                        .session_matches(snap_id, seq.session_hash)
                {
                    self.ssm_snapshots.restore(
                        snap_id,
                        seq.slot_idx,
                        &self.ssm_pool,
                        self.gpu.as_ref(),
                        stream,
                    )?;
                    if snap_tok < matched {
                        tracing::info!(
                            "Marconi intermediate hit: restored from checkpoint at token {} \
                             (skipping {} tokens, recomputing {} SSM tokens to match point {})",
                            snap_tok,
                            snap_tok,
                            matched - snap_tok,
                            matched,
                        );
                    } else {
                        tracing::info!(
                            "Marconi SSM cache hit: {} tokens skipped ({} blocks), snapshot {}",
                            matched,
                            prefix_match.matched_blocks.len(),
                            snap_id,
                        );
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };
            let has_ssm = self.config.num_ssm_layers() > 0;
            if matched > 0 && !skip && has_ssm {
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — recomputing all KV",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            } else if matched > 0 && !skip {
                // F82 (2026-04-30): non-SSM cache-hit skip path.
                skip = true;
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) reused (F82+F83: non-SSM cache-hit skip)",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            }
            // For SSM models: use ssm_snapshot_tokens (not matched) as skip point.
            // Exception: when matched == total (exact prompt match), the snapshot
            // covers the entire prompt, so skip all tokens.
            // For pure attention (MLA/GQA): use matched tokens directly.
            let skip_tokens = if skip && !has_ssm {
                matched
            } else if skip && matched == total {
                matched
            } else if skip {
                prefix_match.ssm_snapshot_tokens
            } else {
                0
            };
            seq.marconi_skip_to = skip_tokens;
            Ok((skip_tokens, skip))
        } else if seq.marconi_skip_to > 0 {
            // Chunk 1+: inherit skip info from chunk 0's prefix cache lookup.
            Ok((seq.marconi_skip_to, true))
        } else {
            Ok((0, false))
        }
    }
}
