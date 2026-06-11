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
    pub(in crate::model) fn prefill_b_prefix_lookup(
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
            // Issue #31: the prefix cache stores per-layer K/V on disk for every
            // matched block (that's the radix-tree invariant — blocks with a
            // non-MAX `disk_block_id` are fully offloaded across every attention
            // layer). Advance every layer's offload cursor to match the new
            // `disk_block_ids.len()` so the slide-before-alloc loop in
            // `block_mgmt::ensure_blocks_through_prefill` doesn't bail later
            // when it discovers `disk_last_offloaded[L] < window_start`. Without
            // this, gbanyan's repro (long prompt + prefix-caching + HSS) tripped
            // `offload_layer_kv` on the first attn layer with `attn_layer_idx=0,
            // logical_pos=0, window_start>0` because the cached blocks pushed
            // `disk_block_ids` and `block_table` forward without notifying the
            // layer cursors.
            let new_total = seq.disk_block_ids.len() as u32;
            for cursor in seq.disk_last_offloaded_per_layer.iter_mut() {
                if *cursor < new_total {
                    *cursor = new_total;
                }
            }
            // Marconi: restore SSM snapshot if available.
            // With intermediate checkpoints, ssm_snapshot_tokens may be less than
            // matched_tokens. We skip SSM computation only up to ssm_snapshot_tokens
            // and recompute SSM for tokens between the checkpoint and
            // matched_tokens. KV for that replay window is NOT rewritten — the
            // layer_kv_write_start floor (forward_layers.rs) skips writes below
            // cached_prefix_tokens, so the shared prefix-cache blocks keep the
            // original values (a non-bit-equal rewrite would poison them).
            let mut skip = if let Some(snap_id) = prefix_match.ssm_snapshot {
                let snap_tok = prefix_match.ssm_snapshot_tokens;
                // Exact full-prompt hit on a hiddenless snapshot (finish
                // leaves never stash a hidden): the exact-snap fixup cannot
                // produce the first token's logits, so fall through to the
                // no-snapshot full-recompute path. Only affects identical
                // retried prompts; multi-turn warm hits have matched < total.
                let exact_without_hidden = snap_tok == matched
                    && matched == total
                    && !self.ssm_snapshots.has_hidden(snap_id);
                if snap_tok > 0
                    && matched <= total
                    && !exact_without_hidden
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
                    if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
                        self.ssm_pool.debug_state_checksum(
                            seq.slot_idx,
                            self.gpu.as_ref(),
                            stream,
                            &format!("restore@{snap_tok}"),
                        );
                    }
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
                        // Exact full-prompt leaf hit (snap_tok == matched ==
                        // total): the last prompt token is re-run for logits,
                        // double-advancing the SSM recurrent state. Flag it so
                        // finalize_last re-restores state@N and emits the first
                        // token from the snapshot's stashed hidden. Only when
                        // the whole prompt matched — a shorter-than-total match
                        // (matched < total) continues forward correctly.
                        if matched == total {
                            seq.marconi_exact_snap = Some(snap_id);
                        }
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };
            // CBD probe (env-gated, default OFF = current behavior): bypass the
            // exact-leaf-hit snapshot shortcut + marconi_exact_snap fixup, routing
            // exact full-prompt hits through full recompute (the proven-correct
            // cache-off-equivalent). Isolates whether the exact-snap stashed-hidden
            // path degrades output quality (cache-ON ws ~23% vs cache-OFF ~60% with
            // give-ups already eliminated). If ws climbs with this set, that path
            // is the residual bug.
            if skip
                && prefix_match.ssm_snapshot_tokens == matched
                && matched == total
                && std::env::var("ATLAS_NO_MARCONI_EXACT").as_deref() == Ok("1")
            {
                skip = false;
                seq.marconi_exact_snap = None;
                tracing::info!(
                    "ATLAS_NO_MARCONI_EXACT: bypassing exact-leaf snapshot shortcut \
                     for {matched}-token full hit — recomputing all KV+SSM"
                );
            }
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
            // Exception: when the snapshot covers the ENTIRE matched prefix
            // (snap_tok == matched) AND the whole prompt matched
            // (matched == total), the restored recurrent state is already
            // at token `total`, so we can skip all tokens (the exact-hit
            // fixup in finalize_last handles the redundant last-token re-run).
            //
            // CRITICAL (warm-hit SSM corruption fix): when an *intermediate*
            // checkpoint matched at full prompt length (snap_tok < matched
            // == total — e.g. the leaf snapshot was evicted from the
            // 16-slot pool under agentic churn, leaving only a block-aligned
            // checkpoint), the restored recurrent state is at token
            // `snap_tok`, NOT `total`. Skipping to `total` here would leave
            // the SSM h_state/conv_state stale by (total - snap_tok) tokens
            // while positions/KV advance to `total`, so the first decoded
            // token reads a misaligned recurrent state → garbage → immediate
            // stop (empty completion). We MUST skip only to `snap_tok` so the
            // suffix-prefill recomputes SSM over [snap_tok, total), exactly
            // like the `matched < total` intermediate path. The redundant KV
            // writes for [snap_tok, matched) are harmless (they duplicate
            // already-cached values).
            //
            // For pure attention (MLA/GQA): use matched tokens directly.
            let snap_tok = prefix_match.ssm_snapshot_tokens;
            let skip_tokens = if skip && !has_ssm {
                matched
            } else if skip && matched == total && snap_tok == matched {
                matched
            } else if skip {
                snap_tok
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
