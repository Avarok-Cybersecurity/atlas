// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/integration/vllm/vllm_v1_adapter.py (the external hit counted in whole blocks past the GPU match, the blocks allocated by the scheduler and filled before the prefill reads them) and the hybrid rule of lmcache/v1/api.py:636-656 (commit 3aefae1, Apache-2.0); no code copied.
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/simple_kv_offload/manager.py (restore into freshly allocated blocks, then hand them to the prefix cache), Copyright contributors to the vLLM project, Apache-2.0; no code copied.

//! Responsibility: the host tier's restore path. [`Live::prefetch`] (synchronous) and
//! [`Live::restore_begin`] / [`Live::restore_poll`] / [`Live::settle`] (lane L27's parked
//! requests) plan which blocks to restore ([`super::plan`]), pin the GPU prefix, allocate the
//! destination blocks, start one batched H2D through the tier and, once the copy completed,
//! insert the restored blocks into the radix tree with exact reference bookkeeping:
//!
//! * the GPU prefix `[0, from)` is pinned by a `lookup` (radix node references, so the
//!   allocation below cannot evict it); its blocks take no extra KV reference;
//! * each destination block is allocated with one reference (the glue's);
//! * on `Ready`: `insert(tokens[..to], prefix ++ dst, matched = from)` creates the nodes
//!   `[from, to)`; the cache takes its own reference on each created node's block
//!   (`cache_acquires_refs`), the glue drops its allocation reference, and `release_matched`
//!   returns every node reference the lookup and the insert took. A destination block whose node
//!   already existed (another request inserted it meanwhile) is thereby freed;
//! * on `Failed` (or any error after the allocation): the destination blocks are freed and the
//!   lookup's node references returned; every GPU block stays valid and the prefill recomputes.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::{BlockKey, RestoreState, RestoreTicket};
use spark_runtime::prefix_cache::PrefixCache;

use super::TierPrefetch;
use super::plan::{self, Plan, PlanInput, PrefetchOutcome};
use super::state::Live;
use crate::model::block_mgmt::{alloc_block_evicting, cache_acquires_refs};

/// What a restore step needs from the model: its backend, its (locked) KV cache and its prefix
/// cache. Tests build it from mock parts.
pub(crate) struct Env<'a> {
    pub(crate) gpu: &'a dyn GpuBackend,
    pub(crate) kv: &'a mut PagedKvCache,
    pub(crate) cache: &'a dyn PrefixCache,
    /// The model has recurrent (GDN / SSM) layers: the snapshot boundary rule applies.
    pub(crate) has_ssm: bool,
}

/// A started restore, waiting for its copy.
pub(crate) struct Pending {
    /// `tokens[..to * block_size]`.
    tokens: Vec<u32>,
    adapter: u64,
    /// The GPU prefix blocks `[0, from)`, pinned by the lookup.
    prefix: Vec<u32>,
    /// The destination blocks `[from, to)`, one glue reference each.
    dst: Vec<u32>,
    from: usize,
    gpu_tokens: usize,
}

impl Pending {
    /// Free the destination blocks (teardown and failure paths).
    pub(crate) fn free_dst(&self, kv: &mut PagedKvCache) {
        for &b in &self.dst {
            kv.dec_ref(b);
        }
    }
}

/// The result of planning and starting a restore.
enum Begun {
    Nothing(TierPrefetch),
    Started(RestoreTicket, Pending),
}

impl Live {
    /// Restore the prompt's host-held prefix and wait for the copy (the synchronous path).
    pub(crate) fn prefetch(
        &self,
        env: &mut Env<'_>,
        tokens: &[u32],
        adapter: u64,
        session_hash: u64,
    ) -> Result<TierPrefetch> {
        match self.begin(env, tokens, adapter, session_hash)? {
            Begun::Nothing(p) => Ok(p),
            Begun::Started(ticket, pending) => {
                let state = match self.tier.wait(env.gpu, &ticket) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("kv_host_tier: restore wait failed: {e:#}");
                        RestoreState::Failed
                    }
                };
                let out = self.finish(env, pending, state);
                self.tier.release(ticket);
                Ok(out)
            }
        }
    }

    /// Start a restore without waiting. `Ok(None)`: nothing to restore.
    pub(crate) fn restore_begin(
        &self,
        env: &mut Env<'_>,
        tokens: &[u32],
        adapter: u64,
        session_hash: u64,
    ) -> Result<Option<RestoreTicket>> {
        match self.begin(env, tokens, adapter, session_hash)? {
            Begun::Nothing(p) => {
                tracing::debug!("kv_host_tier: no async restore ({:?})", p.outcome);
                Ok(None)
            }
            Begun::Started(ticket, pending) => {
                self.books.lock().pending.insert(ticket.id(), pending);
                Ok(Some(ticket))
            }
        }
    }

    /// Non-blocking state of an async restore; finishes it on `Ready` or `Failed`.
    pub(crate) fn restore_poll(
        &self,
        env: &mut Env<'_>,
        t: &RestoreTicket,
    ) -> Result<RestoreState> {
        let state = match self.tier.poll(env.gpu, t) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("kv_host_tier: restore poll failed: {e:#}");
                RestoreState::Failed
            }
        };
        if state != RestoreState::Pending {
            let pending = self.books.lock().pending.remove(&t.id());
            if let Some(p) = pending {
                self.finish(env, p, state);
            }
        }
        Ok(state)
    }

    /// Finish an async restore (waiting for its copy if it still runs) and release its ticket.
    pub(crate) fn settle(&self, env: &mut Env<'_>, t: RestoreTicket) {
        let pending = self.books.lock().pending.remove(&t.id());
        if let Some(p) = pending {
            let state = match self.tier.wait(env.gpu, &t) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("kv_host_tier: restore wait failed: {e:#}");
                    RestoreState::Failed
                }
            };
            self.finish(env, p, state);
        }
        self.tier.release(t);
    }

    /// Plan the restore, pin the GPU prefix, allocate the destination and start the copy.
    fn begin(
        &self,
        env: &mut Env<'_>,
        tokens: &[u32],
        adapter: u64,
        session_hash: u64,
    ) -> Result<Begun> {
        let bs = self.block_size;
        // v1: base-model KV only (the tier's keys fold the adapter in, but stores are adapter 0).
        if adapter != 0 || bs == 0 {
            return Ok(Begun::Nothing(TierPrefetch::default()));
        }
        let usable = plan::usable_blocks(tokens.len(), bs);
        if usable == 0 {
            return Ok(Begun::Nothing(TierPrefetch::nothing(
                PrefetchOutcome::Miss,
                0,
            )));
        }
        let gpu_tokens = env.cache.peek_matched_tokens(tokens, bs, adapter);
        let gpu_blocks = (gpu_tokens / bs).min(usable);
        let gpu_tokens = gpu_blocks * bs;
        if gpu_blocks >= usable {
            return Ok(Begun::Nothing(TierPrefetch::nothing(
                PrefetchOutcome::GpuHit,
                gpu_tokens,
            )));
        }
        let span = &tokens[..usable * bs];
        let keys = BlockKey::chain_blocks(self.ns, 0, span, bs);
        let chunks: Vec<&[u32]> = span.chunks_exact(bs).collect();
        let host_hits = self.tier.lookup(&keys[gpu_blocks..], &chunks[gpu_blocks..]);
        // The in-flight cap (lane L26a) bounds the span before the boundary rule runs, so the
        // snapshot probe below sees the depth that can actually be restored.
        let host = host_hits.min(self.tier.max_restore_blocks());
        if host_hits > 0 && host == 0 {
            return Ok(Begun::Nothing(TierPrefetch::nothing(
                PrefetchOutcome::Declined,
                gpu_tokens,
            )));
        }
        let snapshot_tokens = if env.has_ssm && host > 0 {
            env.cache
                .peek_snapshot_tokens(tokens, (gpu_blocks + host) * bs, session_hash, adapter)
        } else {
            0
        };
        let input = PlanInput {
            usable_blocks: usable,
            gpu_blocks,
            host_hits: host,
            has_ssm: env.has_ssm,
            snapshot_tokens,
            block_size: bs,
            min_tokens: self.tier.min_tokens() as usize,
        };
        let (from, to) = match plan::plan(&input) {
            Plan::Nothing(o) => return Ok(Begun::Nothing(TierPrefetch::nothing(o, gpu_tokens))),
            Plan::Restore { from, to } => (from, to),
        };
        let blocks = to - from;
        if !self.tier.gate(blocks * bs, blocks).restores() {
            return Ok(Begun::Nothing(TierPrefetch::nothing(
                PrefetchOutcome::Declined,
                gpu_tokens,
            )));
        }
        // Pin the GPU prefix: node references keep the allocation below from evicting it.
        let prefix = if from > 0 {
            let m = env
                .cache
                .lookup(&tokens[..from * bs], bs, session_hash, adapter);
            if m.matched_tokens != from * bs || !m.matched_disk_block_ids.is_empty() {
                // The radix changed under the peek, or `--high-speed-swap` owns these blocks
                // (a parallel disk-id table the restore cannot rebuild): restore nothing.
                if m.matched_tokens > 0 {
                    env.cache
                        .release_matched(tokens, bs, m.matched_tokens, adapter);
                }
                return Ok(Begun::Nothing(TierPrefetch::nothing(
                    PrefetchOutcome::Off,
                    gpu_tokens,
                )));
            }
            m.matched_blocks
        } else {
            Vec::new()
        };
        let mut pending = Pending {
            tokens: tokens[..to * bs].to_vec(),
            adapter,
            prefix,
            dst: Vec::with_capacity(blocks),
            from,
            gpu_tokens,
        };
        for _ in 0..blocks {
            match alloc_block_evicting(env.kv, env.cache) {
                Some(b) => pending.dst.push(b),
                None => break,
            }
        }
        if pending.dst.len() < blocks {
            self.abandon(env, &pending);
            return Ok(Begun::Nothing(TierPrefetch::nothing(
                PrefetchOutcome::NoBlocks,
                gpu_tokens,
            )));
        }
        let regions = match env.kv.table_regions(&pending.dst) {
            Ok(r) => r,
            Err(e) => {
                self.abandon(env, &pending);
                return Err(e.context("kv_host_tier: destination block layout"));
            }
        };
        match self.tier.begin_restore(env.gpu, &keys[from..to], &regions) {
            Ok(ticket) => Ok(Begun::Started(ticket, pending)),
            Err(e) => {
                tracing::warn!("kv_host_tier: restore not started ({e:#}); recomputing");
                self.abandon(env, &pending);
                Ok(Begun::Nothing(TierPrefetch::nothing(
                    PrefetchOutcome::Failed,
                    gpu_tokens,
                )))
            }
        }
    }

    /// Free the destination blocks and return the lookup's node references.
    fn abandon(&self, env: &mut Env<'_>, p: &Pending) {
        p.free_dst(env.kv);
        if p.from > 0 {
            env.cache.release_matched(
                &p.tokens,
                self.block_size,
                p.from * self.block_size,
                p.adapter,
            );
        }
    }

    /// Finish a restore whose copy is no longer pending.
    fn finish(&self, env: &mut Env<'_>, p: Pending, state: RestoreState) -> TierPrefetch {
        let bs = self.block_size;
        if state != RestoreState::Ready {
            self.abandon(env, &p);
            return TierPrefetch::nothing(PrefetchOutcome::Failed, p.gpu_tokens);
        }
        let mut table = p.prefix.clone();
        table.extend_from_slice(&p.dst);
        let acquired = env
            .cache
            .insert(&p.tokens, &table, &[], bs, p.from * bs, p.adapter);
        cache_acquires_refs(&acquired, env.kv);
        for &b in &p.dst {
            env.kv.dec_ref(b);
        }
        env.cache
            .release_matched(&p.tokens, bs, p.tokens.len(), p.adapter);
        let restored_blocks = p.dst.len();
        tracing::debug!(
            "kv_host_tier: restored {restored_blocks} block(s) after a GPU match of {} tokens",
            p.gpu_tokens
        );
        TierPrefetch {
            outcome: PrefetchOutcome::Restored,
            gpu_tokens: p.gpu_tokens,
            restored_tokens: restored_blocks * bs,
            restored_blocks,
        }
    }
}
