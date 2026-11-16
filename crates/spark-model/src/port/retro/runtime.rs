// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the selector's mutable state and its per-step host logic:
//! detecting a new decode step, matching the step's rows to live sequences,
//! invalidating indexes whose blocks a write reused, and deciding what each
//! served sequence's index needs for a layer before the plan (the first
//! index, an update, or nothing; the builds themselves are in
//! `runtime_build.rs`), evicting cached or idle indexes when the arena is
//! full. Eager steps only.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::arena::Region;
use super::config::{RetroCfg, RetroParams};
use super::device::{Device, NoRoom};
use super::lifecycle::{IndexStore, LayerSlot, LiveSeq, SeqTable, Sighting};
use super::metrics::Metrics;
use super::shadow::Shadow;
use super::state::{IndexLayout, LayerIndex};
use super::update;
use crate::layers::qwen3_attention::sparse_seam::{KvLayerView, PAD_SEQ_ID, SeamCtx, SeqBatch};

/// One logical sequence of the current decode or verify step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPlan {
    pub uid: u64,
    pub first_row: usize,
    pub rows: usize,
    /// Attention extent of the group's last row (this step's tokens included).
    pub len: u32,
    pub table: Vec<u32>,
}

/// The selector's state, behind one mutex.
#[derive(Default)]
pub struct Runtime {
    pub dev: Option<Device>,
    pub seqs: SeqTable,
    pub store: IndexStore,
    pub step: u64,
    pub last_layer: Option<usize>,
    pub groups: Vec<GroupPlan>,
}

/// Why a layer's index could not be readied for a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotReady {
    Short,
    Budget,
}

impl Runtime {
    /// Free regions, and tell shadow mode a sequence went away.
    pub fn release_seq(&mut self, mut s: LiveSeq, shadow: Option<&Shadow>) {
        let regions = s.release();
        self.free_all(regions);
        if let Some(sh) = shadow {
            sh.forget(s.uid);
        }
    }

    pub fn free_all(&mut self, regions: Vec<Region>) {
        if let Some(d) = self.dev.as_mut() {
            for r in regions {
                d.free(r);
            }
        }
    }

    /// Make sure the device state belongs to `c.gpu`, creating it on the first
    /// eager hook. A different backend (a model swap) drops every index: the
    /// old arena went with its backend. The four retro kernel modules must be
    /// in the image (Ada only); otherwise this fails before allocating.
    pub fn ensure_device(
        &mut self,
        p: &RetroParams,
        c: &SeamCtx,
        s: &SeqBatch,
        shadow: Option<&Shadow>,
    ) -> Result<()> {
        if self.dev.as_ref().is_some_and(|d| d.owned_by(c.gpu)) {
            return Ok(());
        }
        if self.dev.take().is_some() {
            let gone: Vec<u64> = (0..self.seqs.len()).map(|i| self.seqs.get(i).uid).collect();
            self.seqs.clear();
            self.store.clear();
            if let Some(sh) = shadow {
                gone.into_iter().for_each(|u| sh.forget(u));
            }
        }
        let kernels = [
            ("retro_kmeans_ada", super::build::available(c.gpu)),
            ("retro_select_ada", super::select::available(c.gpu)),
            (
                "retro_gather_ada",
                super::gather::GatherKernels::available(c.gpu),
            ),
            ("retro_attn_ada", super::attn::AttnKernels::available(c.gpu)),
        ];
        if let Some((m, _)) = kernels.iter().find(|(_, ok)| !ok) {
            anyhow::bail!(
                "kernel module `{m}` is not in this kernel image (the retro kernels are Ada only)"
            );
        }
        let max_ctx = (s.max_blocks_per_seq.max(1)).saturating_mul(c.kv.block_size.max(1));
        let cfg = RetroCfg::new(
            p.clone(),
            c.kv.kv_heads,
            c.num_q_heads,
            c.kv.head_dim,
            c.inv_sqrt_d,
            c.kv.dtype,
            max_ctx,
            c.kv.block_size,
        )?;
        let mut dev = Device::new(c.gpu, cfg)?;
        if let Err(NoRoom(b)) = dev.ensure_bufs(s.max_blocks_per_seq.max(1)) {
            let _ = GpuBackend::free(c.gpu, dev.base);
            anyhow::bail!(
                "the step buffers need {} MiB, more than --retro-index-cache-mb",
                b >> 20
            );
        }
        tracing::info!(
            "retro: arena {} MiB on the device; capacities per KV head: {} centroids, {} exact tokens, \
             {} retrieval / {} estimation clusters; max context {}",
            p.index_cache_mb,
            dev.cfg.cap_centroids,
            dev.cfg.exact_cap,
            dev.cfg.retrieval_cap,
            dev.cfg.es_cap,
            max_ctx
        );
        self.dev = Some(dev);
        Ok(())
    }

    /// Whether this layer call's geometry is the one the buffers were sized for.
    pub fn serves_shape(&self, c: &SeamCtx) -> bool {
        self.dev.as_ref().is_some_and(|d| {
            let g = &d.cfg;
            g.kv_heads == c.kv.kv_heads
                && g.head_dim == c.kv.head_dim
                && g.num_q_heads == c.num_q_heads
                && g.kv_dtype == c.kv.dtype
                && g.block_size == c.kv.block_size
        })
    }

    /// A new decode step: count it, release stale sequences, and match the
    /// step's rows to live sequences.
    pub fn begin_step(&mut self, s: &SeqBatch, shadow: Option<&Shadow>) {
        self.step += 1;
        for gone in self.seqs.sweep(self.step) {
            self.release_seq(gone, shadow);
        }
        self.groups.clear();
        let rps = s.rows_per_seq.max(1);
        let n_groups = s.n / rps;
        let bs = self.dev.as_ref().map_or(16, |d| d.cfg.block_size).max(1);
        let mut taken = Vec::new();
        let mut writes: Vec<(u64, u32)> = Vec::new();
        for g in 0..n_groups {
            let first_row = g * rps;
            let len = s
                .host_seq_lens
                .get(first_row + rps - 1)
                .copied()
                .unwrap_or(0);
            let Some(table) = s.host_block_tables.get(g).copied() else {
                continue;
            };
            if len == 0 {
                continue;
            }
            let id = s.seq_ids.get(g).copied().filter(|&i| i != PAD_SEQ_ID);
            let sight = self.seqs.sight(id, table, len, self.step, &taken);
            let i = sight.index();
            taken.push(i);
            if let Sighting::Returning(i) = sight {
                let e = self.seqs.get_mut(i);
                let regions = e.release();
                e.covered.clear();
                e.suspect = false;
                e.dense_only = false;
                let uid = e.uid;
                self.free_all(regions);
                if let Some(sh) = shadow {
                    sh.forget(uid);
                }
            }
            let uid = self.seqs.get(i).uid;
            let first = s.first_pos.get(g).copied().unwrap_or(len.saturating_sub(1));
            for pos in first..len {
                if let Some(&b) = table.get((pos / bs) as usize) {
                    writes.push((uid, b));
                }
            }
            self.groups.push(GroupPlan {
                uid,
                first_row,
                rows: rps,
                len,
                table: table.to_vec(),
            });
        }
        writes.dedup();
        // A write into a block another sequence's index covers: that one ended.
        for (writer, b) in writes {
            for gone in self.seqs.invalidate_blocks(&[b], Some(writer)) {
                self.release_seq(gone, shadow);
            }
            let regions = self.store.invalidate_blocks(&[b]);
            self.free_all(regions);
        }
    }

    /// A prefill write: drop indexes covering the written blocks, or mark the
    /// live ones suspect when the blocks are unknown (batched prefill). Cache
    /// entries are fingerprint-checked before every use anyway.
    pub fn prefill_written(
        &mut self,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
        block: u32,
        shadow: Option<&Shadow>,
    ) {
        if s.host_block_tables.is_empty()
            || s.host_block_tables.len() != first_pos.len().min(n_new.len())
        {
            self.seqs.mark_suspect_except(&[]);
            return;
        }
        let bs = block.max(1);
        let mut blocks = Vec::new();
        for ((t, &f), &n) in s.host_block_tables.iter().zip(first_pos).zip(n_new) {
            if n == 0 {
                continue;
            }
            for lb in f / bs..=(f + n - 1) / bs {
                if let Some(&b) = t.get(lb as usize) {
                    blocks.push(b);
                }
            }
        }
        blocks.sort_unstable();
        blocks.dedup();
        for gone in self.seqs.invalidate_blocks(&blocks, None) {
            self.release_seq(gone, shadow);
        }
        let regions = self.store.invalidate_blocks(&blocks);
        self.free_all(regions);
    }

    /// Allocate an index region, evicting cached then idle indexes for room.
    pub(crate) fn alloc_index(
        &mut self,
        layout: IndexLayout,
        shadow: Option<&Shadow>,
    ) -> Option<(LayerIndex, Region)> {
        loop {
            let d = self.dev.as_mut()?;
            if let Some(got) = d.alloc_index(layout) {
                return Some(got);
            }
            if let Some(r) = self.store.evict_one() {
                self.free_all(vec![r]);
                continue;
            }
            let step = self.step;
            let s = self.seqs.evict_idle(step)?;
            self.release_seq(s, shadow);
        }
    }

    /// Ready one sequence's index for attention layer `c.attn_layer`: the
    /// first index (cache install or build), or the update now due.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        p: &RetroParams,
        c: &SeamCtx,
        uid: u64,
        len: u32,
        table: &[u32],
        metrics: &Metrics,
        shadow: Option<&Shadow>,
    ) -> Result<std::result::Result<(), NotReady>> {
        let layer = u32::try_from(c.attn_layer).unwrap_or(u32::MAX);
        let bs = c.kv.block_size.max(1);
        let Some(e) = self.seqs.uid_mut(uid) else {
            return Ok(Err(NotReady::Short));
        };
        if e.dense_only {
            return Ok(Err(NotReady::Budget));
        }
        let counts = e.layers.get(&layer).and_then(LayerSlot::counts);
        let res = match counts {
            None => self.first_index(p, c, uid, len, table, metrics, shadow)?,
            Some(cn) => match update::update_step(&cn, len, p, bs) {
                None => Ok(()),
                Some(step) => {
                    let r =
                        self.grow_and_build(c, uid, layer, len, table, &step, metrics, shadow)?;
                    if r.is_ok() {
                        metrics.count_update();
                    }
                    r
                }
            },
        };
        if let Err(NotReady::Budget) = res
            && let Some(e) = self.seqs.uid_mut(uid)
        {
            e.dense_only = true;
            tracing::warn!(
                "retro: no room in the {} MiB arena for a sequence of {len} tokens; it runs dense attention",
                p.index_cache_mb
            );
        }
        Ok(res)
    }

    /// The index of a group's sequence for `layer`, if ready.
    pub fn index_of(&self, uid: u64, layer: u32) -> Option<&LayerIndex> {
        let i = self.seqs.by_uid(uid)?;
        self.seqs.get(i).layers.get(&layer)?.index()
    }

    /// Bytes the arena hands out, for the gauge.
    pub fn used_bytes(&self) -> u64 {
        self.dev.as_ref().map_or(0, |d| d.used() as u64)
    }

    /// The `KvLayerView` the device was sized with matches this call.
    pub fn kv_ok(&self, kv: &KvLayerView) -> bool {
        self.dev
            .as_ref()
            .is_some_and(|d| d.cfg.head_dim == kv.head_dim && d.cfg.kv_heads == kv.kv_heads)
    }
}
