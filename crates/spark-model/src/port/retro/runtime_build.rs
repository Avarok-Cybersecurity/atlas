// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the index builds the selector's runtime runs for a
//! sequence and layer: the first index (a cached prefix index, checked
//! against its content fingerprint and extended by a segmented append, or a
//! fresh build published to the cache), and every later build into a private
//! index (copy-on-write from the shared one, or a move into a larger region
//! when the current one is full).

use std::sync::Arc;

use anyhow::Result;

use super::config::RetroParams;
use super::device::NoRoom;
use super::lifecycle::{IndexEntry, IndexKey};
use super::metrics::Metrics;
use super::runtime::{NotReady, Runtime};
use super::shadow::Shadow;
use super::state::IndexCounts;
use super::update::{self, BuildStep};
use crate::layers::qwen3_attention::sparse_seam::SeamCtx;

impl Runtime {
    /// The first index of a sequence for `c.attn_layer`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn first_index(
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
        let Some(initial) = update::initial_step(len, p, bs) else {
            return Ok(Err(NotReady::Short));
        };
        let Some(model) = self.dev.as_ref().map(|d| d.model) else {
            return Ok(Err(NotReady::Budget));
        };
        let params = p.build_hash();
        let end = initial.after.indexed_end;
        // 1. A cached prefix index, always checked against its content
        // fingerprint (its blocks may have been freed and reused unseen).
        for cand in self.store.candidates(model, layer, params, table, end) {
            let ok = {
                let d = self.dev.as_ref().expect("checked above");
                d.read_fingerprint(c.gpu, &c.kv, table, cand.fingerprint.pos, c.stream)?
                    == cand.fingerprint
            };
            if !ok {
                if let Some(r) = self.store.discard(&cand) {
                    drop(cand);
                    self.free_all(vec![r]);
                }
                continue;
            }
            let covered = cand.key.blocks.clone();
            let fp = cand.fingerprint.clone();
            let counts = cand.index.counts;
            if let Some(e) = self.seqs.uid_mut(uid) {
                e.covered.extend(covered);
                let slot = e.layers.entry(layer).or_default();
                slot.base = Some(cand);
                slot.fingerprint = Some(fp);
            }
            tracing::debug!(
                "retro: attn layer {layer} reuses a cached index over [{}, {})",
                counts.sink,
                counts.indexed_end
            );
            if let Some(step) = update::append_step(&counts, len, p, bs) {
                return self.grow_and_build(c, uid, layer, len, table, &step, metrics, shadow);
            }
            return Ok(Ok(()));
        }
        // 2. A fresh build, published to the cache.
        let layout = self.dev.as_ref().expect("checked above").layout_for(end);
        let Some((mut index, region)) = self.alloc_index(layout, shadow) else {
            return Ok(Err(NotReady::Budget));
        };
        let d = self.dev.as_mut().expect("checked above");
        let secs = match d.run_build(c.gpu, &c.kv, table, len, &mut index, &initial, c.stream) {
            Ok(Ok(s)) => s,
            Ok(Err(NoRoom(_))) => {
                d.free(region);
                return Ok(Err(NotReady::Budget));
            }
            Err(e) => {
                d.free(region);
                return Err(e);
            }
        };
        metrics.add_build_secs(secs);
        let fingerprint = match d.read_fingerprint(c.gpu, &c.kv, table, end - 1, c.stream) {
            Ok(f) => f,
            Err(e) => {
                d.free(region);
                return Err(e);
            }
        };
        let blocks = table[..(end / bs) as usize].to_vec();
        let entry = Arc::new(IndexEntry {
            key: IndexKey {
                model,
                attn_layer: layer,
                params,
                blocks: blocks.clone(),
                end,
            },
            index,
            region,
            fingerprint: fingerprint.clone(),
            build_secs: secs,
        });
        if let Some(r) = self.store.insert(entry.clone()) {
            self.free_all(vec![r]);
        }
        if let Some(e) = self.seqs.uid_mut(uid) {
            e.covered.extend(blocks);
            let slot = e.layers.entry(layer).or_default();
            slot.base = Some(entry);
            slot.fingerprint = Some(fingerprint);
        }
        tracing::info!(
            "retro: attn layer {layer} index built over [{}, {end}) ({} centroids) in {secs:.3} s",
            initial.first_pos,
            initial.after.n_centroids
        );
        Ok(Ok(()))
    }

    /// Give the slot a private index that can hold `step.after` (copying the
    /// shared or the too-small one), then run `step` into it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn grow_and_build(
        &mut self,
        c: &SeamCtx,
        uid: u64,
        layer: u32,
        len: u32,
        table: &[u32],
        step: &BuildStep,
        metrics: &Metrics,
        shadow: Option<&Shadow>,
    ) -> Result<std::result::Result<(), NotReady>> {
        let fits_own = self
            .seqs
            .uid_mut(uid)
            .and_then(|e| e.layers.get(&layer))
            .and_then(|s| s.own.as_ref())
            .is_some_and(|(i, _)| {
                update::fits(&step.after, i.layout.cap_centroids, i.layout.cap_tokens)
            });
        if !fits_own {
            let mut layout = self
                .dev
                .as_ref()
                .map(|d| d.private_layout(len))
                .ok_or_else(|| anyhow::anyhow!("retro: no device state"))?;
            layout.cap_centroids = layout.cap_centroids.max(step.after.n_centroids);
            layout.cap_tokens = layout.cap_tokens.max(step.after.listed);
            let Some((mut fresh, region)) = self.alloc_index(layout, shadow) else {
                return Ok(Err(NotReady::Budget));
            };
            let Some(e) = self.seqs.uid_mut(uid) else {
                self.free_all(vec![region]);
                return Ok(Err(NotReady::Short));
            };
            let slot = e.layers.entry(layer).or_default();
            let mut freed = Vec::new();
            {
                let src = slot.index().cloned();
                let d = self.dev.as_ref().expect("allocated above");
                match src {
                    Some(src) => d.copy_index(c.gpu, &src, &mut fresh, c.stream)?,
                    None => {
                        fresh.counts = IndexCounts::default();
                        d.write_header(c.gpu, &fresh, c.stream)?;
                    }
                }
            }
            if let Some((_, r)) = slot.own.replace((fresh, region)) {
                freed.push(r);
            }
            if let Some(b) = slot.base.take()
                && Arc::strong_count(&b) == 1
            {
                freed.push(b.region);
            }
            self.free_all(freed);
        }
        let d = self
            .dev
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("retro: no device state"))?;
        let Some(e) = self.seqs.uid_mut(uid) else {
            return Ok(Err(NotReady::Short));
        };
        let bs = d.cfg.block_size.max(1);
        let slot = e.layers.get_mut(&layer).expect("created above");
        let (index, _) = slot.own.as_mut().expect("private index set above");
        match d.run_build(c.gpu, &c.kv, table, len, index, step, c.stream)? {
            Ok(secs) => {
                metrics.add_build_secs(secs);
                let end = index.counts.indexed_end;
                e.covered
                    .extend(table[..(end / bs) as usize].iter().copied());
                Ok(Ok(()))
            }
            Err(NoRoom(_)) => Ok(Err(NotReady::Budget)),
        }
    }
}
