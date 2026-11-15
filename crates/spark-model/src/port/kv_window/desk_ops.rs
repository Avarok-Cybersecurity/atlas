// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): pages leave the desk in batches and come back as deltas of the selected set, taken from a written description of the design; no code copied or fetched.

//! Responsibility: the window desk's two moves for one sequence (lane L36), split from
//! `desk.rs` for the 500-line cap: [`Desk::demote`] (a batch of resident blocks to host pages,
//! the table entries to the hole) and [`Desk::recall`] (a scorer-ranked delta: retire the
//! recalled pages that fell out, fetch the new ones into fresh blocks), plus the lazy creation
//! of the hole block and the host side. Each move does every fallible step before it touches
//! the table, so an `Err` changes nothing.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_cache::block_io::BlockRegions;

use super::super::fetch::{HostSide, Pending};
use super::super::policy::{self, Slot};
use super::super::selector::{PageScorer, ScoreRequest};
use super::{Desk, DeviceHooks, SeqDesk, StepReport};

impl Desk {
    /// The hole block every hidden entry points at: taken once per cache, zeroed, published to
    /// the view kernel before any table holds it. `None` when the pool has no free block.
    pub(super) fn ensure_hole(
        &mut self,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
        dev: &dyn DeviceHooks,
    ) -> Result<Option<u32>> {
        if let Some(h) = self.hole {
            return Ok(Some(h));
        }
        let Some(h) = kv.try_alloc_block() else {
            return Ok(None);
        };
        if let Err(e) = kv
            .zero_block(h, gpu, stream)
            .and_then(|()| dev.publish_hole(gpu, Some(h), stream))
        {
            kv.dec_ref(h);
            return Err(e);
        }
        tracing::info!("kv window desk: block {h} is the hole every hidden page points at");
        self.hole = Some(h);
        Ok(Some(h))
    }

    fn ensure_host(&mut self, kv: &PagedKvCache, gpu: &dyn GpuBackend) -> Result<()> {
        if self.host.is_none() {
            let page_bytes = kv.block_regions(0).bytes();
            if page_bytes == 0 {
                bail!("kv window desk: the KV cache has no attention layers");
            }
            self.host = Some(HostSide::new(
                gpu,
                page_bytes,
                self.cfg.host_budget_bytes,
                self.stats.clone(),
            )?);
        }
        Ok(())
    }

    /// Move the resident blocks `batch` (logical indices, from `policy::demote_batch`) of the
    /// sequence keyed `block0` to host pages.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn demote(
        &mut self,
        sd: &mut SeqDesk,
        block0: u32,
        table: &mut [u32],
        batch: &[u32],
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
        dev: &dyn DeviceHooks,
        scorer: Option<&dyn PageScorer>,
        rep: &mut StepReport,
    ) -> Result<()> {
        if !dev.ready(gpu, stream)? {
            return Ok(());
        }
        let Some(hole) = self.ensure_hole(kv, gpu, stream, dev)? else {
            return Ok(());
        };
        self.ensure_host(kv, gpu)?;
        let Some(host) = self.host.as_ref() else {
            return Ok(());
        };
        let src: HashMap<u32, u32> = batch.iter().map(|&i| (i, table[i as usize])).collect();
        if src.values().any(|&b| b == hole) {
            bail!("kv window desk: a resident slot of block {block0}'s sequence holds the hole");
        }
        let regions = batch
            .iter()
            .map(|i| kv.checked_block_regions(src[i]))
            .collect::<Result<Vec<BlockRegions>>>()?;
        host.arena.reserve(gpu, batch.len());
        if let Some(sc) = scorer {
            let pairs: Vec<(u32, u32)> = batch.iter().map(|i| (*i, src[i])).collect();
            if let Err(e) = sc.on_demote(gpu, stream, block0, &pairs) {
                tracing::warn!("kv window desk: scorer `{}` on_demote: {e:#}", sc.name());
            }
        }
        let report = host.api.demote(gpu, sd.id, batch, &regions, stream)?;
        for &i in report.copied.iter().chain(&report.resident) {
            table[i as usize] = hole;
            kv.inc_ref(hole);
            sd.slots[i as usize] = Slot::Hidden;
        }
        for i in &report.resident {
            kv.dec_ref(src[i]);
            rep.freed += 1;
        }
        if let Some(ticket) = report.ticket {
            let sources = report.copied.iter().map(|i| (*i, src[i])).collect();
            sd.pending.push(Pending::Demote { ticket, sources });
        }
        rep.demoted += report.copied.len() + report.resident.len();
        if !report.refused.is_empty() {
            tracing::debug!(
                "kv window desk: host budget full, {} block(s) of block {block0}'s sequence stay on the GPU",
                report.refused.len()
            );
        }
        Ok(())
    }

    /// One recall round for the sequence keyed `block0`: rank, retire, fetch the delta.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn recall(
        &mut self,
        sd: &mut SeqDesk,
        block0: u32,
        table: &mut [u32],
        kv_tokens: usize,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
        sc: &dyn PageScorer,
        rep: &mut StepReport,
    ) -> Result<()> {
        let (Some(hole), Some(host)) = (self.hole, self.host.as_ref()) else {
            return Ok(());
        };
        let geom = self.geometry(kv.block_size());
        // A page whose host copy is still being written is not a candidate yet.
        let writing: HashSet<u32> = sd
            .pending
            .iter()
            .flat_map(|p| p.demoting().iter().map(|&(i, _)| i))
            .collect();
        let candidates: Vec<u32> = sd
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| s.demoted() && !writing.contains(&(*i as u32)))
            .map(|(i, _)| i as u32)
            .collect();
        if candidates.is_empty() {
            return Ok(());
        }
        let req = ScoreRequest {
            block0,
            table,
            kv_tokens,
            block_size: geom.block_size,
            candidates: &candidates,
            stream,
        };
        let ranked: Vec<u32> = sc
            .rank(gpu, &req, geom.recall_blocks())?
            .into_iter()
            .filter(|p| !writing.contains(p))
            .collect();
        let plan = policy::plan_recall(&geom, &sd.slots, &ranked);
        for &i in &plan.retire {
            let x = table[i as usize];
            if x == hole || !host.api.contains(sd.id, i) {
                continue;
            }
            table[i as usize] = hole;
            kv.inc_ref(hole);
            kv.dec_ref(x);
            sd.slots[i as usize] = Slot::Hidden;
            rep.rehidden += 1;
        }
        let mut staged: Vec<(u32, u32)> = Vec::new();
        for &i in &plan.fetch {
            if !host.api.contains(sd.id, i) {
                continue;
            }
            let Some(x) = kv.try_alloc_block() else {
                break;
            };
            staged.push((i, x));
        }
        if staged.is_empty() {
            return Ok(());
        }
        let release = |kv: &mut PagedKvCache, staged: &[(u32, u32)]| {
            for &(_, x) in staged {
                kv.dec_ref(x);
            }
        };
        let regions = match staged
            .iter()
            .map(|&(_, x)| kv.checked_block_regions(x))
            .collect::<Result<Vec<BlockRegions>>>()
        {
            Ok(r) => r,
            Err(e) => {
                release(kv, &staged);
                return Err(e);
            }
        };
        let pages: Vec<u32> = staged.iter().map(|&(i, _)| i).collect();
        match host.api.fetch(gpu, sd.id, &pages, &regions, stream) {
            Ok(ticket) => {
                for &i in &pages {
                    sd.slots[i as usize] = Slot::Fetching;
                }
                rep.fetch_started += pages.len();
                tracing::debug!("kv_window recall seq={block0} blocks={pages:?}");
                sd.pending.push(Pending::Fetch { ticket, staged });
                Ok(())
            }
            Err(e) => {
                release(kv, &staged);
                Err(e)
            }
        }
    }
}
