// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Rewritten for Atlas's sparse-attention seam; changes:
// one selection shared by all verify rows, a compacted block table in fixed device buffers instead
// of host staging, graph-safe device-side decisions. No prototype source was read.

//! Responsibility: the Quest `KvSelector` (port lane L30, row
//! `attn_page_sparse`, APPROXIMATE, off by default on every target), plugged
//! into the sparse-attention seam (`layers/qwen3_attention/sparse_seam.rs`,
//! master plan 3.7.6).
//!
//! - `on_kv_written`: fold the keys just written into their pages' summaries
//!   (decode and verify: positions from the device lengths, so it is captured
//!   once and stays right on every replay; prefill: the host's written range).
//! - `plan` (decode and verify): score every candidate page with the step's
//!   queries, keep the first `sink` pages, the top-k pages and the recent
//!   window, and return a `View` over selector-owned fixed buffers. Rows of a
//!   verify group share one selection. Below `--attn-sparse-min-ctx`, or when
//!   the budget already covers every page, the kernels write the identity view
//!   (a verbatim copy of the dense table), so the output is bit-identical to
//!   dense attention there and a captured graph never changes topology.
//! - Prefill attends densely (`plan_prefill` keeps the default `Dense`).
//!
//! Every refusal is `Dense` with one log line naming it: no summary layout
//! charged, the kernel module absent (every target but Ada), a layer geometry
//! the layout does not describe, a K pool other than the layer's own, or a
//! step shape larger than the View arena.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::arena::{self, Arena};
use super::budget::{self, Geometry};
use super::config::QuestConfig;
use super::launch::{self, Budget, Kernels, LayerBufs, StepRows, TableRef, UpdateRange, ViewOut};
use super::metrics;
use crate::layers::qwen3_attention::sparse_seam::{
    KvSelector, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
};

/// Seconds between two harvests of the device counters (eager hooks only).
const HARVEST_EVERY: Duration = Duration::from_secs(1);

/// Why a hook did not serve the call (each logged once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoGeometry,
    NoModule,
    CaptureBeforeAlloc,
    LayerGeometry,
    ForeignPool,
    Shape,
    ViewTooLarge,
}

impl Refusal {
    fn why(self) -> &'static str {
        match self {
            Self::NoGeometry => {
                "no page-summary layout was charged to the KV budget by this model build"
            }
            Self::NoModule => {
                "the kernel module attn_page_sparse_ada is not in this image (Ada only)"
            }
            Self::CaptureBeforeAlloc => {
                "a CUDA graph was captured before the first eager hook allocated the summaries"
            }
            Self::LayerGeometry => {
                "an attention layer's KV geometry differs from the charged summary layout"
            }
            Self::ForeignPool => "a call used a K pool other than its layer's own",
            Self::Shape => "a step's rows do not group into whole sequences or heads",
            Self::ViewTooLarge => "a step's View does not fit the 32 MiB selector arena",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The selector. Public so tests and the microtest can drive it.
pub struct QuestSelector {
    cfg: QuestConfig,
    arena: Mutex<Option<Arena>>,
    warned: [AtomicBool; 7],
    announced: AtomicBool,
}

impl QuestSelector {
    pub fn new(cfg: QuestConfig) -> Self {
        Self {
            cfg,
            arena: Mutex::new(None),
            warned: Default::default(),
            announced: AtomicBool::new(false),
        }
    }

    pub fn config(&self) -> &QuestConfig {
        &self.cfg
    }

    /// The selection budget in kernel terms.
    pub fn budget(&self) -> Budget {
        Budget {
            sink_pages: self.cfg.sink_pages,
            recent_tokens: self.cfg.recent_tokens,
            topk_pages: self.cfg.topk_pages,
            min_ctx: self.cfg.min_ctx,
        }
    }

    /// True once `r` was logged (tests read it).
    pub fn refused(&self, r: Refusal) -> bool {
        self.warned[r.index()].load(Ordering::Relaxed)
    }

    fn refuse(&self, r: Refusal) {
        if !self.warned[r.index()].swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "attn_page_sparse is on but Quest page-sparse attention is NOT running for some \
                 calls: {} (those calls attend densely)",
                r.why()
            );
        }
    }

    /// Run `f` with this backend's arena and the layer's buffers, allocating on
    /// an eager hook. `None` (after one log line) when the call is not served.
    fn with_layer<R>(
        &self,
        c: &SeamCtx,
        f: impl FnOnce(&mut Arena, &Kernels, &LayerBufs, &Geometry) -> Result<R>,
    ) -> Result<Option<R>> {
        let Some(geom) = budget::geometry() else {
            self.refuse(Refusal::NoGeometry);
            return Ok(None);
        };
        let mut slot = self
            .arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fresh = !slot
            .as_ref()
            .is_some_and(|a| a.serves(c.gpu, geom.generation));
        if fresh {
            if c.in_graph {
                self.refuse(Refusal::CaptureBeforeAlloc);
                return Ok(None);
            }
            if let Some(old) = slot.take()
                && old.owner == arena::backend_key(c.gpu)
            {
                // Same backend, new model build: free the old arena first.
                old.release(c.gpu, c.stream)?;
            }
            *slot = Some(Arena::new(c.gpu, &geom, c.stream)?);
            metrics::GLOBAL.rebase();
        }
        let a = slot.as_mut().expect("allocated above");
        let Some(k) = a.kernels else {
            self.refuse(Refusal::NoModule);
            return Ok(None);
        };
        let layout = &geom.layout;
        let layer = c.attn_layer;
        let fits = layer < layout.layers.len()
            && layout.layers[layer].kv_heads == c.kv.kv_heads
            && layout.layers[layer].head_dim == c.kv.head_dim
            && layout.block_size == c.kv.block_size;
        if !fits {
            self.refuse(Refusal::LayerGeometry);
            return Ok(None);
        }
        if !a.claim_pool(layer, c.kv.k_pool) {
            self.refuse(Refusal::ForeignPool);
            return Ok(None);
        }
        let eb = c.kv.dtype.bytes_per_elem() as u32;
        let l = LayerBufs::new(
            layout,
            a.summary.base,
            layer,
            c.kv.k_pool,
            c.kv.block_stride_bytes,
            eb,
        );
        f(a, &k, &l, geom.as_ref()).map(Some)
    }

    /// Run `f` on this backend's page-summary arena, read-only. `None` (and `f` not called) when no
    /// eager hook has allocated the arena yet, when it belongs to another backend or model build,
    /// or when no layout was charged. The window desk's recall scorer (`scorer.rs`) reads the
    /// summaries of pages about to leave the GPU through this.
    pub(super) fn with_summaries<R>(
        &self,
        gpu: &dyn GpuBackend,
        f: impl FnOnce(&Arena, &Geometry) -> Result<R>,
    ) -> Result<Option<R>> {
        let Some(geom) = budget::geometry() else {
            return Ok(None);
        };
        let slot = self
            .arena
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match slot.as_ref() {
            Some(a) if a.serves(gpu, geom.generation) => f(a, geom.as_ref()).map(Some),
            _ => Ok(None),
        }
    }

    fn harvest(&self, c: &SeamCtx, a: &Arena) -> Result<()> {
        if c.in_graph || c.attn_layer != 0 {
            return Ok(());
        }
        metrics::GLOBAL.harvest(c.gpu, a.views.stats, c.stream, HARVEST_EVERY, false)
    }

    /// Eager diagnostic: log each group's selection in the line format
    /// `bench/niah_grid.py --selection-log` reads.
    fn log_selection(&self, c: &SeamCtx, s: &SeqBatch, a: &Arena) -> Result<()> {
        let rows = s.n;
        let mut lens_raw = vec![0u8; rows * 4];
        c.gpu
            .copy_d2h_on_stream(a.views.seq_lens, &mut lens_raw, c.stream)?;
        let bs = c.kv.block_size.max(1) as usize;
        for g in 0..rows / s.rows_per_seq {
            let r = g * s.rows_per_seq;
            let len = i32::from_le_bytes(lens_raw[r * 4..r * 4 + 4].try_into().expect("4 bytes"));
            let n = (len.max(0) as usize).div_ceil(bs);
            let mut raw = vec![0u8; n * 4];
            let row = a
                .views
                .block_table
                .offset(r * s.max_blocks_per_seq as usize * 4);
            c.gpu.copy_d2h_on_stream(row, &mut raw, c.stream)?;
            let blocks: Vec<String> = raw
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]).to_string())
                .collect();
            let id = s.seq_ids.get(g).copied().unwrap_or(g as u64);
            tracing::info!(
                "quest select seq={id} layer={} len={len} dense={} blocks=[{}]",
                c.attn_layer,
                s.host_seq_lens
                    .get(r + s.rows_per_seq - 1)
                    .copied()
                    .unwrap_or(0),
                blocks.join(",")
            );
        }
        Ok(())
    }
}

impl KvSelector for QuestSelector {
    fn name(&self) -> &'static str {
        "quest"
    }

    /// The fixed View arena. The page summaries are charged separately, where
    /// the KV budget is computed (`budget::charge_summaries`), because their
    /// size depends on the pool.
    fn reserved_bytes(&self) -> u64 {
        arena::VIEW_ARENA_BYTES as u64
    }

    fn on_kv_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<()> {
        match c.phase {
            SeamPhase::Decode | SeamPhase::Verify => {
                let r = s.rows_per_seq;
                if s.block_table.is_null()
                    || s.seq_lens.is_null()
                    || r == 0
                    || !s.n.is_multiple_of(r)
                {
                    return Ok(());
                }
                self.with_layer(c, |_, k, l, _| {
                    launch::launch_update(
                        c.gpu,
                        k,
                        l,
                        TableRef::Flat(s.block_table),
                        s.max_blocks_per_seq,
                        UpdateRange::Device {
                            seq_lens: s.seq_lens,
                            rows_per_seq: r as u32,
                            groups: (s.n / r) as u32,
                        },
                        c.stream,
                    )
                })
                .map(|_| ())
            }
            SeamPhase::Prefill => {
                if c.in_graph {
                    return Ok(());
                }
                self.with_layer(c, |a, k, l, _| {
                    let bs = l.block_size as usize;
                    for (i, (&first, &count)) in first_pos.iter().zip(n_new).enumerate() {
                        if count == 0 {
                            continue;
                        }
                        let last_block = (first as usize + count as usize - 1) / bs;
                        let (table, stride) = if !s.block_table_ptrs.is_null() {
                            let t = TableRef::Ptrs {
                                ptrs: s.block_table_ptrs,
                                index: i as u32,
                            };
                            (t, s.max_blocks_per_seq)
                        } else if s.n == 1 && !s.block_table.is_null() {
                            (TableRef::Flat(s.block_table), s.max_blocks_per_seq)
                        } else if let Some(t) = s.host_block_tables.get(i).filter(|t| !t.is_empty())
                        {
                            let p = a.upload_table(c.gpu, c.stream, t)?;
                            (TableRef::Flat(p), t.len() as u32)
                        } else {
                            // No table on either side: the score kernel rebuilds these pages.
                            continue;
                        };
                        if last_block >= stride as usize {
                            continue;
                        }
                        launch::launch_update(
                            c.gpu,
                            k,
                            l,
                            table,
                            stride,
                            UpdateRange::Host {
                                first_pos: first,
                                count,
                            },
                            c.stream,
                        )?;
                    }
                    self.harvest(c, a)
                })
                .map(|_| ())
            }
        }
    }

    fn plan(&self, c: &SeamCtx, s: &SeqBatch, q: DevicePtr) -> Result<SparsePlan> {
        let r = s.rows_per_seq;
        let kvh = c.kv.kv_heads;
        if c.phase == SeamPhase::Prefill
            || s.block_table.is_null()
            || s.seq_lens.is_null()
            || s.max_blocks_per_seq == 0
        {
            return Ok(SparsePlan::Dense);
        }
        if r == 0 || !s.n.is_multiple_of(r) || kvh == 0 || !c.num_q_heads.is_multiple_of(kvh) {
            self.refuse(Refusal::Shape);
            return Ok(SparsePlan::Dense);
        }
        let groups = s.n / r;
        let budget = self.budget();
        let gqa = self.cfg.gqa.kernel_flag();
        let log = self.cfg.log_selection && !c.in_graph;
        let served = self.with_layer(c, |a, k, l, _| {
            let stride = launch::score_stride(s.max_blocks_per_seq, l.page_blocks);
            if !arena::view_fits(s.n, s.max_blocks_per_seq as usize, groups, stride as usize) {
                return Ok(None);
            }
            let rows = StepRows {
                block_table: s.block_table,
                seq_lens: s.seq_lens,
                rows_per_seq: r as u32,
                groups: groups as u32,
                row_stride: s.max_blocks_per_seq,
            };
            let v = a.views;
            launch::launch_score(
                c.gpu,
                k,
                l,
                &rows,
                q,
                s.q_row_stride,
                c.num_q_heads,
                v.scores,
                &budget,
                gqa,
                c.stream,
            )?;
            let out = ViewOut {
                block_table: v.block_table,
                seq_lens: v.seq_lens,
                stats: v.stats,
            };
            launch::launch_compact(
                c.gpu,
                k,
                l.block_size,
                l.page_blocks,
                &rows,
                v.scores,
                &out,
                &budget,
                c.attn_layer == 0,
                c.stream,
            )?;
            if log {
                self.log_selection(c, s, a)?;
            }
            self.harvest(c, a)?;
            Ok(Some(v))
        })?;
        let Some(served) = served else {
            return Ok(SparsePlan::Dense);
        };
        let Some(v) = served else {
            self.refuse(Refusal::ViewTooLarge);
            return Ok(SparsePlan::Dense);
        };
        if !self.announced.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "attn_page_sparse: Quest page-sparse decode ACTIVE ({}); APPROXIMATE attention \
                 above the minimum context",
                self.cfg.describe()
            );
        }
        Ok(SparsePlan::View {
            block_table: v.block_table,
            seq_lens: v.seq_lens,
            max_blocks_per_seq: s.max_blocks_per_seq,
        })
    }
}
