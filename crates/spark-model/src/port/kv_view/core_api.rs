// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's interface to its core (port lane L32
//! consuming lane L31): the [`ViewCore`] trait the selector drives, and
//! [`L31Core`], its implementation over L31's per-page key sketch
//! (`spark_runtime::kv_cache::sketch`), query capture
//! (`port::kv_view::query_capture`) and block scorer and picker
//! (`spark_runtime::kv_select`).
//!
//! This file is the only place lane L32 names an L31 item, so a signature
//! change on L31's side is repaired here alone. The selector's flow tests
//! drive a fake core (`tests_flow.rs`).
//!
//! Sketch updates: a prefill write plans page jobs on the host
//! (`plan_jobs`, host tables or the batched pointer tables) and recomputes
//! those pages eagerly; a decode or verify write recomputes the pages of each
//! sequence's last tokens from the step's device table (`update_tail`, the
//! graph-safe form). Rows no job can cover are reported back so the selector
//! skips that sequence's next selection (`sketch_gap`).

use std::ops::Range;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::kv_cache::sketch::{
    DEFAULT_MAX_JOBS, SketchBuffers, SketchConfig, SketchLayer, TailBatch, WrittenRange, plan_jobs,
};
use spark_runtime::kv_select::{self, DEFAULT_SCRATCH_BYTES, GqaAgg, PickCfg, ScoreScratch};

use super::config::BLOCK_TOKENS;
use super::query_capture::{self, QueryCapture, QueryCaptureCfg};
use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, SeamCtx, SeamPhase, SeqBatch};

/// The model geometry the view sizes its buffers by (learned from the model
/// at the first prefill call; see `ModelFacts`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geom {
    /// Tokens per KV page (the paged cache's block).
    pub block_size: u32,
    /// Pages in the KV pool.
    pub num_blocks: u32,
    pub attn_layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_heads: u32,
    pub rotary_dim: u32,
    pub rope_theta: f32,
    /// The model's block-table stride for the longest sequence.
    pub max_blocks_per_seq: u32,
}

impl Geom {
    /// Pages per scoring block (a block is [`BLOCK_TOKENS`] tokens).
    pub fn block_pages(&self) -> u32 {
        (BLOCK_TOKENS / self.block_size.max(1)).max(1)
    }

    /// Tokens per scoring block.
    pub fn block_tokens(&self) -> u32 {
        self.block_pages() * self.block_size.max(1)
    }
}

/// A pick request, in scoring blocks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PickParams {
    pub budget_blocks: u32,
    pub sink_blocks: u32,
    pub recent_blocks: u32,
    /// Blocks the query span touches.
    pub mandatory: Vec<u32>,
}

/// What the selector needs from the core.
pub trait ViewCore: Send {
    /// After a K write, fold the written rows into the per-page key sketch
    /// (inside the graph when decoding). Returns, per logical sequence of
    /// `s`, the written rows no sketch update covered (empty when all were).
    fn record_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<Vec<u32>>;
    /// Claim a capture slot for `slot`'s final user message `span`; the kept
    /// span (its last query rows), or `None` when every slot is taken.
    fn arm(&self, slot: u64, span: Range<u32>) -> Option<Range<u32>>;
    /// Release `slot`'s capture.
    fn disarm(&self, slot: u64);
    /// The prefill `on_query` forwarding (`s.seq_ids` name the streams).
    /// Returns the rows captured.
    fn capture(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
    ) -> Result<usize>;
    /// Score the blocks of a sequence whose pages (physical ids, in order)
    /// are `pages` against `slot`'s captured rows; writes the f32 scores to
    /// `out` on `stream`. `Ok(false)` when the capture is incomplete.
    fn score(
        &self,
        gpu: &dyn GpuBackend,
        pages: &[u32],
        slot: u64,
        out: DevicePtr,
        stream: u64,
    ) -> Result<bool>;
    /// The picked blocks, chronological.
    fn pick(&self, scores: &[f32], p: &PickParams) -> Vec<u32>;
    /// Fraction of the score mass in `picked`.
    fn recall_mass(&self, scores: &[f32], picked: &[u32]) -> f64;
    /// Device bytes the core allocated.
    fn device_bytes(&self) -> u64;
}

/// Device bytes [`L31Core::new`] would allocate for `geom`.
pub fn l31_bytes(geom: &Geom, query_tokens: u32, slots: usize) -> u64 {
    let (s, q) = l31_configs(geom, query_tokens, slots);
    (s.total_bytes() + q.total_bytes() + DEFAULT_SCRATCH_BYTES) as u64
}

fn l31_configs(geom: &Geom, query_tokens: u32, slots: usize) -> (SketchConfig, QueryCaptureCfg) {
    let sketch = SketchConfig {
        num_blocks: geom.num_blocks as usize,
        attn_layers: geom.attn_layers as usize,
        kv_heads: geom.kv_heads as usize,
        head_dim: geom.head_dim as usize,
        block_size: geom.block_size as usize,
        rotary_dim: geom.rotary_dim as usize,
        rope_theta: geom.rope_theta,
        max_jobs: DEFAULT_MAX_JOBS,
    };
    let capture = QueryCaptureCfg {
        attn_layers: geom.attn_layers as usize,
        q_heads: geom.q_heads as usize,
        head_dim: geom.head_dim as usize,
        query_tokens: query_tokens as usize,
        slots,
        rotary_dim: geom.rotary_dim as usize,
        rope_theta: geom.rope_theta,
    };
    (sketch, capture)
}

/// [`ViewCore`] over lane L31's sketch, capture, scorer and picker.
pub struct L31Core {
    geom: Geom,
    sketch: SketchBuffers,
    capture: QueryCapture,
    scratch: ScoreScratch,
}

impl L31Core {
    /// Allocate the sketch for the whole KV pool, `slots` capture slots of
    /// `query_tokens` rows and the scorer's scratch. Frees what it allocated
    /// on failure.
    pub fn new(gpu: &dyn GpuBackend, geom: Geom, query_tokens: u32, slots: usize) -> Result<Self> {
        let (sc, qc) = l31_configs(&geom, query_tokens, slots);
        let sketch = SketchBuffers::new(gpu, sc)?;
        let capture = match QueryCapture::new(gpu, qc) {
            Ok(c) => c,
            Err(e) => {
                let _ = sketch.free(gpu);
                return Err(e);
            }
        };
        let scratch = match ScoreScratch::new(gpu, DEFAULT_SCRATCH_BYTES) {
            Ok(s) => s,
            Err(e) => {
                let _ = sketch.free(gpu);
                let _ = capture.free(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            geom,
            sketch,
            capture,
            scratch,
        })
    }
}

fn sketch_layer(c: &SeamCtx) -> SketchLayer {
    SketchLayer {
        attn_layer: c.attn_layer,
        k_pool: c.kv.k_pool,
        dtype: match c.kv.dtype {
            KvDtypeTag::Bf16 => KvCacheDtype::Bf16,
            KvDtypeTag::Fp8 => KvCacheDtype::Fp8,
        },
        block_stride_bytes: c.kv.block_stride_bytes,
        kv_scale: c.kv.kv_scale,
    }
}

impl ViewCore for L31Core {
    fn record_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<Vec<u32>> {
        let layer = sketch_layer(c);
        if c.phase != SeamPhase::Prefill || c.in_graph {
            if s.block_table.is_null() || s.seq_lens.is_null() || s.n == 0 {
                return Ok(Vec::new());
            }
            let t = TailBatch {
                block_table: s.block_table,
                seq_lens: s.seq_lens,
                max_blocks_per_seq: s.max_blocks_per_seq,
                n_rows: s.n as u32,
                rows_per_seq: s.rows_per_seq.max(1) as u32,
                n_new: n_new.iter().copied().max().unwrap_or(0),
            };
            self.sketch.update_tail(c.gpu, &layer, &t, c.stream)?;
            return Ok(Vec::new());
        }
        let ranges: Vec<WrittenRange> = first_pos
            .iter()
            .zip(n_new)
            .enumerate()
            .map(|(i, (&first_pos, &n_new))| WrittenRange {
                first_pos,
                n_new,
                host_table: s.host_block_tables.get(i).copied(),
            })
            .collect();
        let device_tables = !s.block_table_ptrs.is_null();
        let mut gaps = vec![0u32; ranges.len()];
        for (i, r) in ranges.iter().enumerate() {
            let (_, uncovered) = plan_jobs(
                std::slice::from_ref(r),
                c.kv.block_size as usize,
                device_tables,
            );
            gaps[i] = uncovered as u32;
        }
        let (jobs, _) = plan_jobs(&ranges, c.kv.block_size as usize, device_tables);
        self.sketch.update_jobs(
            c.gpu,
            &layer,
            &jobs,
            s.block_table_ptrs,
            s.max_blocks_per_seq,
            c.stream,
        )?;
        Ok(if gaps.iter().all(|&g| g == 0) {
            Vec::new()
        } else {
            gaps
        })
    }

    fn arm(&self, slot: u64, span: Range<u32>) -> Option<Range<u32>> {
        self.capture
            .arm(slot, span.start, span.end)
            .map(|(a, b)| a..b)
    }

    fn disarm(&self, slot: u64) {
        self.capture.disarm(slot);
    }

    fn capture(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
    ) -> Result<usize> {
        query_capture::capture_rows(&self.capture, c, s, q_pre_rope, q)
    }

    fn score(
        &self,
        gpu: &dyn GpuBackend,
        pages: &[u32],
        slot: u64,
        out: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let Some(rows) = self.capture.rows_view(slot) else {
            return Ok(false);
        };
        let cfg = self
            .scratch
            .cfg(self.geom.block_pages() as usize, GqaAgg::Mean);
        kv_select::score_blocks(gpu, &self.sketch.view(pages), &rows, &cfg, out, stream)?;
        Ok(true)
    }

    fn pick(&self, scores: &[f32], p: &PickParams) -> Vec<u32> {
        let cfg = PickCfg {
            budget_blocks: p.budget_blocks as usize,
            sink_blocks: p.sink_blocks as usize,
            recent_blocks: p.recent_blocks as usize,
            mandatory: p.mandatory.clone(),
        };
        kv_select::pick(scores, scores.len(), &cfg)
    }

    fn recall_mass(&self, scores: &[f32], picked: &[u32]) -> f64 {
        kv_select::recall_mass(scores, picked)
    }

    fn device_bytes(&self) -> u64 {
        (self.sketch.config().total_bytes()
            + self.capture.config().total_bytes()
            + self.scratch.bytes()) as u64
    }
}
