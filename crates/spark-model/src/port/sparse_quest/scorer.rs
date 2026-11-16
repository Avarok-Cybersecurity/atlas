// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Used here to rank the pages of the host-resident
// window desk; changes: summaries copied per logical page before the block leaves the GPU, a
// graph-safe query capture into fixed buffers, a host-side cross-layer ranking, by Atlas
// contributors. No prototype source was read.

//! Responsibility: the Quest page scorer of the window desk (port lane L30, request R3 of lane L36,
//! rows `attn_page_sparse` and `kv_window_desk` both on): a `PageScorer` that rides on the desk's
//! selector, because only one `KvSelector` may be registered per serve.
//!
//! # What it keeps
//! - Page summaries stay where the Quest kernels write them (the arena of `selector.rs`, keyed by
//!   the physical block that leads each 64-token page); this scorer forwards the seam's
//!   `on_kv_written` so they stay current. When the desk moves blocks to host RAM,
//!   `PageScorer::on_demote` copies the summary of every page whose leading block leaves (all layers, all KV heads, only
//!   when the page was completely folded) into a per-logical-page host store, because the physical
//!   block is reused once its copy completed.
//! - The last post-RoPE query of every sequence in the last decode or verify step, per attention
//!   layer. `on_query` enqueues plain device copies into fixed buffers (a memset, a strided copy of
//!   the first block-table column, a copy of the lengths and a strided copy of the query rows), so
//!   it is captured into the decode graph and refreshed on every replay, whatever batch the replay
//!   carries. The sequence is identified by its first physical block, the desk's own key.
//!
//! # Ranking
//! [`rank`](PageScorer::rank) runs eager between steps. It reads the sequence's query rows back
//! (two small copies per layer), scores every candidate page that has a stored summary with the
//! Quest bound and combines the layers (`scorer_rank.rs`). It answers with nothing, never an
//! error, when no summary or no query is available yet (a sequence that has not run a decode step,
//! or one absent from the last step), so the desk simply keeps sinks plus the recent window.
//!
//! # Limits, stated
//! The query is the one of the LAST decode step, not of the step that will read the recalled pages;
//! prefill queries are not captured (their hook runs before RoPE). The summary is trusted by its
//! completeness mask alone (the kernels' key fingerprint is not re-checked): a block restored from
//! a prefix cache could carry a stale one, and the desk refuses the prefix cache. At most
//! [`HOST_CAP_BYTES`] of summaries are kept, least recently used sequences first.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::page_summary::{META_BYTES, full_mask};

use super::arena::{self, Arena};
use super::budget;
use super::config::GqaMode;
use super::reference::bf16_val;
use super::scorer_rank::{self, LayerQuery, PageRecord};
pub use super::scorer_store::HOST_CAP_BYTES;
use super::scorer_store::HostStore;
use super::selector::QuestSelector;
use crate::layers::qwen3_attention::sparse_seam::{KvSelector, SeamCtx, SeamPhase, SeqBatch};
use crate::port::kv_window::selector::{PageScorer, ScoreRequest};

/// Rows of one step whose queries are kept per layer (a larger step is not captured).
pub const MAX_Q_ROWS: usize = 64;
/// Device bytes for the query buffers of every layer.
pub const QUERY_ARENA_BYTES: usize = 16 << 20;
const WHY_ROWS: &str =
    "a step has more rows than the query capture keeps; its queries are not captured";
const WHY_CAPTURED: &str = "a decode graph was captured before the first eager hook allocated the \
                            query buffers; recall has no query";
const WHY_LAYER: &str = "a layer's query buffers do not fit or its shape or K pool changed; that \
                         layer has no query";

/// Per layer: the first block-table column (`MAX_Q_ROWS` i32) then the lengths (`MAX_Q_ROWS` i32).
const COLS_BYTES: usize = 2 * MAX_Q_ROWS * 4;
const LENS_OFF: usize = MAX_Q_ROWS * 4;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static RANKED: AtomicU64 = AtomicU64::new(0);
static NO_QUERY: AtomicU64 = AtomicU64::new(0);
static NO_SUMMARY: AtomicU64 = AtomicU64::new(0);
static PAGES_RANKED: AtomicU64 = AtomicU64::new(0);
static SAVED: AtomicU64 = AtomicU64::new(0);
static UNTRUSTED: AtomicU64 = AtomicU64::new(0);

/// Called by `apply` once the scorer is registered: its series then appear on `/metrics`.
pub(super) fn mark_active() {
    ACTIVE.store(true, Ordering::Relaxed);
}

/// The scorer's `/metrics` series; nothing unless it is registered.
pub fn render_prometheus(out: &mut String) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let rounds = "atlas_attn_sparse_desk_rank_total";
    let _ = writeln!(
        out,
        "# HELP {rounds} Window-desk recall rounds handed to the Quest scorer, by outcome (ranked, \
         no_query: no decode query captured for the sequence yet, no_summary: no candidate page \
         had a stored summary).\n# TYPE {rounds} counter"
    );
    for (outcome, v) in [
        ("ranked", &RANKED),
        ("no_query", &NO_QUERY),
        ("no_summary", &NO_SUMMARY),
    ] {
        let _ = writeln!(
            out,
            "{rounds}{{outcome=\"{outcome}\"}} {}",
            v.load(Ordering::Relaxed)
        );
    }
    let ranked = "atlas_attn_sparse_desk_pages_ranked_total";
    let _ = writeln!(
        out,
        "# HELP {ranked} Pages the Quest scorer ranked for the window desk.\n# TYPE {ranked} counter\n\
         {ranked} {}",
        PAGES_RANKED.load(Ordering::Relaxed)
    );
    let saved = "atlas_attn_sparse_desk_summaries_total";
    let _ = writeln!(
        out,
        "# HELP {saved} Page summaries copied to the host as their blocks left the GPU, by result.\n\
         # TYPE {saved} counter\n{saved}{{result=\"saved\"}} {}\n{saved}{{result=\"untrusted\"}} {}",
        SAVED.load(Ordering::Relaxed),
        UNTRUSTED.load(Ordering::Relaxed)
    );
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One attention layer's query buffers.
#[derive(Clone, Copy)]
struct LayerQ {
    /// `MAX_Q_ROWS` i32 leading block ids, then `MAX_Q_ROWS` i32 lengths.
    cols: DevicePtr,
    /// `[MAX_Q_ROWS][nq * hd]` BF16, rows packed.
    q: DevicePtr,
    nq: u32,
    hd: u32,
    /// The layer's own K pool; a call with another pool (a drafter's cache) is left alone.
    pool: DevicePtr,
}

/// The query arena of one backend and one summary generation.
struct QueryStore {
    owner: usize,
    generation: u64,
    base: DevicePtr,
    next: usize,
    layers: HashMap<usize, LayerQ>,
}

impl QueryStore {
    /// The buffers of `layer`, assigned from the arena on first use. `None` when the arena is full
    /// or the layer was first seen with another shape or pool.
    fn layer(&mut self, layer: usize, nq: u32, hd: u32, pool: DevicePtr) -> Option<LayerQ> {
        if let Some(l) = self.layers.get(&layer) {
            return (l.nq == nq && l.hd == hd && l.pool == pool).then_some(*l);
        }
        let row_bytes = nq as usize * hd as usize * 2;
        let need = (COLS_BYTES + MAX_Q_ROWS * row_bytes).div_ceil(256) * 256;
        if row_bytes == 0 || self.next + need > QUERY_ARENA_BYTES {
            return None;
        }
        let l = LayerQ {
            cols: self.base.offset(self.next),
            q: self.base.offset(self.next + COLS_BYTES),
            nq,
            hd,
            pool,
        };
        self.next += need;
        self.layers.insert(layer, l);
        Some(l)
    }
}

/// The store for `gpu` at `generation`, allocated now on an eager hook (never while capturing).
fn ensure_store<'a>(
    slot: &'a mut Option<QueryStore>,
    gpu: &dyn GpuBackend,
    stream: u64,
    in_graph: bool,
    generation: u64,
) -> Result<Option<&'a mut QueryStore>> {
    let key = arena::backend_key(gpu);
    let fresh = !slot
        .as_ref()
        .is_some_and(|s| s.owner == key && s.generation == generation);
    if fresh {
        if in_graph {
            return Ok(None);
        }
        if let Some(old) = slot.take()
            && old.owner == key
        {
            gpu.synchronize(stream)?;
            gpu.free(old.base)?;
        }
        let base = gpu.alloc(QUERY_ARENA_BYTES)?;
        *slot = Some(QueryStore {
            owner: key,
            generation,
            base,
            next: 0,
            layers: HashMap::new(),
        });
    }
    Ok(slot.as_mut())
}

/// Copy the summary of physical block `phys` (every layer, every KV head) to the host. `None` when
/// the slot is outside the arena or any head's page was not completely folded.
fn read_page(
    gpu: &dyn GpuBackend,
    stream: u64,
    a: &Arena,
    full: u64,
    phys: u32,
) -> Result<Option<PageRecord>> {
    let layout = &a.summary.layout;
    if phys >= layout.cap_blocks {
        return Ok(None);
    }
    let base = a.summary.base;
    let mut halves: Vec<u16> = Vec::with_capacity(scorer_rank::record_halves(&layout.layers));
    for (l, g) in layout.layers.iter().enumerate() {
        let kvh = g.kv_heads as usize;
        let mut meta = vec![0u8; kvh * META_BYTES];
        let at = base.offset(layout.meta_offset(l, phys, 0) as usize);
        gpu.copy_d2h_on_stream(at, &mut meta, stream)?;
        for h in 0..kvh {
            let word: [u8; 8] = meta[h * META_BYTES..h * META_BYTES + 8]
                .try_into()
                .expect("8 bytes");
            if u64::from_le_bytes(word) != full {
                return Ok(None);
            }
        }
        let mut raw = vec![0u8; g.slot_halves() as usize * 2];
        let at = base.offset(layout.vals_offset(l, phys, 0) as usize);
        gpu.copy_d2h_on_stream(at, &mut raw, stream)?;
        halves.extend(
            raw.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]])),
        );
    }
    Ok(Some(Arc::from(halves)))
}

/// The Quest page scorer (see the module doc).
pub struct QuestScorer {
    sel: Arc<QuestSelector>,
    host: Mutex<HostStore>,
    queries: Mutex<Option<QueryStore>>,
    warned: [AtomicBool; 3],
}

impl QuestScorer {
    /// A scorer over `sel`, whose arena and kernels it uses for the summaries. `sel` is not
    /// registered with the seam: the desk's selector is.
    pub fn new(sel: Arc<QuestSelector>) -> Self {
        Self {
            sel,
            host: Mutex::new(HostStore::new(HOST_CAP_BYTES)),
            queries: Mutex::new(None),
            warned: Default::default(),
        }
    }

    pub fn selector(&self) -> &Arc<QuestSelector> {
        &self.sel
    }

    fn warn_once(&self, i: usize, msg: &str) {
        if !self.warned[i].swap(true, Ordering::Relaxed) {
            tracing::warn!("attn_page_sparse (window desk scorer): {msg}");
        }
    }

    /// The query rows of the sequence keyed `block0`, one entry per layer that captured them.
    pub(super) fn read_queries(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        block0: u32,
    ) -> Result<Vec<LayerQuery>> {
        let guard = lock(&self.queries);
        let Some(store) = guard.as_ref() else {
            return Ok(Vec::new());
        };
        if store.owner != arena::backend_key(gpu) {
            return Ok(Vec::new());
        }
        let mut layers: Vec<(usize, LayerQ)> = store.layers.iter().map(|(&l, &q)| (l, q)).collect();
        layers.sort_by_key(|&(l, _)| l);
        let want = block0 as i32;
        let mut out = Vec::new();
        for (layer, lq) in layers {
            let mut cols = [0u8; COLS_BYTES];
            gpu.copy_d2h_on_stream(lq.cols, &mut cols, stream)?;
            let at = |off: usize, r: usize| {
                i32::from_le_bytes(
                    cols[off + r * 4..off + r * 4 + 4]
                        .try_into()
                        .expect("4 bytes"),
                )
            };
            let live = |r: usize| at(LENS_OFF, r) > 0 && at(0, r) == want;
            let Some(r0) = (0..MAX_Q_ROWS).find(|&r| live(r)) else {
                continue;
            };
            let count = (r0..MAX_Q_ROWS).take_while(|&r| live(r)).count();
            let row_bytes = lq.nq as usize * lq.hd as usize * 2;
            let mut raw = vec![0u8; count * row_bytes];
            gpu.copy_d2h_on_stream(lq.q.offset(r0 * row_bytes), &mut raw, stream)?;
            let rows: Vec<Vec<f32>> = raw
                .chunks_exact(row_bytes)
                .map(|row| {
                    row.chunks_exact(2)
                        .map(|b| bf16_val(u16::from_le_bytes([b[0], b[1]])))
                        .collect()
                })
                .collect();
            out.push(LayerQuery {
                layer,
                num_q_heads: lq.nq as usize,
                rows,
            });
        }
        Ok(out)
    }
}

impl PageScorer for QuestScorer {
    fn name(&self) -> &'static str {
        "quest"
    }

    /// The summary arena's fixed View region (allocated with it, used by the selector's own plan
    /// only) and the query buffers.
    fn reserved_bytes(&self) -> u64 {
        (arena::VIEW_ARENA_BYTES + QUERY_ARENA_BYTES) as u64
    }

    fn on_kv_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<()> {
        KvSelector::on_kv_written(self.sel.as_ref(), c, s, first_pos, n_new)
    }

    fn on_query(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
    ) -> Result<()> {
        if q_pre_rope.is_some()
            || c.phase == SeamPhase::Prefill
            || q.is_null()
            || s.block_table.is_null()
            || s.seq_lens.is_null()
            || s.n == 0
        {
            return Ok(());
        }
        if s.n > MAX_Q_ROWS {
            self.warn_once(0, WHY_ROWS);
            return Ok(());
        }
        let (nq, hd) = (c.num_q_heads, c.kv.head_dim);
        let row_elems = nq as usize * hd as usize;
        if row_elems == 0 || (s.q_row_stride as usize) < row_elems {
            return Ok(());
        }
        let Some(geom) = budget::geometry() else {
            return Ok(());
        };
        let mut guard = lock(&self.queries);
        let stored = ensure_store(&mut guard, c.gpu, c.stream, c.in_graph, geom.generation)?;
        let Some(store) = stored else {
            self.warn_once(1, WHY_CAPTURED);
            return Ok(());
        };
        let Some(lq) = store.layer(c.attn_layer, nq, hd, c.kv.k_pool) else {
            self.warn_once(2, WHY_LAYER);
            return Ok(());
        };
        let gpu = c.gpu;
        // Rows of an earlier, larger step must not survive: zero lengths are ignored on read.
        gpu.memset_async(lq.cols, 0, COLS_BYTES, c.stream)?;
        gpu.copy_d2d_2d_async(
            s.block_table,
            s.max_blocks_per_seq as usize * 4,
            lq.cols,
            4,
            4,
            s.n,
            c.stream,
        )?;
        gpu.copy_d2d_async(s.seq_lens, lq.cols.offset(LENS_OFF), s.n * 4, c.stream)?;
        gpu.copy_d2d_2d_async(
            q,
            s.q_row_stride as usize * 2,
            lq.q,
            row_elems * 2,
            row_elems * 2,
            s.n,
            c.stream,
        )
    }

    fn on_demote(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        block0: u32,
        pages: &[(u32, u32)],
    ) -> Result<()> {
        let Some(geom) = budget::geometry() else {
            return Ok(());
        };
        let pb = geom.layout.page_blocks.max(1);
        let leads: Vec<(u32, u32)> = pages
            .iter()
            .copied()
            .filter(|&(i, _)| i % pb == 0)
            .collect();
        if leads.is_empty() {
            return Ok(());
        }
        let full = full_mask(geom.layout.page_tokens() as usize);
        let saved = self.sel.with_summaries(gpu, |a, _| {
            let mut out = Vec::with_capacity(leads.len());
            for &(idx, phys) in &leads {
                out.push((idx / pb, read_page(gpu, stream, a, full, phys)?));
            }
            Ok(out)
        })?;
        // No arena yet: no kernel ever folded a key, so there is nothing to save.
        let Some(saved) = saved else {
            return Ok(());
        };
        let mut host = lock(&self.host);
        host.reset_for(geom.generation);
        for (page, rec) in saved {
            let counter = if rec.is_some() { &SAVED } else { &UNTRUSTED };
            counter.fetch_add(1, Ordering::Relaxed);
            host.put(block0, page, rec);
        }
        host.evict(block0);
        Ok(())
    }

    fn rank(&self, gpu: &dyn GpuBackend, req: &ScoreRequest, k: usize) -> Result<Vec<u32>> {
        let Some(geom) = budget::geometry() else {
            return Ok(Vec::new());
        };
        if k == 0 || req.candidates.is_empty() {
            return Ok(Vec::new());
        }
        let pb = geom.layout.page_blocks.max(1);
        let mut wanted: Vec<u32> = req.candidates.iter().map(|b| b / pb).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let records = {
            let mut host = lock(&self.host);
            host.reset_for(geom.generation);
            host.records(req.block0, &wanted)
        };
        if records.is_empty() {
            NO_SUMMARY.fetch_add(1, Ordering::Relaxed);
            return Ok(Vec::new());
        }
        let queries = self.read_queries(gpu, req.stream, req.block0)?;
        if queries.is_empty() {
            NO_QUERY.fetch_add(1, Ordering::Relaxed);
            return Ok(Vec::new());
        }
        let gqa_sum = self.sel.config().gqa == GqaMode::Sum;
        let ranked = scorer_rank::rank_pages(&geom.layout.layers, &queries, &records, gqa_sum);
        RANKED.fetch_add(1, Ordering::Relaxed);
        PAGES_RANKED.fetch_add(ranked.len() as u64, Ordering::Relaxed);
        Ok(scorer_rank::blocks_of(&ranked, req.candidates, pb, k))
    }
}
