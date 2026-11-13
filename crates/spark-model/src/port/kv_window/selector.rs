// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): attention over the pages on the desk only, every key keeping the RoPE position it was written with, taken from the port plan's description (report 77 section 2.5); no code copied or fetched. Atlas reads the desk through a compacted block table over its paged KV instead of staging pages into ordinary KV cells.

//! Responsibility: the window desk's attention side (port plan lane L36): the `KvSelector`
//! registered with the sparse-attention seam (contract 3.7.6), its device buffers, the
//! compacted-table view the decode and verify attention reads, and the recall scorer seam.
//!
//! # The view
//! A demoted block keeps its logical index: the sequence's block table holds the desk's HOLE
//! block there (one zeroed block the desk owns, reference-counted once per table entry). Every
//! decode and verify step the selector launches `kv_window_view_compact`
//! (`kernels/ada/common/kv_page_fetch_ada.cu`) over the real table and lengths the model just
//! uploaded: each row's table without its hole entries, in order, and its length minus
//! `block_size` per removed (always full) block. Keys keep their RoPE positions, so attending
//! the compacted table is exact attention over the pages on the desk; verify rows of one
//! sequence share one table, so one selection serves every row (contract 3.7.6).
//!
//! # Graph rule
//! The kernel reads the hole id from a device slot and writes into fixed buffers of one
//! 32 MiB arena, so a captured graph replays correctly whatever the desk does later. The arena
//! is allocated on the first eager hook of a backend (the desk's block hook runs before every
//! decode forward) and its hole slot starts at -1 (nothing hidden: the view is the identity). A
//! plan that cannot return a view while capturing marks the desk inert for the serve: a graph
//! captured with the dense table must never meet a hidden page. With pages hidden, a call that
//! cannot build the view fails rather than read the hole densely.
//!
//! # Recall scorer
//! Only one selector runs per serve, so a page scorer (Quest bounds, lane L30) rides on this
//! one through [`PageScorer`]: the desk forwards the seam's KV and query hooks, tells it which
//! blocks leave the GPU, and asks it to rank the demoted pages at each recall round. With no
//! scorer registered the desk is sinks plus the recent window.
//!
//! A Quest scorer (`port::sparse_quest`, lane L30) rebuilds the summary of any page whose
//! fingerprint does not match the K pool, so on a page that lives off the GPU it would read a
//! stale or reused physical block. `ScoreRequest::table` holds the hole block at demoted
//! indices and `candidates` names the demoted logical pages, so a scorer must not look a
//! candidate up in the K pool through that table. It must either copy each page's summary
//! in `on_demote` (the physical block is reused once the copy completed) and keep it keyed by
//! logical page, or score only the resident pages through a table of its own.
//!
//! On a target without the kernel module the selector plans `Dense` and the desk never hides a
//! page (one log line).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

use super::desk::DeviceHooks;
use crate::layers::qwen3_attention::sparse_seam::{
    self, KvSelector, PrefillPlan, SeamCtx, SeqBatch, SparsePlan,
};
use crate::layers::try_target_kernel;

/// The Ada module (`kernels/ada/common/kv_page_fetch_ada.cu`) and its view kernel.
pub const MODULE: &str = "kv_page_fetch_ada";
pub const VIEW_KERNEL: &str = "kv_window_view_compact";

/// Device bytes the selector reserves (charged to the KV-block budget by the seam).
pub const ARENA_BYTES: usize = 32 << 20;
const ALIGN: usize = 256;
const HOLE_OFF: usize = 0;
const PREFILL_OFF: usize = ALIGN;
/// Bytes of the prefill view table (262,144 blocks of 4 bytes).
pub const PREFILL_BYTES: usize = 1 << 20;
const VIEW_OFF: usize = PREFILL_OFF + PREFILL_BYTES;
/// Bytes of the decode and verify view (table and lengths).
pub const VIEW_BYTES: usize = ARENA_BYTES - VIEW_OFF;

const NO_HOLE: u64 = u64::MAX;

struct Device {
    owner: usize,
    base: DevicePtr,
    kernel: KernelHandle,
}

static DEVICE: Mutex<Option<Device>> = Mutex::new(None);
static HOLE: AtomicU64 = AtomicU64::new(NO_HOLE);
static HIDDEN: AtomicBool = AtomicBool::new(false);
static CAPTURED_DENSE: AtomicBool = AtomicBool::new(false);
static WARNED_ABSENT: AtomicBool = AtomicBool::new(false);
static WARNED_FIT: AtomicBool = AtomicBool::new(false);

fn backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu).cast::<()>() as usize
}

fn align(v: usize) -> usize {
    v.div_ceil(ALIGN) * ALIGN
}

/// Bytes a view of `rows` rows of `max_blocks` entries occupies (table, then lengths).
pub fn view_bytes(rows: usize, max_blocks: usize) -> usize {
    align(rows * max_blocks * 4) + align(rows * 4)
}

/// The arena base and the view kernel on `gpu` (allocated now when `may_alloc`; never while
/// capturing). A new backend (a model swap) gets a fresh arena and nothing hidden.
fn device(
    gpu: &dyn GpuBackend,
    may_alloc: bool,
    stream: u64,
) -> Result<Option<(DevicePtr, KernelHandle)>> {
    let key = backend_key(gpu);
    let mut slot = DEVICE.lock();
    if let Some(d) = slot.as_ref()
        && d.owner == key
    {
        return Ok(Some((d.base, d.kernel)));
    }
    if !may_alloc {
        return Ok(None);
    }
    let base = gpu.alloc(ARENA_BYTES)?;
    // Hole slot = -1: nothing hidden, every view is the identity.
    gpu.memset_async(base.offset(HOLE_OFF), 0xFF, 4, stream)?;
    let kernel = try_target_kernel(gpu, MODULE, VIEW_KERNEL);
    if kernel.0 == 0 && !WARNED_ABSENT.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "kv window desk: `{MODULE}::{VIEW_KERNEL}` is not in this build; the desk keeps every \
             page on the GPU (attention stays dense)"
        );
    }
    HOLE.store(NO_HOLE, Ordering::Relaxed);
    *slot = Some(Device {
        owner: key,
        base,
        kernel,
    });
    Ok(Some((base, kernel)))
}

/// The host value of the hole block id, `None` before the desk hid a page.
pub fn hole() -> Option<u32> {
    match HOLE.load(Ordering::Relaxed) {
        NO_HOLE => None,
        v => Some(v as u32),
    }
}

/// Whether the desk holds any page off the GPU now (kept by the desk's block hook).
pub(crate) fn set_hidden(on: bool) {
    HIDDEN.store(on, Ordering::Relaxed);
}

/// The production [`DeviceHooks`]: the seam arena and its hole slot.
pub(crate) struct SeamDevice;

impl DeviceHooks for SeamDevice {
    fn ready(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<bool> {
        let may_alloc = !gpu.stream_is_capturing(stream);
        Ok(match device(gpu, may_alloc, stream)? {
            Some((_, k)) => k.0 != 0 && !CAPTURED_DENSE.load(Ordering::Relaxed),
            None => false,
        })
    }

    fn publish_hole(&self, gpu: &dyn GpuBackend, hole: Option<u32>, stream: u64) -> Result<()> {
        let base = DEVICE
            .lock()
            .as_ref()
            .filter(|d| d.owner == backend_key(gpu))
            .map(|d| d.base);
        let Some(base) = base else {
            // No arena on this backend: its first allocation starts at -1 anyway.
            HOLE.store(NO_HOLE, Ordering::Relaxed);
            return Ok(());
        };
        let v: i32 = hole.map_or(-1, |h| h as i32);
        gpu.copy_h2d_async(&v.to_le_bytes(), base.offset(HOLE_OFF), stream)?;
        HOLE.store(hole.map_or(NO_HOLE, u64::from), Ordering::Relaxed);
        Ok(())
    }
}

/// CPU oracle of `kv_window_view_compact` for one row: the first `ceil(len / bs)` entries of
/// `table` without `hole`, in order, and `len` minus `bs` per removed entry.
pub fn compact_row(table: &[u32], len: u32, block_size: u32, hole: Option<u32>) -> (Vec<u32>, u32) {
    let bs = block_size.max(1);
    let n = (len.div_ceil(bs) as usize).min(table.len());
    let kept: Vec<u32> = table[..n]
        .iter()
        .copied()
        .filter(|&b| Some(b) != hole)
        .collect();
    let removed = (n - kept.len()) as u32;
    (kept, len.saturating_sub(removed * bs))
}

/// What the desk asks a recall scorer (one sequence, between two steps, eager).
pub struct ScoreRequest<'a> {
    /// The sequence's first physical block (the desk's key for it).
    pub block0: u32,
    /// Its block table, hole entries included.
    pub table: &'a [u32],
    pub kv_tokens: usize,
    pub block_size: usize,
    /// Demoted logical block indices, ascending: the pages the scorer may rank.
    pub candidates: &'a [u32],
    pub stream: u64,
}

/// A page ranker riding on the desk (see the module doc). Every method but `rank` has a default.
pub trait PageScorer: Send + Sync {
    fn name(&self) -> &'static str;
    /// Device bytes the scorer allocates (charged with the desk's).
    fn reserved_bytes(&self) -> u64 {
        0
    }
    /// The seam's KV hook, forwarded.
    fn on_kv_written(
        &self,
        _c: &SeamCtx,
        _s: &SeqBatch,
        _first_pos: &[u32],
        _n_new: &[u32],
    ) -> Result<()> {
        Ok(())
    }
    /// The seam's query hook, forwarded.
    fn on_query(
        &self,
        _c: &SeamCtx,
        _s: &SeqBatch,
        _q_pre_rope: Option<DevicePtr>,
        _q: DevicePtr,
    ) -> Result<()> {
        Ok(())
    }
    /// Logical blocks `(index, physical block)` of `block0`'s sequence are leaving the GPU: copy
    /// any per-block summary now (the physical block is reused once its copy completed).
    fn on_demote(
        &self,
        _gpu: &dyn GpuBackend,
        _stream: u64,
        _block0: u32,
        _pages: &[(u32, u32)],
    ) -> Result<()> {
        Ok(())
    }
    /// At most `k` of `req.candidates`, best first.
    fn rank(&self, gpu: &dyn GpuBackend, req: &ScoreRequest, k: usize) -> Result<Vec<u32>>;
    /// The seam's sequence-end hook, forwarded.
    fn on_seq_free(&self, _seq_id: u64) {}
}

static SCORER: OnceLock<Arc<dyn PageScorer>> = OnceLock::new();

/// Register the desk's recall scorer (at most one, before the model is built).
pub fn set_page_scorer(s: Arc<dyn PageScorer>) -> Result<()> {
    let name = s.name();
    SCORER.set(s).map_err(|_| {
        anyhow::anyhow!(
            "kv window desk: scorer `{name}` cannot register, `{}` already ranks the desk's pages",
            SCORER.get().map_or("?", |s| s.name())
        )
    })?;
    tracing::info!("kv window desk: recall pages are ranked by `{name}`");
    Ok(())
}

/// The registered scorer, if any.
pub fn page_scorer() -> Option<&'static dyn PageScorer> {
    SCORER.get().map(|s| s.as_ref())
}

/// The selector the desk registers with the seam.
#[derive(Default)]
pub struct WindowDeskSelector;

impl WindowDeskSelector {
    pub fn new() -> Self {
        Self
    }
}

/// `Dense` where that is safe; marks the desk inert when a graph would bake it.
fn dense(c: &SeamCtx, why: &str) -> Result<SparsePlan> {
    if HIDDEN.load(Ordering::Relaxed) {
        bail!(
            "kv window desk: {why} while pages are on the host; refusing a dense read of the hole"
        );
    }
    if c.in_graph && !CAPTURED_DENSE.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "kv window desk: a decode graph was captured with the dense table ({why}); the desk keeps \
             every page on the GPU for the rest of this serve"
        );
    }
    Ok(SparsePlan::Dense)
}

fn plan_view(c: &SeamCtx, s: &SeqBatch) -> Result<SparsePlan> {
    if s.n == 0 || s.block_table.is_null() || s.seq_lens.is_null() || s.max_blocks_per_seq == 0 {
        return dense(c, "an attention call without a device block table");
    }
    let Some((base, kernel)) = device(c.gpu, !c.in_graph, c.stream)? else {
        return dense(c, "no view buffers yet");
    };
    if kernel.0 == 0 {
        // Inert build: the desk never hides a page, so the real table is the view.
        return Ok(SparsePlan::Dense);
    }
    let mb = s.max_blocks_per_seq as usize;
    if view_bytes(s.n, mb) > VIEW_BYTES {
        if !WARNED_FIT.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "kv window desk: a {}-row x {mb}-block view does not fit the {} MiB arena",
                s.n,
                VIEW_BYTES >> 20
            );
        }
        return dense(c, "the view does not fit its buffers");
    }
    let out_table = base.offset(VIEW_OFF);
    let out_lens = out_table.offset(align(s.n * mb * 4));
    let n = s.n as u32;
    let (nb, mbb, bsb) = (
        n.to_le_bytes(),
        s.max_blocks_per_seq.to_le_bytes(),
        c.kv.block_size.to_le_bytes(),
    );
    c.gpu.launch_typed(
        kernel,
        [n, 1, 1],
        [32, 1, 1],
        0,
        c.stream,
        &[
            KernelArg::Buffer(s.block_table),
            KernelArg::Buffer(s.seq_lens),
            KernelArg::Buffer(out_table),
            KernelArg::Buffer(out_lens),
            KernelArg::Buffer(base.offset(HOLE_OFF)),
            KernelArg::Bytes(&nb),
            KernelArg::Bytes(&mbb),
            KernelArg::Bytes(&bsb),
        ],
    )?;
    if !c.in_graph && c.attn_layer == 0 {
        sparse_seam::record_selected_pages(selected_pages(s, c.kv.block_size));
    }
    Ok(SparsePlan::View {
        block_table: out_table,
        seq_lens: out_lens,
        max_blocks_per_seq: s.max_blocks_per_seq,
    })
}

/// Pages the view reads this step, from the host tables when the site published them.
fn selected_pages(s: &SeqBatch, block_size: u32) -> u64 {
    let bs = block_size.max(1);
    let rps = s.rows_per_seq.max(1);
    let hole = hole();
    s.host_seq_lens
        .iter()
        .enumerate()
        .map(|(r, &len)| match s.host_block_tables.get(r / rps) {
            Some(t) => {
                let n = (len.div_ceil(bs) as usize).min(t.len());
                t[..n].iter().filter(|&&b| Some(b) != hole).count() as u64
            }
            None => u64::from(len.div_ceil(bs)),
        })
        .sum()
}

fn plan_prefill_view(c: &SeamCtx, s: &SeqBatch) -> Result<PrefillPlan> {
    let Some(hole) = hole() else {
        return Ok(PrefillPlan::Dense);
    };
    let (Some(table), Some(&kv_len)) = (s.host_block_tables.first(), s.host_seq_lens.first())
    else {
        return Ok(PrefillPlan::Dense);
    };
    if s.n != 1 || s.block_table.is_null() || !table.contains(&hole) {
        return Ok(PrefillPlan::Dense);
    }
    let (view, len) = compact_row(table, kv_len, c.kv.block_size, Some(hole));
    if view.len() * 4 > PREFILL_BYTES {
        bail!(
            "kv window desk: a {}-block prefill view does not fit its {} KiB buffer",
            view.len(),
            PREFILL_BYTES >> 10
        );
    }
    let Some((base, _)) = device(c.gpu, !c.in_graph, c.stream)? else {
        bail!(
            "kv window desk: a prefill chunk of a sequence with hidden pages before the view buffers exist"
        );
    };
    let raw: Vec<u8> = view
        .iter()
        .flat_map(|&b| (b as i32).to_le_bytes())
        .collect();
    c.gpu
        .copy_h2d_async(&raw, base.offset(PREFILL_OFF), c.stream)?;
    Ok(PrefillPlan::View {
        block_table: base.offset(PREFILL_OFF),
        kv_len: len,
    })
}

impl KvSelector for WindowDeskSelector {
    fn name(&self) -> &'static str {
        "window_desk"
    }

    fn reserved_bytes(&self) -> u64 {
        ARENA_BYTES as u64 + page_scorer().map_or(0, |s| s.reserved_bytes())
    }

    fn on_kv_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<()> {
        // The first eager hook allocates, so the first decode capture finds the buffers.
        device(c.gpu, !c.in_graph, c.stream)?;
        match page_scorer() {
            Some(sc) => sc.on_kv_written(c, s, first_pos, n_new),
            None => Ok(()),
        }
    }

    fn on_query(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        q_pre_rope: Option<DevicePtr>,
        q: DevicePtr,
    ) -> Result<()> {
        match page_scorer() {
            Some(sc) => sc.on_query(c, s, q_pre_rope, q),
            None => Ok(()),
        }
    }

    fn plan(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<SparsePlan> {
        plan_view(c, s)
    }

    fn plan_prefill(&self, c: &SeamCtx, s: &SeqBatch, _q: DevicePtr) -> Result<PrefillPlan> {
        plan_prefill_view(c, s)
    }

    fn on_seq_free(&self, seq_id: u64) {
        if let Some(sc) = page_scorer() {
            sc.on_seq_free(seq_id);
        }
    }
}
