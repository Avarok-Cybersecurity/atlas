// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the sparse-attention seam (lane L29), the one
//! place an approximate KV selector plugs into the Qwen3 full-attention decode,
//! verify and prefill paths: Quest page selection (L30), the bounded view
//! (L31/L32), RetroInfer (L33/L34), the host window desk (L36) and the tree
//! verify device plan. The names below are the seam's contract and are
//! binding, and every addition to it is marked "extension".
//!
//! # Off is byte-identical
//! Every call site is one `if let Some(sel) = sparse_seam::active()` branch in
//! front of the unchanged dense code. With no selector registered (every
//! target, every row off) [`active`] is one atomic load that answers `None`
//! and the dense branch runs exactly as at `52d6b0e9a`.
//!
//! # Hook order per layer call
//! - Prefill (chunk 0 through the cache-skip path, chunk 1+ through the paged
//!   path, single stream or batched): `on_query` before RoPE (with the
//!   pre-RoPE, post-norm Q when the site has it), `on_kv_written` after the
//!   K/V write, `plan_prefill` (paged single-stream chunks only), and
//!   `on_prefill_layer_done` when the layer's attention output is final.
//! - Decode and verify (single sequence and the batched multi-sequence
//!   path): `on_kv_written`, then `on_query` (post-RoPE Q, `q_pre_rope =
//!   None`), then `plan`, whose [`SparsePlan`] decides what the paged-decode
//!   attention reads.
//!
//! # Graph rule
//! Decode and verify steps are captured into CUDA graphs and replayed without
//! running any host code, so a hook runs on the host only at capture time and
//! on eager steps; `SeamCtx::in_graph` says which. While a selector is active
//! it returns `View` on every decode step (an identity view below its minimum
//! context), backed by selector-owned fixed device buffers that its captured
//! kernels refresh on every replay, so a captured graph never changes
//! topology. A `Custom` plan whose `graph_safe()` is false is bypassed while
//! capturing (the dense tables are captured instead) and counted in
//! `atlas_sparse_graph_bypass_total`.
//!
//! # Memory
//! A selector reports its device footprint through `reserved_bytes()` at
//! registration and allocates it after the model build (lazily, on its first
//! eager hook). The seam charges the total to the KV-block budget
//! ([`charge_kv_budget_for_pool`], called from the model build) so
//! `--gpu-memory-utilization` stays honest, and logs it once. A per-page
//! sidecar (`reserved_bytes_per_kv_block`) is charged in proportion to the
//! pool the budget leaves.
//!
//! # Startup refusals
//! [`validate_serve`] (called while the serve resolves its KV cache config,
//! after every selector registered) refuses a KV dtype other than BF16 or FP8,
//! `--high-speed-swap`, and a vision checkpoint served without `--no-vision`
//! (mRoPE vision sequences).

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

pub mod identity;
mod metrics;
mod registry;
pub(crate) mod route;
mod sites;

#[cfg(test)]
#[path = "sparse_seam_geometry_tests.rs"]
mod geometry_tests;
#[cfg(test)]
#[path = "sparse_seam_mock_tests.rs"]
mod mock_tests;
#[cfg(test)]
#[path = "sparse_seam_site_tests.rs"]
mod site_tests;
#[cfg(test)]
#[path = "sparse_seam_tests.rs"]
mod tests;

pub use metrics::{record_selected_pages, render_prometheus};
pub use registry::{StepGuard, publish_step};
pub(crate) use sites::{PrefillSite, SeamDims, SeamRows};

/// Sequence id given to a padding row of a batched step (no sequence behind it).
pub const PAD_SEQ_ID: u64 = u64::MAX;

/// KV storage of the layer a hook is called for. The seam serves only these two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtypeTag {
    Bf16,
    Fp8,
}

impl KvDtypeTag {
    /// The tag of a KV cache dtype the seam serves, `None` for every other one.
    pub fn from_cache(dtype: KvCacheDtype) -> Option<Self> {
        match dtype {
            KvCacheDtype::Bf16 => Some(Self::Bf16),
            KvCacheDtype::Fp8 => Some(Self::Fp8),
            _ => None,
        }
    }

    /// Bytes per stored K or V element.
    pub fn bytes_per_elem(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }
}

/// One attention layer's paged KV pools, in the layout the paged-decode
/// kernels read (`kernels/*/common/paged_decode_attn*.cu`).
#[derive(Clone, Copy, Debug)]
pub struct KvLayerView {
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub dtype: KvDtypeTag,
    pub block_size: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// FP8: the K dequant scale the decode kernels use (calibrated when
    /// calibration is on). 1.0 for BF16.
    pub kv_scale: f32,
    /// Extension: the FP8 V dequant scale. 1.0 for BF16.
    pub v_scale: f32,
    /// Extension: bytes from one block of this layer's pool to the next.
    pub block_stride_bytes: u64,
    /// Extension: the element stride the FP8 paged kernels take
    /// (`PagedKvCache::cache_stride`).
    pub cache_stride: u64,
}

/// Which forward the hook belongs to (extension).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamPhase {
    /// One new token per sequence.
    Decode,
    /// `rows_per_seq > 1` rows of one sequence (speculative verify).
    Verify,
    /// A prompt chunk.
    Prefill,
}

/// The rows one hook call covers.
///
/// Rows are what the paged attention calls "sequences": one per decode
/// sequence, `rows_per_seq` per sequence in verify (consecutive rows, one per
/// verify token, with the same block table and lengths one apart), one per
/// stream in prefill. Host slices are capture-time values under a graph.
pub struct SeqBatch<'a> {
    /// Attention rows.
    pub n: usize,
    /// `[n * max_blocks_per_seq]` i32 block table on the device. NULL in
    /// chunk-0 prefill (the cache-skip attention is contiguous; use
    /// `host_block_tables`) and in batched prefill (use `block_table_ptrs`).
    pub block_table: DevicePtr,
    /// `[n]` i32 attention extent per row on the device (this step's tokens
    /// included). NULL wherever `block_table` is NULL, except batched prefill,
    /// where it is the per-stream KV length array when the step is varlen.
    pub seq_lens: DevicePtr,
    pub max_blocks_per_seq: u32,
    /// Attention extent per row, this step's tokens included; 0 marks a
    /// padding row of a batched step.
    pub host_seq_lens: &'a [u32],
    /// One id per logical sequence (`n / rows_per_seq` entries) when the
    /// caller published them ([`publish_step`]); empty otherwise. Padding
    /// sequences carry [`PAD_SEQ_ID`].
    pub seq_ids: &'a [u64],
    pub rows_per_seq: usize,
    /// Extension: BF16 elements from one row of `q` to the next. In prefill
    /// `q` is `[q_rows, num_q_heads * head_dim]`, stacked in stream order.
    pub q_row_stride: u32,
    /// Extension: rows in `q` (`n` in decode and verify, the chunk's token
    /// count in prefill).
    pub q_rows: usize,
    /// Extension: per logical sequence, the absolute position of its first
    /// new token in this call.
    pub first_pos: &'a [u32],
    /// Extension: per logical sequence, the new tokens in this call (1 in
    /// decode, `rows_per_seq` in verify, the chunk length in prefill, 0 for
    /// padding).
    pub new_tokens: &'a [u32],
    /// Extension: per logical sequence, its host block table (physical block
    /// ids in order); empty where the site has none (batched prefill).
    pub host_block_tables: &'a [&'a [u32]],
    /// Extension: batched prefill only, the `[n]` device array of per-stream
    /// block-table pointers (`BatchedAttnMetadata::block_table_ptrs`); NULL
    /// everywhere else.
    pub block_table_ptrs: DevicePtr,
}

/// What every hook receives about the layer and the stream.
pub struct SeamCtx<'a> {
    pub gpu: &'a dyn GpuBackend,
    /// Attention-layer index (the KV pool index), not the model layer index.
    pub attn_layer: usize,
    pub kv: KvLayerView,
    pub stream: u64,
    /// True while the stream is being captured into a CUDA graph.
    pub in_graph: bool,
    /// Extension: decode, verify or prefill.
    pub phase: SeamPhase,
    /// Extension: query heads of this layer (GQA group = `num_q_heads / kv.kv_heads`).
    pub num_q_heads: u32,
    /// Extension: the attention scale the dense kernels use.
    pub inv_sqrt_d: f32,
}

/// What the decode and verify attention reads this step.
pub enum SparsePlan {
    /// The real block table, unchanged.
    Dense,
    /// A substitute table in selector-owned device buffers: `[n *
    /// max_blocks_per_seq]` i32 block ids and `[n]` i32 lengths, rows in the
    /// same order as the real table. Pages stay in chronological order so the
    /// causal structure of verify rows is kept.
    View {
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
    },
    /// The selector computes the attention output itself.
    Custom(Arc<dyn CustomAttention>),
}

/// Extension: what a paged prefill chunk (chunk 1 and later, single stream)
/// attends to. Prefill runs eager, so the selector knows the extent on the host.
pub enum PrefillPlan {
    Dense,
    /// A substitute `[blocks]` i32 table holding the kept history pages in
    /// chronological order followed by every page this chunk writes, and its
    /// extent `kv_len` (this chunk's tokens included, at the end). The chunk's
    /// queries attend causally inside the last `num_tokens` positions.
    View {
        block_table: DevicePtr,
        kv_len: u32,
    },
}

/// An attention kernel a selector runs in place of the paged decode.
pub trait CustomAttention: Send + Sync {
    /// Write `[n, num_q_heads, head_dim]` BF16 attention output to `out`.
    fn run(&self, c: &SeamCtx, s: &SeqBatch, q: DevicePtr, out: DevicePtr) -> Result<()>;
    /// True when `run` only issues capturable work on `c.stream`.
    fn graph_safe(&self) -> bool;
}

/// An approximate KV selector. At most one is active per serve.
pub trait KvSelector: Send + Sync {
    /// Metric label and log name, for example `"quest"`.
    fn name(&self) -> &'static str;
    /// After this call's K/V rows are in the paged cache (inside the graph
    /// when decoding). `first_pos` and `n_new` are per logical sequence and
    /// describe exactly the rows written (a warm prefix replay writes fewer
    /// than it computes).
    fn on_kv_written(
        &self,
        c: &SeamCtx,
        s: &SeqBatch,
        first_pos: &[u32],
        n_new: &[u32],
    ) -> Result<()>;
    /// The call's Q. `q_pre_rope` is `Some(q)` when the hook runs before RoPE
    /// (prefill sites: `q` then still holds the post-norm, pre-RoPE values and
    /// RoPE rotates it in place right after the call); `None` when only the
    /// post-RoPE Q exists (decode and verify, and a chunk-0 prefill whose Q
    /// RoPE is fused into the deinterleave). Copy what you need during the call.
    fn on_query(
        &self,
        _c: &SeamCtx,
        _s: &SeqBatch,
        _q_pre_rope: Option<DevicePtr>,
        _q: DevicePtr,
    ) -> Result<()> {
        Ok(())
    }
    /// Decode and verify: what the attention reads. With `rows_per_seq > 1`
    /// one selection is shared by all rows of a sequence.
    fn plan(&self, c: &SeamCtx, s: &SeqBatch, q: DevicePtr) -> Result<SparsePlan>;
    /// After the layer's prefill attention for this chunk. `last_chunk` comes
    /// from the caller's [`publish_step`] (false when unpublished).
    fn on_prefill_layer_done(&self, _c: &SeamCtx, _s: &SeqBatch, _last_chunk: bool) -> Result<()> {
        Ok(())
    }
    /// A sequence the caller published ended; drop its state.
    fn on_seq_free(&self, _seq_id: u64) {}
    /// Extension: device bytes the selector will allocate after the model
    /// build. Charged to the KV-block budget.
    fn reserved_bytes(&self) -> u64 {
        0
    }
    /// Extension: the KV pool's geometry, delivered by the model build
    /// ([`charge_kv_budget_for_pool`]) right before it reads
    /// [`reserved_bytes`](Self::reserved_bytes) and
    /// [`reserved_bytes_per_kv_block`](Self::reserved_bytes_per_kv_block), so a
    /// per-page sidecar can be sized before the first prefill. Called at most
    /// once per model build, only on a serve with this selector registered.
    fn on_pool_geometry(&self, _pool: &PoolGeometry) {}
    /// Extension: device bytes the selector allocates per KV page of the pool
    /// (a per-page sidecar). The pool is sized from the budget this charge
    /// leaves, so the charge is proportional: the build gives the sidecar
    /// `pages * per` bytes with `pages = budget / (kv_block_bytes + per)`.
    fn reserved_bytes_per_kv_block(&self) -> u64 {
        0
    }
    /// Extension: paged prefill chunks (chunk 1 and later, single stream).
    fn plan_prefill(&self, _c: &SeamCtx, _s: &SeqBatch, _q: DevicePtr) -> Result<PrefillPlan> {
        Ok(PrefillPlan::Dense)
    }
}

/// Register the serve's selector. At most one: a second registration fails
/// startup with both names, so two approximate rows on at once are refused.
pub fn register(sel: Arc<dyn KvSelector>) -> Result<()> {
    registry::ACTIVE.register(sel)
}

/// The registered selector, `None` on every serve with no approximate row on.
pub fn active() -> Option<&'static dyn KvSelector> {
    registry::ACTIVE.get()
}

/// Device bytes the active selector reserved (0 with none).
pub fn reserved_bytes() -> u64 {
    active().map_or(0, |s| s.reserved_bytes())
}

/// Charge the active selector's reservation to the KV-block budget. Returns
/// the budget unchanged with no selector; refuses a reservation that would
/// leave no KV budget at all.
pub fn charge_kv_budget(kv_budget: usize) -> Result<usize> {
    charge_with(active(), kv_budget)
}

pub(crate) fn charge_with(sel: Option<&dyn KvSelector>, kv_budget: usize) -> Result<usize> {
    let Some(sel) = sel else {
        return Ok(kv_budget);
    };
    let reserve = usize::try_from(sel.reserved_bytes()).unwrap_or(usize::MAX);
    if reserve == 0 {
        return Ok(kv_budget);
    }
    anyhow::ensure!(
        reserve < kv_budget,
        "sparse attention: selector `{}` reserves {:.2} GiB of device memory, more than the \
         whole KV budget left by --gpu-memory-utilization ({:.2} GiB); lower its budget flags \
         or raise --gpu-memory-utilization",
        sel.name(),
        reserve as f64 / f64::from(1u32 << 30),
        kv_budget as f64 / f64::from(1u32 << 30),
    );
    tracing::info!(
        "KV budget: reserving {:.1} MiB for the sparse-attention selector `{}` (charged to the KV-block budget)",
        reserve as f64 / f64::from(1u32 << 20),
        sel.name(),
    );
    Ok(kv_budget - reserve)
}

/// Extension: what the model build knows about the KV pool when it charges
/// the selector (the block byte size and the per-layer geometry).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolGeometry {
    /// Bytes of one KV block across every attention layer, K and V (the
    /// divisor the pool's block count is computed with).
    pub kv_block_bytes: usize,
    /// Tokens per KV block.
    pub block_size: usize,
    /// `(kv_heads, head_dim)` of each attention layer, in pool order.
    pub layer_dims: Vec<(usize, usize)>,
}

impl PoolGeometry {
    /// The geometry of a KV cache config.
    pub fn from_config(cfg: &KvCacheConfig) -> Self {
        Self {
            kv_block_bytes: cfg.block_bytes_kv_all_layers(),
            block_size: cfg.block_size,
            layer_dims: (0..cfg.num_layers).map(|i| cfg.dims_for_layer(i)).collect(),
        }
    }

    /// Attention layers that own a KV pool.
    pub fn attn_layers(&self) -> usize {
        self.layer_dims.len()
    }

    /// `(kv_heads, head_dim)` when every attention layer has the same, else `None`.
    pub fn uniform_dims(&self) -> Option<(usize, usize)> {
        let first = *self.layer_dims.first()?;
        self.layer_dims.iter().all(|d| *d == first).then_some(first)
    }
}

/// [`charge_kv_budget`] plus the per-page charge (extension): the model build
/// calls this one, with the KV config it sizes the pool from. With no selector
/// registered it returns the budget unchanged and builds nothing.
pub fn charge_kv_budget_for_pool(cfg: &KvCacheConfig, kv_budget: usize) -> Result<usize> {
    match active() {
        None => Ok(kv_budget),
        Some(sel) => charge_pool_with(Some(sel), kv_budget, &PoolGeometry::from_config(cfg)),
    }
}

/// The fixed charge (`reserved_bytes`), then the proportional one: the selector
/// gets `reserved_bytes_per_kv_block` for each page the pool will hold,
/// `pages = budget / (kv_block_bytes + per)`, so the pool built from what is
/// left holds at least `pages` pages (it can hold one more from rounding, and
/// later charges only shrink it). A selector that returns 0 per page gets
/// exactly the fixed charge.
pub(crate) fn charge_pool_with(
    sel: Option<&dyn KvSelector>,
    kv_budget: usize,
    pool: &PoolGeometry,
) -> Result<usize> {
    let Some(sel) = sel else {
        return Ok(kv_budget);
    };
    sel.on_pool_geometry(pool);
    let kv_budget = charge_with(Some(sel), kv_budget)?;
    let per = usize::try_from(sel.reserved_bytes_per_kv_block()).unwrap_or(usize::MAX);
    if per == 0 || pool.kv_block_bytes == 0 {
        return Ok(kv_budget);
    }
    let pages = kv_budget / pool.kv_block_bytes.saturating_add(per);
    anyhow::ensure!(
        pages > 0,
        "sparse attention: selector `{}` needs {per} B of device memory per KV page, which leaves \
         no room for a KV block in the budget left by --gpu-memory-utilization ({:.2} GiB); \
         lower its budget flags or raise --gpu-memory-utilization",
        sel.name(),
        kv_budget as f64 / f64::from(1u32 << 30),
    );
    let charge = pages.saturating_mul(per);
    tracing::info!(
        "KV budget: reserving {:.1} MiB for the sparse-attention selector `{}` per-page sidecar \
         ({pages} pages x {per} B, charged to the KV-block budget)",
        charge as f64 / f64::from(1u32 << 20),
        sel.name(),
    );
    Ok(kv_budget - charge)
}

/// The serve facts the seam's startup refusals read.
pub struct ServeFacts<'a> {
    /// The base KV dtype and every per-layer override.
    pub kv_dtypes: &'a [KvCacheDtype],
    pub high_speed_swap: bool,
    /// The checkpoint has a vision tower and `--no-vision` is not set.
    pub vision_input: bool,
}

/// Refuse a serve the active selector cannot run on. Ok with no selector.
pub fn validate_serve(f: &ServeFacts) -> Result<()> {
    validate_with(active(), f)
}

pub(crate) fn validate_with(sel: Option<&dyn KvSelector>, f: &ServeFacts) -> Result<()> {
    let Some(sel) = sel else {
        return Ok(());
    };
    let name = sel.name();
    if let Some(bad) = f
        .kv_dtypes
        .iter()
        .find(|d| KvDtypeTag::from_cache(**d).is_none())
    {
        anyhow::bail!(
            "sparse attention: selector `{name}` reads BF16 or FP8 KV only, but this serve stores \
             {bad:?} KV; use --kv-cache-dtype bf16 or fp8"
        );
    }
    anyhow::ensure!(
        !f.high_speed_swap,
        "sparse attention: selector `{name}` cannot run with --high-speed-swap (the streamed \
         window is not in the paged pool the selector reads)"
    );
    anyhow::ensure!(
        !f.vision_input,
        "sparse attention: selector `{name}` does not serve mRoPE vision sequences; pass \
         --no-vision to serve this checkpoint as text only"
    );
    Ok(())
}

/// A sequence the caller published through [`publish_step`] ended.
pub fn seq_freed(seq_id: u64) {
    if let Some(sel) = active() {
        sel.on_seq_free(seq_id);
    }
}
