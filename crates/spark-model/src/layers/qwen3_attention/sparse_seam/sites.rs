// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the seam's layer glue. Each function gathers what one call
//! site of `Qwen3AttentionLayer` already has in hand (the KV pools, the
//! attention metadata, the host lengths and block tables), builds the
//! `SeamCtx` and `SeqBatch` of the seam contract, and calls the selector's
//! hooks. A site calls these only inside its `if let Some(sel) =
//! sparse_seam::active()` branch; a `None` result means "the seam does not
//! serve this layer, run the dense code", so a layer the seam cannot serve
//! (MLA, QSA, hyper-connections, a sliding window, a KV dtype other than BF16
//! or FP8, `--high-speed-swap`) behaves exactly as with no selector.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::registry::current_step;
use super::route::{self, DecodeRoute, PrefillView, RouteEnv};
use super::{KvDtypeTag, KvLayerView, KvSelector, SeamCtx, SeamPhase, SeqBatch};
use crate::layer::{AttnMetadataDev, BatchedAttnMetadata, ForwardContext};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// The host rows of a batched decode or verify call: the orchestrator's
/// per-row pre-append lengths and block tables. [`SeamRows::NONE`] (a site
/// without them) keeps the seam off for that call.
#[derive(Clone, Copy)]
pub(crate) struct SeamRows<'a> {
    pub(crate) seq_lens: &'a [usize],
    pub(crate) block_tables: &'a [Vec<u32>],
}

impl SeamRows<'static> {
    pub(crate) const NONE: Self = Self {
        seq_lens: &[],
        block_tables: &[],
    };
}

/// A layer call's head geometry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeamDims {
    pub(crate) nq: u32,
    pub(crate) nkv: u32,
    pub(crate) hd: u32,
}

/// A prefill call's site inputs (chunk 0 through the cache-skip path, or a
/// paged chunk; single stream or batched).
pub(crate) struct PrefillSite<'a> {
    pub(crate) num_tokens: usize,
    /// Tokens before this chunk (0 for chunk 0).
    pub(crate) seq_len_start: usize,
    /// Local index of the first token whose K/V the call writes (the warm
    /// prefix replay floor; 0 on a cold prefill).
    pub(crate) kv_write_start: usize,
    /// Single stream: the sequence's host block table.
    pub(crate) host_block_table: &'a [u32],
    /// Single-stream paged chunk: the device tables. `None` at chunk 0.
    pub(crate) meta: Option<&'a AttnMetadataDev>,
    pub(crate) batched: Option<&'a BatchedAttnMetadata>,
    /// The call is chunk 0 (the cache-skip path: no device block table).
    pub(crate) chunk0: bool,
    /// `[num_tokens, nq * hd]` BF16 Q.
    pub(crate) q: DevicePtr,
    pub(crate) dims: SeamDims,
    pub(crate) stream: u64,
}

/// Per-stream geometry of a prefill call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefillRows {
    pub(crate) extent: Vec<u32>,
    pub(crate) q_first: Vec<u32>,
    pub(crate) q_len: Vec<u32>,
    pub(crate) written_first: Vec<u32>,
    pub(crate) written_len: Vec<u32>,
}

fn to_u32(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// Ids the caller published for this forward, mapped onto the groups.
fn step_ids(padding: &[bool]) -> (Vec<u64>, bool) {
    match current_step() {
        Some(step) => (route::assign_ids(&step.seq_ids, padding), step.last_chunk),
        None => (Vec::new(), false),
    }
}

/// Per-stream geometry: stream lengths come from the varlen prefix sums when
/// they cover the call, else from the uniform chunk length; extents from the
/// per-stream KV lengths when present. The write floor applies to the stacked
/// token index, so it trims the first streams only.
pub(crate) fn prefill_rows(
    num_tokens: usize,
    seq_len_start: usize,
    kv_write_start: usize,
    batched: Option<&BatchedAttnMetadata>,
) -> PrefillRows {
    let lens: Vec<usize> = match batched {
        None => vec![num_tokens],
        Some(b) => {
            let cu = &b.cu_seqlens_host;
            let covers = cu.len() >= 2
                && cu[0] == 0
                && usize::try_from(*cu.last().unwrap_or(&0)).ok() == Some(num_tokens)
                && cu.windows(2).all(|w| w[1] >= w[0]);
            if covers {
                cu.windows(2).map(|w| (w[1] - w[0]) as usize).collect()
            } else if b.chunk_len > 0 && num_tokens.is_multiple_of(b.chunk_len as usize) {
                vec![b.chunk_len as usize; num_tokens / b.chunk_len as usize]
            } else {
                vec![num_tokens]
            }
        }
    };
    let kv_lens: &[i32] = batched.map_or(&[], |b| b.kv_lens_host.as_slice());
    let floor = kv_write_start.min(num_tokens);
    let mut rows = PrefillRows::default();
    let mut offset = 0usize;
    for (i, &len) in lens.iter().enumerate() {
        let extent = if kv_lens.len() == lens.len() {
            usize::try_from(kv_lens[i]).unwrap_or(0).max(len)
        } else {
            seq_len_start + len
        };
        let first = extent - len;
        let skip = floor.saturating_sub(offset).min(len);
        rows.extent.push(to_u32(extent));
        rows.q_first.push(to_u32(first));
        rows.q_len.push(to_u32(len));
        rows.written_first.push(to_u32(first + skip));
        rows.written_len.push(to_u32(len - skip));
        offset += len;
    }
    rows
}

impl Qwen3AttentionLayer {
    /// The pools a selector reads for this layer, or `None` when the seam
    /// does not serve it.
    pub(crate) fn seam_kv_view(&self, kv_cache: &PagedKvCache, d: SeamDims) -> Option<KvLayerView> {
        if self.mla.is_some()
            || self.qsa.is_some()
            || self.hc.is_some()
            || matches!(self.sliding_window, Some(w) if w > 0)
            || kv_cache.config().cache_blocks_per_seq.is_some()
        {
            return None;
        }
        let dtype = KvDtypeTag::from_cache(self.kv_dtype)?;
        let (kv_scale, v_scale) = match dtype {
            KvDtypeTag::Bf16 => (1.0, 1.0),
            KvDtypeTag::Fp8 => self.effective_fp8_scales(),
        };
        Some(KvLayerView {
            k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
            v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
            dtype,
            block_size: to_u32(kv_cache.block_size()),
            kv_heads: d.nkv,
            head_dim: d.hd,
            kv_scale,
            v_scale,
            block_stride_bytes: kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
            cache_stride: kv_cache.cache_stride() as u64,
        })
    }

    fn seam_ctx<'a>(
        &self,
        ctx: &'a ForwardContext,
        kv: KvLayerView,
        d: SeamDims,
        phase: SeamPhase,
        stream: u64,
    ) -> SeamCtx<'a> {
        SeamCtx {
            gpu: ctx.gpu,
            attn_layer: self.attn_layer_idx,
            kv,
            stream,
            in_graph: ctx.graph_capture,
            phase,
            num_q_heads: d.nq,
            inv_sqrt_d: self.effective_attn_scale(d.hd),
        }
    }

    /// Single-sequence decode (`decode/attention_forward.rs`), after the K/V
    /// write: `on_kv_written`, `on_query`, `plan`. `seq_len` is the
    /// pre-append length (the new token's position).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seam_decode_single(
        &self,
        sel: &dyn KvSelector,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        meta: &AttnMetadataDev,
        q: DevicePtr,
        out: DevicePtr,
        seq_len: usize,
        block_table: &[u32],
        d: SeamDims,
        stream: u64,
    ) -> Result<Option<DecodeRoute>> {
        let Some(kv) = self.seam_kv_view(kv_cache, d) else {
            return Ok(None);
        };
        let c = self.seam_ctx(ctx, kv, d, SeamPhase::Decode, stream);
        let host_seq_lens = [to_u32(seq_len + 1)];
        let first_pos = [to_u32(seq_len)];
        let new_tokens = [1u32];
        let tables = [block_table];
        let (ids, _) = step_ids(&[false]);
        let s = SeqBatch {
            n: 1,
            block_table: meta.block_table,
            seq_lens: meta.seq_len,
            max_blocks_per_seq: meta.max_blocks_per_seq,
            host_seq_lens: &host_seq_lens,
            seq_ids: &ids,
            rows_per_seq: 1,
            q_row_stride: d.nq * d.hd,
            q_rows: 1,
            first_pos: &first_pos,
            new_tokens: &new_tokens,
            host_block_tables: &tables,
            block_table_ptrs: DevicePtr::NULL,
        };
        attend(sel, &c, &s, q, out).map(Some)
    }

    /// The batched multi-sequence decode and K-row verify
    /// (`trait_impl/multi_seq/attn.rs`), after the K/V write. Rows group into
    /// sequences by [`route::group_rows`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seam_decode_multi(
        &self,
        sel: &dyn KvSelector,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        meta: &AttnMetadataDev,
        rows: SeamRows<'_>,
        n: usize,
        q: DevicePtr,
        q_stride: u32,
        out: DevicePtr,
        d: SeamDims,
        stream: u64,
    ) -> Result<Option<DecodeRoute>> {
        if n == 0 || rows.seq_lens.len() != n || rows.block_tables.len() != n {
            return Ok(None);
        }
        let Some(kv) = self.seam_kv_view(kv_cache, d) else {
            return Ok(None);
        };
        let groups = route::group_rows(rows.seq_lens, rows.block_tables);
        let phase = if groups.rows_per_seq > 1 {
            SeamPhase::Verify
        } else {
            SeamPhase::Decode
        };
        let c = self.seam_ctx(ctx, kv, d, phase, stream);
        let (ids, _) = step_ids(&groups.padding);
        let tables: Vec<&[u32]> = groups
            .first_row
            .iter()
            .map(|&r| rows.block_tables[r].as_slice())
            .collect();
        let s = SeqBatch {
            n,
            block_table: meta.block_table,
            seq_lens: meta.seq_len,
            max_blocks_per_seq: meta.max_blocks_per_seq,
            host_seq_lens: &groups.host_seq_lens,
            seq_ids: &ids,
            rows_per_seq: groups.rows_per_seq,
            q_row_stride: q_stride,
            q_rows: n,
            first_pos: &groups.first_pos,
            new_tokens: &groups.new_tokens,
            host_block_tables: &tables,
            block_table_ptrs: DevicePtr::NULL,
        };
        attend(sel, &c, &s, q, out).map(Some)
    }

    /// Run `f` over a prefill call's `SeamCtx` and `SeqBatch`; `None` when the
    /// seam does not serve the layer or the call has no tokens.
    fn with_prefill<R>(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        site: &PrefillSite<'_>,
        f: impl FnOnce(&SeamCtx, &SeqBatch, &PrefillRows, bool) -> Result<R>,
    ) -> Result<Option<R>> {
        if site.num_tokens == 0 {
            return Ok(None);
        }
        let Some(kv) = self.seam_kv_view(kv_cache, site.dims) else {
            return Ok(None);
        };
        let c = self.seam_ctx(ctx, kv, site.dims, SeamPhase::Prefill, site.stream);
        let rows = prefill_rows(
            site.num_tokens,
            site.seq_len_start,
            site.kv_write_start,
            site.batched,
        );
        let (ids, last_chunk) = step_ids(&vec![false; rows.q_len.len()]);
        let tables: Vec<&[u32]> = if site.batched.is_none() {
            vec![site.host_block_table]
        } else {
            Vec::new()
        };
        let (block_table, seq_lens, max_blocks_per_seq, block_table_ptrs) =
            match (site.batched, site.meta) {
                (Some(b), _) => (
                    DevicePtr::NULL,
                    b.kv_lens,
                    b.max_blocks_per_seq,
                    if site.chunk0 {
                        DevicePtr::NULL
                    } else {
                        b.block_table_ptrs
                    },
                ),
                (None, Some(m)) if !site.chunk0 => (
                    m.block_table,
                    m.seq_len,
                    m.max_blocks_per_seq,
                    DevicePtr::NULL,
                ),
                _ => (DevicePtr::NULL, DevicePtr::NULL, 0, DevicePtr::NULL),
            };
        let s = SeqBatch {
            n: rows.q_len.len(),
            block_table,
            seq_lens,
            max_blocks_per_seq,
            host_seq_lens: &rows.extent,
            seq_ids: &ids,
            rows_per_seq: 1,
            q_row_stride: site.dims.nq * site.dims.hd,
            q_rows: site.num_tokens,
            first_pos: &rows.q_first,
            new_tokens: &rows.q_len,
            host_block_tables: &tables,
            block_table_ptrs,
        };
        f(&c, &s, &rows, last_chunk).map(Some)
    }

    /// Prefill `on_query`, before RoPE. `pre_rope` is false when the site's
    /// Q RoPE already ran (fused into the deinterleave).
    pub(crate) fn seam_prefill_query(
        &self,
        sel: &dyn KvSelector,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        site: &PrefillSite<'_>,
        pre_rope: bool,
    ) -> Result<()> {
        let q = site.q;
        self.with_prefill(ctx, kv_cache, site, |c, s, _, _| {
            sel.on_query(c, s, pre_rope.then_some(q), q)
        })
        .map(|_| ())
    }

    /// Prefill `on_kv_written`, after the K/V write.
    pub(crate) fn seam_prefill_kv_written(
        &self,
        sel: &dyn KvSelector,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        site: &PrefillSite<'_>,
    ) -> Result<()> {
        self.with_prefill(ctx, kv_cache, site, |c, s, rows, _| {
            sel.on_kv_written(c, s, &rows.written_first, &rows.written_len)
        })
        .map(|_| ())
    }

    /// The single-stream paged prefill plan (`prefill/paged.rs`, chunk 1 and
    /// later). `None` keeps the dense table. Head dims above 256 always stay
    /// dense: their paged dispatch branches on `seq_len_start == 0`, which a
    /// compacted extent could reach.
    pub(crate) fn seam_prefill_route(
        &self,
        sel: &dyn KvSelector,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        site: &PrefillSite<'_>,
        dense_kv_len: u32,
    ) -> Result<Option<PrefillView>> {
        if site.batched.is_some() || site.meta.is_none() || site.chunk0 || site.dims.hd > 256 {
            return Ok(None);
        }
        let n = to_u32(site.num_tokens);
        let q = site.q;
        Ok(self
            .with_prefill(ctx, kv_cache, site, |c, s, _, _| {
                route::prefill(sel, c, s, q, n, dense_kv_len, &RouteEnv::global())
            })?
            .flatten())
    }

    /// Prefill `on_prefill_layer_done`, when the layer's attention output is final.
    pub(crate) fn seam_prefill_done(
        &self,
        sel: &dyn KvSelector,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        site: &PrefillSite<'_>,
    ) -> Result<()> {
        self.with_prefill(ctx, kv_cache, site, |c, s, _, last_chunk| {
            sel.on_prefill_layer_done(c, s, last_chunk)
        })
        .map(|_| ())
    }
}

/// The decode and verify hook sequence: K/V written, the post-RoPE query,
/// then the plan.
fn attend(
    sel: &dyn KvSelector,
    c: &SeamCtx,
    s: &SeqBatch,
    q: DevicePtr,
    out: DevicePtr,
) -> Result<DecodeRoute> {
    sel.on_kv_written(c, s, s.first_pos, s.new_tokens)?;
    sel.on_query(c, s, None, q)?;
    route::decode(sel, c, s, q, out, &RouteEnv::global())
}
