// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the bridge from the chunk-1+ paged prefill attention dispatch (`paged_attn.rs`) to
//! the block-sparse prefill (`crate::port::sparse_prefill`, lane L35, APPROXIMATE, row
//! `prefill_sparse_attn`, off on every target).
//!
//! A child of `paged_attn.rs` (declared there by `#[path]`) so it can read the layer's private fields
//! (`kv_dtype`, `attn_layer_idx`, the FP8 scales) and the dispatch's `PagedAttnArgs` without widening
//! either. It turns them into the plain-value `sparse_prefill::Call` and returns whether the sparse
//! path produced `attn_out`. With the row off it returns `Ok(false)` after one sealed-config read,
//! before touching anything else, so the dense dispatch that follows is the base's, byte for byte.

use anyhow::Result;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::super::Qwen3AttentionLayer;
use super::PagedAttnArgs;
use crate::layer::ForwardContext;
use crate::port::sparse_prefill::{self, Call, KvKind};

/// Run this chunk's attention block-sparse if the row is on and the chunk qualifies. `Ok(true)`: the
/// output is written and the caller skips its dense kernels.
pub(super) fn try_run(
    layer: &Qwen3AttentionLayer,
    kv_cache: &PagedKvCache,
    ctx: &ForwardContext,
    a: &PagedAttnArgs,
) -> Result<bool> {
    let cfg = sparse_prefill::config();
    if !cfg.enabled {
        return Ok(false);
    }
    let kv = match layer.kv_dtype {
        KvCacheDtype::Bf16 => KvKind::Bf16,
        KvCacheDtype::Fp8 => KvKind::Fp8,
        _ => KvKind::Other,
    };
    let (k_scale, v_scale) = if kv == KvKind::Fp8 {
        layer.effective_fp8_scales()
    } else {
        (1.0, 1.0)
    };
    let call = Call {
        q: a.q_contiguous,
        out: a.attn_out,
        k_pool: kv_cache.k_pool_ptr(layer.attn_layer_idx),
        v_pool: kv_cache.v_pool_ptr(layer.attn_layer_idx),
        block_table: a.meta.block_table,
        q_len: a.n,
        kv_len: a.kv_len,
        q_offset: a.seq_len_start as u32,
        nq: a.nq,
        nkv: a.nkv,
        hd: a.hd,
        block_size: a.bs as u32,
        cache_stride: kv_cache.cache_stride() as u64,
        kv,
        k_scale,
        v_scale,
        inv_sqrt_d: a.inv_sqrt_d,
        sliding_window: layer.sliding_window.unwrap_or(0),
        graph_capture: ctx.graph_capture,
        seam_active: crate::layers::qwen3_attention::sparse_seam::active().is_some(),
        layer: layer.attn_layer_idx,
    };
    sparse_prefill::run(ctx.gpu, cfg, &call, a.stream)
}
