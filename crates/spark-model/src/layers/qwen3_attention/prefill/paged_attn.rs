// SPDX-License-Identifier: AGPL-3.0-only

//! Paged Flash Attention dispatch for `prefill_attention_paged` chunk 1+.
//! Picks one of: HSS streaming (when --high-speed-swap engaged),
//! HDIM=512 contiguous/paged kernel (Gemma-4), or one of the
//! NVFP4 / Turbo / BF16 / FP8 paged-attention paths (BR=64 long-context
//! variant when N>=256). Extracted to keep `paged.rs` ≤500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) struct PagedAttnArgs<'a> {
    pub q_contiguous: DevicePtr,
    pub k_contiguous: DevicePtr,
    pub v_contiguous: DevicePtr,
    pub attn_out: DevicePtr,
    pub n: u32,
    pub seq_len_start: usize,
    pub num_tokens: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub bs: usize,
    pub bf16: usize,
    pub inv_sqrt_d: f32,
    pub kv_len: u32,
    pub meta: &'a AttnMetadataDev,
    pub block_table: &'a Vec<u32>,
    pub disk_block_ids: &'a mut Vec<u32>,
    pub disk_last_offloaded_per_layer: &'a mut Vec<u32>,
    pub stream: u64,
}

/// Outcome of the dispatch — `EarlyReturn` means the caller should
/// short-circuit (HSS streaming branch, which already produced the final
/// output). `Continue` means the caller should run sections 9 + 10
/// (sigmoid-gate × attn_out + O-projection).
#[allow(dead_code)]
pub(super) enum PagedAttnOutcome {
    EarlyReturn(DevicePtr),
    Continue,
}

impl Qwen3AttentionLayer {
    /// Run the chunk-1+ flash-attention path. Returns either an attention
    /// output pointer (early return) or `Continue` to defer to the caller.
    pub(super) fn prefill_attention_paged_attn(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &mut PagedAttnArgs,
    ) -> Result<PagedAttnOutcome> {
        // Issue #31 (2026-05-08): the HSS streaming prefill branch
        // (`hss.attend_layer_on_stream_with_q_pos`) was an early attempt at
        // Phase 6.2.b. It reads K/V from DISK for every prior position via
        // `disk_block_ids[..n_blocks]`. But the CURRENT chunk's blocks
        // haven't been offloaded yet at attention-compute time (offload
        // runs after attention in `prefill_inner`), so the disk reads for
        // those blocks return zeros/stale bytes → silently-wrong output
        // (gbanyan's repro: prompt > cap×bs produces garbage even after
        // the slide-during-prefill fix).
        //
        // With the companion change in `factory/build.rs` (HSS pool sized
        // to `max(cap+1, max_seq_len_blocks)` per seq) plus the no-slide-
        // during-prefill change in `block_mgmt`, every prior chunk's K/V
        // remains HBM-resident through the entire prefill. Fall through
        // to the normal paged-attention paths which read from
        // `kv_cache.{k,v}_pool_ptr` (HBM) and produce correct output.
        //
        // The `high_speed_swap_offload_new_blocks` call still runs at the
        // end of `prefill_inner` so blocks reach disk for the orchestrator-
        // tiled DECODE attention (which IS correctly wired up).
        let PagedAttnArgs {
            q_contiguous,
            k_contiguous: _,
            v_contiguous: _,
            attn_out,
            n,
            seq_len_start,
            num_tokens: _,
            nq,
            nkv,
            hd,
            bs,
            bf16: _,
            inv_sqrt_d,
            kv_len,
            meta,
            block_table: _,
            ref mut disk_block_ids,
            ref mut disk_last_offloaded_per_layer,
            stream,
        } = *args;
        let _ = &disk_block_ids; // unused after issue #31 (HSS-streaming branch removed)
        let _ = &disk_last_offloaded_per_layer;

        let bs_u = bs as u32;

        // HDIM=512 path: Gemma-4 long-attention layers.
        if hd > 256 && self.prefill_attn_512_k.0 != 0 && seq_len_start == 0 {
            ops::prefill_attention(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_contiguous,
                args.k_contiguous,
                args.v_contiguous,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                stream,
            )?;
        } else if hd > 256 && seq_len_start > 0 {
            if self.kv_dtype != KvCacheDtype::Bf16 {
                anyhow::bail!(
                    "Gemma-4 HDIM=512 chunked prefill only supports BF16 KV cache \
                     (layer {}, seq_len_start={}, kv_dtype={:?}).",
                    self.attn_layer_idx,
                    seq_len_start,
                    self.kv_dtype
                );
            }
            if self.prefill_attn_paged_512_k.0 == 0 {
                anyhow::bail!(
                    "Gemma-4 HDIM=512 paged prefill kernel not loaded \
                     (inferspark_prefill_paged_512). Rebuild required."
                );
            }
            ops::prefill_attention_paged_512(
                ctx.gpu,
                self.prefill_attn_paged_512_k,
                q_contiguous,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                n,
                kv_len,
                seq_len_start as u32,
                nq,
                nkv,
                hd,
                bs_u,
                self.sliding_window.unwrap_or(0),
                inv_sqrt_d,
                stream,
            )?;
        } else {
            let use_br64 = n >= 256;
            let (fp8_k_scale, fp8_v_scale) = self.effective_fp8_scales();
            match (self.kv_dtype, use_br64) {
                (KvCacheDtype::Nvfp4, true) => ops::prefill_attention_paged_nvfp4_64(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?,
                (KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 | KvCacheDtype::Turbo8, true) => {
                    let data_bytes = match self.kv_dtype {
                        KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                        KvCacheDtype::Turbo8 => kv_cache.turbo8_data_bytes() as u64,
                        _ => kv_cache.turbo4_data_bytes() as u64,
                    };
                    ops::prefill_attention_paged_nvfp4_64(
                        ctx.gpu,
                        self.prefill_attn_paged_nvfp4_64_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start as u32,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        self.sliding_window.unwrap_or(0),
                        inv_sqrt_d,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        data_bytes,
                        stream,
                    )?
                }
                (KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 | KvCacheDtype::Turbo8, false) => {
                    let data_bytes = match self.kv_dtype {
                        KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                        KvCacheDtype::Turbo8 => kv_cache.turbo8_data_bytes() as u64,
                        _ => kv_cache.turbo4_data_bytes() as u64,
                    };
                    ops::prefill_attention_paged_nvfp4(
                        ctx.gpu,
                        self.prefill_attn_paged_nvfp4_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start as u32,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        self.sliding_window.unwrap_or(0),
                        inv_sqrt_d,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        data_bytes,
                        stream,
                    )?
                }
                (KvCacheDtype::Nvfp4, false) => ops::prefill_attention_paged_nvfp4(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?,
                (KvCacheDtype::Bf16, true) => ops::prefill_attention_paged_64(
                    ctx.gpu,
                    self.prefill_attn_paged_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?,
                (KvCacheDtype::Bf16, false) => ops::prefill_attention_paged(
                    ctx.gpu,
                    self.prefill_attn_paged_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?,
                (_, true) => ops::prefill_attention_paged_fp8_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.cache_stride() as u64,
                    stream,
                )?,
                (_, false) => ops::prefill_attention_paged_fp8(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.cache_stride() as u64,
                    stream,
                )?,
            }
        }

        Ok(PagedAttnOutcome::Continue)
    }
}
