// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `super::super::decode.rs` for file-size budget.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in super::super) fn high_speed_swap_engaged(
        &self,
        kv_cache: &spark_runtime::kv_cache::PagedKvCache,
    ) -> bool {
        if kv_cache.config().cache_blocks_per_seq.is_none() {
            return false;
        }
        if !spark_storage::local_installed() {
            return false;
        }
        // Phase 6.2.c — proper: every quantization variant now has a host-side
        // dequant path that produces BF16 for the orchestrator's tiled-attention
        // kernel. For Turbo3/4/8 the cache stores WHT(K)/WHT(V) which round-trip
        // correctly because production already applies WHT(Q) and iWHT(out)
        // around the orchestrator call (decode.rs WHT bookends).
        matches!(
            kv_cache.dtype_for_layer(self.attn_layer_idx),
            KvCacheDtype::Bf16
                | KvCacheDtype::Fp8
                | KvCacheDtype::Nvfp4
                | KvCacheDtype::Turbo3
                | KvCacheDtype::Turbo4
                | KvCacheDtype::Turbo8,
        )
    }

    /// Catch up: alloc a disk_block_id and offload K/V to disk for every
    /// block_table entry that doesn't yet have a disk_id. Idempotent —
    /// re-running on a sequence that's fully caught up is a no-op.
    /// Called after every K/V write from both decode and prefill paths.
    pub(in super::super) fn high_speed_swap_offload_new_blocks(
        &self,
        kv_cache: &mut PagedKvCache,
        block_table: &Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        nkv: u32,
        hd: u32,
        bs: usize,
    ) -> Result<()> {
        if !self.high_speed_swap_engaged(kv_cache) {
            return Ok(());
        }
        let layer_u32 = self.attn_layer_idx as u32;
        let block_floats = bs * (nkv as usize) * (hd as usize);

        // Phase 6.3: disk_block_ids growth lives in the alloc helper
        // (`model.rs::ensure_blocks_through_decode` / `_prefill`). Here we
        // assert the invariant `disk_block_ids.len() == hss_window_start +
        // block_table.len()` (window_start derivable from the lengths) and
        // proceed straight to the per-layer K/V offload.
        debug_assert!(
            disk_block_ids.len() >= block_table.len(),
            "Phase 6.3 invariant: alloc helper must keep disk_block_ids ≥ block_table.len() \
             (got disk={} bt={})",
            disk_block_ids.len(),
            block_table.len()
        );

        // Step 2: per-layer catch-up. THIS layer's offloaded count
        // (`disk_last_offloaded_per_layer[L]`) lags `disk_block_ids.len()`
        // by however many new blocks have been allocated since this layer
        // last ran. For each missing block, the layer's K/V is currently
        // in the production HBM cache at `block_table[bt_idx]` for
        // `bt_idx = logical_pos - window_start`. Read it back, push to disk.
        if disk_last_offloaded_per_layer.len() <= self.attn_layer_idx {
            // Defensive: caller didn't size the vec. Resize on first hit.
            disk_last_offloaded_per_layer.resize(self.attn_layer_idx + 1, 0);
        }
        let last = disk_last_offloaded_per_layer[self.attn_layer_idx] as usize;
        let total = disk_block_ids.len();
        if total == 0 {
            return Ok(());
        }
        // Window of HBM-resident blocks: block_table[0..block_table.len()]
        // covers logical positions [total - block_table.len(), total).
        let window_start = total.saturating_sub(block_table.len());
        // Always re-offload the ACTIVE (last) block on every call. Decode
        // writes one new slot per step into the existing last block — its
        // HBM contents change without disk_block_ids.len() growing, so the
        // naive `last..total` range would skip it and the streaming kernel
        // would read stale (zero-init) bytes for slots 1..15 → degenerate
        // attention → "the the the" loop. Setting `start = last.min(total-1)`
        // guarantees the active block is re-pushed every step.
        let start = last.min(total - 1);
        for logical_pos in start..total {
            if logical_pos < window_start {
                // Issue #31: the slide-before-alloc loop in
                // `block_mgmt::ensure_blocks_through_prefill` advanced the
                // sliding window past `logical_pos` before this layer got
                // a chance to offload. The invariant declared at line 122-127
                // of `block_mgmt.rs` (every attention layer must catch up
                // its offloads before any slide) is debug-asserted only —
                // release builds silently let the slide proceed, then this
                // check trips at the next offload pass.
                //
                // Practical fix until Phase 6.2.b lands chunked-prefill
                // reads through the HSS orchestrator: ensure
                // `--high-speed-swap-cache-blocks-per-seq × --block-size`
                // is large enough that the per-chunk prefill never grows
                // `disk_block_ids` past `block_table.len()` faster than the
                // per-layer offload can keep up. Drop --high-speed-swap if
                // KV fits HBM at this batch size.
                anyhow::bail!(
                    "high-speed-swap: layer {} block {} was evicted before this layer offloaded \
                     it (issue #31). \n\
                     Diagnostic state: attn_layer_idx={}, logical_pos={}, \
                     window_start={}, total=disk_block_ids.len()={}, \
                     block_table.len()={}, this_layer.last_offloaded={}, \
                     all_layer_cursors={:?}.\n\
                     This means the sliding-window eviction loop advanced past disk slot \
                     {} before attention layer {} could push its K/V there. The slide-before-alloc \
                     invariant in block_mgmt.rs (every attention layer must offload before any \
                     slide) is debug-asserted only — release builds skip it.\n\
                     Workaround: raise --high-speed-swap-cache-blocks-per-seq so \
                     `cap × block_size` ≥ your largest prompt, OR drop --high-speed-swap \
                     entirely if KV fits HBM at this batch/quant.",
                    self.attn_layer_idx,
                    logical_pos,
                    self.attn_layer_idx,
                    logical_pos,
                    window_start,
                    total,
                    block_table.len(),
                    last,
                    disk_last_offloaded_per_layer,
                    logical_pos,
                    self.attn_layer_idx,
                );
            }
            let bt_idx = logical_pos - window_start;
            let phys_blk = block_table[bt_idx];
            let disk_id = disk_block_ids[logical_pos];
            let k_block_dev = kv_cache.k_cache_ptr(self.attn_layer_idx, phys_blk).0;
            let v_block_dev = kv_cache.v_cache_ptr(self.attn_layer_idx, phys_blk).0;
            let mut k_host = vec![half::bf16::from_f32(0.0); block_floats];
            let mut v_host = vec![half::bf16::from_f32(0.0); block_floats];
            // Phase 6.2.c proper — dispatch on layer dtype. BF16 streams the
            // bytes directly; quantized variants read raw bytes then dequant
            // on the host before disk-write (the streaming kernel reads BF16).
            let layer_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
            let layer_block_bytes = kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx);
            let bs_us = bs;
            let nkv_us = nkv as usize;
            let hd_us = hd as usize;
            match layer_dtype {
                KvCacheDtype::Bf16 => {
                    // copy_d2h_on_stream: orders the D2H after WHT+reshape_and_cache
                    // on the production stream. copy_d2h would race (default-stream
                    // sync only) and read torn bytes — Turbo8 race fix, 2026-04-28.
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(k_block_dev),
                        unsafe {
                            std::slice::from_raw_parts_mut(
                                k_host.as_mut_ptr() as *mut u8,
                                block_floats * 2,
                            )
                        },
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(v_block_dev),
                        unsafe {
                            std::slice::from_raw_parts_mut(
                                v_host.as_mut_ptr() as *mut u8,
                                block_floats * 2,
                            )
                        },
                        stream,
                    )?;
                }
                KvCacheDtype::Fp8 => {
                    let mut k_raw = vec![0u8; block_floats];
                    let mut v_raw = vec![0u8; block_floats];
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    let (k_scale, v_scale) = self.effective_fp8_scales();
                    dequant_fp8_to_bf16(&k_raw, k_scale, &mut k_host);
                    dequant_fp8_to_bf16(&v_raw, v_scale, &mut v_host);
                }
                KvCacheDtype::Nvfp4 | KvCacheDtype::Turbo4 => {
                    let mut k_raw = vec![0u8; layer_block_bytes];
                    let mut v_raw = vec![0u8; layer_block_bytes];
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    let lut = if layer_dtype == KvCacheDtype::Nvfp4 {
                        &NVFP4_E2M1_LUT
                    } else {
                        &TURBO4_LUT
                    };
                    dequant_4bit_block_to_bf16(&k_raw, bs_us, nkv_us, hd_us, lut, &mut k_host);
                    dequant_4bit_block_to_bf16(&v_raw, bs_us, nkv_us, hd_us, lut, &mut v_host);
                }
                KvCacheDtype::Turbo3 => {
                    let mut k_raw = vec![0u8; layer_block_bytes];
                    let mut v_raw = vec![0u8; layer_block_bytes];
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    dequant_turbo3_block_to_bf16(&k_raw, bs_us, nkv_us, hd_us, &mut k_host);
                    dequant_turbo3_block_to_bf16(&v_raw, bs_us, nkv_us, hd_us, &mut v_host);
                }
                KvCacheDtype::Turbo8 => {
                    let mut k_raw = vec![0u8; layer_block_bytes];
                    let mut v_raw = vec![0u8; layer_block_bytes];
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(k_block_dev),
                        &mut k_raw,
                        stream,
                    )?;
                    ctx.gpu.copy_d2h_on_stream(
                        spark_runtime::gpu::DevicePtr(v_block_dev),
                        &mut v_raw,
                        stream,
                    )?;
                    dequant_turbo8_block_to_bf16(&k_raw, bs_us, nkv_us, hd_us, &mut k_host);
                    dequant_turbo8_block_to_bf16(&v_raw, bs_us, nkv_us, hd_us, &mut v_host);
                }
            }
            spark_storage::with_local(|hss| {
                match layer_dtype {
                    KvCacheDtype::Bf16 => hss.offload_block_on_stream(
                        stream,
                        layer_u32,
                        disk_id,
                        k_block_dev,
                        &k_host,
                        &v_host,
                    ),
                    // Quantized: skip predictor projection — the BF16 kernel
                    // would OOB-read on a non-BF16 layout. Eviction degrades
                    // to LRU for these blocks; correctness preserved.
                    _ => hss.offload_block_no_predict_on_stream(
                        stream, layer_u32, disk_id, &k_host, &v_host,
                    ),
                }
            })
            .expect("local_installed checked in high_speed_swap_engaged")?;
        }
        disk_last_offloaded_per_layer[self.attn_layer_idx] = total as u32;
        Ok(())
    }
}
