// SPDX-License-Identifier: AGPL-3.0-only

//! The FP8 arm of `write_kv_cache`: the calibration window's view of the
//! write, then the write. Split out of `write_kv_cache.rs` when #919's write
//! target pushed that file past the 500-line cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layers::fp8_calibration::Fp8KvWriteTarget;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// #919: `observe` accumulates the amax over the requested
    /// `--fp8-kv-calibration-tokens` window ACROSS requests and, when the
    /// window closes, requantizes the entries the window wrote — so it needs
    /// to know where this write is going. It runs BEFORE the write, so
    /// `effective_fp8_scales()` returns exactly the scale the batch is about
    /// to be written with (and every later read dequantizes with). A layer
    /// with no calibration window observes nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_fp8_kv_write(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(ref cal) = self.fp8_calibration else {
            return Ok(());
        };
        let target = Fp8KvWriteTarget {
            kernel: self.reshape_cache_k,
            k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
            v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
            block_size,
            cache_stride: kv_cache.cache_stride() as u64,
            key_stride,
            value_stride,
            slot,
        };
        cal.observe(
            gpu,
            k,
            v,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
            &target,
        )
    }
    /// Observe (outside graph capture), then write with the scales the window
    /// settled on.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_kv_cache_fp8(
        &self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        key_stride: u32,
        value_stride: u32,
        stream: u64,
        graph_capture: bool,
    ) -> Result<()> {
        if !graph_capture {
            self.observe_fp8_kv_write(
                gpu,
                k,
                v,
                kv_cache,
                slot,
                num_tokens,
                num_kv_heads,
                head_dim,
                block_size,
                key_stride,
                value_stride,
                stream,
            )?;
        }
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::reshape_and_cache_fp8(
            gpu,
            self.reshape_cache_k,
            k,
            v,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            slot,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            k_scale,
            v_scale,
            key_stride,
            value_stride,
            kv_cache.cache_stride() as u64,
            stream,
        )
    }
}
