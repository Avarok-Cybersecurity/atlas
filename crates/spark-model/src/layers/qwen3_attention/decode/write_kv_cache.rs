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
    pub(in super::super) fn write_kv_cache(
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
        match self.kv_dtype {
            KvCacheDtype::Nvfp4 => ops::reshape_and_cache_nvfp4(
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
                key_stride,
                value_stride,
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                stream,
            ),
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 | KvCacheDtype::Turbo8 => {
                // Apply WHT to K and V before writing to turbo cache.
                // K and V are laid out as `[num_tokens, num_kv_heads, head_dim]`
                // BF16; the WHT kernel takes `[num_heads, head_dim]` and runs
                // one CTA per head. Grid must cover ALL (token × kv_head)
                // pairs — using `num_kv_heads` alone only WHTs the first
                // token's heads and leaves the rest of prefill un-WHT'd in
                // the cache, which collapses long-context decode (the cache
                // mixes WHT'd reads of Q with un-WHT'd K/V for tokens 1+).
                // WHT bookend (Turbo3/4/8 with Walsh-Hadamard decorrelation).
                // 2026-04-28: was temporarily gated behind ATLAS_TURBO_ENABLE_WHT=1
                // because FP8 per-group scales (~12% precision) compounded WHT
                // round-trip errors catastrophically. Resolved by upgrading
                // Turbo8 scales to BF16 (~0.4% precision); WHT is back on by
                // default. Turbo3/4 still use FP8 scales — they're affected
                // less because their LUTs already have lower precision targets.
                if self.wht_bf16_k.0 != 0 && (head_dim == 128 || head_dim == 256 || head_dim == 512)
                {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let total_heads = num_kv_heads * num_tokens;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(k)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                    KernelLaunch::new(gpu, self.wht_bf16_k)
                        .grid([total_heads, 1, 1])
                        .block([32, 1, 1])
                        .arg_ptr(v)
                        .arg_u32(head_dim)
                        .launch(stream)?;
                }
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo8 => kv_cache.turbo8_data_bytes() as u64,
                    KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                    _ => kv_cache.nvfp4_data_bytes() as u64, // turbo4 same as nvfp4
                };
                ops::reshape_and_cache_nvfp4(
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
                    key_stride,
                    value_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    stream,
                )
            }
            KvCacheDtype::Bf16 => ops::reshape_and_cache(
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
                key_stride,
                value_stride,
                kv_cache.cache_stride() as u64,
                stream,
            ),
            _ => {
                // FP8 KV cache
                if !graph_capture && let Some(ref cal) = self.fp8_calibration {
                    cal.observe(gpu, k, v, num_tokens, num_kv_heads, head_dim, stream)?;
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
    }
}
