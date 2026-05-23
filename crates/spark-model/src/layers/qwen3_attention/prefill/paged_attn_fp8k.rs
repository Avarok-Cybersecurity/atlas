// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ asym Fp8K + TurboNV prefill helper.
//!
//! Sibling of `paged_attn.rs`: extracts the 3 Fp8K dispatch arms into one
//! method so `paged_attn.rs` stays under the 500-LoC cap. Same shape as the
//! 3 bf16k_turbo*v inline branches, but with FP8 K-side `k_scale` threaded
//! through to the kernel call.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(super) fn prefill_fp8k_turbo_nv(
        &self,
        ctx: &ForwardContext,
        kv_cache: &mut PagedKvCache,
        q_contiguous: DevicePtr,
        attn_out: DevicePtr,
        block_table: DevicePtr,
        n: u32,
        kv_len: u32,
        seq_len_start: usize,
        nq: u32,
        nkv: u32,
        hd: u32,
        bs_u: u32,
        inv_sqrt_d: f32,
        fp8_k_scale: f32,
        stream: u64,
    ) -> Result<()> {
        let v_block_stride = kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
        match self.kv_dtype {
            KvCacheDtype::Fp8KTurbo3V => {
                if self.prefill_attn_paged_fp8k_turbo3v_64_k.0 == 0 {
                    anyhow::bail!(
                        "Fp8KTurbo3V prefill kernel not loaded (layer {}); rebuild kernels.",
                        self.attn_layer_idx
                    );
                }
                ops::prefill_attention_paged_fp8k_turbo3v_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8k_turbo3v_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table,
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
                    v_block_stride,
                    kv_cache.turbo3_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Fp8KTurbo4V => {
                if self.prefill_attn_paged_fp8k_turbo4v_64_k.0 == 0 {
                    anyhow::bail!(
                        "Fp8KTurbo4V prefill kernel not loaded (layer {}); rebuild kernels.",
                        self.attn_layer_idx
                    );
                }
                ops::prefill_attention_paged_fp8k_turbo4v_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8k_turbo4v_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table,
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
                    v_block_stride,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Fp8KTurbo2V => {
                if self.prefill_attn_paged_fp8k_turbo2v_64_k.0 == 0 {
                    anyhow::bail!(
                        "Fp8KTurbo2V prefill kernel not loaded (layer {}); rebuild kernels.",
                        self.attn_layer_idx
                    );
                }
                ops::prefill_attention_paged_fp8k_turbo2v_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8k_turbo2v_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table,
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
                    v_block_stride,
                    kv_cache.turbo2_data_bytes() as u64,
                    stream,
                )
            }
            _ => unreachable!("prefill_fp8k_turbo_nv called with non-Fp8K dtype"),
        }
    }
}
