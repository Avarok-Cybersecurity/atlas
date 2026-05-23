// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ both-sides-asym Turbo*K + Turbo*V prefill helper.
//!
//! Sibling of `paged_attn.rs` / `paged_attn_fp8k.rs`: extracts the 3
//! Turbo*K dispatch arms into one method so `paged_attn.rs` stays under
//! the 500-LoC cap. Each combo threads BOTH K-side and V-side
//! (block_stride, data_section) pairs into the kernel call.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_turbok_turbo_v(
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
        stream: u64,
    ) -> Result<()> {
        let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
        let k_bs = kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
        let v_bs = kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
        let sw = self.sliding_window.unwrap_or(0);
        let layer = self.attn_layer_idx;
        match self.kv_dtype {
            KvCacheDtype::Turbo4KTurbo3V => {
                if self.prefill_attn_paged_turbo4k_turbo3v_64_k.0 == 0 {
                    anyhow::bail!(
                        "Turbo4KTurbo3V prefill kernel not loaded (layer {layer}); rebuild kernels."
                    );
                }
                ops::prefill_attention_paged_turbo4k_turbo3v_64(
                    ctx.gpu,
                    self.prefill_attn_paged_turbo4k_turbo3v_64_k,
                    q_contiguous,
                    k_pool,
                    v_pool,
                    attn_out,
                    block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    sw,
                    inv_sqrt_d,
                    k_bs,
                    kv_cache.nvfp4_data_bytes() as u64,
                    v_bs,
                    kv_cache.turbo3_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Turbo4KTurbo8V => {
                if self.prefill_attn_paged_turbo4k_turbo8v_64_k.0 == 0 {
                    anyhow::bail!(
                        "Turbo4KTurbo8V prefill kernel not loaded (layer {layer}); rebuild kernels."
                    );
                }
                ops::prefill_attention_paged_turbo4k_turbo8v_64(
                    ctx.gpu,
                    self.prefill_attn_paged_turbo4k_turbo8v_64_k,
                    q_contiguous,
                    k_pool,
                    v_pool,
                    attn_out,
                    block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    sw,
                    inv_sqrt_d,
                    k_bs,
                    kv_cache.nvfp4_data_bytes() as u64,
                    v_bs,
                    kv_cache.turbo8_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Turbo3KTurbo8V => {
                if self.prefill_attn_paged_turbo3k_turbo8v_64_k.0 == 0 {
                    anyhow::bail!(
                        "Turbo3KTurbo8V prefill kernel not loaded (layer {layer}); rebuild kernels."
                    );
                }
                ops::prefill_attention_paged_turbo3k_turbo8v_64(
                    ctx.gpu,
                    self.prefill_attn_paged_turbo3k_turbo8v_64_k,
                    q_contiguous,
                    k_pool,
                    v_pool,
                    attn_out,
                    block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    sw,
                    inv_sqrt_d,
                    k_bs,
                    kv_cache.turbo3_data_bytes() as u64,
                    v_bs,
                    kv_cache.turbo8_data_bytes() as u64,
                    stream,
                )
            }
            _ => unreachable!(),
        }
    }
}
