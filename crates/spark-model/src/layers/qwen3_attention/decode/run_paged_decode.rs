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
    pub(in super::super) fn run_paged_decode(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        use atlas_core::device::sm121::NUM_SMS;

        match self.kv_dtype {
            KvCacheDtype::Nvfp4 => {
                // V4-Flash uses MLA with compressed KV cache (576 dims: 512 latent + 64 rope)
                // Detection: V4-Flash has rope > 0 (64 dims), V3 has rope = 0
                let is_v4_flash = self.mla.as_ref().map(|m| m.rope > 0).unwrap_or(false);
                
                if is_v4_flash {
                    // Use MLA decode kernel for V4-Flash
                    let kv_cache_dim = (self.mla.as_ref().unwrap().kv_lora_rank 
                        + self.mla.as_ref().unwrap().rope) as u32; // 512 + 64 = 576
                    tracing::info!(
                        "V4-Flash MLA decode: using MLA kernel, q_head_dim={}, kv_cache_dim={}",
                        head_dim,
                        kv_cache_dim
                    );
                    ops::mla_paged_decode_nvfp4(
                        gpu,
                        self.mla_paged_decode_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        kv_cache_dim,
                        block_size,
                        inv_sqrt_d,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        num_seqs,
                        stream,
                    )
                } else {
                    // Standard MLA (V3) or full attention
                    let current_ctas = num_q_heads * num_seqs;
                    let num_splits = if current_ctas >= NUM_SMS {
                        1u32
                    } else {
                        NUM_SMS / current_ctas
                    };
                    tracing::info!(
                        "V3 MLA decode: NVFP4 paged_decode_k.handle={}, head_dim={}, num_splits={}",
                        self.paged_decode_k.0,
                        head_dim,
                        num_splits
                    );

                    if num_splits > 1 {
                        let splitk_k = self
                            .paged_decode_splitk_k
                            .expect("split-K kernel required for NVFP4");
                        let reduce_k = self
                            .paged_decode_reduce_k
                            .expect("reduce kernel required for NVFP4");
                        ops::paged_decode_attn_splitk_nvfp4(
                            gpu,
                            splitk_k,
                            q,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            workspace,
                            block_table,
                            seq_lens,
                            max_blocks_per_seq,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            block_size,
                            inv_sqrt_d,
                            num_splits,
                            q_stride,
                            kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                            kv_cache.nvfp4_data_bytes() as u64,
                            num_seqs,
                            stream,
                        )?;
                        ops::paged_decode_attn_reduce_nvfp4(
                            gpu,
                            reduce_k,
                            workspace,
                            output,
                            seq_lens,
                            num_q_heads,
                            head_dim,
                            num_splits,
                            num_seqs,
                            stream,
                        )
                    } else {
                        ops::paged_decode_attn_nvfp4(
                            gpu,
                            self.paged_decode_k,
                            q,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            output,
                            block_table,
                            seq_lens,
                            max_blocks_per_seq,
                            num_seqs,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            block_size,
                            inv_sqrt_d,
                            q_stride,
                            kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                            kv_cache.nvfp4_data_bytes() as u64,
                            stream,
                        )
                    }
                }
            }
            // Turbo4/3: same 4-bit interface as NVFP4 (block_stride + data_section layout).
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                    _ => kv_cache.turbo4_data_bytes() as u64,
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    stream,
                )
            }
            // Turbo8: WHT + FP8 — 1 byte per element + per-group FP8 scales.
            KvCacheDtype::Turbo8 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo8_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Fp8 => {
                // V4-Flash uses MLA with compressed KV cache (576 dims: 512 latent + 64 rope)
                // Detection: V4-Flash has rope > 0 (64 dims), V3 has rope = 0
                let is_v4_flash = self.mla.as_ref().map(|m| m.rope > 0).unwrap_or(false);
                
                if is_v4_flash {
                    // Use MLA decode kernel for V4-Flash with FP8 KV cache
                    let kv_cache_dim = (self.mla.as_ref().unwrap().kv_lora_rank 
                        + self.mla.as_ref().unwrap().rope) as u32; // 512 + 64 = 576
                    tracing::info!(
                        "V4-Flash MLA decode (FP8): using MLA kernel, q_head_dim={}, kv_cache_dim={}",
                        head_dim,
                        kv_cache_dim
                    );
                    let (k_scale, v_scale) = self.effective_fp8_scales();
                    ops::mla_paged_decode_fp8(
                        gpu,
                        self.mla_paged_decode_fp8_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        kv_cache_dim,
                        block_size,
                        inv_sqrt_d,
                        k_scale,
                        v_scale,
                        kv_cache.cache_stride() as u64,
                        num_seqs,
                        stream,
                    )
                } else {
                    // Standard FP8 decode (V3 or full attention)
                    let current_ctas = num_q_heads * num_seqs;
                    let num_splits = if current_ctas >= NUM_SMS {
                        1u32
                    } else {
                        NUM_SMS / current_ctas
                    };

                    let (k_scale, v_scale) = self.effective_fp8_scales();

                    if num_splits > 1 {
                        let splitk_k = self
                            .paged_decode_splitk_k
                            .expect("split-K kernel required for FP8");
                        let reduce_k = self
                            .paged_decode_reduce_k
                            .expect("reduce kernel required for FP8");
                        ops::paged_decode_attn_splitk_fp8(
                            gpu,
                            splitk_k,
                            q,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            workspace,
                            block_table,
                            seq_lens,
                            max_blocks_per_seq,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            block_size,
                            inv_sqrt_d,
                            num_splits,
                            k_scale,
                            v_scale,
                            q_stride,
                            kv_cache.cache_stride() as u64,
                            num_seqs,
                            stream,
                        )?;
                        ops::paged_decode_attn_reduce_fp8(
                            gpu,
                            reduce_k,
                            workspace,
                            output,
                            seq_lens,
                            num_q_heads,
                            head_dim,
                            num_splits,
                            num_seqs,
                            stream,
                        )
                    } else {
                        // Use HDIM=512 kernel for Gemma-4 full-attention layers
                        let fp8_kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                            self.paged_decode_512_k
                        } else {
                            self.paged_decode_k
                        };
                        ops::paged_decode_attn_fp8(
                            gpu,
                            fp8_kernel,
                            q,
                            kv_cache.k_pool_ptr(self.attn_layer_idx),
                            kv_cache.v_pool_ptr(self.attn_layer_idx),
                            output,
                            block_table,
                            seq_lens,
                            max_blocks_per_seq,
                            num_seqs,
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            block_size,
                            inv_sqrt_d,
                            k_scale,
                            v_scale,
                            q_stride,
                            kv_cache.cache_stride() as u64,
                            stream,
                        )
                    }
                }
            }
            KvCacheDtype::Bf16 => {
                // BF16 paged decode — no Split-K (not implemented for BF16 yet)
                // Use HDIM=512 kernel for Gemma-4 full-attention layers (head_dim > 256)
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                // Gemma-4 sliding layers attend only to the last `window_size`
                // KV positions; full layers (and all non-Gemma-4 models) pass 0.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    sliding,
                    stream,
                )
            }
        }
    }
}
