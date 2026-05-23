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
                // Split count derived from the configured max batch (constant),
                // not the runtime co-batched count, so a sequence's reduction
                // tree is identical alone vs co-batched (determinism fix).
                let current_ctas = num_q_heads * super::super::split_ref_seqs(num_seqs);
                let num_splits = if current_ctas >= NUM_SMS {
                    1u32
                } else {
                    NUM_SMS / current_ctas
                };

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
            // Turbo4/3: same 4-bit interface as NVFP4 (block_stride + data_section layout).
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 | KvCacheDtype::Turbo2 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                    KvCacheDtype::Turbo2 => kv_cache.turbo2_data_bytes() as u64,
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
            KvCacheDtype::Bf16KTurbo3V => {
                // TurboQuant+ safer-asym Bf16K + Turbo3V combined paged decode.
                // K read as BF16 NHD (vector loads), V read as turbo3 (3-bit
                // packed + FP8 group scale, sparse-V threshold on batched +
                // remainder paths). Single combined kernel per HDIM variant.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo3v(
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
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo3_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo4V => {
                // TurboQuant+ safer-asym Bf16K + Turbo4V combined paged decode.
                // K read as BF16 NHD, V read as turbo4 (4-bit packed + FP8
                // group scale, sparse-V threshold on batched + remainder paths).
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo4v(
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
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo2V => {
                // TurboQuant+ safer-asym Bf16K + Turbo2V (6.4x V compression)
                // combined paged decode. K read as BF16 NHD, V read as turbo2
                // (2-bit packed + FP8 group scale, sparse-V threshold).
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo2v(
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
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo2_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V
            | KvCacheDtype::Turbo3KTurbo8V => {
                // TurboQuant+ both-sides asym: K and V both turbo. Pass per-side
                // (block_stride, data_section) pairs since K and V pools have
                // independent byte layouts.
                let sliding = self.sliding_window.unwrap_or(0);
                let k_block_stride =
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Turbo4KTurbo3V => ops::paged_decode_attn_turbo4k_turbo3v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
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
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Turbo4KTurbo8V => ops::paged_decode_attn_turbo4k_turbo8v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
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
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Turbo3KTurbo8V => ops::paged_decode_attn_turbo3k_turbo8v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
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
                        k_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Fp8KTurbo3V | KvCacheDtype::Fp8KTurbo4V | KvCacheDtype::Fp8KTurbo2V => {
                // TurboQuant+ asym for FP8 models: K=fp8 (per-tensor scale),
                // V=turbo{3,4,2} with sparse-V threshold on batched + remainder.
                let sliding = self.sliding_window.unwrap_or(0);
                let (k_scale, _) = self.effective_fp8_scales();
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Fp8KTurbo3V => ops::paged_decode_attn_fp8k_turbo3v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
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
                        q_stride,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo4V => ops::paged_decode_attn_fp8k_turbo4v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
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
                        q_stride,
                        v_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo2V => ops::paged_decode_attn_fp8k_turbo2v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
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
                        q_stride,
                        v_block_stride,
                        kv_cache.turbo2_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    _ => unreachable!(),
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
            _ => {
                // FP8 paged decode. Split count from configured max batch
                // (constant), not runtime co-batched count → deterministic
                // reduction tree alone vs co-batched (determinism fix).
                let current_ctas = num_q_heads * super::super::split_ref_seqs(num_seqs);
                let num_splits = if current_ctas >= NUM_SMS {
                    1u32
                } else {
                    NUM_SMS / current_ctas
                };

                // DIAGNOSTIC (ATLAS_ATTN_DBG): split-K reduction structure for the
                // active row depends on `num_seqs` (co-batched count) via num_splits.
                // Print only when co-batched (num_seqs>1) to confirm whether a
                // serial-looking workload ever shares a batch (root-cause probe for
                // batch>1 temp-0 nondeterminism).
                if num_seqs != 1 && std::env::var("ATLAS_ATTN_DBG").is_ok() {
                    eprintln!(
                        "ATTN_DBG L{} num_seqs={} num_splits={} (NUM_SMS={} nq={})",
                        self.attn_layer_idx, num_seqs, num_splits, NUM_SMS, num_q_heads
                    );
                }

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
    }
}
