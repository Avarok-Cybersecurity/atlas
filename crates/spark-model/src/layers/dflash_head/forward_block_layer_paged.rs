// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2 (Option B) γ-only per-layer body. Replaces the current
//! `forward_block_layer` for the paged-attention path. The current body
//! runs over `n_attn = γ + ctx` rows recomputing ctx K/V every layer;
//! this body runs over γ rows only and reads ctx K/V from the drafter's
//! paged BF16 KV cache (populated once per propose by `precompute_ctx_kv`).
//!
//! Pipeline per layer (γ rows only):
//!   3a. input_layernorm.rms_norm(stream_buf → norm_buf), γ rows
//!   3b. q/k/v_proj.dense_gemm over γ rows
//!   3c. q_norm / k_norm per-head over γ rows
//!   3d. rope_yarn(Q, K) at positions [position .. position+γ)
//!   3e. reshape_and_cache writes γ K/V into the layer's paged cache at
//!       slots [ctx_count .. ctx_count + γ]
//!   3f. prefill_attention_paged_dflash: q_len=γ, kv_len=ctx_count+γ,
//!       q_offset=ctx_count, reads K/V from paged cache pool
//!   3g. o_proj.dense_gemm over γ rows
//!   3h. residual_add
//!   3i. post_attention_layernorm.rms_norm
//!   3j. gate_proj + up_proj + silu_mul + down_proj (γ rows)
//!   3k. residual_add
//!
//! No ctx slots, no ctx K/V recomputation. All scratch buffer
//! `n_attn`-dependent rows become γ. Saves ~17 launches × 5 layers =
//! ~85 launches per propose, plus the per-layer MLP runs over γ rows
//! instead of γ+ctx (~3x fewer FLOPs at ctx=32).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashLayer};
use crate::layer::ForwardContext;

/// Inputs to the γ-only paged-attention per-layer body.
#[allow(clippy::too_many_arguments)]
pub(super) struct PagedLayerArgs {
    pub layer_idx: usize,
    /// `ctx_count` from the proposer state — number of paged-cache slots
    /// already populated with ctx K/V for this drafter layer. Determines
    /// `kv_len` and `q_offset` for the paged attention call, plus the
    /// starting slot for this propose's γ K/V writes.
    pub ctx_count: u32,
    pub h: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub inter: u32,
    pub inv_sqrt_d: f32,
    /// Slot mapping [γ] i32 — device pointer to the cache slot indices
    /// where this layer should write γ K/V via reshape_and_cache. Same
    /// across all drafter layers (block_table is shared).
    pub slot_mapping_gamma: DevicePtr,
    /// Block table device pointer for the active sequence — same across
    /// all drafter layers. Maps logical block indices to physical pool
    /// block indices for the paged attention kernel.
    pub block_table_dev: DevicePtr,
    pub stream: u64,
}

impl BlockDiffusionDraftHead {
    /// γ-only paged-attention per-layer body. See module docstring.
    pub(super) fn forward_block_layer_paged(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        use crate::layers::ops;

        let PagedLayerArgs {
            layer_idx,
            ctx_count,
            h,
            q_dim,
            kv_dim,
            inter,
            inv_sqrt_d,
            slot_mapping_gamma,
            block_table_dev,
            stream,
        } = *args;
        let gpu = ctx.gpu;
        let g = self.gamma as u32;
        let kv_len = ctx_count + g;
        let _ = layer_idx; // reserved for future per-layer debug branches

        // 3a. input_layernorm — γ rows.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            g,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        // 3b. q/k/v projections — γ rows.
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.norm_buf,
            &layer.q_proj,
            self.scratch.q_buf,
            g,
            q_dim,
            h,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.norm_buf,
            &layer.k_proj,
            self.scratch.k_buf,
            g,
            kv_dim,
            h,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.norm_buf,
            &layer.v_proj,
            self.scratch.v_buf,
            g,
            kv_dim,
            h,
            stream,
        )?;

        // 3c. q_norm / k_norm — per-head RMS over head_dim.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.q_buf,
            &layer.q_norm,
            self.scratch.q_buf,
            g * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            &layer.k_norm,
            self.scratch.k_buf,
            g * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;

        // 3d. yarn RoPE over γ positions [position..position+γ).
        // position_ids buffer's first γ entries are the γ noise positions
        // (built by forward_block when ATLAS_DFLASH_OPTION_B=1).
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.position_ids,
            g,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;

        // 3e. reshape_and_cache — write γ K/V into the layer's paged cache
        // at slots [ctx_count .. ctx_count + γ]. Slot mapping is provided
        // by the caller (built once per propose for the γ rows).
        let (k_pool, v_pool) = {
            let cache = self.kv_cache.lock();
            (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
        };
        ops::reshape_and_cache(
            gpu,
            self.kernels.reshape_cache_bf16,
            self.scratch.k_buf,
            self.scratch.v_buf,
            k_pool,
            v_pool,
            slot_mapping_gamma,
            g,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            16, // block_size — matches from_weights.rs:68
            kv_dim,
            kv_dim,
            0,
            stream,
        )?;

        // ── Stage 4 cache readback diagnostic ──
        // ATLAS_DFLASH_OPTION_B_DIAG=1 reads back layer 0's first cached
        // K row at the slot we just wrote and compares first 8 BF16 values
        // against the source k_buf row 0. If they differ, the cache write
        // landed in the wrong slot or with the wrong layout. ONE-SHOT.
        if layer_idx == 0
            && std::env::var("ATLAS_DFLASH_OPTION_B_DIAG").ok().as_deref() == Some("1")
        {
            static DIAG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !DIAG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                gpu.synchronize(stream)?;

                // Read slot 0's physical index from slot_mapping (i64).
                let mut slot0_bytes = [0u8; 8];
                gpu.copy_d2h(slot_mapping_gamma, &mut slot0_bytes)?;
                let slot0 = i64::from_le_bytes(slot0_bytes);
                let block_size: usize = 16;
                let phys_block = slot0 / block_size as i64;
                let block_off = slot0 % block_size as i64;

                // Compute K row 0's physical address inside the pool:
                //   k_pool + phys_block * (block_size * num_kv_heads * head_dim) +
                //          + block_off * (num_kv_heads * head_dim)
                let n_elems = self.num_kv_heads * self.head_dim; // BF16 elements per slot
                let block_stride_bytes = block_size * n_elems * 2;
                let row_stride_bytes = n_elems * 2;
                let cache_row_ptr = k_pool.offset(
                    (phys_block as usize) * block_stride_bytes
                        + (block_off as usize) * row_stride_bytes,
                );

                // Read first 8 BF16 from source k_buf row 0 and from cached row 0.
                let read8 = |p: spark_runtime::gpu::DevicePtr| -> Result<Vec<f32>> {
                    let mut b = [0u8; 16];
                    gpu.copy_d2h(p, &mut b)?;
                    Ok(b.chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect())
                };
                let src = read8(self.scratch.k_buf)?;
                let cached = read8(cache_row_ptr)?;
                tracing::info!(
                    "DFLASH OPTION_B DIAG: γ K layer0 slot0={} phys_block={} off={} \
                     src[0..8]={:?} cached[0..8]={:?}",
                    slot0,
                    phys_block,
                    block_off,
                    src,
                    cached,
                );

                // Also check ctx_count and the block_table[0..4].
                let mut bt_bytes = [0u8; 16];
                gpu.copy_d2h(block_table_dev, &mut bt_bytes)?;
                let bt: Vec<u32> = bt_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                tracing::info!(
                    "DFLASH OPTION_B DIAG: ctx_count={} block_table[0..4]={:?} kv_len={}",
                    ctx_count,
                    bt,
                    kv_len,
                );

                // Read ctx slot 0 from the same K pool — should be the
                // first ctx token's K row 0 (written by precompute_ctx_kv).
                // If it's zero, the ctx write missed entirely.
                if ctx_count > 0 {
                    let ctx0_ptr = k_pool; // physical slot 0 = block_table[0] * stride + 0
                    let ctx0_phys_block = bt[0] as usize;
                    let ctx0_addr = k_pool.offset(ctx0_phys_block * block_stride_bytes);
                    let ctx0 = read8(ctx0_addr)?;
                    let ctx0_max = ctx0.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                    tracing::info!(
                        "DFLASH OPTION_B DIAG: ctx K layer0 slot0 (phys_block={}) values={:?} max_abs={:.4}",
                        ctx0_phys_block,
                        ctx0,
                        ctx0_max,
                    );
                    let _ = ctx0_ptr;
                }
            }
        }

        // 3f. paged attention — q_len=γ, kv_len=ctx_count+γ.
        //
        // Phase 5 (CUDA graph): kv_len and q_offset are read from
        // `option_b_indirect_args_dev` at kernel entry rather than passed
        // as scalar args, so the captured launch survives per-call value
        // changes. Host writes the 8-byte pair in forward_block.rs
        // pre-graph; replays pick up whatever's there.
        ops::prefill_attention_paged_dflash_bf16_indirect(
            gpu,
            self.kernels.prefill_attn_dflash_bf16_indirect,
            self.scratch.q_buf,
            k_pool,
            v_pool,
            self.scratch.attn_out,
            block_table_dev,
            g,
            self.scratch.option_b_indirect_args_dev,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            16, // cache_block_size
            0,  // sliding_window — drafter not windowed for now
            inv_sqrt_d,
            stream,
        )?;
        // Suppress unused-var warning: kv_len and ctx_count are still
        // computed for slot-mapping and slot-position arithmetic above;
        // the indirect kernel pulls them from device memory at entry.
        let _ = (kv_len, ctx_count);

        // 3g. o_proj — γ rows.
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            g,
            h,
            q_dim,
            stream,
        )?;

        // 3h. residual: stream_buf += stream_acc (γ rows).
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            self.scratch.stream_acc,
            g * h,
            stream,
        )?;

        // 3i. post_attention_layernorm.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            g,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        // 3j. MLP gate + up + silu_mul + down — γ rows.
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.norm_buf,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            g,
            inter,
            h,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.norm_buf,
            &layer.up_proj,
            self.scratch.mlp_up,
            g,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            g * inter,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            g,
            h,
            inter,
            stream,
        )?;

        // 3k. residual.
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            self.scratch.stream_acc,
            g * h,
            stream,
        )?;

        Ok(())
    }
}
