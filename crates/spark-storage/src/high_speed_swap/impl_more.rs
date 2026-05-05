// SPDX-License-Identifier: AGPL-3.0-only
//
//! Additional `HighSpeedSwap` methods (offload + attention orchestration).

use anyhow::Result;
use std::ffi::c_void;

use super::HighSpeedSwap;
use crate::backend::{ReadRequest, StorageBackend};
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, copy_d_to_h_async, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, KvKind};
use crate::predictor::Predictor;
use crate::scratch_pool::{ResidentKey, ScratchPool};

impl HighSpeedSwap {
    /// Persist a freshly-written KV block to disk and update the predictor's
    /// per-block K_lr. K block layout is `[block_size, num_kv_heads, head_dim]`
    /// BF16 in both `*_dev` (used for projection) and `*_host` (used for the
    /// per-(kv_head) disk stripe).
    pub fn offload_block(
        &mut self,
        ctx: &CudaCtx,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        self.offload_block_on_stream(
            ctx.stream,
            layer,
            block,
            k_block_dev,
            k_block_host,
            v_block_host,
        )
    }

    /// Stream-only variant for production callers (spark-model decode path).
    /// `stream` must already be bound to the current thread's CUDA context.
    pub fn offload_block_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        // True when the production HBM buffer at `k_block_dev` is BF16-laid-out;
        // the predictor's project_kv_block kernel reads it as BF16. Non-BF16
        // callers must use `offload_block_no_predict_on_stream`.
        self.offload_block_inner_on_stream(
            stream,
            layer,
            block,
            k_block_dev,
            k_block_host,
            v_block_host,
            true,
        )
    }

    /// FP8/quantized callers: identical to `offload_block_on_stream` but skips
    /// the predictor's per-block K projection (since `k_block_dev` is not
    /// BF16-laid-out — running the BF16 kernel on it would OOB-read into
    /// adjacent blocks). Eviction policy degrades to LRU-only for these
    /// blocks; correctness is preserved.
    pub fn offload_block_no_predict_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> Result<()> {
        self.offload_block_inner_on_stream(
            stream,
            layer,
            block,
            0,
            k_block_host,
            v_block_host,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn offload_block_inner_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
        do_predict: bool,
    ) -> Result<()> {
        if do_predict {
            self.predictor.project_kv_block_on_stream(
                stream,
                layer as usize,
                block as usize,
                k_block_dev,
            )?;
        }
        let bs = self.model.block_size as usize;
        let nkv = self.model.num_kv_heads as usize;
        let hd = self.model.head_dim as usize;
        if k_block_host.len() != bs * nkv * hd || v_block_host.len() != bs * nkv * hd {
            anyhow::bail!(
                "offload_block: host buffers must be {} BF16 elements",
                bs * nkv * hd
            );
        }
        for kh in 0..nkv {
            let mut k_stripe = Vec::with_capacity(bs * hd * 2);
            let mut v_stripe = Vec::with_capacity(bs * hd * 2);
            for tok in 0..bs {
                let base = (tok * nkv + kh) * hd;
                for x in &k_block_host[base..base + hd] {
                    k_stripe.extend_from_slice(&x.to_le_bytes());
                }
                for x in &v_block_host[base..base + hd] {
                    v_stripe.extend_from_slice(&x.to_le_bytes());
                }
            }
            self.backend
                .write_from_host(GroupKey::new(layer, block, kh as u16, KvKind::K), &k_stripe)?;
            self.backend
                .write_from_host(GroupKey::new(layer, block, kh as u16, KvKind::V), &v_stripe)?;
        }
        // Drop the resident-cache copy (if any). The on-disk image was just
        // overwritten; without invalidation, attend_layer_on_stream would
        // keep serving the stale slot. Critical for decode where the active
        // block is re-offloaded every step with new slots filled.
        self.pool.invalidate(ResidentKey { layer, block });
        Ok(())
    }

    /// Run streaming attention for one (layer, sequence). `q_dev` is the
    /// full [num_q_heads × head_dim] BF16 query for this step;
    /// `seq_block_ids` is the sequence's full block list; `output_dev`
    /// receives the [num_q_heads × head_dim] BF16 attention output.
    pub fn attend_layer(
        &mut self,
        ctx: &CudaCtx,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> Result<()> {
        self.attend_layer_on_stream(ctx.stream, layer, seq_block_ids, q_dev, output_dev)
    }

    /// Stream-only variant for production callers (spark-model decode path).
    /// `stream` must already be bound to the current thread's CUDA context.
    ///
    /// Backwards-compat: defaults `last_block_valid_slots` to `block_size`,
    /// i.e. no causal masking — appropriate for decode where the active
    /// block's stale slots are zero-init from `zero_block`. For prefill,
    /// callers MUST use `attend_layer_on_stream_with_q_pos` to pass the
    /// query's absolute position, otherwise future tokens within the
    /// active block leak into past queries.
    pub fn attend_layer_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> Result<()> {
        let bs = self.model.block_size as i32;
        self.attend_layer_on_stream_with_q_pos(stream, layer, seq_block_ids, q_dev, output_dev, bs)
    }

    /// Causal-masking variant: `last_block_valid_slots` controls how many
    /// slots of the LAST block in `seq_block_ids` are consumed by the
    /// attention kernel. For prefill query at absolute position `q_pos`,
    /// pass `(q_pos % block_size) + 1` to mask out future positions in
    /// the active block.
    pub fn attend_layer_on_stream_with_q_pos(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
        last_block_valid_slots: i32,
    ) -> Result<()> {
        // 1. Project Q. 2. Score every block at this layer (only seq subset
        //    is consumed; the rest is wasted compute but score_blocks is µs).
        self.predictor
            .project_q_on_stream(stream, q_dev, self.q_proj.ptr)?;
        let m = &self.model;
        let layer_a_g = self.predictor.a_g_dev_ptr()
            + (layer as u64)
                * (m.max_blocks_per_layer as u64)
                * (m.num_kv_heads as u64)
                * (m.block_size as u64)
                * (self.cfg.rank as u64)
                * 2;
        self.predictor.score_blocks_on_stream(
            stream,
            self.q_proj.ptr,
            layer_a_g,
            self.block_scores_dev.ptr,
            m.max_blocks_per_layer as usize,
        )?;
        copy_d_to_h_async(
            self.score_host_buf.as_mut_ptr() as *mut c_void,
            self.block_scores_dev.ptr,
            self.score_host_buf.len() * 4,
            stream,
        )?;
        stream_sync(stream)?;

        // 3. Tile loop.
        self.attn.begin_step_on_stream(stream, 1)?;
        let tile_cap = self.cfg.resident_blocks as usize;
        let mut tile_idx = 0;
        while tile_idx < seq_block_ids.len() {
            let tile_end = (tile_idx + tile_cap).min(seq_block_ids.len());
            let tile = &seq_block_ids[tile_idx..tile_end];

            // Pin slots already resident for tile blocks; mark them touched.
            let mut block_table = vec![0_i32; tile_cap];
            let mut pinned: Vec<u32> = Vec::new();
            // First pass: identify which tile blocks are missing.
            let mut missing: Vec<u32> = Vec::new();
            for (i, &blk) in tile.iter().enumerate() {
                let key = ResidentKey { layer, block: blk };
                if let Some(slot) = self.pool.lookup(key) {
                    block_table[i] = slot as i32;
                    pinned.push(slot);
                    self.eviction.touch(slot);
                } else {
                    missing.push(blk);
                }
            }
            // Second pass: assign + read missing blocks.
            let mut reqs: Vec<ReadRequest> = Vec::new();
            for &blk in &missing {
                let key = ResidentKey { layer, block: blk };
                let candidates = self.eviction.rank(&pinned);
                let slot = self.pool.assign(key, &candidates)?;
                pinned.push(slot);
                self.eviction.touch(slot);
                self.eviction
                    .record_score(slot, self.score_host_buf[blk as usize]);
                // Find this block's index in the tile so the block_table is right.
                let idx = tile.iter().position(|&x| x == blk).unwrap();
                block_table[idx] = slot as i32;
                for kh in 0..self.model.num_kv_heads {
                    reqs.push(ReadRequest {
                        group: GroupKey::new(layer, blk, kh, KvKind::K),
                        dst_dev_ptr: self.pool.slot_k_ptr(slot, kh),
                    });
                    reqs.push(ReadRequest {
                        group: GroupKey::new(layer, blk, kh, KvKind::V),
                        dst_dev_ptr: self.pool.slot_v_ptr(slot, kh),
                    });
                }
            }
            self.backend.read(&reqs, stream)?;

            // 4. Tiled attention launch.
            let counts = [(tile.len()) as i32];
            copy_h_to_d_async(
                self.block_table_dev.ptr,
                block_table.as_ptr() as *const c_void,
                tile_cap * 4,
                stream,
            )?;
            copy_h_to_d_async(
                self.counts_dev.ptr,
                counts.as_ptr() as *const c_void,
                4,
                stream,
            )?;
            let (s_blk, s_tok, s_kvh) = self.attn.scratch_pool_strides();
            let v_off = (self.model.num_kv_heads as u64)
                * (self.model.block_size as u64)
                * (self.model.head_dim as u64)
                * 2;
            // Causal mask: only apply on the FINAL tile of the seq's block
            // list. Earlier tiles are full blocks of historical K/V.
            let lbvs = if tile_end == seq_block_ids.len() {
                last_block_valid_slots
            } else {
                self.model.block_size as i32
            };
            self.attn.step_tile_on_stream(
                stream,
                q_dev,
                self.pool.pool_dev_ptr(),
                self.pool.pool_dev_ptr() + v_off,
                self.block_table_dev.ptr,
                self.counts_dev.ptr,
                1,
                s_blk,
                s_tok,
                s_kvh,
                lbvs,
            )?;
            tile_idx = tile_end;
        }
        self.attn.finalize_on_stream(stream, output_dev, 1)?;
        Ok(())
    }

    /// Test/diag accessors.
    pub fn pool(&self) -> &ScratchPool {
        &self.pool
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
    pub fn config(&self) -> &HighSpeedSwapConfig {
        &self.cfg
    }
}
