// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash Option B — once-per-propose precompute of drafter ctx K/V.
//!
//! Lifts ctx-side work out of the γ-block layer loop so the per-layer
//! attention path runs over `q_len = γ` rows instead of `n_attn = γ + ctx`.
//! The drafter's paged BF16 KV cache holds the ctx K/V across propose
//! calls; this module appends the *new* ctx slots (delta since the last
//! propose) once, fused across all L drafter layers.
//!
//! Pipeline per call — matches vLLM `precompute_and_store_context_kv`
//! (`qwen3_dflash.py:342-434`) op-for-op:
//!   1. Batched `fc` projection: `[n, L_t * h_t] → [n, h]`.
//!   2. `hidden_norm` (RMS) over `[n, h]`.
//!   3. Single fused KV GEMM: `[n, h] × [h, L * 2 * kv_dim]
//!      → [n, L * 2 * kv_dim]`.  (py:381–392)
//!   4. Compact all L layers' K into contiguous `[L*n, kv_dim]` staging.
//!      (py:386–391 permute+contiguous)
//!   5. Per-layer k_norm over `[n, kv_dim]` blocks.  (py:393–401)
//!   6. **Single** fused RoPE over `[L*n, kv_dim]` at repeated positions.
//!      (py:403–418 — one `ops.rotary_embedding` call for all layers)
//!   7. Per-layer `reshape_and_cache` writing K/V into the layer's paged
//!      cache at the appropriate slot mapping.  (py:420–434)
//!
//! Scratch buffers borrowed (all available before the γ-block layer loop):
//!   `mlp_intermediate` → all_k_stage `[L*n, kv_dim]` BF16
//!   `norm_buf`         → extended_positions `[L*n]` i32 (first L*n*4 bytes)
//!   `k_buf` / `v_buf`  → per-layer K/V staging for cache write

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use crate::layer::ForwardContext;
use crate::weight_map::DenseWeight;

impl BlockDiffusionDraftHead {
    /// Project new ctx hidden states through `fc + hidden_norm`, derive
    /// fused K/V across all drafter layers, apply per-layer k_norm + fused
    /// RoPE, and write the results into the per-layer paged KV cache slots.
    ///
    /// Mirrors `precompute_and_store_context_kv` (qwen3_dflash.py:342).
    ///
    /// `ctx_base_ptr`: base of the captured-target-hidden accumulator
    ///   (`[max_ctx_len, L_t * h_t]` BF16).
    /// `start_slot`: first ctx slot to project (inclusive).
    /// `new_ctx_count`: number of contiguous ctx slots starting at
    ///   `start_slot` to feed through.
    /// `slot_positions`: `&[i32]` of length `new_ctx_count` — the TRUE
    ///   fixed RoPE position of each row being computed (stamped at append
    ///   time, vLLM convention).
    /// `slot_mapping_dev`: device pointer to an `i32[new_ctx_count]`
    ///   array of paged-cache slot indices.
    /// `commit`: when `true`, write the computed K/V into the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn precompute_ctx_kv(
        &self,
        ctx_base_ptr: DevicePtr,
        start_slot: usize,
        new_ctx_count: usize,
        slot_positions: &[i32],
        slot_mapping_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        commit: bool,
    ) -> Result<()> {
        use crate::layers::ops;

        if new_ctx_count == 0 {
            return Ok(());
        }

        let Some(fused_kv) = self.fused_kv_weight else {
            anyhow::bail!(
                "DFlash precompute_ctx_kv called without fused_kv_weight — build-order bug"
            );
        };

        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let h = self.hidden_size as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let n = new_ctx_count as u32;
        let l_total = self.num_layers;
        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * bf16;
        let kv_slab_bytes = (kv_dim as usize) * bf16;
        // Stride (bytes) between adjacent rows in the fused KV GEMM output.
        let row_stride = l_total * 2 * kv_slab_bytes;

        // One-shot diagnostic dump (ATLAS_DFLASH_PRECOMPUTE_DUMP=1).
        static PRECOMPUTE_DUMP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        let dump = std::env::var("ATLAS_DFLASH_PRECOMPUTE_DUMP")
            .ok()
            .as_deref()
            == Some("1")
            && !PRECOMPUTE_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed);
        let dump_buf = |label: &str, ptr: DevicePtr, bytes: usize| -> Result<()> {
            if !dump {
                return Ok(());
            }
            let mut buf = vec![0u8; bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let path = format!("/tmp/atlas_precompute_{label}.bin");
            if let Err(e) = std::fs::write(&path, &buf) {
                tracing::warn!("precompute dump {label} write failed: {e}");
            } else {
                tracing::info!("precompute dump {label}: {} bytes → {}", bytes, path);
            }
            Ok(())
        };

        // ── Step 1: batched fc projection ────────────────────────────
        // py:175  `target_hidden = self.hidden_norm(self.fc(target_hidden))`
        //   first half: fc maps [n, L_t*h_t] → [n, h].
        let src = ctx_base_ptr.offset(start_slot * ctx_slot_bytes);
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            src,
            &self.fc,
            self.scratch.fc_proj,
            n,
            h,
            target_hidden_dim as u32,
            stream,
        )?;
        dump_buf(
            "fc_proj",
            self.scratch.fc_proj,
            new_ctx_count * self.hidden_size * bf16,
        )?;

        // ── Step 2: hidden_norm RMS in-place ─────────────────────────
        // py:375–380  `ops.rms_norm(normed_context_states, context_states,
        //               self._hidden_norm_weight, self._rms_norm_eps)`
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.fc_proj,
            &self.hidden_norm,
            self.scratch.fc_proj,
            n,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        dump_buf(
            "fc_proj_normed",
            self.scratch.fc_proj,
            new_ctx_count * self.hidden_size * bf16,
        )?;

        // ── Step 3: fused KV GEMM across all layers ──────────────────
        // py:381–392  `all_kv_flat = F.linear(normed_context_states,
        //                self._fused_kv_weight, self._fused_kv_bias)`
        // [n, h] × [h, L * 2 * kv_dim] → [n, L * 2 * kv_dim].
        // Layout per row: [K_0 | V_0 | K_1 | V_1 | … | K_{L-1} | V_{L-1}].
        let fused_w = DenseWeight { weight: fused_kv };
        let fused_n_cols = (l_total as u32) * 2 * kv_dim;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.fc_proj,
            &fused_w,
            self.scratch.fused_kv_out,
            n,
            fused_n_cols,
            h,
            stream,
        )?;
        dump_buf(
            "fused_kv_out",
            self.scratch.fused_kv_out,
            new_ctx_count * l_total * 2 * kv_slab_bytes,
        )?;

        // ── Step 4: build extended position array for fused RoPE ─────
        // py:407  `positions_repeated = context_positions.repeat(L)`
        //   = slot_positions ×L = [p0..p_{n-1}, p0..p_{n-1}, …] (L copies).
        // Stored in norm_buf (first L*n*4 bytes; norm_buf = 2 MB >>  this).
        // norm_buf is not needed until step 3a of the γ-block layer loop
        // (forward_block_layer_pre_attn), which runs after precompute returns.
        debug_assert_eq!(slot_positions.len(), new_ctx_count);
        {
            let repeated_bytes: Vec<u8> = slot_positions
                .iter()
                .cloned()
                .cycle()
                .take(l_total * new_ctx_count)
                .flat_map(|p: i32| p.to_le_bytes())
                .collect();
            gpu.copy_h2d(&repeated_bytes, self.scratch.norm_buf)?;
        }

        // ── Step 5: compact all L layers' K → all_k_stage ────────────
        // py:386–391  `all_kv = all_kv_flat.view(n,L,2,nkv,hd)
        //                          .permute(2,1,0,3,4).contiguous()`
        //              `all_k = all_kv[0]`  → [L, n, nkv, hd] contiguous.
        // Atlas: copy_d2d row-by-row to build the same [L, n, kv_dim] layout
        // in mlp_intermediate (borrowed; not used until step 3j of the
        // γ-block layer loop). Capacity: n_attn × inter × 2 >> L×n×kv_dim×2.
        let all_k_stage = self.scratch.mlp_intermediate;
        for l in 0..l_total {
            for row in 0..new_ctx_count {
                let k_src = self
                    .scratch
                    .fused_kv_out
                    .offset(row * row_stride + l * 2 * kv_slab_bytes);
                let k_dst = all_k_stage.offset((l * new_ctx_count + row) * kv_slab_bytes);
                gpu.copy_d2d(k_src, k_dst, kv_slab_bytes)?;
            }
        }

        // ── Step 6: per-layer k_norm ──────────────────────────────────
        // py:393–401  `for i in range(L): ops.rms_norm(all_k_normed[i],
        //               all_k[i], self._k_norm_weights[i], eps)`
        // Each block: all_k_stage[l*n .. (l+1)*n] shape [n, kv_dim]
        //   → treated as [n * num_kv_heads, head_dim] for per-head norm.
        for l in 0..l_total {
            let k_l = all_k_stage.offset(l * new_ctx_count * kv_slab_bytes);
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                k_l,
                &self.layers[l].k_norm,
                k_l,
                n * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;
        }

        // ── Step 7: single fused RoPE across all L layers ─────────────
        // py:403–418  `all_k_flat = all_k_normed.view(L * n, kv)`
        //              `ops.rotary_embedding(positions_repeated, all_k_flat,
        //                None, head_size, cos_sin_cache, is_neox)`
        // Atlas: rope_yarn with seq_len=L*n, num_q_heads=0 (K-only).
        //   K buffer = all_k_stage[0..L*n*kv_dim].
        //   positions = norm_buf (L*n i32 written in step 4).
        // Grid: [num_kv_heads, ceil(L*n / pos_per_block), 1] — all CTAs
        //   process K heads across the full L*n row range.
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            all_k_stage, // Q — unread when num_q_heads=0
            all_k_stage, // K = all_k_flat [L*n, kv_dim]
            self.scratch.norm_buf,
            l_total as u32 * n,
            0, // num_q_heads=0 → K-only (rope.cu:46-48)
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;

        if dump {
            dump_buf(
                "layer0_k_post_rope",
                all_k_stage,
                new_ctx_count * kv_slab_bytes,
            )?;
        }

        // ── Step 8: per-layer V compaction + reshape_and_cache ────────
        // py:420–434  per-layer `attn.impl.do_kv_cache_update(...)`.
        // Compact V_l inline (no norm/RoPE applied to V — oracle matches).
        // K is read from all_k_stage[l*n..]; V from fused_kv_out (GEMM output).
        let v_stage = self.scratch.v_buf;
        for l in 0..l_total {
            let k_l = all_k_stage.offset(l * new_ctx_count * kv_slab_bytes);

            // Compact V_l from the fused GEMM output.
            for row in 0..new_ctx_count {
                let v_src = self
                    .scratch
                    .fused_kv_out
                    .offset(row * row_stride + l * 2 * kv_slab_bytes + kv_slab_bytes);
                let v_dst = v_stage.offset(row * kv_slab_bytes);
                gpu.copy_d2d(v_src, v_dst, kv_slab_bytes)?;
            }

            if dump && l == 0 {
                dump_buf("layer0_v", v_stage, new_ctx_count * kv_slab_bytes)?;
            }

            if commit {
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(l), cache.v_pool_ptr(l))
                };
                ops::reshape_and_cache(
                    gpu,
                    self.kernels.reshape_cache_bf16,
                    k_l,
                    v_stage,
                    k_pool,
                    v_pool,
                    slot_mapping_dev,
                    n,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16, // block_size — matches from_weights.rs
                    kv_dim,
                    kv_dim,
                    0,
                    stream,
                )?;
            }
        }

        if dump {
            PRECOMPUTE_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }
}
