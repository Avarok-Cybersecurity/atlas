// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash Option B — once-per-propose precompute of drafter ctx K/V.
//!
//! Lifts ctx-side work out of the γ-block layer loop so the per-layer
//! attention path runs over `q_len = γ` rows instead of `n_attn = γ + ctx`.
//! The drafter's paged BF16 KV cache holds the ctx K/V across propose
//! calls; this module appends the *new* ctx slots (delta since the last
//! propose) once, fused across all L drafter layers.
//!
//! Pipeline per call (mirrors vLLM `precompute_and_store_context_kv` in
//! `qwen3_dflash.py:342-434`):
//!   1. Batched `fc` projection: `[new_ctx, L_t * h_t] → [new_ctx, h]`.
//!   2. `hidden_norm` (RMS) over `[new_ctx, h]`.
//!   3. **Single** fused KV GEMM: `[new_ctx, h] × [h, L * 2 * kv_dim]
//!      → [new_ctx, L * 2 * kv_dim]`. Layout per layer: `[K_i, V_i]`.
//!   4. Per-layer k_norm (5 launches at L=5 — vLLM matches; deferred fuse).
//!   5. Per-layer RoPE-on-K (5 launches — keys only, queries derived in
//!      the γ-block).
//!   6. Per-layer `reshape_and_cache` writing K/V into the layer's paged
//!      cache at the appropriate slot mapping.
//!
//! Per-step kernel launches (independent of γ): `~1 + 1 + 1 + 5 + 5 + 5
//! = 18` for L=5, vs the current ~17 per layer × 5 layers = 85 (a 4.7×
//! reduction on the ctx side alone). The real win is that none of these
//! launches scale with γ — every γ row is removed from the layer body.
//!
//! Stage 3 wires this in behind `ATLAS_DFLASH_PRECOMPUTE=1`. The pyref
//! diff harness compares dumped K/V against the canonical
//! `~/atlas_dflash_pyref.py` PyTorch reference; once cosine ≥ 0.9999 we
//! flip the default and rip out the in-layer ctx-K/V-override path.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use crate::layer::ForwardContext;
use crate::weight_map::DenseWeight;

impl BlockDiffusionDraftHead {
    /// Project new ctx hidden states through `fc + hidden_norm`, derive
    /// fused K/V across all drafter layers, apply per-layer k_norm + RoPE,
    /// and write the results into the per-layer paged KV cache slots.
    ///
    /// `ctx_base_ptr`: base of the captured-target-hidden accumulator
    ///   (`[max_ctx_len, L_t * h_t]` BF16).
    /// `start_slot`: first ctx slot to project (inclusive).
    /// `new_ctx_count`: number of contiguous ctx slots starting at
    ///   `start_slot` to feed through.
    /// `absolute_start_pos`: logical sequence position of `start_slot`
    ///   (used as the RoPE position for the first new K row).
    /// `slot_mapping_dev`: device pointer to an `i32[new_ctx_count]`
    ///   array of paged-cache slot indices, pre-filled by the caller
    ///   via `ops::fill_slots_from_block_table` against the drafter's
    ///   block table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn precompute_ctx_kv(
        &self,
        ctx_base_ptr: DevicePtr,
        start_slot: usize,
        new_ctx_count: usize,
        absolute_start_pos: usize,
        slot_mapping_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        if new_ctx_count == 0 {
            return Ok(());
        }

        let Some(fused_kv) = self.fused_kv_weight else {
            anyhow::bail!(
                "DFlash precompute_ctx_kv called without fused_kv_weight allocated — \
                 from_weights() should populate it; this is a build-order bug."
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

        // Optional dump for pyref bit-exact diff. Gated by env var so it's
        // zero-cost off-path. Writes raw BF16 LE bytes — pyref reads them
        // with `np.frombuffer(..., dtype=np.uint16).view(<bf16 emulation>)`.
        // ONE-SHOT: pyref reads a single consistent snapshot (the meta JSON
        // from forward_block's DUMP_FULL is also one-shot). Each subsequent
        // propose call would overwrite with a different new_ctx_count,
        // making shape mismatch errors in the harness.
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

        // ── Step 1: batched fc projection ─────────────────────────
        // [new_ctx, L_t * h_t] × fc^T → [new_ctx, h]. dense_gemm runs
        // over the whole batch in one launch (vs the current per-row
        // GEMV loop in forward_block.rs:160-173).
        let src = ctx_base_ptr.offset(start_slot * ctx_slot_bytes);
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
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

        // ── Step 2: hidden_norm RMS over fc_proj in-place ────────
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

        // ── Step 3: fused KV GEMM across all layers ──────────────
        // [n, h] × [h, L * 2 * kv_dim] → [n, L * 2 * kv_dim].
        // dense_gemm consumes a &DenseWeight — wrap the raw fused
        // pointer in a transient DenseWeight (it's just a wrapper).
        let fused_w = DenseWeight { weight: fused_kv };
        let fused_n_cols = (l_total as u32) * 2 * kv_dim;
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
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
            new_ctx_count * (l_total * 2 * (kv_dim as usize)) * bf16,
        )?;

        // Per-row stride through the fused output (in bytes) for layer slicing.
        let row_stride = (l_total * 2 * (kv_dim as usize)) * bf16;
        let kv_slab_bytes = (kv_dim as usize) * bf16;

        // ── Step 4: positions buffer for RoPE-on-K ───────────────
        // Reuse scratch.position_ids (i32[ctx_window+γ]); first n
        // entries are the absolute positions of the new ctx rows.
        let pos_host: Vec<i32> = (0..new_ctx_count)
            .map(|i| (absolute_start_pos + i) as i32)
            .collect();
        let pos_bytes: Vec<u8> = pos_host.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&pos_bytes, self.scratch.position_ids)?;

        // ── Step 5: per-layer k_norm + RoPE + reshape_and_cache ──
        //
        // For each layer l we slice into the fused output buffer to
        // get K_l and V_l (contiguous in memory: K then V). Strides
        // are computed in Rust (no fattening of DenseWeight needed).
        //
        // The KV slabs are not contiguous across rows for a single
        // layer (every row interleaves all layers' K/V), so the
        // per-layer kernels operate on a "logical view" — we hand
        // them the base pointer of layer l's K (or V) in row 0, but
        // the kernels assume contiguous `[n, kv_dim]` BF16. That
        // requires either (a) a per-layer compaction copy, or (b) a
        // stride-aware variant of each kernel.
        //
        // ── Compaction path (chosen for stage 3) ──────────────────
        // Compact each layer's K and V slabs into a contiguous
        // `[n, kv_dim]` BF16 buffer using `copy_d2d` row-by-row. At
        // n=512 ctx tokens × L=5 layers × 2 (K+V) = 5120 d2d launches
        // worst case; the alternative (stride-aware k_norm/rope/
        // reshape) costs more engineering and is the optimization
        // path for a later phase. For now: correctness first.
        //
        // We reuse `scratch.k_buf` / `scratch.v_buf` as the per-layer
        // contiguous staging area (sized n_attn = ctx_window + γ ≥ n,
        // so it fits any new_ctx_count up to ctx_window).
        let k_stage = self.scratch.k_buf;
        let v_stage = self.scratch.v_buf;

        for l in 0..l_total {
            // (a) Compact K_l and V_l from the fused output.
            for row in 0..new_ctx_count {
                let row_base = self.scratch.fused_kv_out.offset(row * row_stride);
                let layer_base = row_base.offset(l * 2 * kv_slab_bytes);
                let k_src = layer_base;
                let v_src = layer_base.offset(kv_slab_bytes);
                let k_dst = k_stage.offset(row * kv_slab_bytes);
                let v_dst = v_stage.offset(row * kv_slab_bytes);
                gpu.copy_d2d(k_src, k_dst, kv_slab_bytes)?;
                gpu.copy_d2d(v_src, v_dst, kv_slab_bytes)?;
            }

            // (b) k_norm — per-head RMSNorm over head_dim. Treats
            // the [n, num_kv_heads, head_dim] tensor as n*num_kv_heads
            // rows of length head_dim.
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                k_stage,
                &self.layers[l].k_norm,
                k_stage,
                n * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;

            // (c) RoPE-on-K. num_q_heads=0 means the kernel processes
            // only K blocks (see kernels/gb10/common/rope.cu:46-48).
            // Pass k_stage as both Q and K — Q is unread when nq=0.
            ops::rope_yarn(
                gpu,
                self.kernels.rope_qwen3,
                k_stage, // Q pointer — unused with num_q_heads=0
                k_stage,
                self.scratch.position_ids,
                n,
                0,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rotary_dim as u32,
                self.yarn_inv_freq,
                self.rope_theta,
                stream,
            )?;

            if dump && l == 0 {
                dump_buf(
                    "layer0_k_post_rope",
                    k_stage,
                    new_ctx_count * (kv_dim as usize) * bf16,
                )?;
                dump_buf(
                    "layer0_v",
                    v_stage,
                    new_ctx_count * (kv_dim as usize) * bf16,
                )?;
            }

            // (d) reshape_and_cache — write into the layer's paged
            // K and V pools at the supplied slot mapping. Skipped in
            // dump-only mode (ATLAS_DFLASH_PRECOMPUTE_DUMP=1 with no
            // ATLAS_DFLASH_PRECOMPUTE_COMMIT=1) because stage 3 has
            // not yet allocated block_table entries on the drafter's
            // paged cache — reshape would walk into unmapped slots.
            let commit = std::env::var("ATLAS_DFLASH_PRECOMPUTE_COMMIT")
                .ok()
                .as_deref()
                == Some("1");
            if commit {
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(l), cache.v_pool_ptr(l))
                };
                ops::reshape_and_cache(
                    gpu,
                    self.kernels.reshape_cache_bf16,
                    k_stage,
                    v_stage,
                    k_pool,
                    v_pool,
                    slot_mapping_dev,
                    n,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16, // block_size — matches from_weights.rs:68
                    kv_dim,
                    kv_dim,
                    0,
                    stream,
                )?;
            }
        }

        // Mark one-shot dump complete so subsequent propose calls don't
        // overwrite the snapshot with a different new_ctx_count.
        if dump {
            PRECOMPUTE_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }
}
