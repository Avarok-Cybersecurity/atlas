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
//!
//! **Phase F.1 split** (2026-05-28): the monolithic body is now split
//! into three pieces — `forward_block_layer_pre_attn` (3a–3e),
//! `forward_block_layer_attention` (3f), and
//! `forward_block_layer_post_attn` (3g–3k). The legacy
//! `forward_block_layer_paged` is preserved as a thin orchestrator that
//! calls all three in sequence; behaviour is identical. F.2 will wrap
//! the pre and post halves in their own CUDA graph captures, leaving
//! attention eager (vLLM piecewise pattern).

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
    /// Friday 2026-06-11 (id259 next-action): when true, this propose is the
    /// armed one-shot per-layer block-forward parity dump. Each layer dumps
    /// its noise-block intermediates (post-input_norm, post-qkv, post-qknorm,
    /// post-rope, post-attn, post-mlp) to /tmp/atlas_blk_L{layer}_{stage}.bin
    /// so atlas_dflash_block_parity.py can localize the FIRST op that diverges
    /// from the z-lab PyTorch reference (cos<0.999). Gated by
    /// ATLAS_DFLASH_BLOCK_DUMP=1 upstream; forces the eager path (graph
    /// capture cannot contain the D2H/sync this dump injects).
    pub block_dump: bool,
}

impl BlockDiffusionDraftHead {
    /// γ-only paged-attention per-layer body. See module docstring.
    ///
    /// **Phase F.1**: this is now a thin orchestrator that calls the three
    /// split halves — `forward_block_layer_pre_attn`, the attention kernel
    /// itself, and `forward_block_layer_post_attn`. Behaviour is identical
    /// to the pre-F.1 monolithic body; the split exists so Phase F.2 can
    /// wrap each half in its own CUDA graph capture (mirroring vLLM's
    /// piecewise pattern — attention stays eager, everything else
    /// captures).
    ///
    /// **Phase F.2**: this method is no longer on the hot path —
    /// `forward_block` now calls `forward_block_layer_pre_attn`,
    /// `forward_block_layer_attention`, and `forward_block_layer_post_attn`
    /// directly so it can wrap each in its own capture region. Kept
    /// available so future ablation paths (and tests) can run the
    /// monolithic body without rebuilding the call site.
    #[allow(dead_code)]
    pub(super) fn forward_block_layer_paged(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let (k_pool, v_pool) = self.forward_block_layer_pre_attn(layer, args, ctx)?;
        self.forward_block_layer_attention(args, ctx, k_pool, v_pool)?;
        self.forward_block_layer_post_attn(layer, args, ctx)?;
        Ok(())
    }

    /// Friday 2026-06-11 (id259): one-shot per-layer block-forward parity
    /// dump helper. Copies `rows*cols` BF16 values from `src` (γ-row noise
    /// block scratch) to `/tmp/atlas_blk_L{layer_idx}_{stage}.bin`. Reused at
    /// each pipeline boundary (post-input_norm, post-qkv, post-qknorm,
    /// post-rope, post-attn, post-mlp) so atlas_dflash_block_parity.py can
    /// walk the layers and flag the FIRST stage with cos<0.999 vs the z-lab
    /// PyTorch reference. Synchronous (sync + D2H) — only ever runs on the
    /// armed eager propose (graph capture is disabled when block_dump=true).
    pub(super) fn block_dump_buf(
        &self,
        ctx: &ForwardContext,
        src: DevicePtr,
        layer_idx: usize,
        stage: &str,
        rows: u32,
        cols: u32,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let n_bytes = rows as usize * cols as usize * bf16;
        gpu.synchronize(stream)?;
        let mut buf = vec![0u8; n_bytes];
        gpu.copy_d2h(src, &mut buf)?;
        let path = format!("/tmp/atlas_blk_L{layer_idx}_{stage}.bin");
        if let Err(e) = std::fs::write(&path, &buf) {
            tracing::warn!("DFLASH BLOCK_DUMP per-layer: write {path} failed: {e}");
        } else if layer_idx == 0 {
            tracing::info!("DFLASH BLOCK_DUMP per-layer: wrote {path} ({rows}x{cols} BF16)");
        }
        Ok(())
    }

    /// Phase F pre-attention half: input_layernorm → q_proj → q_norm →
    /// k_proj → k_norm → v_proj → rope → reshape_and_cache (steps 3a–3e).
    ///
    /// Op order is byte-faithful to dflash.py lines 68-80 (noise half):
    ///   125 input_layernorm → 68 q_proj → 70 q_norm → 72 k_proj →
    ///   77(noise) k_norm → 74 v_proj → 80 rope → 75-76 cache write.
    ///
    /// Returns `(k_pool, v_pool)` so the caller can invoke attention
    /// without re-locking the KV cache.
    ///
    /// **Capture eligibility**: this body is a pure sequence of compute
    /// kernels reading from stable scratch pointers + the locked
    /// (k_pool, v_pool) pointers. Safe to capture as a CUDA graph
    /// EXCEPT when the layer-0 `ATLAS_DFLASH_OPTION_B_DIAG=1` debug
    /// block runs — that path injects D2H + sync, but it's gated by an
    /// env var that already disables graph eligibility upstream
    /// (`forward_block.rs:438`).
    pub(super) fn forward_block_layer_pre_attn(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<(DevicePtr, DevicePtr)> {
        use crate::layers::ops;

        let PagedLayerArgs {
            layer_idx,
            ctx_count,
            h,
            q_dim,
            kv_dim,
            slot_mapping_gamma,
            block_table_dev,
            stream,
            ..
        } = *args;
        let gpu = ctx.gpu;
        let g = self.gamma as u32;
        let kv_len = ctx_count + g;

        // 3a. input_layernorm — γ rows.
        // dflash.py:125  hidden_states = self.input_layernorm(hidden_states)
        //   stream_buf holds the noise hidden states (the layer's residual);
        //   norm_buf receives the layernorm output fed to all projections.
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

        // id259 per-layer dump: post-input_norm (γ × h).
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.norm_buf,
                layer_idx,
                "input_norm",
                g,
                h,
                stream,
            )?;
        }

        // Phase G: when self.quant == Fp8Weights, swap each dense_gemm
        // for fp8_gemm_n128_row_scaled against the FP8 mirror weight.
        // Per-row f32 scales (built at load time by quantize_bf16_to_fp8)
        // are applied inside the GEMM at write-out. Fp8 mirror None →
        // fall back to BF16 (defensive, shouldn't fire if G.2 ran).
        let use_fp8 = matches!(self.quant, super::DflashQuantization::Fp8Weights);
        let gemm_swap = |w_bf16: &crate::weight_map::DenseWeight,
                         w_fp8: &Option<crate::weight_map::Fp8DenseWeight>,
                         src: spark_runtime::gpu::DevicePtr,
                         dst: spark_runtime::gpu::DevicePtr,
                         n_out: u32,
                         k_in: u32|
         -> Result<()> {
            if use_fp8 && let Some(fp8) = w_fp8 {
                return ops::fp8_gemm_n128_row_scaled(
                    gpu,
                    self.kernels.fp8_gemm_n128_row_scaled,
                    src,
                    fp8,
                    dst,
                    g,
                    n_out,
                    k_in,
                    stream,
                );
            }
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                src,
                w_bf16,
                dst,
                g,
                n_out,
                k_in,
                stream,
            )
        };

        // 3b-q / 3c-q. Q branch: q_proj then q_norm — faithful to dflash.py:68-70.
        // dflash.py:68  q = self.q_proj(hidden_states)
        // dflash.py:70  q = self.q_norm(q.view(..., head_dim)).transpose(1,2)
        //   Atlas: [γ, q_dim] tokens-first; q_norm over [γ*num_q_heads, head_dim].
        gemm_swap(
            &layer.q_proj,
            &layer.q_proj_fp8,
            self.scratch.norm_buf,
            self.scratch.q_buf,
            q_dim,
            h,
        )?;
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.q_buf,
                layer_idx,
                "q_postproj",
                g,
                q_dim,
                stream,
            )?;
        }
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
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.q_buf,
                layer_idx,
                "q_postnorm",
                g,
                q_dim,
                stream,
            )?;
        }

        // 3b-k / 3c-k. K_noise branch: k_proj then k_norm — faithful to
        // dflash.py:72 + 77 (noise portion).
        // dflash.py:72  k_noise = self.k_proj(hidden_states)
        // dflash.py:77  k = self.k_norm(cat([k_ctx, k_noise]))
        //   — per-head normalization is independent per token; applying k_norm
        //     to noise K alone is equivalent to the joint cat application.
        //     ctx K is normed separately in precompute_ctx_kv (same weight).
        gemm_swap(
            &layer.k_proj,
            &layer.k_proj_fp8,
            self.scratch.norm_buf,
            self.scratch.k_buf,
            kv_dim,
            h,
        )?;
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.k_buf,
                layer_idx,
                "k_postproj",
                g,
                kv_dim,
                stream,
            )?;
        }
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
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.k_buf,
                layer_idx,
                "k_postnorm",
                g,
                kv_dim,
                stream,
            )?;
        }

        // 3b-v. V_noise branch: v_proj — faithful to dflash.py:74.
        // dflash.py:74  v_noise = self.v_proj(hidden_states)
        //   No v_norm: dflash.py:78 `v = v.transpose(1,2)` — transpose only.
        gemm_swap(
            &layer.v_proj,
            &layer.v_proj_fp8,
            self.scratch.norm_buf,
            self.scratch.v_buf,
            kv_dim,
            h,
        )?;

        // 3d. yarn RoPE over γ positions [position..position+γ) for Q and K_noise.
        // dflash.py:79-80  cos, sin = position_embeddings
        //                  q, k = apply_rotary_pos_emb(q, k, cos, sin)
        //   z-lab: Q uses cos[..,-q_len:,:] (last γ noise positions);
        //          K (full ctx+noise) uses full cos. Atlas equivalent:
        //          ctx K is RoPE-rotated at its fixed slot positions in
        //          precompute_ctx_kv; noise K is rotated here at
        //          [position..position+γ) — same positions as Q.
        //   position_ids buffer: γ entries = [position, ..., position+γ-1].
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

        // id259 per-layer dump: post-RoPE (the rotated q/k actually fed to attn).
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.q_buf,
                layer_idx,
                "q_postrope",
                g,
                q_dim,
                stream,
            )?;
            self.block_dump_buf(
                ctx,
                self.scratch.k_buf,
                layer_idx,
                "k_postrope",
                g,
                kv_dim,
                stream,
            )?;
            self.block_dump_buf(ctx, self.scratch.v_buf, layer_idx, "v", g, kv_dim, stream)?;
        }

        // 3e. reshape_and_cache — write γ K/V into the layer's paged cache
        // at slots [ctx_count .. ctx_count + γ].
        // dflash.py:75-76  k = cat([k_ctx, k_noise], dim=1)
        //                  v = cat([v_ctx, v_noise], dim=1)
        //   Atlas equivalent: ctx K/V already at slots [0..ctx_count),
        //   noise K/V written here at slots [ctx_count..ctx_count+γ).
        //   Paged attention then reads the whole kv_len=ctx_count+γ range.
        // Slot mapping is provided by the caller (built once per propose).
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
        // Suppress unused-var warnings: kv_len is computed for diagnostics
        // above; the indirect kernel reads it from device memory at entry.
        let _ = kv_len;

        Ok((k_pool, v_pool))
    }

    /// Phase F attention boundary: γ-rows paged attention call. Always
    /// eager (vLLM convention — attention is the natural sync barrier
    /// between captured subgraphs). `k_pool`/`v_pool` are passed in
    /// from `forward_block_layer_pre_attn` so we don't re-lock the KV
    /// cache.
    ///
    /// **ATLAS_DFLASH_CONTIG_ATTN=1**: bypasses the paged-indirect kernel
    /// and runs the contiguous-gather path (`forward_block_layer_attention_contig`)
    /// which matches dflash.py:75-97 op-for-op. Default path is untouched.
    pub(super) fn forward_block_layer_attention(
        &self,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
    ) -> Result<()> {
        use crate::layers::ops;

        // ATLAS_DFLASH_CONTIG_ATTN=1: cat([k_ctx, k_noise]) gather + contiguous
        // non-causal prefill_attention — matches dflash.py:75-97 op-for-op.
        // Default (env unset): paged-indirect kernel, unchanged.
        static USE_CONTIG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *USE_CONTIG
            .get_or_init(|| std::env::var("ATLAS_DFLASH_CONTIG_ATTN").ok().as_deref() == Some("1"))
        {
            return self.forward_block_layer_attention_contig(args, ctx, k_pool, v_pool);
        }

        let PagedLayerArgs {
            block_table_dev,
            stream,
            inv_sqrt_d,
            ..
        } = *args;
        let gpu = ctx.gpu;
        let g = self.gamma as u32;

        // 3f. paged attention — q_len=γ, kv_len=ctx_count+γ, causal=false.
        // dflash.py:84-97  `attn_output, _ = attn_fn(self, q, k, v, ...)`
        //   with `self.is_causal = False` (line 39).  Q(γ) attends over
        //   full K/V(ctx+γ) bidirectionally — identical semantics.
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

        // id259 per-layer dump: post-attention output (pre o_proj), γ × q_dim.
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.attn_out,
                args.layer_idx,
                "attn_out",
                g,
                args.q_dim,
                stream,
            )?;
        }

        Ok(())
    }

    /// ATLAS_DFLASH_CONTIG_ATTN=1 attention path.
    ///
    /// Replicates dflash.py:75-97 op-for-op:
    ///   1. Gather ctx K/V from paged cache slots [0..ctx_count] → CPU.
    ///   2. Cat with γ noise K/V from scratch → one contiguous
    ///      `[ctx_count+γ, num_kv_heads, head_dim]` BF16 buffer.
    ///   3. Pad Q to `[ctx_count+γ, num_q_heads, head_dim]` with zeros for
    ///      the ctx rows (their outputs are discarded).
    ///   4. Run `ops::prefill_attention(causal=false, seq_len=ctx_count+γ)`.
    ///   5. Extract γ noise output rows → `attn_out[0..γ]` for post_attn.
    ///
    /// No kernel changes. CPU gather + sync H2D/D2H (diagnostic path —
    /// performance is not a concern; correctness parity with z-lab is).
    fn forward_block_layer_attention_contig(
        &self,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
    ) -> Result<()> {
        use crate::layers::ops;

        let PagedLayerArgs {
            ctx_count,
            q_dim,
            kv_dim,
            block_table_dev,
            stream,
            inv_sqrt_d,
            ..
        } = *args;
        let gpu = ctx.gpu;
        let g = self.gamma as u32;
        let ctx_us = ctx_count as usize;
        let g_us = g as usize;
        let seq_len = ctx_count + g;
        const BF16: usize = 2;
        const BLOCK_SIZE: usize = 16; // paged cache block_size (matches reshape_and_cache_flash)
        let kv_slot = kv_dim as usize * BF16; // bytes per KV token row
        let q_slot = q_dim as usize * BF16; // bytes per Q/output token row

        // Guard: scratch buffers are sized n_attn = ctx_window + γ.
        // ctx_count must not exceed ctx_window or we'd overflow k_buf/v_buf/q_buf.
        anyhow::ensure!(
            ctx_us <= self.ctx_window,
            "CONTIG_ATTN: ctx_count({ctx_us}) > ctx_window({}); \
             scratch buffers sized for {} rows — reduce ctx or raise ATLAS_DFLASH_CTX_WINDOW",
            self.ctx_window,
            self.ctx_window + g_us,
        );

        // Sync: ensure all pre_attn kernel writes are retired before D2H.
        gpu.synchronize(stream)?;

        // ── Fast path: ctx=0 — K/V/Q already in γ-row scratch, no gather needed ──
        // dflash.py:75  k = cat([k_ctx, k_noise]) ≡ k_noise when ctx_count=0.
        if ctx_count == 0 {
            ops::prefill_attention(
                gpu,
                self.kernels.prefill_attn,
                self.scratch.q_buf,
                self.scratch.k_buf,
                self.scratch.v_buf,
                self.scratch.attn_out,
                g,
                1,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                inv_sqrt_d,
                false, // is_causal=false  (dflash.py:39)
                0,
                stream,
            )?;
            if args.block_dump {
                self.block_dump_buf(
                    ctx,
                    self.scratch.attn_out,
                    args.layer_idx,
                    "attn_out",
                    g,
                    q_dim,
                    stream,
                )?;
            }
            return Ok(());
        }

        // ── D2H: γ noise K, V, Q from scratch (pre_attn output) ──────────────
        let mut noise_k = vec![0u8; g_us * kv_slot];
        let mut noise_v = vec![0u8; g_us * kv_slot];
        let mut noise_q = vec![0u8; g_us * q_slot];
        gpu.copy_d2h(self.scratch.k_buf, &mut noise_k)?;
        gpu.copy_d2h(self.scratch.v_buf, &mut noise_v)?;
        gpu.copy_d2h(self.scratch.q_buf, &mut noise_q)?;

        // ── D2H: block table (ceil(ctx_count / BLOCK_SIZE) u32 entries) ───────
        let num_ctx_blocks = ctx_us.div_ceil(BLOCK_SIZE);
        let mut bt_raw = vec![0u8; num_ctx_blocks * 4];
        gpu.copy_d2h(block_table_dev, &mut bt_raw)?;
        let block_table: Vec<u32> = bt_raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // ── Gather ctx K, V from paged cache (one D2H per physical block) ─────
        // Paged layout (from reshape_and_cache_flash):
        //   k_cache: [num_phys_blocks, BLOCK_SIZE, num_kv_heads, head_dim] BF16
        //   = [num_phys_blocks, BLOCK_SIZE, kv_dim] in flat BF16 bytes.
        // For logical slot s: phys_block = block_table[s / BLOCK_SIZE],
        //   offset in pool = phys_block * BLOCK_SIZE * kv_slot + (s % BLOCK_SIZE) * kv_slot.
        let phys_block_bytes = BLOCK_SIZE * kv_slot;
        let mut ctx_k = vec![0u8; ctx_us * kv_slot];
        let mut ctx_v = vec![0u8; ctx_us * kv_slot];
        for b in 0..num_ctx_blocks {
            let phys = block_table[b] as usize;
            let pool_off = phys * phys_block_bytes;
            let mut blk_k = vec![0u8; phys_block_bytes];
            let mut blk_v = vec![0u8; phys_block_bytes];
            gpu.copy_d2h(k_pool.offset(pool_off), &mut blk_k)?;
            gpu.copy_d2h(v_pool.offset(pool_off), &mut blk_v)?;
            let slot_start = b * BLOCK_SIZE;
            let slot_end = (slot_start + BLOCK_SIZE).min(ctx_us);
            for s in slot_start..slot_end {
                let src = (s - slot_start) * kv_slot;
                let dst = s * kv_slot;
                ctx_k[dst..dst + kv_slot].copy_from_slice(&blk_k[src..src + kv_slot]);
                ctx_v[dst..dst + kv_slot].copy_from_slice(&blk_v[src..src + kv_slot]);
            }
        }

        // ── Build contiguous K = [ctx_K, noise_K]; H2D → k_buf ────────────────
        // dflash.py:75  k = cat([k_ctx, k_noise], dim=1)
        // k_buf sized n_attn*kv_slot = (ctx_window+γ)*kv_slot ≥ (ctx+γ)*kv_slot.
        let total_kv = (ctx_us + g_us) * kv_slot;
        let mut k_contig = vec![0u8; total_kv];
        let mut v_contig = vec![0u8; total_kv];
        k_contig[..ctx_us * kv_slot].copy_from_slice(&ctx_k);
        k_contig[ctx_us * kv_slot..].copy_from_slice(&noise_k);
        v_contig[..ctx_us * kv_slot].copy_from_slice(&ctx_v);
        v_contig[ctx_us * kv_slot..].copy_from_slice(&noise_v);
        gpu.copy_h2d(&k_contig, self.scratch.k_buf)?;
        gpu.copy_h2d(&v_contig, self.scratch.v_buf)?;

        // ── Build padded Q = [zeros(ctx_count), noise_Q]; H2D → q_buf ─────────
        // z-lab Q rows [ctx_count..ctx_count+γ] are the noise queries.
        // Rows [0..ctx_count] are zero-padded; their attention output is discarded.
        let total_q = (ctx_us + g_us) * q_slot;
        let mut q_contig = vec![0u8; total_q]; // zero-initialized
        q_contig[ctx_us * q_slot..].copy_from_slice(&noise_q);
        gpu.copy_h2d(&q_contig, self.scratch.q_buf)?;

        // ── Contiguous non-causal attention ────────────────────────────────────
        // dflash.py:84-97  eager_attention_forward(q, k, v, is_causal=False)
        //   q_buf:    [ctx+γ, q_dim]   (padded — ctx rows zero)
        //   k_buf:    [ctx+γ, kv_dim]  (gathered ctx + noise K)
        //   v_buf:    [ctx+γ, kv_dim]  (gathered ctx + noise V)
        //   attn_out: [ctx+γ, q_dim]   kernel writes all rows; we keep [ctx..ctx+γ]
        ops::prefill_attention(
            gpu,
            self.kernels.prefill_attn,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.v_buf,
            self.scratch.attn_out,
            seq_len,
            1,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            inv_sqrt_d,
            false, // is_causal=false  (dflash.py:39)
            0,
            stream,
        )?;

        // ── Extract γ noise query outputs → attn_out[0..γ] ────────────────────
        // attn_out[ctx..ctx+γ] are the valid noise-query outputs; shift to [0..γ]
        // so post_attn (o_proj etc.) can read from offset 0 as usual.
        gpu.synchronize(stream)?;
        let mut noise_attn = vec![0u8; g_us * q_slot];
        gpu.copy_d2h(
            self.scratch.attn_out.offset(ctx_us * q_slot),
            &mut noise_attn,
        )?;
        gpu.copy_h2d(&noise_attn, self.scratch.attn_out)?;

        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.attn_out,
                args.layer_idx,
                "attn_out",
                g,
                q_dim,
                stream,
            )?;
        }

        Ok(())
    }

    /// Phase F post-attention half: o_proj → residual_add →
    /// post_attention_layernorm → gate/up_proj → silu_mul → down_proj
    /// → residual_add (steps 3g–3k).
    ///
    /// Op order is byte-faithful to dflash.py decoder-layer lines 138-142:
    ///   99 o_proj → 138 residual+attn → 140 post_attn_norm →
    ///   141 mlp(gate+up+silu_mul+down) → 142 residual+mlp.
    ///
    /// **Capture eligibility**: pure compute over stable scratch
    /// pointers + layer weights. No D2H, no sync, no env-var
    /// branches. Safe to capture unconditionally when the upstream
    /// `graph_eligible` gate is set.
    pub(super) fn forward_block_layer_post_attn(
        &self,
        layer: &DflashLayer,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
    ) -> Result<()> {
        use crate::layers::ops;

        let PagedLayerArgs {
            h,
            q_dim,
            inter,
            stream,
            ..
        } = *args;
        let gpu = ctx.gpu;
        let g = self.gamma as u32;

        // Phase G — same swap helper as pre_attn (q/k/v). Single call
        // site per logical GEMM; the row-scaled FP8 GEMM kernel applies
        // the per-row scale internally at write-out.
        let use_fp8 = matches!(self.quant, super::DflashQuantization::Fp8Weights);
        let gemm_swap = |w_bf16: &crate::weight_map::DenseWeight,
                         w_fp8: &Option<crate::weight_map::Fp8DenseWeight>,
                         src: spark_runtime::gpu::DevicePtr,
                         dst: spark_runtime::gpu::DevicePtr,
                         n_out: u32,
                         k_in: u32|
         -> Result<()> {
            if use_fp8 && let Some(fp8) = w_fp8 {
                return ops::fp8_gemm_n128_row_scaled(
                    gpu,
                    self.kernels.fp8_gemm_n128_row_scaled,
                    src,
                    fp8,
                    dst,
                    g,
                    n_out,
                    k_in,
                    stream,
                );
            }
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                src,
                w_bf16,
                dst,
                g,
                n_out,
                k_in,
                stream,
            )
        };

        // 3g. o_proj — γ rows, [q_dim → h].
        // dflash.py:98-99  attn_output = attn_output.reshape(bsz, q_len, -1)
        //                  attn_output = self.o_proj(attn_output)
        //   stream_acc ← o_proj(attn_out); stream_buf still holds the
        //   PRE-layernorm residual (saved implicitly — stream_buf was not
        //   modified since 3a wrote norm_buf from it).
        gemm_swap(
            &layer.o_proj,
            &layer.o_proj_fp8,
            self.scratch.attn_out,
            self.scratch.stream_acc,
            h,
            q_dim,
        )?;

        // 3h. First residual add: hidden = residual + attn_output.
        // dflash.py:138  hidden_states = residual + hidden_states
        //   stream_buf (residual = pre-3a noise hidden states)
        //       += stream_acc (attn_output = o_proj output).
        //   After: stream_buf = residual + attn_output.
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            self.scratch.stream_acc,
            g * h,
            stream,
        )?;

        // 3i. post_attention_layernorm — input is stream_buf after 3h.
        // dflash.py:139-140  residual = hidden_states
        //                    hidden_states = self.post_attention_layernorm(hidden_states)
        //   stream_buf (= residual + attn_output) serves as both the new
        //   residual (line 139) and the input to post_attention_layernorm
        //   (line 140). norm_buf receives the layernorm output fed to MLP.
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

        // 3j. MLP: gate_proj + up_proj + silu_mul + down_proj — γ rows.
        // dflash.py:141  hidden_states = self.mlp(hidden_states)
        //   Qwen3MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).
        //   gate_proj and up_proj both read from norm_buf (post-3i).
        //   silu_mul: silu(mlp_intermediate) * mlp_up → mlp_intermediate.
        //   down_proj: mlp_intermediate → stream_acc.
        gemm_swap(
            &layer.gate_proj,
            &layer.gate_proj_fp8,
            self.scratch.norm_buf,
            self.scratch.mlp_intermediate,
            inter,
            h,
        )?;
        gemm_swap(
            &layer.up_proj,
            &layer.up_proj_fp8,
            self.scratch.norm_buf,
            self.scratch.mlp_up,
            inter,
            h,
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
        gemm_swap(
            &layer.down_proj,
            &layer.down_proj_fp8,
            self.scratch.mlp_intermediate,
            self.scratch.stream_acc,
            h,
            inter,
        )?;

        // 3k. Second residual add: hidden = (residual + attn) + mlp_output.
        // dflash.py:142  hidden_states = residual + hidden_states
        //   stream_buf (= residual + attn_output, the line-139 residual)
        //       += stream_acc (mlp_output = down_proj output).
        //   After: stream_buf = residual + attn_output + mlp_output — the
        //   final hidden state passed to the next layer or to run_tail.
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            self.scratch.stream_acc,
            g * h,
            stream,
        )?;

        // id259 per-layer dump: final layer output (post-MLP residual), γ × h.
        // This is the hidden_states the NEXT layer consumes — the per-layer
        // chain the harness walks to find the first diverging op.
        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.stream_buf,
                args.layer_idx,
                "layer_out",
                g,
                h,
                stream,
            )?;
        }

        Ok(())
    }
}
