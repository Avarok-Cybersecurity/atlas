// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash γ-block forward (Phase 2 kernel chain). Split out of
//! `dflash_head.rs` for file-size budget — body still exceeds the
//! 500 LoC target because the per-step kernel chain (fc → pos →
//! 8 drafter layers → final norm/lm_head/argmax → D2H) shares
//! many locals with no clean extraction boundary.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use crate::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    /// `option_b`: when `Some((block_table_dev, ctx_count))`, run the
    /// Phase 2 γ-only paged-attention path. ctx K/V is precomputed into
    /// the drafter's paged cache from `ctx_buffer` at slots
    /// `[0..ctx_count)`, γ K/V is written by the layer body at slots
    /// `[ctx_count..ctx_count+γ)`, attention reads all of
    /// `kv_len = ctx_count + γ` from the cache.
    pub(super) fn forward_block(
        &self,
        last_token: u32,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
        ctx_buffer: Option<(DevicePtr, usize)>,
        option_b: Option<(DevicePtr, u32)>,
    ) -> Result<Vec<u32>> {
        use crate::layers::ops;

        let g = self.gamma as u32;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let bf16 = 2usize;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let gpu = ctx.gpu;

        // Determine effective ctx_len: capped by the configured ctx_window
        // and the accumulator's actual fill. Use the LAST `eff_ctx` ctx
        // positions (most recent) — drafter trained on locally recent
        // context, distant history adds noise to attention.
        // ATLAS_DFLASH_DEBUG_CTX_OFF=1 disables ctx entirely (eff_ctx=0)
        // for A/B testing whether the drafter actually responds to ctx.
        let force_no_ctx = std::env::var("ATLAS_DFLASH_DEBUG_CTX_OFF").ok().as_deref() == Some("1");
        let force_ctx_used: Option<usize> = std::env::var("ATLAS_DFLASH_DEBUG_CTX_USED")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let (ctx_base_ptr, ctx_total, eff_ctx) = match ctx_buffer {
            Some(_) if force_no_ctx => (None, 0, 0),
            Some((p, n)) => {
                let eff = match force_ctx_used {
                    Some(forced) => forced.min(n).min(self.ctx_window),
                    None => n.min(self.ctx_window),
                };
                (Some(p), n, eff)
            }
            None => (None, 0, 0),
        };

        // Phase 2 Option B: ctx K/V already lives in the paged cache
        // (precompute_ctx_kv ran in propose.rs before forward_block).
        // Force eff_ctx=0 to disable the in-layer ctx K/V recomputation
        // and the ctx-side of the stream_buf / position_ids / fc_proj
        // paths. The layer body runs over γ rows only and reads ctx
        // K/V from the cache via the paged-attention dispatcher.
        let (option_b_block_table, option_b_ctx_count) = match option_b {
            Some((bt, cc)) => (Some(bt), cc),
            None => (None, 0),
        };
        let option_b_on = option_b_block_table.is_some();
        let eff_ctx = if option_b_on { 0 } else { eff_ctx };
        let _ = ctx_base_ptr; // Option B doesn't read ctx from this path
        let n_attn = (eff_ctx + self.gamma) as u32;
        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * bf16;

        // Debug dump gated by env var: prints first 10 BF16 floats of key
        // intermediates so a Python reference run on the same checkpoint
        // can be compared element-wise. Use ATLAS_DFLASH_DEBUG_DUMP=1.
        let debug_dump = std::env::var("ATLAS_DFLASH_DEBUG_DUMP").ok().as_deref() == Some("1");
        let dump_bf16 = |label: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        // ── Phase 2 Option B precompute (stage 3 — dump-only) ──────
        // When ATLAS_DFLASH_PRECOMPUTE=1, run the new precompute_ctx_kv
        // path in parallel to (not replacing) the existing fc gemv loop
        // below. The precompute writes BF16 dump files to /tmp for the
        // pyref diff harness; it does NOT yet feed the layer body's
        // attention. Stage 4 will swap the layer body to read from the
        // paged cache and remove the per-row gemv path entirely.
        //
        // Requires ATLAS_DFLASH_PRECOMPUTE_DUMP=1 to actually emit
        // dump files; otherwise the kernel chain runs and discards
        // intermediates (useful for perf-only A/B).
        if std::env::var("ATLAS_DFLASH_PRECOMPUTE").ok().as_deref() == Some("1") {
            if let Some(base) = ctx_base_ptr {
                if eff_ctx > 0 {
                    let start_slot = ctx_total.saturating_sub(eff_ctx);
                    let abs_start = position.saturating_sub(eff_ctx);
                    // Dump-only diagnostic path: reconstruct the legacy
                    // sliding positions (abs_start + i) as a slice. The
                    // production path (propose.rs) uses per-slot fixed
                    // positions from ctx_positions instead.
                    let slot_positions: Vec<i32> =
                        (0..eff_ctx).map(|i| (abs_start + i) as i32).collect();
                    // slot_mapping unused in dump-only mode; pass the
                    // scratch buffer so reshape_and_cache has a valid
                    // pointer if ATLAS_DFLASH_PRECOMPUTE_COMMIT=1.
                    self.precompute_ctx_kv(
                        base,
                        start_slot,
                        eff_ctx,
                        &slot_positions,
                        self.scratch.slot_mapping_dev,
                        ctx,
                        stream,
                    )?;
                }
            }
        }

        // ── Step 0: fc projection of captured target hiddens ──
        // For each of the `eff_ctx` most-recent ctx positions, run a GEMV
        // through `self.fc` (input: 10240 BF16 → output: 2048 BF16) and
        // then per-row RMSNorm through `self.hidden_norm`. Results land
        // contiguously in `scratch.fc_proj` shaped `[eff_ctx, hidden]`.
        if let Some(base) = ctx_base_ptr {
            // Walk the LAST `eff_ctx` slots of the accumulator.
            let start_slot = ctx_total.saturating_sub(eff_ctx);
            // ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1 overwrites the captured
            // target_hidden_stack with a deterministic test pattern so a
            // PyTorch reference run on the same input produces directly
            // comparable intermediates. Pattern: row i, col j contains
            // `0.01 * (i+1) * (j+1) / target_hidden` BF16. Mirrors
            // `dflash_pytorch_reference.py:make_input_target_hidden_stack`.
            let force_pattern = std::env::var("ATLAS_DFLASH_DEBUG_FORCE_PATTERN")
                .ok()
                .as_deref()
                == Some("1");
            if force_pattern && eff_ctx > 0 {
                let n_rows = self.target_layer_ids.len();
                let n_cols = self.target_hidden_size;
                let mut bytes = Vec::with_capacity(n_rows * n_cols * 2);
                for i in 0..n_rows {
                    for j in 0..n_cols {
                        let v = 0.01_f32 * ((i + 1) as f32) * ((j + 1) as f32) / (n_cols as f32);
                        // f32 → bf16 (truncate-to-zero of low 16 bits).
                        let bits = v.to_bits();
                        let bf16_bits = (bits >> 16) as u16;
                        bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                    }
                }
                gpu.copy_h2d(&bytes, base.offset(start_slot * ctx_slot_bytes))?;
            }
            // Dump the FIRST ctx slot's input target_hidden_stack (first 10 floats).
            if eff_ctx > 0 {
                dump_bf16(
                    "step0.input.target_hidden_stack[0]",
                    base.offset(start_slot * ctx_slot_bytes),
                    10,
                )?;
            }
            // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: write the full 10240-element
            // target_hidden_stack (one ctx slot) to /tmp/atlas_target_hidden.bin
            // so a Python reference can run dflash.py forward on the same
            // input and compare predicted draft tokens vs Atlas drafts.
            // Also dumps last_token + drafter outputs separately for the
            // bisect script. ONE-SHOT: writes only the first propose() call.
            static FULL_DUMP_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if eff_ctx > 0
                && !FULL_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
                && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                    .ok()
                    .as_deref()
                    == Some("1")
            {
                // Dump ALL eff_ctx slots — needed to reproduce the
                // multi-token ctx in PyTorch reference. Layout:
                // contiguous BF16, eff_ctx slots × 5 layers × 2048 dims.
                let n_bytes = eff_ctx * ctx_slot_bytes;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(base.offset(start_slot * ctx_slot_bytes), &mut buf)?;
                if let Err(e) = std::fs::write("/tmp/atlas_target_hidden.bin", &buf) {
                    tracing::warn!("DFLASH DUMP_FULL: target_hidden write failed: {e}");
                } else {
                    tracing::info!(
                        "DFLASH DUMP_FULL: wrote {} bytes ({} ctx slots × {} BF16 elements) to /tmp/atlas_target_hidden.bin (last_token={}, position={}, eff_ctx={})",
                        n_bytes,
                        eff_ctx,
                        ctx_slot_bytes / 2,
                        last_token,
                        position,
                        eff_ctx,
                    );
                }
                FULL_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);

                // Write companion meta JSON for the pyref diff harness.
                // Shapes/strides Atlas knows but the Python side can't
                // infer from the .bin alone. Written once alongside the
                // target_hidden dump so harness runs read a consistent
                // snapshot.
                let meta = format!(
                    "{{\n  \"last_token\": {},\n  \"position\": {},\n  \"eff_ctx\": {},\n  \"n_layers_captured\": {},\n  \"target_hidden_size\": {},\n  \"gamma\": {},\n  \"hidden_size\": {},\n  \"num_kv_heads\": {},\n  \"head_dim\": {},\n  \"num_drafter_layers\": {},\n  \"rope_theta\": {}\n}}\n",
                    last_token,
                    position,
                    eff_ctx,
                    self.target_layer_ids.len(),
                    self.target_hidden_size,
                    self.gamma,
                    self.hidden_size,
                    self.num_kv_heads,
                    self.head_dim,
                    self.num_layers,
                    self.rope_theta,
                );
                if let Err(e) = std::fs::write("/tmp/atlas_dflash_meta.json", &meta) {
                    tracing::warn!("DFLASH DUMP_FULL: meta JSON write failed: {e}");
                } else {
                    tracing::info!(
                        "DFLASH DUMP_FULL: wrote /tmp/atlas_dflash_meta.json companion to target_hidden"
                    );
                }
            }
            for i in 0..eff_ctx {
                let src_slot = base.offset((start_slot + i) * ctx_slot_bytes);
                let dst_slot = self.scratch.fc_proj.offset(i * self.hidden_size * bf16);
                ops::dense_gemv(
                    gpu,
                    self.kernels.dense_gemv,
                    src_slot,
                    &self.fc,
                    dst_slot,
                    h,
                    target_hidden_dim as u32,
                    stream,
                )?;
            }
            if eff_ctx > 0 {
                dump_bf16("step0.fc_proj.pre_norm[0]", self.scratch.fc_proj, 10)?;
                ops::rms_norm(
                    gpu,
                    self.kernels.rms_norm,
                    self.scratch.fc_proj,
                    &self.hidden_norm,
                    self.scratch.fc_proj,
                    eff_ctx as u32,
                    h,
                    self.rms_norm_eps,
                    stream,
                )?;
                dump_bf16(
                    "step0.fc_proj.post_hidden_norm[0]",
                    self.scratch.fc_proj,
                    10,
                )?;
            }
        }

        // ── Step 1: build position ids ──
        // Layout: [ctx_pos_0, ..., ctx_pos_{eff_ctx-1}, seq_pos, ..., seq_pos+γ-1].
        // ctx_pos_i = position - eff_ctx + i — the absolute target indices
        // of the captured positions in chronological order.
        let ctx_start = position.saturating_sub(eff_ctx);
        let pos_host: Vec<i32> = (0..eff_ctx)
            .map(|i| (ctx_start + i) as i32)
            .chain((0..self.gamma).map(|i| (position + i) as i32))
            .collect();
        let pos_bytes: Vec<u8> = pos_host.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&pos_bytes, self.scratch.position_ids)?;
        if debug_dump {
            tracing::info!(
                "DFLASH DUMP positions: eff_ctx={} ctx_total={} position={} pos_ids[0..min(8,n_attn)]={:?}",
                eff_ctx,
                ctx_total,
                position,
                &pos_host[..pos_host.len().min(8)]
            );
        }

        // ── Step 2: stream_buf layout ──
        // First eff_ctx rows: zero (Q-side ctx is zero; K/V-side gets
        // overwritten in step 3b' below). Next γ rows: embed of
        // [last_token, mask, mask, ..., mask].
        // Total stream_buf width = n_attn rows.
        if eff_ctx > 0 {
            gpu.memset(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
            )?;
        }
        let token_ids_host: Vec<i32> = std::iter::repeat_n(0i32, eff_ctx)
            .chain(std::iter::once(last_token as i32))
            .chain(std::iter::repeat_n(
                self.mask_token_id as i32,
                self.gamma - 1,
            ))
            .collect();
        if debug_dump {
            tracing::info!(
                "DFLASH DUMP token_ids_host: last_token={} mask={} eff_ctx={} ids[0..8]={:?}",
                last_token,
                self.mask_token_id,
                eff_ctx,
                &token_ids_host[..token_ids_host.len().min(8)],
            );
        }
        let tid_bytes: Vec<u8> = token_ids_host
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        gpu.copy_h2d(&tid_bytes, self.scratch.draft_tokens_dev)?;
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            self.scratch.draft_tokens_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            n_attn,
            h,
            stream,
        )?;
        // Re-zero ctx slots (batched_embed wrote token-0 embedding to them).
        if eff_ctx > 0 {
            gpu.memset(
                self.scratch.stream_buf,
                0,
                eff_ctx * self.hidden_size * bf16,
            )?;
        }
        // ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1: overwrite noise rows
        // [eff_ctx..n_attn) with a deterministic pattern matching the
        // PyTorch reference. Lets us compare layer-0 q/k/v post-projection
        // when both Atlas and PyTorch see identical input.
        let force_noise_pattern = std::env::var("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN")
            .ok()
            .as_deref()
            == Some("1");
        if force_noise_pattern {
            let mut bytes = Vec::with_capacity(self.gamma * self.hidden_size * 2);
            for t in 0..self.gamma {
                for j in 0..self.hidden_size {
                    let v =
                        0.001_f32 * ((t + 1) as f32) * ((j + 1) as f32) / (self.hidden_size as f32);
                    let bf16_bits = (v.to_bits() >> 16) as u16;
                    bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                }
            }
            gpu.copy_h2d(
                &bytes,
                self.scratch
                    .stream_buf
                    .offset(eff_ctx * self.hidden_size * bf16),
            )?;
        }

        // ── Step 3: 8 drafter layers ──
        //
        // All compute runs on `n_attn = eff_ctx + γ` rows. Slots [0..eff_ctx]
        // are CTX (Q-zero / KV from fc_proj projection) and slots
        // [eff_ctx..n_attn] are NOISE (full Q/K/V from embeddings).
        // Per-layer flow follows `dflash.py:Qwen3DFlashDecoderLayer.forward`.
        // Body extracted to `forward_block_layer.rs` for the 500-LoC budget.
        //
        // Option B: layer body runs over γ rows only, reads ctx K/V from
        // the paged cache. Slot mapping for the γ K/V writes is built
        // once and reused across all drafter layers.
        let slot_mapping_gamma_opt = if option_b_on {
            let bt = option_b_block_table.unwrap();
            // Build γ slot indices starting at logical position ctx_count.
            ops::fill_slots_from_block_table(
                gpu,
                self.kernels.fill_slots,
                self.scratch.slot_mapping_dev,
                bt,
                option_b_ctx_count,
                self.gamma as u32,
                16,
                stream,
            )?;
            // Phase 5 (CUDA graph) pre-graph write: stash the per-propose
            // dynamic `[kv_len, q_offset]` pair into the indirect-args
            // buffer. The graph-captured paged-attention launch reads from
            // this pointer at kernel entry, so a single graph instance can
            // be replayed across propose calls with new values written here.
            // Phase C uses it eagerly (no graph yet) to gate correctness.
            let kv_len = option_b_ctx_count + self.gamma as u32;
            let q_offset = option_b_ctx_count;
            let indirect_bytes: [u8; 8] = {
                let mut b = [0u8; 8];
                b[0..4].copy_from_slice(&kv_len.to_ne_bytes());
                b[4..8].copy_from_slice(&q_offset.to_ne_bytes());
                b
            };
            gpu.copy_h2d(&indirect_bytes, self.scratch.option_b_indirect_args_dev)?;
            Some(self.scratch.slot_mapping_dev)
        } else {
            None
        };

        // ── Phase D: CUDA graph capture/replay wraps the layer loop +
        // post-norm + lm_head + argmax (all the per-propose compute). The
        // pre-graph H2D writes above stash dynamic values into stable
        // device pointers; the captured graph reads from those pointers
        // every replay, so a single graph instance is reused across all
        // propose calls.
        //
        // Eligibility: option_b path only (legacy non-paged path isn't
        // graph-ready), suppress_graphs not set, none of the debug dumps
        // enabled (those inject D2H/sync into the region and would taint
        // the graph). Default warm-up N=2 (override
        // `ATLAS_DFLASH_PROPOSE_WARMUP_N`) so PTX→SASS JIT, GB10 clock
        // ramp, and L2 warming all happen eagerly before capture freezes
        // a steady-state SASS pick.
        let graph_eligible = option_b_on
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !debug_dump
            && std::env::var("ATLAS_DFLASH_PROPOSE_NO_GRAPH").is_err()
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL").is_err()
            && std::env::var("ATLAS_DFLASH_OPTION_B_DIAG").is_err()
            && std::env::var("ATLAS_DFLASH_PRECOMPUTE_DUMP").is_err()
            && std::env::var("ATLAS_DFLASH_VERIFY_TRACE").is_err()
            && std::env::var("ATLAS_DFLASH_LOG_DRAFTS").is_err()
            && std::env::var("ATLAS_DFLASH_DEBUG_FORCE_PATTERN").is_err()
            && std::env::var("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN").is_err()
            && std::env::var("ATLAS_DFLASH_DEBUG_CTX_OFF").is_err()
            && std::env::var("ATLAS_DFLASH_DEBUG_CTX_USED").is_err();

        let warmup_target: usize = std::env::var("ATLAS_DFLASH_PROPOSE_WARMUP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        // Helper closures: run each piecewise subgraph eagerly. Phase F.2
        // splits the old monolithic captured region into per-layer halves
        // (pre_attn + post_attn) plus a tail (final norm + lm_head +
        // argmax). The attention call between pre and post stays eager.
        let bf16_local = bf16;
        let inv_sqrt_d_local = inv_sqrt_d;
        let h_local = h;
        let n_attn_local = n_attn;
        let q_dim_local = q_dim;
        let kv_dim_local = kv_dim;
        let inter_local = inter;
        let eff_ctx_local = eff_ctx;
        let noise_byte_offset_local = eff_ctx * self.hidden_size * bf16;
        let stream_noise_local = self.scratch.stream_buf.offset(noise_byte_offset_local);
        let norm_noise_local = self.scratch.norm_buf.offset(noise_byte_offset_local);

        // Build PagedLayerArgs once per layer — same args for pre_attn,
        // attention, and post_attn (the kernel only reads what it needs).
        let make_paged_args =
            |layer_idx: usize| -> Option<super::forward_block_layer_paged::PagedLayerArgs> {
                if !option_b_on {
                    return None;
                }
                let bt = option_b_block_table?;
                let slot_mapping = slot_mapping_gamma_opt?;
                Some(super::forward_block_layer_paged::PagedLayerArgs {
                    layer_idx,
                    ctx_count: option_b_ctx_count,
                    h: h_local,
                    q_dim: q_dim_local,
                    kv_dim: kv_dim_local,
                    inter: inter_local,
                    inv_sqrt_d: inv_sqrt_d_local,
                    slot_mapping_gamma: slot_mapping,
                    block_table_dev: bt,
                    stream,
                })
            };

        // Legacy (non-paged, n_attn rows) per-layer body — kept whole
        // because the legacy path is debug-only and not graph-capture
        // ready. Runs all of 3a–3k inline.
        let run_legacy_layer = |layer_idx: usize, layer: &super::DflashLayer| -> Result<()> {
            let args = super::forward_block_layer::LayerArgs {
                layer_idx,
                n_attn: n_attn_local,
                eff_ctx: eff_ctx_local,
                h: h_local,
                q_dim: q_dim_local,
                kv_dim: kv_dim_local,
                inter: inter_local,
                bf16: bf16_local,
                inv_sqrt_d: inv_sqrt_d_local,
                stream,
            };
            self.forward_block_layer(layer, &args, ctx, debug_dump)
        };

        // Tail: final norm + lm_head + argmax over γ rows. Captured as
        // the last piecewise subgraph (slot index = num_layers * 2).
        let run_tail = || -> Result<()> {
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                stream_noise_local,
                &self.norm,
                norm_noise_local,
                self.gamma as u32,
                h_local,
                self.rms_norm_eps,
                stream,
            )?;
            // Phase G: lm_head GEMM. Largest GEMM in the drafter
            // (γ × vocab=248320). FP8 path uses the small-M kernel
            // fp8_gemm_t_row_scaled_m16 (M_TILE=16, 1 warp/CTA) against
            // the FP8 mirror of the shared lm_head weight. The earlier
            // 0%-accept bug was a half-loaded smem_A K-tile in that
            // kernel (fixed: 2-round A-load covers all 32 K-cols).
            // BF16 path (default, or Fp8 mirror missing) unchanged.
            let lm_head_fp8 = matches!(self.quant, super::DflashQuantization::Fp8Weights);
            if lm_head_fp8 {
                if let Some(fp8) = self.lm_head_shared_fp8.as_ref() {
                    ops::fp8_gemm_n128_row_scaled_m16(
                        gpu,
                        self.kernels.fp8_gemm_n128_row_scaled_m16,
                        norm_noise_local,
                        fp8,
                        self.scratch.logits,
                        self.gamma as u32,
                        self.vocab_size as u32,
                        h_local,
                        stream,
                    )?;
                } else {
                    ops::dense_gemm(
                        gpu,
                        self.kernels.dense_gemm,
                        norm_noise_local,
                        &crate::weight_map::DenseWeight {
                            weight: self.lm_head_shared,
                        },
                        self.scratch.logits,
                        self.gamma as u32,
                        self.vocab_size as u32,
                        h_local,
                        stream,
                    )?;
                }
            } else {
                ops::dense_gemm(
                    gpu,
                    self.kernels.dense_gemm,
                    norm_noise_local,
                    &crate::weight_map::DenseWeight {
                        weight: self.lm_head_shared,
                    },
                    self.scratch.logits,
                    self.gamma as u32,
                    self.vocab_size as u32,
                    h_local,
                    stream,
                )?;
            }
            for i in 0..self.gamma {
                let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16_local);
                let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
                ops::argmax_bf16(
                    gpu,
                    self.kernels.argmax,
                    logits_row,
                    token_slot,
                    self.vocab_size as u32,
                    stream,
                )?;
            }
            Ok(())
        };

        // Run all subgraphs eagerly, no capture — used for warm-up and
        // for the non-graph-eligible path.
        let run_all_eager = || -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if option_b_on {
                    let args = make_paged_args(layer_idx).expect("option_b args available");
                    let (k_pool, v_pool) = self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                    self.forward_block_layer_attention(&args, ctx, k_pool, v_pool)?;
                    self.forward_block_layer_post_attn(layer, &args, ctx)?;
                } else {
                    run_legacy_layer(layer_idx, layer)?;
                }
            }
            run_tail()
        };

        // Phase F.2: piecewise capture/replay path. Only enabled for
        // option_b (paged) — legacy path stays single-shot eager since
        // it's not graph-ready and exists only for ablation.
        if graph_eligible && option_b_on {
            // Subgraph slot layout: [pre_0, post_0, ..., pre_{N-1}, post_{N-1}, tail].
            // 2 × num_layers + 1 slots total.
            let num_layers = self.layers.len();
            let total_slots = num_layers * 2 + 1;
            let tail_slot = num_layers * 2;

            let mut g = self.propose_graphs.lock();
            let cached_ready = matches!(*g, Some(ref v) if v.len() == total_slots);

            if cached_ready {
                // Hot replay path: launch each cached subgraph in order,
                // running attention eagerly between pre and post.
                let graphs = g.as_ref().unwrap();
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let args = make_paged_args(layer_idx).expect("option_b args available");

                    let pre_handle = graphs[layer_idx * 2];
                    if pre_handle.0 != 0 {
                        gpu.launch_graph(pre_handle, stream)?;
                    } else {
                        // Empty-capture sentinel: this slot fell back to
                        // eager at capture time. Replay eager forever.
                        self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                    }

                    // Attention is always eager — but we need k_pool/v_pool
                    // for the call. Re-lock the cache here (the captured
                    // pre_attn already holds the pointers internally; this
                    // is just for the attention boundary).
                    let (k_pool, v_pool) = {
                        let cache = self.kv_cache.lock();
                        (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
                    };
                    self.forward_block_layer_attention(&args, ctx, k_pool, v_pool)?;

                    let post_handle = graphs[layer_idx * 2 + 1];
                    if post_handle.0 != 0 {
                        gpu.launch_graph(post_handle, stream)?;
                    } else {
                        self.forward_block_layer_post_attn(layer, &args, ctx)?;
                    }
                }

                let tail_handle = graphs[tail_slot];
                if tail_handle.0 != 0 {
                    gpu.launch_graph(tail_handle, stream)?;
                } else {
                    run_tail()?;
                }
            } else {
                let warmed = self
                    .propose_warmup_count
                    .load(std::sync::atomic::Ordering::Relaxed);
                if warmed < warmup_target {
                    // Warm-up: eager only, no capture.
                    self.propose_warmup_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    run_all_eager()?;
                } else {
                    // Capture pass: build all subgraphs in one propose
                    // call, then immediately replay them via the launches
                    // below. End-cap returns GraphHandle(0) as the
                    // empty-capture sentinel; we store the zero so the
                    // replay path falls back to eager for that slot.
                    tracing::info!(
                        "DFlash piecewise capture: starting (warmup_count={}, target={}, slots={})",
                        warmed,
                        warmup_target,
                        total_slots
                    );
                    let mut new_graphs: Vec<spark_runtime::gpu::GraphHandle> =
                        Vec::with_capacity(total_slots);

                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        let args = make_paged_args(layer_idx).expect("option_b args available");

                        // pre_attn subgraph
                        gpu.begin_capture(stream)?;
                        let _captured = self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                        let pre_graph = gpu.end_capture(stream)?;
                        new_graphs.push(pre_graph);
                        if pre_graph.0 != 0 {
                            gpu.launch_graph(pre_graph, stream)?;
                        } else {
                            tracing::warn!(
                                "DFlash piecewise: pre_attn layer {} empty capture — eager fallback",
                                layer_idx
                            );
                            self.forward_block_layer_pre_attn(layer, &args, ctx)?;
                        }

                        // attention — eager, never captured
                        let (k_pool, v_pool) = {
                            let cache = self.kv_cache.lock();
                            (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
                        };
                        self.forward_block_layer_attention(&args, ctx, k_pool, v_pool)?;

                        // post_attn subgraph
                        gpu.begin_capture(stream)?;
                        self.forward_block_layer_post_attn(layer, &args, ctx)?;
                        let post_graph = gpu.end_capture(stream)?;
                        new_graphs.push(post_graph);
                        if post_graph.0 != 0 {
                            gpu.launch_graph(post_graph, stream)?;
                        } else {
                            tracing::warn!(
                                "DFlash piecewise: post_attn layer {} empty capture — eager fallback",
                                layer_idx
                            );
                            self.forward_block_layer_post_attn(layer, &args, ctx)?;
                        }
                    }

                    // tail subgraph
                    gpu.begin_capture(stream)?;
                    run_tail()?;
                    let tail_graph = gpu.end_capture(stream)?;
                    new_graphs.push(tail_graph);
                    if tail_graph.0 != 0 {
                        gpu.launch_graph(tail_graph, stream)?;
                    } else {
                        tracing::warn!("DFlash piecewise: tail empty capture — eager fallback");
                        run_tail()?;
                    }

                    let success_count = new_graphs.iter().filter(|g| g.0 != 0).count();
                    tracing::info!(
                        "DFlash piecewise capture: complete ({}/{} subgraphs captured)",
                        success_count,
                        total_slots
                    );
                    *g = Some(new_graphs);
                }
            }
        } else {
            run_all_eager()?;
        }

        // ── Step 6: D2H γ × 4 bytes ──
        //
        // Phase E.2: async D2H to a pinned host buffer, recorded against a
        // dedicated event. The host blocks on the event (not the stream)
        // just before reading the bytes, so any verify-side work the
        // scheduler queues on the same stream after this point can be
        // issued concurrently with the copy completing.
        //
        // Why pinned: cuMemcpyDtoHAsync against a pageable destination
        // silently falls back to a synchronous bounce-buffer copy; the
        // async DMA fast path requires page-locked host memory. nsys
        // confirmed cuMemcpyDtoHAsync_v2 was 64% of API time post-E.1 —
        // pinned memory lets the copy actually pipeline.
        //
        // Why event vs stream sync: cuStreamSynchronize waits for ALL
        // work on the stream; cuEventSynchronize waits only for work
        // recorded up to the event. After Phase E.4 lifts more work
        // into capture, the gap matters.
        let pinned_ptr = self
            .scratch
            .draft_tokens_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        let host_buf: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(pinned_ptr, self.gamma * 4) };
        gpu.copy_d2h_on_stream(self.scratch.draft_tokens_dev, host_buf, stream)?;
        gpu.record_event(self.scratch.draft_tokens_event, stream)?;
        gpu.event_synchronize(self.scratch.draft_tokens_event)?;
        let drafts: Vec<u32> = host_buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1 (one-shot): log all γ drafts so
        // we can compare against the PyTorch reference run on the same
        // captured target_hidden. Static guard mirrors the input dump.
        static DRAFTS_DUMP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !DRAFTS_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
            && (std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
                || std::env::var("ATLAS_DFLASH_LOG_DRAFTS").ok().as_deref() == Some("1"))
        {
            tracing::info!(
                "DFLASH DUMP_FULL drafts (γ={}, last_token={}, position={}, eff_ctx={}): {:?}",
                self.gamma,
                last_token,
                position,
                eff_ctx,
                drafts,
            );
            DRAFTS_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let _ = g; // suppress unused
        Ok(drafts)
    }
}
