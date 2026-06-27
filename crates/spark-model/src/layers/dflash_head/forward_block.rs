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
    pub(super) fn forward_block(
        &self,
        last_token: u32,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
        ctx_buffer: Option<(DevicePtr, usize)>,
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
        // ctx_pos_i = start_slot + i — the ACTUAL absolute position of the
        // i-th used accumulator slot.
        //
        // The ctx accumulator is indexed by absolute sequence position:
        //   acc[abs_pos] = hidden captured at sequence position abs_pos.
        // start_slot = ctx_total - eff_ctx = the first slot we use.
        // So the i-th ctx slot represents actual position (start_slot + i).
        //
        // WRONG formula: position - eff_ctx. This gives position=23, eff_ctx=20
        // → ctx_start=3, but slot 0 is actually at sequence position 0, not 3.
        // Using wrong position IDs corrupts the RoPE rotations for all ctx K
        // vectors, breaking attention and causing "." collapse in predictions.
        let start_slot = ctx_total.saturating_sub(eff_ctx);
        let ctx_start = start_slot;
        // Noise position layout matching SGLang's DFlash drafter training:
        //   noise0 (conditioning token, last_token): position - 1
        //   noise1..gamma-1 (mask tokens): position, position+1, ..., position+gamma-2
        // Block diffusion only trains masked positions; noise0's output is untrained
        // (conditioning). Its position must match the actual last-token position so
        // RoPE is consistent with target ctx K vectors at the same slot.
        let noise0_pos = (position as i64 - 1).max(0) as i32;
        let pos_host: Vec<i32> = (0..eff_ctx)
            .map(|i| (ctx_start + i) as i32)
            .chain(std::iter::once(noise0_pos))
            .chain((0..self.gamma - 1).map(|i| (position + i) as i32))
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
        //
        // The drafter is trained with the last accepted (bonus) token at
        // noise position 0 and mask tokens at positions 1..γ-1. This gives
        // the bidirectional attention a critical conditioning signal: the
        // other mask positions can attend to the known last_token and
        // calibrate their predictions accordingly. Using mask_token_id for
        // position 0 as well would break this conditioning → 0% accept rate.
        // Reference: dflash_worker.py line 549:
        //   block_ids[:, 0].copy_(draft_input.bonus_tokens)
        //   block_ids[:, 1:].fill_(mask_token_id)
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
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let args = super::forward_block_layer::LayerArgs {
                layer_idx,
                n_attn,
                eff_ctx,
                h,
                q_dim,
                kv_dim,
                inter,
                bf16,
                inv_sqrt_d,
                stream,
            };
            self.forward_block_layer(layer, &args, ctx, debug_dump)?;
        }
        // Drop the original inline loop body — extracted to helper.

        // ── Step 4: final RMSNorm + LM head on noise rows only ──
        // Skip ctx slots [0..eff_ctx] (their stream_buf is garbage from
        // layer accumulation). Read from offset `eff_ctx * h * bf16`.
        let noise_byte_offset = eff_ctx * self.hidden_size * bf16;
        let stream_noise = self.scratch.stream_buf.offset(noise_byte_offset);
        let norm_noise = self.scratch.norm_buf.offset(noise_byte_offset);
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            stream_noise,
            &self.norm,
            norm_noise,
            self.gamma as u32,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            norm_noise,
            &crate::weight_map::DenseWeight {
                weight: self.lm_head_shared,
            },
            self.scratch.logits,
            self.gamma as u32,
            self.vocab_size as u32,
            h,
            stream,
        )?;

        // Optional full-stream dump after final norm (debug; before lm_head).
        if debug_dump {
            dump_bf16("final.norm_buf[noise0]", norm_noise, 10)?;
            // Sanity-check: dump first 10 BF16 values of target's lm_head_shared.
            // If this returns zeros or garbage, the BF16 lm_head was freed by
            // factory.rs's NVFP4 quantization step.
            dump_bf16("final.lm_head_shared[0..10]", self.lm_head_shared, 10)?;
        }

        // ── Step 5: argmax per row → γ-1 token ids (skip noise0) ──
        // Block diffusion: the model is only trained at MASKED positions.
        // noise0 (conditioning slot, input = last_token) is UNTRAINED — its
        // logits are arbitrary garbage. SGLang skips it:
        //   `draft_next = greedy_sample(draft_hidden[:, 1:, :])`
        //   `draft_tokens[:, 0] = block_ids[:, 0]`  (= last_token itself)
        // We match that: extract argmax from noise1..noise{gamma-1} only,
        // writing to draft_tokens_dev slots 0..gamma-2 (γ-1 valid drafts).
        if debug_dump {
            dump_bf16("final.logits[noise0]", self.scratch.logits, 10)?;
            dump_bf16(
                "final.logits[noise1]",
                self.scratch.logits.offset(self.vocab_size * bf16),
                10,
            )?;
        }
        for i in 1..self.gamma {
            let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16);
            let token_slot = self.scratch.draft_tokens_dev.offset((i - 1) * 4);
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                logits_row,
                token_slot,
                self.vocab_size as u32,
                stream,
            )?;
        }

        // ── Step 6: D2H (γ-1) × 4 bytes ──
        let mut host_buf = vec![0u8; (self.gamma - 1) * 4];
        let t_pre_sync = std::time::Instant::now();
        gpu.synchronize(stream)?;
        let t_post_sync = std::time::Instant::now();
        if dflash_prof {
            let step0_us = t_step0_done.map(|t| t.elapsed().as_micros()).unwrap_or(0);
            let layers_us = t_layers_done
                .and_then(|tl| t_step0_done.map(|t0| tl.duration_since(t0).as_micros()))
                .unwrap_or(0);
            let sync_wait_us = t_post_sync.duration_since(t_pre_sync).as_micros();
            tracing::info!(
                "DFlash forward_block PROF: pre_sync(step0+layers+head+argmax)={}ms sync_wait={}ms step0_gpu={}ms layers_gpu={}ms eff_ctx={} n_attn={}",
                t_pre_sync.elapsed().as_millis(),
                sync_wait_us / 1000,
                step0_us / 1000,
                layers_us / 1000,
                eff_ctx,
                n_attn,
            );
        }
        gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut host_buf)?;
        let drafts: Vec<u32> = host_buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1 (one-shot): log γ-1 valid drafts so
        // we can compare against the PyTorch reference run on the same
        // captured target_hidden. Static guard mirrors the input dump.
        static DRAFTS_DUMP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !DRAFTS_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
        {
            tracing::info!(
                "DFLASH DUMP_FULL drafts (γ-1={}, last_token={}, position={}, eff_ctx={}): {:?}",
                self.gamma - 1,
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
