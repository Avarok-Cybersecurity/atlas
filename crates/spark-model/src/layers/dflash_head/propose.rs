// SPDX-License-Identifier: AGPL-3.0-only

//! `DraftProposer::propose` body for [`super::BlockDiffusionDraftHead`].
//!
//! Split out of `dflash_head.rs` for file-size budget. Trait impl
//! delegates to [`BlockDiffusionDraftHead::propose_drafts`].

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;
use crate::speculative::ProposerState;

impl BlockDiffusionDraftHead {
    pub(super) fn propose_drafts(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        _num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        _stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;

        // ── I/O-PARITY DUMP: full ctx_hidden_acc accumulator at propose entry ──
        // Gated ATLAS_DFLASH_CTX_PARITY_DUMP=1. One-shot. Writes the ENTIRE
        // accumulated 5×target_hidden context the drafter conditions on, so a
        // PyTorch/vLLM reference can diff slot-count + values against
        // `target_hidden_states[:num_context]` (vLLM feeds num_context = ALL
        // accepted-prefix tokens; this proves whether Atlas's accumulator has
        // the same BREADTH and the same per-slot 5-layer values).
        //
        // Layout of /tmp/atlas_ctx_parity.bin: contiguous BF16,
        // ctx_len slots × target_layer_ids.len() layers × target_hidden_size,
        // i.e. ctx_len × ctx_slot_bytes bytes. Companion JSON carries
        // ctx_len, n_layers, target_hidden_size, position, last_token so the
        // harness reconstructs shape without log-scraping.
        {
            static CTX_PARITY_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if std::env::var("ATLAS_DFLASH_CTX_PARITY_DUMP")
                .ok()
                .as_deref()
                == Some("1")
                && dstate.ctx_len > 0
                && !CTX_PARITY_DONE.load(std::sync::atomic::Ordering::Relaxed)
            {
                let n_bytes = dstate.ctx_len * dstate.ctx_slot_bytes;
                let mut buf = vec![0u8; n_bytes];
                ctx.gpu.synchronize(_stream)?;
                ctx.gpu.copy_d2h(dstate.ctx_hidden_acc, &mut buf)?;
                match std::fs::write("/tmp/atlas_ctx_parity.bin", &buf) {
                    Ok(()) => {
                        let elems_per_slot = dstate.ctx_slot_bytes / 2;
                        let meta = format!(
                            "{{\"ctx_len\":{},\"ctx_slot_bytes\":{},\"elems_per_slot\":{},\"position\":{},\"last_token\":{},\"n_bytes\":{}}}",
                            dstate.ctx_len,
                            dstate.ctx_slot_bytes,
                            elems_per_slot,
                            position,
                            last_token,
                            n_bytes,
                        );
                        let _ = std::fs::write("/tmp/atlas_ctx_parity.json", meta);
                        tracing::info!(
                            "DFLASH CTX_PARITY: wrote {} bytes — ctx_len={} slots × {} BF16 elems/slot (position={}, last_token={}) to /tmp/atlas_ctx_parity.bin",
                            n_bytes,
                            dstate.ctx_len,
                            dstate.ctx_slot_bytes / 2,
                            position,
                            last_token,
                        );
                    }
                    Err(e) => {
                        tracing::warn!("DFLASH CTX_PARITY: write failed: {e}");
                    }
                }
                CTX_PARITY_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // ── Phase 2.5b kernel-chain scaffold (commented for next-session
        // fill-in; current path falls through to empty-Vec stub below) ──
        //
        // Reference: `dflash.py` (in the drafter's HF snapshot) lines 60-95
        // for the per-layer attention pattern. Per-layer flow (one call into
        // Atlas's existing op wrappers per bullet):
        //
        // For each layer in `self.layers`:
        //   ops::rms_norm(self.kernels.rms_norm, stream_buf, layer.input_layernorm,
        //                 norm_buf, gamma, hidden_size, eps)
        //   ops::dense_gemm(self.kernels.dense_gemm, norm_buf, layer.q_proj.weight,
        //                   q_buf, gamma, q_dim, hidden_size)        // [γ, 32*128]
        //   ops::dense_gemm(self.kernels.dense_gemm, norm_buf, layer.k_proj.weight,
        //                   k_buf, gamma, kv_dim, hidden_size)        // [γ, 4*128]
        //   ops::dense_gemm(self.kernels.dense_gemm, norm_buf, layer.v_proj.weight,
        //                   v_buf, gamma, kv_dim, hidden_size)        // [γ, 4*128]
        //   per-head q_norm: ops::rms_norm over each [γ, head_dim] slice
        //   per-head k_norm: ops::rms_norm over each [γ, head_dim] slice
        //   ops::rope_yarn(self.kernels.rope_qwen3, q_buf, k_buf, position_ids,
        //                  gamma, num_q_heads, num_kv_heads, head_dim, rotary_dim,
        //                  inv_freq, theta)
        //   ops::prefill_attention(prefill_attn_kernel, q_buf, k_buf, v_buf,
        //                          attn_out, gamma, 1, num_q_heads, num_kv_heads,
        //                          head_dim, inv_sqrt_d, /* causal = */ false,
        //                          /* sliding_window = */ 0)
        //   ops::dense_gemm(dense_gemm, attn_out, layer.o_proj.weight,
        //                   stream_buf_acc, gamma, hidden_size, q_dim)
        //   ops::residual_add(self.kernels.residual_add, stream_buf, stream_buf_acc,
        //                     stream_buf, gamma * hidden_size)
        //   ops::rms_norm(self.kernels.rms_norm, stream_buf, layer.post_attn_norm,
        //                 norm_buf, gamma, hidden_size, eps)
        //   ops::dense_gemm(dense_gemm, norm_buf, layer.gate_proj.weight,
        //                   gate_out, gamma, intermediate_size, hidden_size)
        //   ops::dense_gemm(dense_gemm, norm_buf, layer.up_proj.weight,
        //                   up_out, gamma, intermediate_size, hidden_size)
        //   ops::silu_mul(self.kernels.silu_mul, gate_out, up_out, mlp_intermediate,
        //                 gamma * intermediate_size)
        //   ops::dense_gemm(dense_gemm, mlp_intermediate, layer.down_proj.weight,
        //                   stream_buf_acc, gamma, hidden_size, intermediate_size)
        //   ops::residual_add(stream_buf, stream_buf_acc, stream_buf,
        //                     gamma * hidden_size)
        //
        // After the layer loop:
        //   ops::rms_norm(rms_norm, stream_buf, self.norm, norm_buf, gamma,
        //                 hidden_size, eps)
        //   ops::dense_gemm(dense_gemm, norm_buf, self.lm_head_shared, logits,
        //                   gamma, vocab_size, hidden_size)
        //   ops::argmax_bf16(self.kernels.argmax, logits, draft_tokens_dev,
        //                    gamma, vocab_size)
        //   gpu.copy_d2h(draft_tokens_dev, &mut host_buf, gamma * 4)
        //   parse host_buf as [u32; γ]
        //
        // Required additional state on the head (not yet allocated):
        //   - position_ids: [γ] u32 device buffer (positions = state.seq_len..+γ)
        //   - inv_freq: [head_dim/2] f32 yarn-scaled frequencies (pre-computed
        //     from drafter's rope_scaling: factor=64, beta_fast=32, beta_slow=1,
        //     original_max_position_embeddings=4096)
        //   - per-rms-norm eps from drafter config (Qwen3 default 1e-6)
        //
        // Open design questions for ctx-conditioned drafting (later iter):
        //   1. ctx_len = ? — vLLM accumulates per-token captures across all
        //      decoded positions; Atlas currently captures only the latest
        //      step's 5 hiddens (model-level single slot). Per-sequence
        //      accumulator needs to land in DflashProposerState.
        //   2. Asymmetric q_len (γ) vs k_len (γ + ctx_len) — either pad q
        //      with a dummy row or use the paged attention with a 1-block
        //      scratch cache for ctx K/V.
        //   3. RoPE position offsets — ctx K positions map to the prior
        //      decoded positions; q/noise K positions map to seq_len..+γ.

        let _ = (ctx, position, last_token);

        // Phase 2.5 stub. Real propose() implementation roadmap:
        //
        // ── Step 0: validate inputs ──
        // - target_hidden_stack must be Some(ptr) — shape [N, target_hidden]
        //   BF16 where N = self.target_layer_ids.len() (5 for Qwen3.6-DFlash).
        // - dstate.prefill_done must be true OR this is the first call after
        //   target prefill (in which case run precompute_and_store_context_kv
        //   to populate drafter KV cache from the prompt-time captures).
        //
        // ── Step 1: project current target hiddens through `fc` ──
        // - Input:  target_hidden_stack: [N * target_hidden] BF16 = [10240]
        // - Op:     dense_gemv_bf16(fc, in)         → [draft_hidden] = [2048]
        // - Op:     rms_norm(hidden_norm)           → [2048] BF16
        // - Op:     reshape_and_cache(K, V at slot dstate.seq_len) into the
        //           drafter's first layer's paged KV cache (this represents
        //           ONE token of context, written through layer 0's K/V proj
        //           → RoPE → cache slot at logical position dstate.seq_len).
        // - Note:   vLLM's `precompute_and_store_context_kv` does this for
        //           the *full* prompt prefix on the first call, and one
        //           token per step thereafter. We follow the same pattern.
        //
        // ── Step 2: build γ-token query input ──
        // - Allocate [γ, draft_hidden] scratch buffer.
        // - Embed token 0 as `last_token` via shared embed_tokens_shared.
        // - Embed tokens 1..γ as `mask_token_id` via shared embed_tokens_shared.
        // - Add the projected fc context to position 0 (Qwen3-DFlash
        //   `combine_hidden_states` semantics — verify against vLLM
        //   `qwen3_dflash.py:DFlashQwen3Model.forward`).
        //
        // ── Step 3: run γ tokens through 8 drafter layers ──
        // For each layer i in 0..self.num_layers:
        //   a. input_layernorm.rms_norm(input → x_norm)
        //   b. q_proj.gemm(x_norm → q [γ, num_q_heads * head_dim])
        //      k_proj.gemm(x_norm → k [γ, num_kv_heads * head_dim])
        //      v_proj.gemm(x_norm → v [γ, num_kv_heads * head_dim])
        //   c. q_norm.rms_norm per-head, k_norm.rms_norm per-head
        //   d. rope(q, k, position+0..γ-1)
        //   e. reshape_and_cache(k, v) into layer i's paged FP8 cache at
        //      slot positions [dstate.seq_len + 1 .. + γ]
        //   f. ops::prefill_attention_paged_fp8_dflash(...) — γ queries,
        //      bidirectional in-block + full prefix attention. Optional
        //      sliding window via self.window_size.
        //   g. o_proj.gemm(attn_out → o)
        //   h. residual_add(input, o)
        //   i. post_attention_layernorm.rms_norm
        //   j. gate_proj+up_proj+silu_mul+down_proj  (Qwen3 SwiGLU)
        //   k. residual_add
        //
        // ── Step 4: final RMSNorm + LM head ──
        // - self.norm.rms_norm
        // - dense_gemm(lm_head_shared) → [γ, vocab_size]
        // - argmax per row → γ candidate token IDs (DEVICE)
        // - copy_d2h γ × 4 bytes
        //
        // ── Step 5: state update ──
        // - dstate.seq_len += γ + 1   (drafter cache now holds prefix + γ + 1)
        //   note: the +1 is for the bonus-token slot we just appended in Step 1
        // - dstate.last_num_drafted = γ
        //
        // ── Required kernel handles (resolved via ctx.gpu.kernel(...)) ──
        // rms_norm, dense_gemv_bf16, dense_gemm_bf16, rope_qwen3_yarn,
        // reshape_and_cache_fp8, prefill_attention_paged_fp8_dflash,
        // silu_mul, residual_add, argmax_bf16, batched_embed
        //

        // Append the model's latest single-slot ctx capture into the
        // per-seq accumulator. Skip when `target_hidden_stack` is None
        // (e.g. EP=2 worker rank or the very first call before any
        // capture has fired). Capping at `max_ctx_len` to keep within
        // allocated bounds — drafter quality plateaus past a few hundred
        // ctx positions anyway.
        //
        // ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND=1 disables the post-decode
        // append. The captured target_hidden_stack is the K-1 token of
        // the last K=2 verify (the draft, NOT the bonus). On REJECT
        // (the typical case during cold-start training-distribution
        // mismatch) the draft was never accepted, so appending its
        // hiddens to the accumulator poisons the ctx for subsequent
        // propose() calls. Setting this flag uses ONLY prefill captures
        // — clean ctx isolation for diagnosing real-traffic acceptance.
        // EAGLE-fix: the K=2 accept path appends row 0 + row 1 in EAGLE order
        // BEFORE calling propose and sets this one-shot flag, so propose must
        // NOT also decode-append row 0 (would duplicate it). Consume the flag
        // here. K=gamma/K=4 never set it -> their decode-append is unaffected.
        let eagle_skip = dstate.skip_next_decode_append;
        dstate.skip_next_decode_append = false;
        let skip_decode_append = std::env::var("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND")
            .ok()
            .as_deref()
            == Some("1");
        if !skip_decode_append
            && !eagle_skip
            && let Some(latest_ctx) = target_hidden_stack
            && dstate.ctx_len < dstate.max_ctx_len
        {
            let dst_offset = dstate.ctx_len * dstate.ctx_slot_bytes;
            ctx.gpu.copy_d2d_async(
                latest_ctx,
                dstate.ctx_hidden_acc.offset(dst_offset),
                dstate.ctx_slot_bytes,
                _stream,
            )?;
            // Phase I (v2): stamp this slot's TRUE absolute position, fixed
            // forever. The just-decoded token sits at `position - 1` (the
            // full-rebuild formula assigns slot ctx_len the position
            // (position - (ctx_len+1)) + ctx_len == position - 1). Keeping
            // ctx_positions parallel to ctx_len lets precompute rope each
            // slot by its own fixed position instead of a sliding base.
            debug_assert_eq!(dstate.ctx_positions.len(), dstate.ctx_len);
            dstate.ctx_positions.push(position.saturating_sub(1) as i32);
            dstate.ctx_len += 1;
        }

        // ── Phase 2 Option B: lazy block_table allocation ─────────────
        // When ATLAS_DFLASH_OPTION_B=1 and the proposer hasn't yet
        // allocated paged blocks, do it now. We allocate enough blocks
        // to cover the full ctx_hidden_acc plus a safety margin for γ.
        // Block_size matches from_weights.rs:68 (=16).
        let option_b_enabled = std::env::var("ATLAS_DFLASH_OPTION_B").ok().as_deref() == Some("1");
        let option_b_arg: Option<(DevicePtr, u32)> = if option_b_enabled {
            // Lazy block table init. ctx slots come from precompute over the
            // accumulated target hiddens; γ slots come from the layer body.
            // We need ceil((max_ctx_len + γ) / block_size) blocks.
            const BLOCK_SIZE: usize = 16;
            let blocks_needed = (dstate.max_ctx_len + self.gamma + 1).div_ceil(BLOCK_SIZE);
            if dstate.block_table_dev.is_none() {
                let mut cache = self.kv_cache.lock();
                dstate.block_table.clear();
                for _ in 0..blocks_needed {
                    match cache.try_alloc_block() {
                        Some(b) => dstate.block_table.push(b),
                        None => {
                            anyhow::bail!(
                                "DFlash Option B: paged KV cache exhausted at block {}/{}",
                                dstate.block_table.len(),
                                blocks_needed
                            );
                        }
                    }
                }
                drop(cache);
                // Copy block_table to device.
                let bt_bytes: Vec<u8> = dstate
                    .block_table
                    .iter()
                    .flat_map(|b| b.to_le_bytes())
                    .collect();
                let bt_dev = ctx.gpu.alloc(bt_bytes.len())?;
                ctx.gpu.copy_h2d(&bt_bytes, bt_dev)?;
                dstate.block_table_dev = Some(bt_dev);
                dstate.max_ctx_count_drafter = blocks_needed * BLOCK_SIZE;
                tracing::info!(
                    "DFlash Option B: allocated {} blocks ({} slots) for drafter paged cache",
                    blocks_needed,
                    dstate.max_ctx_count_drafter
                );
            }
            // Phase I (v2) — incremental ctx precompute (design doc §18).
            // Only the new tail [ctx_committed..ctx_len) needs its K/V
            // computed; slots [0..ctx_committed) are already valid in the
            // paged cache and — because each slot ropes at its OWN fixed
            // position (stamped at append, see ctx_positions) — they never
            // go stale when later accepts move the live `position`. The old
            // path rebuilt the whole prefix every step (O(ctx_len²)).
            //
            // Escape hatch: ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE=1 forces a
            // full recompute (committed=0) for A/B accept-rate parity.
            let force_full = std::env::var("ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE")
                .ok()
                .as_deref()
                == Some("1");
            // Clamp watermark defensively: a rewind should have reset it,
            // but never start past ctx_len.
            let committed = if force_full {
                0
            } else {
                dstate.ctx_committed.min(dstate.ctx_len)
            };
            let new_count = dstate.ctx_len - committed;
            if dstate.ctx_len > 0 && new_count > 0 {
                // Ctx-holes follow-up (2026-07-08): the precompute scratch
                // (fc_proj / fused_kv_out / slot_mapping_dev) is sized for
                // ctx_window rows. A serial-append stretch (think-gated /
                // adaptive-suspended) can grow the uncommitted tail far past
                // that (MinHeap: 846 think tokens vs ctx_window=512); one
                // precompute over the whole tail then reads/writes past the
                // scratch allocations -> cuMemcpyDtoDAsync
                // CUDA_ERROR_INVALID_VALUE on EVERY re-probe (ctx_committed
                // never advances, so each propose retries the same oversized
                // tail). Chunk the tail to ctx_window rows per pass — the
                // paged cache itself covers max_ctx_len + gamma, so only the
                // per-pass scratch needs bounding.
                anyhow::ensure!(
                    self.ctx_window > 0,
                    "DFlash precompute: ctx_window=0 but ctx tail of {} slots \
                     needs precompute — scratch has no capacity",
                    new_count,
                );
                let slot_mapping = &self.scratch.slot_mapping_dev;
                let mut chunk_start = committed;
                while chunk_start < dstate.ctx_len {
                    let chunk_count = (dstate.ctx_len - chunk_start).min(self.ctx_window);
                    // Build slot_mapping for this chunk
                    // [chunk_start .. chunk_start + chunk_count).
                    crate::layers::ops::fill_slots_from_block_table(
                        ctx.gpu,
                        self.kernels.fill_slots,
                        *slot_mapping,
                        dstate.block_table_dev.unwrap(),
                        chunk_start as u32,
                        chunk_count as u32,
                        BLOCK_SIZE as u32,
                        _stream,
                    )?;
                    // The fixed positions for exactly the rows we're
                    // computing. ctx_positions is parallel to ctx slots.
                    let slot_positions =
                        &dstate.ctx_positions[chunk_start..chunk_start + chunk_count];
                    self.precompute_ctx_kv(
                        dstate.ctx_hidden_acc,
                        chunk_start,
                        chunk_count,
                        slot_positions,
                        *slot_mapping,
                        ctx,
                        _stream,
                        true, // commit: always write to paged cache on production path
                    )?;
                    chunk_start += chunk_count;
                }
                // Tail is now committed in the paged cache. Committed slots
                // hold fixed-position rope, so no base re-stamp is needed.
                dstate.ctx_committed = dstate.ctx_len;
            }
            dstate.ctx_count_drafter = dstate.ctx_len;
            // ── SERIAL-APPEND BOUNDARY PROOF (2026-07-08) ──
            // With ATLAS_DFLASH_CTXLEN_PROBE=1, validate the ctx position
            // stamps are STRICTLY INCREASING across all populated slots —
            // contiguous appends, no double-append, no dropped stretch.
            // The think→spec seam (re-probe after a serial stretch) is
            // where a violation would surface. Host-side Vec scan,
            // probe-gated, zero cost when off.
            if std::env::var("ATLAS_DFLASH_CTXLEN_PROBE").ok().as_deref() == Some("1")
                && let Some(i) = dstate.ctx_positions.windows(2).position(|w| w[1] <= w[0])
            {
                tracing::warn!(
                    "DFLASH CTX_POSITIONS VIOLATION: slot {} pos {} -> slot {} pos {} \
                     (not strictly increasing: double-append or seam hole)",
                    i,
                    dstate.ctx_positions[i],
                    i + 1,
                    dstate.ctx_positions[i + 1],
                );
            }
            // ── CTX-LEN STALL PROBE (Friday 2026-06-11, id252) ──
            // q_offset = ctx_count_drafter = ctx_len drives the γ query RoPE
            // position. id252 found q_offset stuck at ~21 while position=301
            // (drafter querying 280 positions behind reality). This logs
            // ctx_len vs position EVERY propose so we can confirm whether
            // ctx_len GROWS with position (healthy) or STALLS at prompt length
            // (the bug — likely thinking-mode tokens not appending to ctx).
            // Gated ATLAS_DFLASH_CTXLEN_PROBE=1, rate-limited to ~1/16 steps
            // to avoid log flood.
            if std::env::var("ATLAS_DFLASH_CTXLEN_PROBE").ok().as_deref() == Some("1")
                && position.is_multiple_of(16)
            {
                tracing::info!(
                    "DFLASH CTXLEN_PROBE: position={} ctx_len={} q_offset(=ctx_len)={} GAP={} (position - ctx_len; healthy≈prompt_len, BUG if grows unbounded)",
                    position,
                    dstate.ctx_len,
                    dstate.ctx_len,
                    position.saturating_sub(dstate.ctx_len),
                );
            }
            // Ablation: ATLAS_DFLASH_OPTION_B_NO_CTX=1 forces ctx_count=0
            // in the layer body so paged attention only sees the γ K/V
            // we write in-layer. If accept rate is bad even here, the
            // bug is in the cache write/read path, not in precompute.
            let ablate_no_ctx = std::env::var("ATLAS_DFLASH_OPTION_B_NO_CTX")
                .ok()
                .as_deref()
                == Some("1");
            let effective_ctx_count = if ablate_no_ctx {
                0
            } else {
                dstate.ctx_count_drafter as u32
            };
            Some((dstate.block_table_dev.unwrap(), effective_ctx_count))
        } else {
            None
        };

        let drafts = self
            .forward_block(
                last_token,
                position,
                ctx,
                _stream,
                // Pass the accumulator's start pointer + `ctx_len` so
                // forward_block knows how many ctx positions to project.
                if dstate.ctx_len > 0 {
                    Some((dstate.ctx_hidden_acc, dstate.ctx_len))
                } else {
                    None
                },
                option_b_arg,
            )
            .map_err(|e| {
                tracing::warn!("DFlash forward_block failed, falling back to no-spec: {e:#}");
                e
            })?;
        // Default cap = γ. The nologik spec_ssm merge provides the WY-chunkwise
        // GDN kernels (gdn_decode_wy17 for K=17, wy2/wy3/wy4 for smaller K)
        // that snapshot per-position h/conv intermediates into the SSM pool.
        // commit_verify_state_async (verify_dflash_step.rs) reads those
        // intermediates to roll back to the accepted prefix on partial reject.
        // SSM pool is pre-allocated for num_intermediates=17 (impl_a1.rs:129)
        // and the WY17 strided layout (inter_stride_floats = h_bytes/4) maps
        // 1:1 to ssm_pool.h_intermediate(layer, slot, i). Override with
        // ATLAS_DFLASH_DRAFT_CAP=N (N=1 to force K=2 path for ablation).
        let cap: usize = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.gamma);

        // ATLAS_DFLASH_VERIFY_TRACE=1: log all γ drafts BEFORE the cap so we
        // can see whether the drafter echoes only at position 0 or across
        // every noise row. Pairs with K2 TRACE in the scheduler.
        if std::env::var("ATLAS_DFLASH_VERIFY_TRACE").ok().as_deref() == Some("1") {
            tracing::info!(
                "DFLASH TRACE drafts: token_in={} position={} γ={} drafts_pre_cap={:?}",
                last_token,
                position,
                drafts.len(),
                drafts,
            );
        }

        // Block-diffusion drafter convention: noise_row[0]'s input is
        // `last_token`, and the drafter denoises it trivially back to
        // itself — that's the "bonus" position. The first USEFUL draft
        // lives at noise_row[1] (input = mask, predicts position+1).
        // vLLM ignores row 0 via `token_indices_to_sample`. Atlas was
        // reading row 0 as draft[0], giving 0% K=2 accept on z-lab
        // DFlash drafters; dropping it lifts accept to ~80%.
        //
        // Gated on `mask_token_id` presence in the drafter config —
        // that's the diffusion-drafter signal. Autoregressive drafters
        // (e.g. EAGLE) have no mask token and should keep row 0.
        let drafts = if self.mask_token_id != 0 && drafts.len() > 1 {
            drafts[1..].to_vec()
        } else {
            drafts
        };

        let drafts = drafts.into_iter().take(cap).collect::<Vec<_>>();
        dstate.last_num_drafted = drafts.len();
        Ok(drafts)
    }
}
