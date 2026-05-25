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
        num_drafts: usize,
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
        // Phase 2.5b first-iteration impl.
        //
        // Runs the γ-block forward chain (8 layers × Qwen3-decoder,
        // non-causal self-attention) through `forward_block`. Returns
        // **only the first draft** (1 token) so the scheduler routes
        // through the proven `step_verify_k2` path which already handles
        // SSM state rollback via the K=2 graphed verify kernel's
        // populated intermediates.
        //
        // Why cap at 1? Atlas's K=γ eager verify path (`decode_verify`)
        // does NOT populate `h_state_intermediates` — those are only
        // written by the K=2/3/4 specialized GDN kernels. So a γ-token
        // verify with partial accept (the typical case) produces garbage
        // SSM state on hybrid models like Qwen3.6-A3B (30 GDN layers).
        // Capping at 1 makes drafts.len()=1, scheduler picks K=2 verify
        // which DOES populate intermediates, SSM rollback works correctly.
        //
        // This loses the γ-parallel speedup (DFlash's main advantage)
        // but produces correct output with acceptance >0 — strict
        // improvement over no-spec when drafts match. The full γ-parallel
        // path needs either:
        //   (a) a K=γ specialized GDN verify kernel that populates
        //       intermediates per position (multi-week kernel work), or
        //   (b) restricting DFlash to pure-attention targets (Gemma-4,
        //       MiniMax-M2 dense) where SSM rollback isn't needed.
        //
        // Quality note: drafter runs WITHOUT context conditioning
        // (`ctx_len=0`) — it was trained with 5×target_hidden ctx, so
        // first-token acceptance will be poor (<<70%). Adding ctx is the
        // next iteration on top of `forward_block`.
        let _ = num_drafts;

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
        let skip_decode_append = std::env::var("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND")
            .ok()
            .as_deref()
            == Some("1");
        if !skip_decode_append
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
            // Precompute over the full accumulated ctx into paged cache
            // slots [0..ctx_len). This is wasteful (rebuilds every step)
            // but keeps stage-4 correctness simple — incremental append
            // is a later optimization.
            if dstate.ctx_len > 0 {
                // Build slot_mapping for the ctx slots [0..ctx_len).
                let slot_mapping = &self.scratch.slot_mapping_dev;
                crate::layers::ops::fill_slots_from_block_table(
                    ctx.gpu,
                    self.kernels.fill_slots,
                    *slot_mapping,
                    dstate.block_table_dev.unwrap(),
                    0,
                    dstate.ctx_len as u32,
                    BLOCK_SIZE as u32,
                    _stream,
                )?;
                // Run precompute with COMMIT path. Caller of precompute
                // gates commit on ATLAS_DFLASH_PRECOMPUTE_COMMIT — we
                // force-set it here since Option B requires committed K/V.
                // SAFETY: setting env var process-wide; OK since the
                // server is single-purpose during Option B benches.
                let prev_commit = std::env::var("ATLAS_DFLASH_PRECOMPUTE_COMMIT").ok();
                // SAFETY: env mutation here is single-threaded per process
                // for the DFlash bench; we restore the prior value after.
                unsafe {
                    std::env::set_var("ATLAS_DFLASH_PRECOMPUTE_COMMIT", "1");
                }
                let abs_start = position.saturating_sub(dstate.ctx_len);
                let precompute_res = self.precompute_ctx_kv(
                    dstate.ctx_hidden_acc,
                    0,
                    dstate.ctx_len,
                    abs_start,
                    *slot_mapping,
                    ctx,
                    _stream,
                );
                // Restore env var.
                match prev_commit {
                    Some(v) => unsafe {
                        std::env::set_var("ATLAS_DFLASH_PRECOMPUTE_COMMIT", v);
                    },
                    None => unsafe { std::env::remove_var("ATLAS_DFLASH_PRECOMPUTE_COMMIT") },
                }
                precompute_res?;
            }
            dstate.ctx_count_drafter = dstate.ctx_len;
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
        // Phase 2.5e scaffolding: K=γ verify path is implemented in model.rs
        // (decode_verify_graphed_kgamma) and dispatched via step_verify_dflash
        // when drafts.len()>=4. However, per-step output corruption (output
        // starts correct then degenerates to gibberish at K=5) indicates an
        // SSM state-management mismatch between the generic K=γ path and the
        // hand-tuned K=2/3/4 specializations: the K=N!=2/3/4 fallback writes
        // intermediates differently from the WY-chunkwise kernels, causing
        // partial-accept rollback to land on stale state.
        //
        // Until the SSM intermediate semantics are reconciled (kernel work),
        // cap drafts at 1 → forces scheduler to use step_verify_k2 which
        // produces correct output. Set ATLAS_DFLASH_DRAFT_CAP=N to override
        // (N=γ to test the K=γ path; N=1 is the safe default).
        let cap: usize = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

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
