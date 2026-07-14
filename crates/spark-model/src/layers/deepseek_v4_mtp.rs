// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash Multi-Token-Prediction (MTP) draft proposer.
//!
//! Implements [`DraftProposer`] over the `DeepseekV4MtpModule` loaded by
//! `load_v4_mtp_module`. Unlike the
//! Qwen-shaped [`crate::layers::MtpHead`] (a hand-rolled single attention +
//! MoE block), the V4 MTP module's body is a full reused V4 layer
//! (MLA + manifold-constrained hyper-connections (mHC) + 256-expert NVFP4
//! MoE). The proposer therefore delegates the bulk of the forward to
//! `body.decode()` and only wraps it with the MTP-specific pieces.
//!
//! Forward (`propose()`, K = 1 since `num_nextn_predict_layers == 1`):
//!
//! ```text
//!   embed   = embed_tokens[last_token]                       // [hidden] BF16
//!   h_in    = e_proj · rms_norm(embed,  enorm)
//!           + h_proj · rms_norm(hidden, hnorm)               // combiner
//!   hc_expand(h_in → hc_streams)                             // is_first mHC
//!   body.decode(hc_streams, …, mtp_kv_cache, state.seq_len)  // MIDDLE mHC + MLA + MoE
//!   hc_head(hc_streams → h_out)                              // is_last mHC
//!   logits  = lm_head(rms_norm(h_out, norm))
//!   draft   = argmax(logits)                                 // grammar-masked when Some
//! ```
//!
//! The body was assembled with `layer_idx = num_hidden_layers`, so its
//! `decode_inner_hc` sees `is_first_layer == false` AND `is_last_layer ==
//! false`: it runs the middle mHC mixing (hc_pre → attn → hc_post → hc_pre →
//! ffn → hc_post) reading/writing `hc_streams`, but does NOT call `hc_expand`
//! or `hc_head`. The proposer supplies both ends.
//!
//! ## Separate KV cache + distinct metadata offset
//!
//! The MTP attention writes into its OWN single-layer MLA-shaped
//! [`PagedKvCache`] (num_kv_heads = 1, head_dim = kv_lora_rank +
//! qk_rope_head_dim), never the target's. The V4-Flash decode attention
//! (`attention_forward_v4`) reads positions / slot / seq_len / block_table
//! from `ctx.attn_metadata`, so the proposer uploads MTP-specific metadata to
//! `scratch().offset(MTP_META_OFFSET)` — distinct from the target metadata at
//! `32768` — and threads it through a derived [`ForwardContext`].

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::deepseek_v4::DeepseekV4MtpModule;
use crate::weight_map::DenseWeight;

/// Scratch-buffer byte offset for the MTP attention metadata. Must be distinct
/// from the target model's metadata (`32768`) so a `propose()` call does not
/// clobber the in-flight target `attn_metadata`. Mirrors the Qwen `MtpHead`
/// choice of `49152` (the Qwen head uploads its own packed header there too).
const MTP_META_OFFSET: usize = 49152;

/// Per-sequence state for the DeepSeek-V4 MTP proposer.
pub struct DeepseekV4MtpProposerState {
    /// Block table for the MTP module's OWN KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in the MTP KV cache.
    pub seq_len: usize,
    /// Drafts produced by the last `propose()` (for `after_verify` trimming).
    pub last_num_drafted: usize,
    /// Per-layer state for the reused V4 body. MLA attention layers use
    /// `EmptyLayerState`, but we allocate it via `body.alloc_state` so any
    /// future stateful body type is handled correctly (no hard-coded assumption).
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for DeepseekV4MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// DeepSeek-V4 MTP draft proposer.
pub struct DeepseekV4MtpHead {
    /// The loaded MTP module: reused V4 body + combiner + final norm + hc_head.
    module: DeepseekV4MtpModule,
    /// Shared token embedding table (BF16), from the target model.
    embed_tokens: DenseWeight,
    /// Shared LM head (BF16 — DeepSeek-V4-Flash keeps the head in BF16), from the
    /// target model. Every draft is re-verified by the target's head, so the
    /// draft head only affects acceptance, never an accepted token.
    lm_head: DenseWeight,
    /// Reduced vocab size for the draft LM-head GEMV (0 = full vocab).
    mtp_vocab_size: u32,
    /// Single-layer MLA-shaped KV cache for the MTP attention.
    kv_cache: Mutex<PagedKvCache>,

    // Kernel handles.
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl DeepseekV4MtpHead {
    /// Build the proposer from a loaded `DeepseekV4MtpModule` and the shared
    /// embedding + NVFP4 LM head.
    pub fn new(
        module: DeepseekV4MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        // MTP KV cache: single MLA-absorbed attention layer. Matches the
        // target's MLA cache shape (num_kv_heads = 1, head_dim = kv_lora_rank
        // + qk_rope_head_dim) so `write_kv_cache` / `run_paged_decode` in the
        // reused V4 body land at the correct strides. BF16 (the MTP cache is
        // one tiny layer — BF16 cost is negligible and avoids the FP8 unit-
        // scale collapse seen on the Qwen path).
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        // The MTP body is a single layer, but it was built with
        // `attn_layer_idx = num_hidden_layers` (so its mHC/hash/compressor logic
        // takes the "interior, no-compressor" path), and its decode indexes the
        // KV cache pool at THAT index. So the cache pool must have
        // `num_hidden_layers + 1` layer slots even though only the last is used.
        // The extra slots are tiny (one MLA layer each at this seq len, ~2 MB).
        let num_layers = config.num_hidden_layers + 1;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: mla_cache_dim,
            num_layers,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            // V4 ships HF-vanilla norm weights (enorm/hnorm/norm are loaded
            // exactly) — the offset-from-1 kernel would apply `1 + w`.
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    /// Allocate per-sequence state. Mirrors the body's own `alloc_state` for
    /// the body sub-state.
    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<DeepseekV4MtpProposerState> {
        Ok(DeepseekV4MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    /// One MTP draft step. Returns the drafted token id.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut DeepseekV4MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_mult = ctx.config.hc_mult as u32;
        let row_bytes = h as usize * 2;

        // ── 1. Embed last token (D2D gather from the shared table) ──
        let embed_out = ctx.buffers.ssm_qkvz();
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // ── 2. Combiner: h_in = e_proj·rms_norm(embed,enorm)
        //                       + h_proj·rms_norm(target_hidden,hnorm) ──
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.module.enorm,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        let normed_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.module.hnorm,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        // e_proj / h_proj are square [hidden, hidden] dense BF16. Compute the
        // embedding branch into `h_in`, the hidden branch into a temp, then
        // accumulate (`bf16_residual_add` does h_in += temp in place).
        let h_in = ctx.buffers.hidden_states();
        let h_branch = ctx.buffers.norm_output();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed_embed,
            &self.module.e_proj,
            h_in,
            h,
            h,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed_hidden,
            &self.module.h_proj,
            h_branch,
            h,
            h,
            stream,
        )?;
        ops::residual_add(ctx.gpu, self.residual_add_k, h_in, h_branch, h, stream)?;

        // ── 3. mHC expand: replicate h_in into hc_mult streams (is_first) ──
        let hc_streams = ctx.buffers.hc_streams();
        ops::hc_expand(
            ctx.gpu,
            self.hc_expand_k,
            h_in,
            hc_streams,
            1,
            h,
            hc_mult,
            stream,
        )?;

        // ── 4. Body decode: MIDDLE mHC + MLA attention (writes MTP KV cache)
        //       + MoE. Reads/writes `hc_streams` (hidden is a single-stream
        //       scratch). The body NEVER calls hc_expand/hc_head (layer_idx =
        //       num_hidden_layers ⇒ is_first == is_last == false). ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        // Upload MTP-specific attention metadata at the distinct scratch offset
        // so it does not clobber the target metadata at 32768. Layout mirrors
        // the target's `AttnMetadataDev`: pos(u32)@0, slot(i64)@8,
        // seq_len(i32)@16, block_table(i32[])@256.
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;

        let bt_i32: Vec<i32> = state.block_table.iter().map(|&b| b as i32).collect();
        let bt_len = bt_i32.len() * 4;
        let mut meta_buf = vec![0u8; 256 + bt_len];
        meta_buf[0..4].copy_from_slice(&(position as u32).to_le_bytes());
        meta_buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
        meta_buf[16..20].copy_from_slice(&actual_seq_len.to_le_bytes());
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_len) };
        meta_buf[256..256 + bt_len].copy_from_slice(bt_bytes);
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
        };

        // The body's hash-MoE (if any) reads the decode token id from
        // `token_ids[0]`; upload this draft's input token there. The main
        // decode loop uploaded the target token earlier in the step, so we
        // must overwrite it for the MTP forward (and the main loop re-uploads
        // before the next target step / graph replay).
        if let Some(tid_buf) = ctx.token_ids {
            ctx.gpu
                .copy_h2d_async(&token.to_le_bytes(), tid_buf, stream)?;
        }

        // Derive a ForwardContext carrying the MTP metadata. CUDA-graph capture
        // is forced off for the MTP forward (its block-table / metadata are
        // host-built per call and the H2D uploads above are illegal under
        // capture).
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            // comm = None: the MTP draft runs ONLY on rank 0, so its MoE must NOT
            // issue an EP all-reduce (rank 1 never participates → the collective
            // hangs ~35s then corrupts CUDA). The MTP body is loaded with ALL
            // experts local (force_all_experts), so the no-EP MoE is correct.
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
        };

        // `decode_inner_hc` reads the persistent multi-stream state from
        // `ctx.buffers.hc_streams()` directly (already populated by `hc_expand`
        // above) and uses the `hidden` ARG as a single-stream scratch (hc_pre
        // collapses into it). So `hidden` must be a SEPARATE buffer, NOT
        // `hc_streams` — aliasing them corrupts the persistent state. Reuse
        // `hidden_states()` (= the now-consumed `h_in` scratch).
        let body_scratch = ctx.buffers.hidden_states();
        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        let residual = ctx.buffers.residual();
        self.module.body.decode(
            body_scratch,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        // ── 5. mHC head: collapse hc_mult streams → single h_out (is_last) ──
        let h_out = ctx.buffers.hidden_states();
        if let Some(ref head) = self.module.hc_head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                h_out,
                1,
                h,
                hc_mult,
                eps,
                ctx.config.hc_eps,
                stream,
            )?;
        } else {
            // No mHC (hc_mult == 0): hc_expand was a no-op replicate of 1 ⇒
            // the body left the result in hc_streams' single stream.
            ctx.gpu
                .copy_d2d_async(hc_streams, h_out, row_bytes, stream)?;
        }

        // ── 6. Final norm + shared LM head → logits ──
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            h_out,
            &self.module.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            final_normed,
            &self.lm_head,
            logits,
            v,
            h,
            stream,
        )?;

        // ── 7. Argmax (grammar-masked when a bitmask is supplied) ──
        let out_ptr = ctx.buffers.scratch();
        let token_id = if let Some(bitmask) = grammar_bitmask {
            argmax_grammar_masked(ctx.gpu, logits, v as usize, bitmask, position)?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            u32::from_le_bytes(buf)
        };

        state.seq_len += 1;
        Ok(token_id)
    }
}

/// CPU grammar-masked argmax over the BF16 logits (mirrors `MtpHead`): D2H the
/// logit vector, mask off (→ -inf) tokens the grammar rejects, argmax on CPU.
/// Returns `0` (pad) when the matcher's allowed set is empty so the draft is
/// rejected at verify rather than emitting a possibly-special token.
fn argmax_grammar_masked(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    bitmask: &[i32],
    position: usize,
) -> Result<u32> {
    let mut bf16_buf = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut bf16_buf)?;

    let mut best_tok = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    let mut any_allowed = false;
    for tok in 0..vocab {
        let word = tok / 32;
        let bit = tok % 32;
        let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
        if !allowed {
            continue;
        }
        any_allowed = true;
        // BF16 → f32: BF16 is the upper 16 bits of an f32.
        let hi = u16::from_le_bytes([bf16_buf[2 * tok], bf16_buf[2 * tok + 1]]);
        let val = f32::from_bits((hi as u32) << 16);
        if val > best_val {
            best_val = val;
            best_tok = tok as u32;
        }
    }
    if !any_allowed {
        tracing::warn!(
            "V4 MTP grammar mask allowed zero tokens at pos {position}; \
             returning 0 as pad-draft (will be rejected at verify)."
        );
        return Ok(0);
    }
    Ok(best_tok)
}

impl DraftProposer for DeepseekV4MtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;
        for i in 0..num_drafts {
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "V4 MTP grammar-masked drafting with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                v4_state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            tracing::debug!(
                "V4 MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                v4_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // Subsequent drafts feed on the MTP head's own collapsed hidden.
            current_hidden = ctx.buffers.hidden_states();
        }
        v4_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        // Trim `drafted - accepted` rejected entries from the MTP KV cache by
        // rolling back `seq_len` (the slots are overwritten on the next
        // propose). Mirrors `MtpHead::after_verify`.
        let num_drafted = v4_state.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        let old_sl = v4_state.seq_len;
        if num_to_trim > 0 {
            v4_state.seq_len = v4_state.seq_len.saturating_sub(num_to_trim);
        }
        tracing::debug!(
            "V4 MTP after_verify: accepted={num_accepted} drafted={num_drafted} \
             trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            v4_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        if !v4_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&v4_state.block_table);
            v4_state.block_table.clear();
        }
        v4_state.seq_len = 0;
        Ok(())
    }
}
