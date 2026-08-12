// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H MTP (Multi-Token-Prediction) draft proposer.
//!
//! Implements [`DraftProposer`] over the `NemotronMtpModule` loaded by
//! `weight_loader::nemotron::load_nemotron_mtp_module` (Nemotron-3.5
//! Lightning's DeepSeek-style 1-step head under `mtp.layers.*`). The Qwen
//! [`crate::layers::MtpHead`] cannot serve this family — its forward is
//! hard-wired to gated attention (interleaved Q+gate), q/k norms, SwiGLU
//! experts and softmax routing, none of which Nemotron has. Instead this
//! proposer follows the DeepSeek-V4 pattern: it delegates each block to the
//! SAME `TransformerLayer::decode` implementations the backbone runs
//! (`Qwen3AttentionLayer` ungated/NoPE + `NemotronMoeLayer` relu²/sigmoid
//! router), and only hand-rolls the MTP-specific pieces.
//!
//! Forward (`propose()`, K-1 drafts chained autoregressively):
//!
//! ```text
//!   e       = embed_tokens[last_token]                     // [H] BF16
//!   x       = eh_proj · cat(rms(e, enorm), rms(hidden, hnorm))   // combiner
//!   x       = attn_block.decode(x)     // norm + ungated GQA (NoPE) + residual
//!   x       = moe_block.decode(x)      // norm + relu² MoE (sigmoid+bias) + residual
//!   logits  = lm_head_nvfp4 · rms(x, final_layernorm)
//!   draft   = argmax(logits)           // grammar-masked when Some
//! ```
//!
//! ## Separate KV cache + distinct metadata offset
//!
//! The MTP attention writes its OWN single-layer BF16 GQA [`PagedKvCache`]
//! (`attn_layer_idx = 0` at load), never the target's. BF16 KV deliberately:
//! the Qwen MTP head's FP8 path with unit scales collapsed drafts to a
//! constant. Attention metadata is uploaded at `scratch() + MTP_META_OFFSET`
//! — the same slab layout and offset as `MtpHead` / `DeepseekV4MtpHead`, via
//! the shared packer — and threaded through a derived [`ForwardContext`].

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, TransformerLayer};
use crate::layers::deepseek_v4_mtp::argmax_grammar_masked;
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::nemotron::NemotronMtpModule;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Per-sequence state for the Nemotron MTP proposer.
pub struct NemotronMtpProposerState {
    /// Block table for the MTP module's OWN KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in the MTP KV cache (compacted row space:
    /// accepted pairs only — no drafter prompt-prefill on this family).
    pub seq_len: usize,
    /// Drafts produced by the last `propose()` (for `after_verify` trimming).
    pub last_num_drafted: usize,
    /// Attention block state (`EmptyLayerState` — KV lives in the cache).
    pub attn_state: Box<dyn LayerState>,
    /// MoE block state (`EmptyLayerState`).
    pub moe_state: Box<dyn LayerState>,
}

impl ProposerState for NemotronMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Nemotron-H MTP draft proposer.
pub struct NemotronMtpHead {
    /// The loaded MTP module: combiner + attention block + MoE block + norm.
    module: NemotronMtpModule,
    /// Shared token embedding table (BF16), from the target model.
    embed_tokens: DenseWeight,
    /// Shared NVFP4 LM head (the Lightning checkpoint ships it prepacked).
    /// Drafts are re-verified by the target, so the draft head only affects
    /// acceptance, never an accepted token.
    lm_head_nvfp4: QuantizedWeight,
    /// Reduced vocab size for the draft LM-head GEMV (0 = full vocab).
    mtp_vocab_size: u32,
    /// Single-layer BF16 GQA KV cache for the MTP attention block.
    kv_cache: Mutex<PagedKvCache>,

    // Kernel handles. `("norm","rms_norm")` resolves to the nemotron kernel
    // target's ABSOLUTE-formula override (x·w/rms) — the same handle the
    // backbone layers use on this family, matching its vanilla-stored norms.
    rms_norm_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl NemotronMtpHead {
    /// Build the proposer from a loaded module + the shared embedding and
    /// NVFP4 LM head.
    pub fn new(
        module: NemotronMtpModule,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Single-layer GQA cache matching the target's attention shape
        // (2 KV heads × 128). The attention block was built with
        // `attn_layer_idx = 0`, so a 1-layer pool is exactly what it indexes.
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
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
            lm_head_nvfp4,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    /// One MTP draft step. Returns the drafted token id.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut NemotronMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h as usize * 2;

        // Scratch note: the Qwen MtpHead's `ssm_gates`/`ssm_ba` picks are
        // GDN-SIZED — on a Mamba-2 config (`linear_*` fields all 0) those
        // buffers collapse to the 256-byte floor. Use scratch that is
        // guaranteed row-capacity on this family instead: `ssm_qkvz` /
        // `ssm_deinterleaved` (sized from mamba2 in_proj / d_xBC ≥ H) and the
        // attention buffers (M-row, consumed only AFTER the combiner is done).

        // ── 1. Embed last token (D2D gather from the shared table) ──
        let embed_out = ctx.buffers.ssm_qkvz();
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // ── 2. Combiner: eh_proj · cat(rms(e,enorm), rms(hidden,hnorm)) ──
        // Concat order is embed-first (DeepSeek-V3 convention, which the
        // Nemotron MTP module copies wholesale, tensor names included).
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
        // [H] — qkv_output is [M, q+2kv(+gate)] BF16; row capacity ≥ H and the
        // attention block only claims it after the combiner consumed this.
        let normed_hidden = ctx.buffers.qkv_output();
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
        // [2H] — attn_output is [M, nq*hd] BF16 (M rows contiguous, so the
        // first 2H elements are always in-bounds); free until the attention
        // block's own decode, by which time eh_proj has consumed this.
        let concat_out = ctx.buffers.attn_output();
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            normed_embed,
            normed_hidden,
            concat_out,
            h,
            stream,
        )?;
        // hidden = the drafter's residual stream from here on.
        let hidden = ctx.buffers.hidden_states();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            concat_out,
            &self.module.eh_proj,
            hidden,
            h,
            h * 2,
            stream,
        )?;

        // ── 3. MTP attention metadata + KV block allocation ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            actual_seq_len,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
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
            seq_slot: DevicePtr(0),
            // MTP is a non-batched path: null => the MoE fold hooks fall back
            // to the request-granularity `moe_route_gate`.
            moe_row_adapter: DevicePtr(0),
        };

        // Derive a ForwardContext carrying the MTP metadata. Graph capture is
        // forced off (host-built metadata + H2D uploads are illegal under
        // capture); comm = None (the draft runs on rank 0 only).
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            routed_lora_layers: None,
            midchunk_capture: None,
            // Inherit the owning request's MoE-LoRA fold decision; the
            // drafter's own MoE layer carries no adapter, so the fold hooks
            // short-circuit either way.
            moe_lora_route: ctx.moe_lora_route,
        };

        // ── 4. Attention block: norm + ungated GQA (NoPE) + residual ──
        // `TransformerLayer::decode` reads and writes `hidden` in place; the
        // proposer's own 1-layer cache + block table stand in for the
        // target's. Disk-offload vectors are empty (no HSS on the drafter).
        let residual = ctx.buffers.residual();
        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        self.module.attn_layer.decode(
            hidden,
            residual,
            state.attn_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        // ── 5. MoE block: norm + sigmoid-routed relu² MoE + residual ──
        // The MoE block never touches the KV cache; hand it the proposer's
        // cache lock-free scratch (it only needs the trait signature).
        let mut moe_kv = self.kv_cache.lock();
        self.module.moe_layer.decode(
            hidden,
            residual,
            state.moe_state.as_mut(),
            &mut moe_kv,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(moe_kv);

        // ── 6. Final norm + shared NVFP4 LM head → logits ──
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.module.final_norm,
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
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            final_normed,
            &self.lm_head_nvfp4,
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

impl DraftProposer for NemotronMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(NemotronMtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            attn_state: self.module.attn_layer.alloc_state(gpu)?,
            moe_state: self.module.moe_layer.alloc_state(gpu)?,
        }))
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
        let nem_state = state
            .as_any_mut()
            .downcast_mut::<NemotronMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Nemotron MTP proposer state"))?;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;
        for i in 0..num_drafts {
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "Nemotron MTP grammar-masked drafting with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                nem_state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            tracing::debug!(
                "Nemotron MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                nem_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // Subsequent drafts feed on the MTP head's own residual stream
            // (pre-final-norm hidden left in `hidden_states()` by step 5).
            current_hidden = ctx.buffers.hidden_states();
        }
        nem_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let nem_state = state
            .as_any_mut()
            .downcast_mut::<NemotronMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Nemotron MTP proposer state"))?;
        // Roll back the rejected drafts' KV rows (slots are overwritten on
        // the next propose). Mirrors `MtpHead::after_verify`.
        let num_drafted = nem_state.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        let old_sl = nem_state.seq_len;
        if num_to_trim > 0 {
            nem_state.seq_len = nem_state.seq_len.saturating_sub(num_to_trim);
        }
        tracing::debug!(
            "Nemotron MTP after_verify: accepted={num_accepted} drafted={num_drafted} \
             trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            nem_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let nem_state = state
            .as_any_mut()
            .downcast_mut::<NemotronMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Nemotron MTP proposer state"))?;
        if !nem_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&nem_state.block_table);
            nem_state.block_table.clear();
        }
        nem_state.seq_len = 0;
        Ok(())
    }
}
