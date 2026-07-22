// SPDX-License-Identifier: AGPL-3.0-only

//! Batched MTP drafter context prefill (ATLAS_MTP_DRAFTER_PREFILL).
//!
//! Why this exists: without it the drafter's KV cache starts EMPTY at decode —
//! the drafter is blind to the prompt through its own attention, and measured
//! per-position acceptance DEGRADES with context (pos0 77→73%, pos1 45→40% at
//! 50→2449-token prompts) while vLLM's prompt-prefilled MTP drafter IMPROVES
//! (83→91% pos0). This pass writes one drafter KV entry per prompt position
//! before the first propose(), mirroring vLLM's proposer prefill.
//!
//! Key structural fact that keeps this cheap: the drafter is a SINGLE decoder
//! layer, so its K/V at position i are pure functions of its input pair
//! `x_i = fc(concat(norm(embed(t_{i+1})), norm(target_hidden_i)))` — they do
//! not depend on the drafter's own attention outputs. The prefill therefore
//! needs NO attention pass: embed-gather → norms → concat → fc →
//! input_layernorm → k/v projections → k_norm → RoPE → reshape_and_cache,
//! all batched over [`PREFILL_CHUNK`]-row chunks with existing kernels.
//!
//! Row/position convention (matches `forward_one` exactly): drafter row i
//! pairs `embed(t_{i+1})` with `hidden_i`, RoPE position `i+1`, KV slot `i`,
//! for i = 0..P-2. The first decode propose() then appends
//! `(first_sampled_token, hidden_{P-1})` at position P, slot P-1 — gapless.
//!
//! v1 scope (explicit): BF16 MTP head (`--mtp-quantization bf16`, the
//! recommended/highest-acceptance config) with BF16 drafter KV. Other quants
//! warn once and no-op — propose() then behaves exactly as before.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{MtpHead, MtpProposerState, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::speculative::ProposerState;

/// Rows processed per batched pass. Sizes the dedicated scratch (~50 MB at
/// h=5120 / nq=32 / hd=256); one full weight read per chunk per projection.
pub(crate) const PREFILL_CHUNK: usize = 512;

impl MtpHead {
    /// Batch-prefill the drafter KV over `prompt_tokens` using per-position
    /// target hiddens. Returns rows written (P-1), or 0 when unsupported /
    /// already prefilled. See module docs for the exact row convention.
    pub(crate) fn prefill_drafter_impl(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.drafter_rows_impl(prompt_tokens, hiddens, 0, 1, state, ctx, stream)
    }

    /// Row/position generalization of the drafter prefill: appends drafter
    /// rows at KV slots `row_base ..` with RoPE positions `pos_base ..`
    /// (row r = pair `(embed(prompt_tokens[r+1]), hiddens row r)`, RoPE
    /// `pos_base + r` = its sequence pair key + 1). Slots and positions are
    /// decoupled because without drafter prefill the row space is COMPACTED
    /// (slots dense, RoPE sequence-space with gaps — matching `forward_one`).
    /// `row_base = 0, pos_base = 1` is the classic whole-prompt prefill;
    /// the catch-up feed (ATLAS_MTP_CATCHUP) appends at `row_base = seq_len`
    /// with the fed pairs' true sequence positions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn drafter_rows_impl(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let mtp_state = match state.as_any_mut().downcast_mut::<MtpProposerState>() {
            Some(s) => s,
            None => return Ok(0),
        };
        // Rows must append exactly at the drafter's current length: for the
        // classic prefill that means a fresh drafter (seq_len == 0); for the
        // catch-up feed, row_base == seq_len. Anything else would leave a
        // hole or overwrite live rows.
        if mtp_state.seq_len != row_base || prompt_tokens.len() < 2 {
            return Ok(0);
        }
        let scratch = match self.prefill_scratch.as_ref() {
            Some(s) => s,
            None => return Ok(0),
        };
        // v1: BF16 head + BF16 drafter KV only (see module docs).
        let (fc_w, k_w, v_w) = match (&self.fc, &self.k_proj, &self.v_proj) {
            (ProjectionWeight::Bf16(fc), ProjectionWeight::Bf16(k), ProjectionWeight::Bf16(v))
                if self.kv_bf16 && self.dense_gemm_k.0 != 0 =>
            {
                (fc, k, v)
            }
            _ => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        "ATLAS_MTP_DRAFTER_PREFILL: drafter prefill supports the BF16 \
                         MTP head (--mtp-quantization bf16) with BF16 KV only; \
                         continuing WITHOUT drafter context prefill."
                    );
                });
                return Ok(0);
            }
        };

        let t0 = std::time::Instant::now();
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let kv_dim = (nkv * hd) as usize;
        let bf16 = 2usize;
        let rows_total = prompt_tokens.len() - 1; // pairs (t_{i+1}, h_i), i=0..P-2

        // Grow the drafter block table to cover all prefill slots up front.
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (row_base + rows_total - 1) / bs + 1;
        while mtp_state.block_table.len() < blocks_needed {
            mtp_state.block_table.push(kv_cache.alloc_block()?);
        }

        let mut done = 0usize;
        while done < rows_total {
            let c = (rows_total - done).min(PREFILL_CHUNK);

            // 1. Gather embeddings of t_{i+1} for rows done..done+c.
            for r in 0..c {
                let tok = prompt_tokens[done + r + 1] as usize;
                self_copy_embed_row(self, ctx, tok, scratch.embed, r, h, stream)?;
            }

            // 2. Pre-fc norms: embedding rows and target-hidden rows.
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                scratch.embed,
                &self.pre_fc_norm_embedding,
                scratch.normed_embed,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hiddens.offset(done * h * bf16),
                &self.pre_fc_norm_hidden,
                scratch.normed_hidden,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;

            // 3. Per-row concat [normed_embed | normed_hidden] → [c, 2h].
            for r in 0..c {
                ops::bf16_concat(
                    ctx.gpu,
                    self.bf16_concat_k,
                    scratch.normed_embed.offset(r * h * bf16),
                    scratch.normed_hidden.offset(r * h * bf16),
                    scratch.concat.offset(r * 2 * h * bf16),
                    h as u32,
                    stream,
                )?;
            }

            // 4. fc: [c, 2h] → [c, h], then input layernorm.
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                scratch.concat,
                fc_w,
                scratch.fc_out,
                c as u32,
                h as u32,
                (2 * h) as u32,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                scratch.fc_out,
                &self.input_layernorm,
                scratch.normed2,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;

            // 5. K/V projections (Q is not needed — outputs are discarded).
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                scratch.normed2,
                k_w,
                scratch.k_out,
                c as u32,
                nkv * hd,
                h as u32,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                scratch.normed2,
                v_w,
                scratch.v_out,
                c as u32,
                nkv * hd,
                h as u32,
                stream,
            )?;
            if !self.k_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    scratch.k_out,
                    &self.k_norm,
                    scratch.k_out,
                    c as u32 * nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }

            // 6. RoPE positions pos_base+r and KV slots row_base+r, uploaded
            //    per chunk (decoupled — see fn docs).
            let positions: Vec<u32> = (0..c).map(|r| (pos_base + done + r) as u32).collect();
            let pos_bytes =
                unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, c * 4) };
            ctx.gpu.copy_h2d_async(pos_bytes, scratch.pos_dev, stream)?;
            let slots: Vec<i64> = (0..c)
                .map(|r| {
                    let i = row_base + done + r;
                    (mtp_state.block_table[i / bs] as i64) * (bs as i64) + (i % bs) as i64
                })
                .collect();
            let slot_bytes =
                unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, c * 8) };
            ctx.gpu
                .copy_h2d_async(slot_bytes, scratch.slot_dev, stream)?;

            ops::rope(
                ctx.gpu,
                self.rope_k,
                scratch.q_scratch,
                scratch.k_out,
                scratch.pos_dev,
                c as u32,
                nq,
                nkv,
                hd,
                ctx.config.rotary_dim() as u32,
                ctx.config.rope_theta as f32,
                stream,
            )?;

            // 7. Write K/V into the drafter's paged BF16 cache.
            ops::reshape_and_cache(
                ctx.gpu,
                self.reshape_cache_k,
                scratch.k_out,
                scratch.v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                scratch.slot_dev,
                c as u32,
                nkv,
                hd,
                bs as u32,
                kv_dim as u32,
                kv_dim as u32,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            // The async H2D sources (positions/slots) are Vec-backed; the
            // driver has queued them, but sync before drop for safety —
            // one sync per 512-row chunk is negligible next to the GEMMs.
            ctx.gpu.synchronize(stream)?;

            done += c;
        }

        mtp_state.seq_len = row_base + rows_total;
        // Last fed row has RoPE pos_base + rows_total - 1 = pair key + 1.
        mtp_state.last_pair_key = Some(pos_base + rows_total - 2);
        tracing::info!(
            "MTP drafter prefill: {} positions ({} prompt tokens) in {:.1} ms",
            rows_total,
            prompt_tokens.len(),
            t0.elapsed().as_secs_f64() * 1e3,
        );
        Ok(rows_total)
    }
}

/// Copy one embedding-table row into `dst` row `r`. Free function to keep the
/// borrow of `self` fields inside the loop simple.
fn self_copy_embed_row(
    head: &MtpHead,
    ctx: &ForwardContext,
    token: usize,
    dst: DevicePtr,
    r: usize,
    h: usize,
    stream: u64,
) -> Result<()> {
    let row_bytes = h * 2;
    let src = head.embed_tokens.weight.offset(token * row_bytes);
    ctx.gpu
        .copy_d2d_async(src, dst.offset(r * row_bytes), row_bytes, stream)
}
