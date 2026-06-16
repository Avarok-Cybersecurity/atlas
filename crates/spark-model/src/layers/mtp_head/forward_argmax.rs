// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-token selection (step 13) for `MtpHead::forward_one`.
//!
//! Hoisted from `forward.rs` to keep that file under the 500 LoC cap.
//! [`MtpHead::argmax_draft_token`] mirrors the original step-13 argmax
//! block 1:1 — same grammar-masked CPU argmax path, same GPU argmax
//! fast path, same `embed_from_argmax` staging and deferred readback.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::MtpHead;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MtpHead {
    /// Select the draft token from `logits` (vocab size `v`).
    ///
    /// When `grammar_bitmask` is `Some`, runs a grammar-masked CPU argmax;
    /// otherwise a GPU argmax. When `draft_embed_target` is `Some`, the
    /// chosen token's embedding is staged on GPU and `0` is returned as a
    /// placeholder (caller reads the real id later); otherwise the token id
    /// is read back synchronously.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn argmax_draft_token(
        &self,
        logits: DevicePtr,
        v: u32,
        h: u32,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let out_ptr = ctx.buffers.scratch();

        let token_id = if let Some(bitmask) = grammar_bitmask {
            // Grammar-masked CPU argmax path.
            //
            // D2H the full logits vector (BF16), apply the XGrammar bitmask
            // (mask off ⇒ -inf), argmax on CPU. This adds ~200μs per draft
            // vs the GPU argmax, but the unmasked path sees ~0% draft
            // acceptance inside tool-call JSON — a 200μs overhead beats a
            // wasted 13.5ms verify step.
            //
            // We then H2D the chosen token id into `out_ptr` so the
            // downstream `embed_from_argmax` kernel can still gather the
            // embedding from the token table on GPU without a new kernel.
            let vocab = v as usize;
            let mut bf16_buf = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h(logits, &mut bf16_buf)?;

            // BF16 → f32 conversion. BF16 is the upper 16 bits of an f32.
            let mut f32_logits = vec![0.0f32; vocab];
            for i in 0..vocab {
                let lo = 0u16;
                let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                f32_logits[i] = f32::from_bits(((hi as u32) << 16) | (lo as u32));
            }

            // Apply mask: bit `tok` set ⇒ allowed; unset ⇒ -inf.
            let mut any_allowed = false;
            for tok in 0..vocab {
                let word = tok / 32;
                let bit = tok % 32;
                let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
                if allowed {
                    any_allowed = true;
                } else {
                    f32_logits[tok] = f32::NEG_INFINITY;
                }
            }

            // Degenerate case: matcher gave us an empty allowed set. Don't
            // propose a real draft — return 0 (pad) as a sentinel. The
            // verifier almost certainly returns a non-zero target token, the
            // draft gets rejected, and the step falls through to target-only
            // decode. This is safer than re-emitting `last_token`, which
            // could be a special token (e.g. `<|im_end|>`) that the verifier
            // might happen to also pick — duplicating a role-boundary
            // token would poison the model's own context.
            if !any_allowed {
                tracing::warn!(
                    "MTP grammar mask allowed zero tokens at pos {position}; \
                     returning 0 as pad-draft (will be rejected at verify)."
                );
                0u32
            } else {
                // CPU argmax over masked logits.
                let mut best_tok = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in f32_logits.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_tok = i as u32;
                    }
                }

                // If caller wants the embedding staged on GPU, stage the
                // chosen token id into `out_ptr` (4 bytes) and reuse the
                // existing embed_from_argmax kernel — it reads the argmax
                // result from `out_ptr` and gathers the embedding on GPU.
                if let Some(embed_target) = draft_embed_target {
                    let tok_bytes = best_tok.to_le_bytes();
                    ctx.gpu.copy_h2d(&tok_bytes, out_ptr)?;
                    ops::embed_from_argmax(
                        ctx.gpu,
                        self.embed_from_argmax_k,
                        out_ptr,
                        self.embed_tokens.weight,
                        embed_target,
                        self.draft_token_id_dev,
                        h,
                        stream,
                    )?;
                }
                best_tok
            }
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            if let Some(embed_target) = draft_embed_target {
                // GPU-side embedding: write draft embedding to verify input buffer
                // and token ID to deferred readback buffer. No D2H sync needed.
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_target,
                    self.draft_token_id_dev,
                    h,
                    stream,
                )?;
                // Return 0 as placeholder — caller reads actual ID later via
                // read_deferred_draft_token().
                0u32
            } else {
                // Fallback: synchronous D2H readback.
                let mut buf = [0u8; 4];
                ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
                u32::from_le_bytes(buf)
            }
        };

        Ok(token_id)
    }
}
