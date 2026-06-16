// SPDX-License-Identifier: AGPL-3.0-only

//! Contiguous-Q/K/V Flash Attention dispatch for the cache-skip prefill path.
//!
//! Extracted verbatim from `cache_skip.rs` to keep that file under the
//! 500-LoC cap. Kernel selection and launch order are unchanged; the result
//! is written into `ctx.buffers.attn_output()`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Contiguous Q/K/V buffers + dims for one prefill chunk's Flash Attention.
pub(super) struct CacheSkipAttnArgs {
    pub q_contiguous: DevicePtr,
    pub k_contiguous: DevicePtr,
    pub v_contiguous: DevicePtr,
    pub attn_out: DevicePtr,
    pub n: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// ── 8. Flash Attention on contiguous Q/K/V (BR=64 for long sequences) ──
    ///
    /// HDIM>256 layers (Gemma-4 long attention) use the scalar reference
    /// kernel (BR=16) and always pass `sliding_window=0`; all others use the
    /// BR=64 kernel honoring `self.sliding_window`.
    pub(super) fn prefill_attention_cache_skip_attn(
        &self,
        ctx: &ForwardContext,
        args: &CacheSkipAttnArgs,
    ) -> Result<()> {
        let CacheSkipAttnArgs {
            q_contiguous,
            k_contiguous,
            v_contiguous,
            attn_out,
            n,
            nq,
            nkv,
            hd,
            stream,
        } = *args;

        let inv_sqrt_d = self.effective_attn_scale(hd);
        if hd > 256 && self.prefill_attn_512_k.0 != 0 {
            // HDIM=512: use scalar reference kernel (BR=16, correct for any head_dim)
            // Full-attention layers (this path) always pass sliding_window=0.
            ops::prefill_attention(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                0,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("prefill_512 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
            })?;
        } else {
            ops::prefill_attention_64(
                ctx.gpu,
                self.prefill_attn_64_k,
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("flash_attn_64 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
            })?;
        }
        Ok(())
    }
}
