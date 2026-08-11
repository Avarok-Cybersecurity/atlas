// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-step diagnostic helper split out of `decode_a.rs` (LoC budget).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// Decode-step diagnostic for Gemma-4 degeneration analysis. Only does work
    /// when `ATLAS_DIAG_GEMMA4=1` (which also disables CUDA graphs upstream, so
    /// the d2h sync here is safe). Reads the top-5 tokens by logit so we can see
    /// whether the LM head produced a near-tie or a confident bad pick.
    /// (B4 — Creative haiku degeneration loop diagnostic.)
    pub(super) fn diag_gemma4_decode_logits(&self, token: u32, stream: u64) -> Result<()> {
        if !std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
            return Ok(());
        }
        self.gpu.synchronize(stream)?;
        let n_logits = self.config.vocab_size;
        // Read the buffer the LM head actually wrote to. With Gemma-4 dense the
        // single-token decode lm_head produces FP32 in `logits_fp32_buf`; the
        // BF16 buffer would be all zeros there.
        let logit_vals: Vec<f32> = if self.use_fp32_logits {
            let mut buf = vec![0u8; n_logits * 4];
            if let Err(e) = self.gpu.copy_d2h(self.logits_fp32_buf, &mut buf) {
                tracing::error!("ATLAS_DIAG_GEMMA4: copy_d2h(logits_fp32_buf): {e:#}");
            }
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            let mut buf = vec![0u8; n_logits * 2];
            if let Err(e) = self.gpu.copy_d2h(self.buffers.logits(), &mut buf) {
                tracing::error!("ATLAS_DIAG_GEMMA4: copy_d2h(logits BF16): {e:#}");
            }
            buf.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        };
        let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
        idx.sort_by(|&a, &b| {
            logit_vals[b]
                .partial_cmp(&logit_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
        tracing::warn!(
            "DIAG decode logits: max={max:.4} min={min:.4} prev_token={token} top5={top5:?}",
        );
        Ok(())
    }
}
