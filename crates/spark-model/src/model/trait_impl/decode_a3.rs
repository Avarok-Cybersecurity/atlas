// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! `TransformerModel::decode_forward_body` — hoisted from `decode_a.rs`
//! to keep that file under the 500 LoC cap (same move as `decode_a2.rs`).
//! The body is unchanged: per-layer decode + periodic SSM state
//! normalization + final RMS norm + LM head, run either eagerly or
//! inside a CUDA-graph capture region.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, TransformerLayer};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// Single-token decode forward body: per-layer decode + periodic SSM
    /// state normalization + final RMS norm + LM head.
    ///
    /// Runs once per decode step — either eagerly or inside a CUDA graph
    /// capture region — and a SECOND time as the eager fallback when
    /// `end_capture` fails (capture records without executing, so a re-run
    /// performs the step exactly once). Everything here is stream-ordered on
    /// `stream`; the only host syncs are gated on `!use_graphs` /
    /// `probe_layers` (both false under capture).
    pub(super) fn decode_forward_body(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        probe_layers: bool,
        use_graphs: bool,
        stream: u64,
    ) -> Result<()> {
        for (i, layer) in self.layers.iter().enumerate() {
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
            // CBD per-layer hidden fingerprint at decode step 0 (eager only).
            // Localizes the FIRST layer whose post-layer hidden diverges
            // cold-vs-ON / ON-vs-ON → pins the bug to that layer's read set.
            if probe_layers {
                self.gpu.synchronize(stream).ok();
                let mut hb = vec![0u8; self.config.hidden_size * 2];
                if self.gpu.copy_d2h(hidden, &mut hb).is_ok() {
                    let mut s = 0f64;
                    for c in hb.chunks_exact(2) {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        let v = f32::from_bits((bits as u32) << 16) as f64;
                        if v.is_finite() {
                            s += v.abs();
                        }
                    }
                    tracing::warn!("ATLAS_LAYER_H[step0] L{i} hidden_sabs={s:.6}");
                }
            }
            // Per-layer hidden fingerprint under the probe env (eager only —
            // a stream sync is illegal inside a capture region). Pairs with the
            // identical call in `verify_hc_rows` so the FIRST layer whose
            // post-layer hidden differs between decode() and prefill() is
            // named, instead of only the logits at the end.
            if !use_graphs {
                self.hidden_probe_layer("decode", i, 0, hidden, stream);
            }
            // DFlash 5-layer hidden capture (no-op when proposer is not DFlash).
            // Single-token decode: row 0 of `hidden_states()` holds the post-layer
            // activation. Cheap d2d when the layer index matches; otherwise a
            // hashmap-free position() probe over a 5-element vec.
            self.try_dflash_capture(i, 0, stream)?;
        }
        // MLA absorbed attention: defensive sync before final norm in eager
        // mode. Skipped under graph capture because cuStreamSynchronize is
        // illegal inside a capture region (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
        // status 900). The sync is redundant when all kernels run on the same
        // stream — they are already sequenced — so the removal is safe for
        // both eager (retains sync as paranoia) and graph mode.
        if self.config.kv_lora_rank > 0 && !use_graphs {
            self.gpu.synchronize(stream)?;
        }

        // Periodic SSM state normalization during decode.
        // Mamba-2 has no per-token gate clamping (unlike GDN), so state can drift
        // from accumulated BF16 input truncation. Normalize every 64 tokens.
        if self.config.mamba_num_heads > 0
            && seq.seq_len > 0
            && seq.seq_len.is_multiple_of(64)
            && let Err(e) = self.normalize_ssm_states(seq, stream)
        {
            tracing::warn!("Periodic SSM state normalization failed: {e:#}");
        }

        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        // ATLAS_LOGIT_PROBE=1: the DECODE side of the hidden-state A/B against
        // the K-row verify. Taken pre-norm so it isolates the layer stack.
        self.hidden_probe("decode", 0, hidden, stream);
        self.final_norm_apply(hidden, normed, 1, h, eps, stream)?;

        // LM head reads from normed directly (no D2D copy needed)
        self.lm_head(normed, stream)?;
        // ATLAS_LOGIT_PROBE=1: the DECODE side of the row-by-row A/B against
        // the K-row verify (`verify_hc.rs`). `decode_logits_ptr` /
        // `decode_logits_fp32` because this path may write the FP32 scratch.
        self.logit_probe(
            "decode",
            0,
            self.decode_logits_ptr(),
            self.decode_logits_fp32(),
            stream,
        );
        Ok(())
    }
}
