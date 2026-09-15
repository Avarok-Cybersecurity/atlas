// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Final norm + LM head pass for `mixed_forward_dispatch`.
//!
//! Hoisted from `decode_b.rs` to keep that file under the 500 LoC cap.
//! Single helper `mixed_final_norm_lm_head` runs the post-layer-loop
//! reductions: RMS norm + per-position LM-head GEMV for the decode batch,
//! plus a single GEMV for the prefill last-token logits when `is_last`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layers::ops;

/// Returns `(decode_logits, prefill_logits)`. `prefill_logits` is
/// `DevicePtr::NULL` when `prefill_is_last == false`.
pub(super) struct MixedHeadOut {
    pub decode_logits: DevicePtr,
    pub prefill_logits: DevicePtr,
}

impl TransformerModel {
    pub(super) fn mixed_final_norm_lm_head(
        &self,
        hidden: DevicePtr,
        prefill_hidden: DevicePtr,
        padded_n: usize,
        proc_count: usize,
        prefill_is_last: bool,
        h: usize,
        bf16: usize,
        fp32: usize,
        stream: u64,
    ) -> Result<MixedHeadOut> {
        // ── 7. Final norm + LM head ──
        //
        // Decode: RMS norm on N tokens → N logits
        // Prefill: RMS norm on last token only (if is_last) → 1 logit
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;

        // 7a. Decode logits: norm [padded_n, H] → GEMV × padded_n
        self.final_norm_apply(hidden, normed, padded_n as u32, h as u32, eps, stream)?;

        // The SAME ladder the pure-decode head uses (`lm_head_batched.rs`).
        //
        // This loop used to run `padded_n` separate `w4a16_gemv` calls, each
        // streaming the entire vocab weight — ~N x 254 MB/step on a
        // [248320, 5120] NVFP4 head, on live continuous-batching traffic,
        // while `decode_a2` had had the batched ladder for weeks. Sharing one
        // function is what keeps the two heads from picking different kernels
        // (and therefore different numerics) at the same `padded_n`.
        //
        // Site credit: @rsafier, #332.
        let logits = self.lm_head_project_batched(normed, padded_n, h, bf16, stream)?;
        let v = self.config.vocab_size;
        let decode_logits = logits;

        // 7b. Prefill logits: norm last token → 1 GEMV (if is_last)
        let prefill_logits = if prefill_is_last {
            let last_hidden = prefill_hidden.offset((proc_count - 1) * h * fp32);
            // Place prefill normed output after decode normed to avoid overlap
            let prefill_normed = normed.offset(padded_n * h * bf16);
            self.final_norm_apply(last_hidden, prefill_normed, 1, h as u32, eps, stream)?;

            // Place prefill logits after decode logits
            let prefill_logits_ptr = logits.offset(padded_n * v * bf16);
            if let Some(ref exl3) = self.lm_head_exl3 {
                // Native EXL3 head (LEADING arm). Destination is logits row
                // `padded_n` — disjoint from the decode rows 0..padded_n the
                // batched projection above used, so the row-keyed rotation
                // scratch is disjoint too (and both run on `stream` anyway).
                self.lm_head_exl3_project(exl3, prefill_normed, 1, prefill_logits_ptr, stream)?;
            } else if let Some(ref fp8) = self.lm_head_fp8 {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    prefill_normed,
                    fp8,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                ops::w4a16_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    prefill_normed,
                    nvfp4,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    prefill_normed,
                    &self.lm_head_weight,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
            prefill_logits_ptr
        } else {
            DevicePtr::NULL
        };

        Ok(MixedHeadOut {
            decode_logits,
            prefill_logits,
        })
    }
}
