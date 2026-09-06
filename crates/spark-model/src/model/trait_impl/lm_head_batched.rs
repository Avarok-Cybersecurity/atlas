// SPDX-License-Identifier: AGPL-3.0-only

//! The decode LM head, batched — ONE source of truth for two call sites.
//!
//! `decode_a2.rs` (the pure-decode batch) and `decode_b2.rs`
//! (`mixed_final_norm_lm_head`, the prefill+decode co-dispatch head reached
//! from `decode_b.rs` via `mixed_forward_dispatch`) both finish a step with
//! RMS-norm then the vocab projection. They had drifted: `decode_a2` grew the
//! full ladder while `decode_b2` still looped `padded_n` times through
//! `ops::w4a16_gemv`, re-reading the whole vocab weight once per row —
//! ~N x 254 MB/step on live continuous-batching traffic.
//!
//! Credit for spotting the site: @rsafier in #332, which fixed it with a bare
//! default-OFF `batch16`. This lifts `decode_a2`'s ladder instead, so the two
//! heads cannot diverge NUMERICALLY at the same batch width — two
//! independently-maintained ladders would be a second source of truth for
//! which kernel a given `padded_n` lands on, and the first thing to go wrong
//! would be a silent accuracy difference between the pure-decode and
//! co-dispatch paths.
//!
//! HONESTY: this is a bandwidth-accounting argument, not a measurement. The
//! mixed path has never been A/B'd. Any throughput claim for it must be gated
//! spec-OFF or on reversed-order pairs (a single spec-ON pair drifts +/-2%).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layers::ops;

/// Batched-GEMV decode lm_head: **ON by default**, disabled by
/// `ATLAS_NO_LM_HEAD_BATCH_GEMV=1`.
///
/// Strict `== "1"` on an `ATLAS_NO_*` name, not a presence check — presence
/// flags in this codebase are ENABLED by `=0`. Read once; this is a per-step
/// site.
pub(super) fn lm_head_batch_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_LM_HEAD_BATCH_GEMV").as_deref() != Ok("1"))
}

impl TransformerModel {
    /// Project `normed` [padded_n, H] into `logits` [padded_n, V].
    ///
    /// `v` is read from `self.config.vocab_size` rather than passed: it is the
    /// same number at both call sites and a parameter would be a second place
    /// for it to be wrong.
    pub(super) fn lm_head_project_batched(
        &self,
        normed: DevicePtr,
        padded_n: usize,
        h: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        if let Some(ref exl3) = self.lm_head_exl3 {
            // Native EXL3 head (LEADING arm, mirrors `lm_head_batched`).
            // Cooperative launches — this path never runs under graph capture
            // (installing the head vetoes decode graphs at both decision
            // points; see model/lm_head_exl3.rs module docs).
            self.lm_head_exl3_project(exl3, normed, padded_n, logits, stream)?;
            return Ok(logits);
        }
        if let Some(ref fp8) = self.lm_head_fp8 {
            for i in 0..padded_n {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    normed.offset(i * h * bf16),
                    fp8,
                    logits.offset(i * v * bf16),
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // Batched GEMV for the decode head. The base M64-tile
            // `w4a16_gemm` below wastes most of its MMA tile here: at
            // padded_n=16 only 16 of 64 tile-rows carry data, and the same
            // nsys note the verify path records (impl_a3.rs) measured it at
            // 19.3 ms on this [248320, 5120] NVFP4 head vs ~2.5 ms for the
            // batched GEMV streaming the same 636 MB once. That cost is
            // FLAT in n, so it sits in the fixed term at every batch size.
            //
            // Tier by padded_n exactly as the SSM mixer does: batch4 (M<=4)
            // / batch8 (M<=8) / batch16 (M<=16). A 0-handle on any tier
            // falls through to the GEMM, so targets lacking the kernel are
            // unaffected.
            // Tile GEMM at padded_n >= 5 over the PADDED transposed twin.
            // padded_n <= 4 stays on the GEMV, which measures 3174 us =
            // 226 GB/s = 98.3% of the memory roofline on this shape and is
            // therefore unimprovable; the tile GEMM LOSES there.
            if padded_n >= 5
                && self.w4a16_gemm_t_bf16_kernel.0 != 0
                && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
            {
                // LOSSLESS path: BF16 MMA, no activation downcast.
                ops::w4a16_gemm_n128_m128_bf16_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_bf16_kernel,
                    normed,
                    nvfp4_t,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    ldb,
                    stream,
                )?;
            } else if padded_n >= 5
                && self.w4a16_gemm_t_kernel.0 != 0
                && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
            {
                ops::w4a16_gemm_n128_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_kernel,
                    normed,
                    nvfp4_t,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    ldb,
                    stream,
                )?;
            } else {
                let narrow = self.w4a16_batchm.kernel(padded_n as u32);
                let gemv_k = if narrow.0 != 0 {
                    narrow
                } else if padded_n <= 16 {
                    self.w4a16_gemv_batch16_kernel
                } else {
                    spark_runtime::gpu::KernelHandle(0)
                };
                if gemv_k.0 != 0 && lm_head_batch_gemv_enabled() {
                    ops::w4a16_gemv_batchm(
                        self.gpu.as_ref(),
                        gemv_k,
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        self.gpu.as_ref(),
                        self.w4a16_gemm_kernel,
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                normed,
                &self.lm_head_weight,
                logits,
                padded_n as u32,
                v as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(logits)
    }
}
