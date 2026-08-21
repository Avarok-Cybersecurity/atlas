// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Presence kill switch for the wide (9..=VERIFY_ROW_CAP row) batched LM head arm:
/// `ATLAS_NO_LMHEAD_BATCHED_WIDE` restores the M64-tile `w4a16_gemm` fallback.
/// Presence, not value — `=0` is NOT "off" (see `atlas_env_presence_check_trap`).
fn lmhead_batched_wide_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_LMHEAD_BATCHED_WIDE").is_err())
}

impl TransformerModel {
    pub(super) fn embed(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let row_bytes = h * 2; // BF16 embedding row
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        self.gpu.copy_d2d_async(src, output, row_bytes, stream)?;
        // Scale embeddings (Gemma-4: sqrt(hidden_size))
        self.scale_embeddings(output, 1, stream)
    }

    /// Scale in-place embeddings by config.embed_scale. The residual stream
    /// is always BF16, so this dispatches `embed_scale::bf16_scale_inplace`.
    pub(super) fn scale_embeddings(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        self.scale_embeddings_bf16(data, num_tokens, stream)
    }

    pub(super) fn scale_embeddings_bf16(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), self.embed_scale_kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    /// Wide batched LM head, `num_tokens` in 9..=`VERIFY_ROW_CAP` (96) — exactly
    /// the batched-verify row regime (`can_batch_verify` bounds `Σks` at
    /// `VERIFY_ROW_CAP`; the 32:2 depth-at-width shape's n=32 × k=3 rows hits
    /// 96 dead on).
    /// Below 9 the existing GEMV ladder (batch2/4/8) already owns the dispatch.
    /// When this arm was capped at 32 (pre-2026-07-30), the R=64 verify step
    /// fell through to the base `w4a16_gemm` below at 23.9 ms/step (nsys,
    /// spec32n) vs 3.65 ms for the same 636 MB via this tile GEMM at R=32.
    ///
    /// Why it matters: every row count in that range fell through to the M64-tile
    /// `w4a16_gemm`, whose own comment upstream documents the cost — nsys measured
    /// **19.9 ms per verify step** on the [248320, 5120] NVFP4 head, ~33 GB/s
    /// effective, because 94%+ of the M-tile is padding. The transposed-twin tile
    /// GEMM streams the same 636 MB once at near-roofline. This is the identical
    /// ladder `decode_a2` already runs at `padded_n >= 5`, so the verify head now
    /// uses the SAME kernel as the non-speculative decode head it is standing in
    /// for (it previously did not — a silent numerics divergence between the spec
    /// and non-spec paths at the same batch width).
    ///
    /// Returns `false` when no wide kernel resolves (no NVFP4 head, no twin, no
    /// batch16 handle), in which case the caller keeps today's `w4a16_gemm`.
    fn lm_head_batched_wide(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        let Some(ref nvfp4) = self.lm_head_nvfp4 else {
            return Ok(false);
        };
        if let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t {
            // LOSSLESS path (ATLAS_LMHEAD_LOSSLESS): BF16 MMA, no activation
            // downcast. Mirrors decode_a2 so the two heads never disagree.
            if self.w4a16_gemm_t_bf16_kernel.0 != 0 {
                ops::w4a16_gemm_n128_m128_bf16_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_bf16_kernel,
                    hidden,
                    nvfp4_t,
                    logits,
                    num_tokens,
                    v,
                    h,
                    ldb,
                    stream,
                )?;
                return Ok(true);
            }
            if self.w4a16_gemm_t_kernel.0 != 0 {
                ops::w4a16_gemm_n128_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_kernel,
                    hidden,
                    nvfp4_t,
                    logits,
                    num_tokens,
                    v,
                    h,
                    ldb,
                    stream,
                )?;
                return Ok(true);
            }
        }
        // No twin (ATLAS_NO_LMHEAD_TGEMM): the M<=16 weight-streaming GEMV still
        // beats the M64 tile. M=17..32 has no single-read form, so it keeps the
        // GEMM rather than paying two full weight passes.
        if num_tokens <= 16 && self.w4a16_gemv_batch16_kernel.0 != 0 {
            ops::w4a16_gemv_batchm(
                self.gpu.as_ref(),
                self.w4a16_gemv_batch16_kernel,
                hidden,
                nvfp4,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
            return Ok(true);
        }
        Ok(false)
    }

    /// LM head for K tokens: hidden[K, H] → logits[K, V].
    pub(super) fn lm_head_batched(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        logits_dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        // Caller picks the destination so co-dispatched prefill streams can each
        // write their own logits row (was a single shared buffer = cross-stream
        // aliasing: all streams' first token collapsed to one). Verify/decode
        // callers pass `self.buffers.logits()` (base) — unchanged behaviour.
        let logits = logits_dst;
        if let Some(ref fp8) = self.lm_head_fp8 {
            // FP8 E4M3 LM head. The dual-GEMV (batch=2) reads the FP8 weight
            // once for both K=2 verify tokens — bit-identical to two M=1 GEMVs
            // but halves the full-vocab weight bandwidth. Falls back to the
            // per-token loop for K!=2 or when the kernel is absent.
            self.fp8_vocab_project(fp8, hidden, logits, num_tokens, v, h, stream)?;
        } else if num_tokens == 2 {
            // Double-GEMV: reads weights once, computes 2 outputs.
            // GEMM M=2 with 64×64 tiles wastes 97% of M-dimension → ~3× slower.
            if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                ops::w4a16_gemv_batch2(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_batch2_kernel,
                    hidden,
                    nvfp4,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            } else {
                // Dense fallback: 2× GEMV. Stays BF16 even when
                // use_fp32_logits is on — the FP32 path is decode-only
                // (single-token `lm_head`); batched-decode/prefill keeps
                // BF16 because the bug it fixes only manifests at decode
                // step 1 (first-token argmax tiebreak).
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden,
                    &self.lm_head_weight,
                    logits,
                    v,
                    h,
                    stream,
                )?;
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden.offset(h as usize * 2),
                    &self.lm_head_weight,
                    logits.offset(v as usize * 2),
                    v,
                    h,
                    stream,
                )?;
            }
        } else if (3..=8).contains(&num_tokens)
            && self.w4a16_batchm.kernel(num_tokens).0 != 0
            && let Some(ref nvfp4) = self.lm_head_nvfp4
        {
            // K=3..8 verify lm_head: one weight read for all rows via the
            // narrowest batched-GEMV tier covering the row count. nsys
            // (2026-07-18, drafts=3 serve): the base M64-tile `w4a16_gemm`
            // below cost 19.3 ms/verify-step on the [248320, 5120] NVFP4
            // lm_head at M=4 — 94% of the M-tile is padding, ~33 GB/s
            // effective. The batch GEMV streams the same 636 MB once at
            // near-peak (~2.5 ms), the single largest slice of the K=4
            // verify-vs-K=2 cost gap; the M=5..8 tiers extend that to the
            // chain-verify rows (batchm_bench).
            ops::w4a16_gemv_batchm(
                self.gpu.as_ref(),
                self.w4a16_batchm.kernel(num_tokens),
                hidden,
                nvfp4,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        } else if (9..=super::trait_impl::verify_e2::VERIFY_ROW_CAP as u32).contains(&num_tokens)
            && lmhead_batched_wide_enabled()
            && self.lm_head_batched_wide(hidden, num_tokens, logits, stream)?
        {
            // Handled by the wide arm; nothing launched when it returns false.
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            ops::w4a16_gemm(
                self.gpu.as_ref(),
                self.w4a16_gemm_kernel,
                hidden,
                nvfp4,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        }
        // Feature-2: overlay overridden logit columns AFTER the base projection,
        // BEFORE softcap. Uniform-active route (seq_slot NULL); BF16 logits.
        // No-op when no overlay is installed.
        self.apply_lmhead_overlay(hidden, DevicePtr(0), logits, num_tokens, false, stream)?;
        // Apply logit softcapping: logits = cap * tanh(logits / cap)
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    pub(super) fn lm_head(&self, hidden: DevicePtr, stream: u64) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        // Pick the output buffer: FP32 scratch when use_fp32_logits is on,
        // shared BF16 buffer otherwise. The sampler must use the matching
        // dtype — see `decode_logits_dtype()`.
        let (logits, fp32) = if self.use_fp32_logits {
            (self.logits_fp32_buf, true)
        } else {
            (self.buffers.logits(), false)
        };
        if let Some(ref fp8) = self.lm_head_fp8 {
            // FP8 E4M3 LM head (`--lm-head-dtype fp8`). `w8a16_gemv` has no
            // FP32-output variant — it writes to whichever buffer is passed.
            // `use_fp32_logits` is false in production, so `logits` is the
            // shared BF16 buffer; the FP32-logits path is unused here.
            ops::dense_gemv_fp8w(
                self.gpu.as_ref(),
                self.dense_gemv_fp8w_kernel,
                hidden,
                fp8,
                logits,
                v,
                h,
                stream,
            )?;
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // Pick FP32-output variant when the FP32 logits buffer is the
            // destination. Same packed-NVFP4 weights, same activation, but the
            // accumulator is NOT downcast to BF16 — closes the 0.125-logit
            // BF16-rounding tiebreak flip that triggers Gemma-4-31B's
            // creative-collapse stop-word loop.
            let kernel = if fp32 {
                self.w4a16_gemv_logits_kernel
            } else {
                self.w4a16_gemv_kernel
            };
            ops::w4a16_gemv(
                self.gpu.as_ref(),
                kernel,
                hidden,
                nvfp4,
                logits,
                v,
                h,
                stream,
            )?;
        } else if fp32 {
            // FP32-output dense GEMV: same precision-preservation reason as
            // the NVFP4 variant above. Used when Gemma keeps the LM head
            // as BF16 (skip_lm_head_quantization=true).
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_fp32out_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        }
        // Feature-2: overlay overridden logit columns AFTER the base projection,
        // BEFORE softcap. Single-token; `fp32` selects the f32-logits kernel.
        // No-op when no overlay is installed.
        self.apply_lmhead_overlay(hidden, DevicePtr(0), logits, 1, fp32, stream)?;
        // Apply logit softcapping: logits = cap * tanh(logits / cap)
        if self.logit_softcap_kernel.0 != 0 || self.logit_softcap_fp32_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            self.apply_logit_softcap_dtype(logits, v, cap, fp32, stream)?;
        }
        Ok(logits)
    }

    /// Apply logit softcapping in-place: `logits[i] = cap * tanh(logits[i] / cap)`.
    /// BF16 path. Use `apply_logit_softcap_dtype` to dispatch by buffer dtype.
    pub(super) fn apply_logit_softcap(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), self.logit_softcap_kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// Dtype-aware softcap dispatcher. Picks the BF16 or FP32 kernel based on
    /// whether the buffer holds FP32 logits. No-op when softcap is disabled
    /// (cap == 0). Used by the single-token decode `lm_head` to keep the FP32
    /// path symmetrical when `use_fp32_logits` is on.
    pub(super) fn apply_logit_softcap_dtype(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        is_fp32: bool,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let kernel = if is_fp32 {
            self.logit_softcap_fp32_kernel
        } else {
            self.logit_softcap_kernel
        };
        if kernel.0 == 0 {
            return Ok(());
        }
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// True when single-token decode `lm_head` writes FP32 logits to
    /// `logits_fp32_buf`. Callers that consume those logits (sampler) MUST
    /// read with the matching dtype. Prefill / batched-decode lm_head still
    /// produce BF16, so this only applies to the `lm_head` (single-token)
    /// return value.
    pub fn decode_logits_fp32(&self) -> bool {
        self.use_fp32_logits
    }

    /// Buffer pointer the single-token decode `lm_head` last wrote to. FP32
    /// scratch when `use_fp32_logits`, otherwise the shared BF16 logits
    /// buffer. Callers that previously hard-coded `self.buffers.logits()`
    /// after `self.lm_head(...)` must use this so the sampler reads the
    /// correct buffer dtype (the BF16 buffer is stale/empty in the FP32
    /// path because lm_head writes elsewhere). Pair with
    /// `logits_ptr_is_fp32` / `decode_logits_fp32` for dtype-aware reads.
    pub fn decode_logits_ptr(&self) -> DevicePtr {
        if self.use_fp32_logits {
            self.logits_fp32_buf
        } else {
            self.buffers.logits()
        }
    }
}
