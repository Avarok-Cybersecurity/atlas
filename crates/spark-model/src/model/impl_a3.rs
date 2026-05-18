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

impl TransformerModel {
    pub(super) fn embed(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let row_bytes = h * 2; // BF16 embedding row
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        if self.bf16_to_f32_kernel.0 != 0 {
            // FP32 residual: embed BF16 to scratch, convert to FP32 output.
            // The scratch buffer is norm_output which is BF16 regardless of
            // residual dtype — use the BF16 scaler explicitly.
            let scratch = self.buffers.norm_output();
            self.gpu.copy_d2d_async(src, scratch, row_bytes, stream)?;
            self.scale_embeddings_bf16(scratch, 1, stream)?;
            crate::layers::ops::bf16_to_f32(
                self.gpu.as_ref(),
                self.bf16_to_f32_kernel,
                scratch,
                output,
                h as u32,
                stream,
            )
        } else {
            self.gpu.copy_d2d_async(src, output, row_bytes, stream)?;
            // Scale embeddings (Gemma-4: sqrt(hidden_size))
            self.scale_embeddings(output, 1, stream)
        }
    }

    /// Scale in-place embeddings by config.embed_scale. Picks the kernel
    /// matching `data`'s actual dtype:
    ///   - when `use_fp32_residual()` is true, `hidden` is FP32 and we
    ///     dispatch `embed_scale::f32_scale_inplace`
    ///   - otherwise (`hidden` is BF16) we dispatch the usual
    ///     `embed_scale::bf16_scale_inplace`
    ///
    /// For the rare case of scaling a BF16 buffer while FP32 residual is
    /// ALSO active (e.g. the decode embed() scratch which is deliberately
    /// BF16 before a bf16_to_f32 cast), use `scale_embeddings_bf16`.
    pub(super) fn scale_embeddings(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.config.use_fp32_residual() {
            self.scale_embeddings_fp32(data, num_tokens, stream)
        } else {
            self.scale_embeddings_bf16(data, num_tokens, stream)
        }
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

    pub(super) fn scale_embeddings_fp32(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let kernel = self.gpu.kernel("embed_scale", "f32_scale_inplace")?;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    /// LM head for K tokens: hidden[K, H] → logits[K, V].
    pub(super) fn lm_head_batched(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        let logits = self.buffers.logits();
        if num_tokens == 2 {
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
        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
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
