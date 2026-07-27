// SPDX-License-Identifier: AGPL-3.0-only

//! Resolved kernel handles for the served NLLB bf16 runtime. All handles are
//! looked up once at model construction from the `nllb_encoder` module (built
//! from `kernels/gb10/common/nllb_encoder.cu`) plus the shared `gemm`/`argmax`
//! modules — the same set the `nllb_cuda_bf16` example uses.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// The bf16 NLLB kernel handles (bf16 storage, f32 accumulation).
pub(super) struct NllbKernels {
    pub embed: KernelHandle,
    pub scale: KernelHandle,
    pub add: KernelHandle,
    pub relu: KernelHandle,
    pub ln: KernelHandle,
    /// Out-of-place layer norm (reads `in`, writes `out`) — lets the beam decode
    /// fuse the pre-LN `dh->normed` copy away.
    pub ln_oop: KernelHandle,
    pub bias: KernelHandle,
    pub attn: KernelHandle,
    /// Tensor-core pipelined GEMM (`gemm` module) for the multi-row encoder.
    pub gemm: KernelHandle,
    /// Warp-per-row GEMV (`nllb_encoder`) for M=1 decode projections + lm_head.
    pub gemv: KernelHandle,
    /// On-device argmax over bf16 logits.
    pub argmax: KernelHandle,
    // ── batched-beam kernels (used only by generate_beam_batch) ──
    /// Batched decode attention over B beams, per-beam key length `tk[b]`.
    pub attn_bdecode: KernelHandle,
    /// Write `src[B,d]` into a batch-major cache `[B,stride,d]` at row `pos`.
    pub scatter: KernelHandle,
    /// Reorder beam caches `dst[i] = src[perm[i]]` (HF `_reorder_cache`).
    pub gather: KernelHandle,
    /// Broadcast-add one position row across a `[B,d]` batch.
    pub add_row: KernelHandle,
    /// Phase-d on-device beam candidate reduction: per row, log-sum-exp over the
    /// full vocab + the top-K `(value, token)` pairs (shrinks the per-step D2H
    /// from `B*vocab` to `B*K`).
    pub beam_topk: KernelHandle,
}

impl NllbKernels {
    pub(super) fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            embed: gpu.kernel("nllb_encoder", "nllb_embed_bf16")?,
            scale: gpu.kernel("nllb_encoder", "nllb_scale_bf16")?,
            add: gpu.kernel("nllb_encoder", "nllb_add_bf16")?,
            relu: gpu.kernel("nllb_encoder", "nllb_relu_bf16")?,
            ln: gpu.kernel("nllb_encoder", "nllb_layernorm_bf16")?,
            ln_oop: gpu.kernel("nllb_encoder", "nllb_layernorm_oop_bf16")?,
            bias: gpu.kernel("nllb_encoder", "nllb_bias_bf16")?,
            attn: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
            gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            gemv: gpu.kernel("nllb_encoder", "nllb_gemv_bf16")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            attn_bdecode: gpu.kernel("nllb_encoder", "nllb_attn_bdecode")?,
            scatter: gpu.kernel("nllb_encoder", "nllb_scatter_batched")?,
            gather: gpu.kernel("nllb_encoder", "nllb_gather_batched")?,
            add_row: gpu.kernel("nllb_encoder", "nllb_add_row_bf16")?,
            beam_topk: gpu.kernel("nllb_encoder", "nllb_beam_topk")?,
        })
    }
}
