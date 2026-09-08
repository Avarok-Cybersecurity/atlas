// SPDX-License-Identifier: AGPL-3.0-only

//! The vision-tower seam: which ViT a multimodal model carries.
//!
//! TWO towers, not one widened one. The Qwen3-VL tower and the GLM-5.3 tower
//! agree on almost nothing below the top-level shape: the block norm is
//! LayerNorm-with-bias vs RMSNorm, the MLP is `fc1/GELU/fc2` vs a clamped
//! SwiGLU with separate gate/up projections, GLM adds per-head q/k RMSNorm
//! between the QKV GEMM and RoPE, GLM has no learned position embedding and no
//! DeepStack, and its merger is a conv2d downsample followed by a five-tensor
//! projection block rather than a two-layer MLP.
//!
//! Widening `VisionEncoder` with an `is_glm` flag would put six untested
//! branches inside the hot per-block path of a tower that works today. Every
//! one of them would be a Qwen regression risk for a model whose output cannot
//! be validated by a unit test — a wrong branch there yields a fluent, WRONG
//! description of the image and logs nothing. An enum keeps the two apart and
//! makes the compiler enumerate every call site.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm5next_vision::Glm5NextVisionEncoder;
use super::vision_encoder::VisionEncoder;

/// A loaded vision tower. Built by `ModelWeightLoader::load_vision_encoder`,
/// held by `TransformerModel`, and reached only through the four methods
/// below — the splice sites must not index device buffers themselves.
#[allow(clippy::large_enum_variant)]
pub enum VisionTower {
    /// Qwen3-VL / Qwen3.5 / Qwen3.8 / Mistral / LongCat — the historical tower.
    Qwen(VisionEncoder),
    /// GLM-5.3-Flash (`vision_config.model_type = "glm5_next_vision"`).
    Glm(Glm5NextVisionEncoder),
}

impl VisionTower {
    /// Encode N images packed into the tower's shared scratch, returning each
    /// image's `(post_merge_h, post_merge_w, merged_patch_rows)` in image order.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        match self {
            Self::Qwen(ve) => ve.forward_batched(images, gpu, stream),
            Self::Glm(ve) => ve.forward_batched(images, gpu, stream),
        }
    }

    /// Width of one encoded row, in elements. Equals the LLM's hidden size.
    pub fn out_hidden_size(&self) -> usize {
        match self {
            Self::Qwen(ve) => ve.out_hidden_size,
            Self::Glm(ve) => ve.out_hidden_size,
        }
    }

    /// Device pointer to packed output row `row`.
    ///
    /// This exists so the two splice sites stop reaching through
    /// `ve.scratch().buf_out.offset(...)`. `scratch()` PANICS when the encode
    /// entry point has not run, and only the encode path can guarantee that, so
    /// the offset arithmetic belongs behind the enum rather than in
    /// `embed_chunk.rs` and `prefill_c.rs`, where a `pending > 0` test is the
    /// only thing standing between a stale count and that panic.
    pub fn out_row(&self, row: usize) -> DevicePtr {
        match self {
            Self::Qwen(ve) => ve.scratch().buf_out.offset(row * ve.out_hidden_size * 2),
            Self::Glm(ve) => ve.scratch().buf_out.offset(row * ve.out_hidden_size * 2),
        }
    }

    /// Pre-merge patch-row capacity of the shared scratch (Σp across a batch).
    /// The scheduler's vision co-dispatch pre-pass books against this.
    pub fn p_max(&self) -> usize {
        match self {
            Self::Qwen(ve) => ve.p_max,
            Self::Glm(ve) => ve.p_max,
        }
    }
}
