// SPDX-License-Identifier: AGPL-3.0-only

//! The vision tower a built model holds, whichever family bound it.
//!
//! Atlas serves two structurally different towers: the Qwen3-VL-shaped
//! [`VisionEncoder`] that six loaders bind, and GLM-5.3's [`GlmVit`]. They
//! agree on exactly three things, and those three are the whole interface the
//! rest of the engine needs:
//!
//! * `forward_batched(images) -> per-image (post_h, post_w, merged_p)`,
//! * a packed `[Σmerged_p, out_hidden_size]` BF16 output buffer, image-ordered,
//! * `out_hidden_size` itself.
//!
//! So the enum lives here rather than a trait object: two variants, three
//! methods, no dynamic dispatch on a path that already costs a GEMM, and —
//! more to the point — no way for a third tower to be added without deciding
//! out loud what its output contract is.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm_vit::GlmVit;
use super::vision_encoder::VisionEncoder;

pub enum VisionTower {
    /// Qwen3-VL / Qwen3.5 / Qwen3.6 / Qwen4 / LongCat / Mistral.
    Qwen(Box<VisionEncoder>),
    /// GLM-5.3-Flash (`glm5_next`).
    Glm(Box<GlmVit>),
}

impl VisionTower {
    pub fn qwen(encoder: VisionEncoder) -> Self {
        Self::Qwen(Box::new(encoder))
    }

    pub fn glm(encoder: GlmVit) -> Self {
        Self::Glm(Box::new(encoder))
    }

    /// Encode a batch of `(pixels, grid_h, grid_w)` temporal groups. Returns
    /// `(post_merge_h, post_merge_w, merged_patches)` per group, in order.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        match self {
            Self::Qwen(e) => e.forward_batched(images, gpu, stream),
            Self::Glm(e) => e.forward_batched(images, gpu, stream),
        }
    }

    /// Width of one merged embedding row — the LLM's hidden size.
    pub fn out_hidden_size(&self) -> usize {
        match self {
            Self::Qwen(e) => e.out_hidden_size,
            Self::Glm(e) => e.out_hidden_size,
        }
    }

    /// Allocate the encoder's device scratch if it is not allocated yet.
    ///
    /// Normally the first image does this. A TENSOR-PARALLEL WORKER never sees
    /// one — it receives the merged rows over NCCL instead — so it has to ask
    /// for the buffers explicitly before `out_row` can name a destination.
    pub fn ensure_scratch(&self, gpu: &dyn GpuBackend) -> Result<()> {
        match self {
            Self::Qwen(e) => e.scratch_init(gpu),
            Self::Glm(e) => e.scratch_init(gpu),
        }
    }

    /// Device pointer to merged row `row` of the packed output. Valid only
    /// after `forward_batched` or `ensure_scratch` has allocated the scratch.
    pub fn out_row(&self, row: usize) -> DevicePtr {
        match self {
            Self::Qwen(e) => e.scratch().buf_out.offset(row * e.out_hidden_size * 2),
            Self::Glm(e) => e.out_row(row),
        }
    }
}
