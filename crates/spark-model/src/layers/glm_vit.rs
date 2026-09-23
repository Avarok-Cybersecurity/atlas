// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash vision tower (`Glm5NextVisionModel`) — 24-block ViT with
//! per-head QK-RMSNorm, pure 2D axial RoPE, clamped SwiGLU, a strided-conv 2×2
//! downsample and a 4-stage SwiGLU merger.
//!
//! A SEPARATE encoder from [`super::VisionEncoder`] (the Qwen3-VL tower) rather
//! than a set of flags on it. The two share no op: GLM norms are weight-only
//! RMSNorm where Qwen's are biased LayerNorm, GLM's attention normalises Q and
//! K per head where Qwen's does not, GLM's MLP is 3-matrix clamped SwiGLU where
//! Qwen's is 2-matrix GELU, and GLM's spatial merge is a convolution where
//! Qwen's is a concatenation. Threading `if glm` through the Qwen path would
//! have put a branch inside every kernel launch in it.
//!
//! What IS shared: the tensor-core GEMM (`gemm::dense_gemm_bf16_pipelined`),
//! the f32-out score GEMM, the packed `buf_out` contract the embedding splice
//! reads, and the `(post_h, post_w, merged_p)` return shape — so the server
//! side (`msg_entry` → `VisionItem` → `embed_chunk` splice) is reused unchanged.
//!
//! Reference math and its citations: `runs/ws2-vision/REFERENCE.md` in the
//! spark-bench repo, derived from `transformers` `glm5_next` / `glm_ocr`.

use spark_runtime::gpu::{DevicePtr, KernelHandle};

/// One ViT block's weights. Every tensor is BF16 on disk and stays BF16 — the
/// tower is NOT quantised in `nvidia/GLM-5.3-Flash-NVFP4` (347/347 tensors are
/// BF16; only the text/MoE stack is NVFP4).
pub struct GlmVitBlock {
    pub norm1_w: DevicePtr,
    pub qkv_w: DevicePtr,
    pub qkv_b: DevicePtr,
    /// Per-head RMSNorm on Q, `[head_dim]`. Applied after the QKV split and
    /// BEFORE the rotary transform — the order is the reference's.
    pub q_norm_w: DevicePtr,
    pub k_norm_w: DevicePtr,
    pub proj_w: DevicePtr,
    pub proj_b: DevicePtr,
    pub norm2_w: DevicePtr,
    pub gate_w: DevicePtr,
    pub gate_b: DevicePtr,
    pub up_w: DevicePtr,
    pub up_b: DevicePtr,
    pub down_w: DevicePtr,
    pub down_b: DevicePtr,
}

/// Everything after the last block: the post-norm, the 2×2 conv downsample and
/// the merger MLP whose output is what gets spliced into the LM.
pub struct GlmVitMerger {
    /// Weight-only RMSNorm over `hidden_size`, applied before the downsample.
    pub post_layernorm_w: DevicePtr,
    /// `Conv2d(hidden, out_hidden, k=2, s=2)` flattened to `[out_hidden,
    /// hidden*4]` in `(in_channel, kh, kw)` order — the conv's own layout, which
    /// is what makes the im2col below a plain reshape.
    pub downsample_w: DevicePtr,
    pub downsample_b: DevicePtr,
    pub proj_w: DevicePtr,
    /// `post_projection_norm` is a real mean-subtracting LayerNorm with a bias
    /// — the ONLY such norm in this tower. Every other one is weight-only
    /// RMSNorm. The checkpoint census settles it: this is the only norm that
    /// ships a `.bias`.
    pub norm_w: DevicePtr,
    pub norm_b: DevicePtr,
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub down_w: DevicePtr,
}

/// Device scratch, allocated on the FIRST image rather than at load — a
/// text-only GLM serve never pays a byte of it, which matters because the
/// quadratic score matrix dominates the group.
pub struct GlmVitScratch {
    pub(crate) buf_f32: DevicePtr,
    pub(crate) buf_h1: DevicePtr,
    pub(crate) buf_h2: DevicePtr,
    pub(crate) buf_gate: DevicePtr,
    pub(crate) buf_up: DevicePtr,
    pub(crate) buf_qr: DevicePtr,
    pub(crate) buf_kr: DevicePtr,
    pub(crate) buf_vt: DevicePtr,
    pub(crate) buf_scores: DevicePtr,
    pub(crate) buf_probs: DevicePtr,
    pub(crate) buf_o_stage: DevicePtr,
    pub(crate) buf_rope_cos: DevicePtr,
    pub(crate) buf_rope_sin: DevicePtr,
    pub(crate) buf_merge_a: DevicePtr,
    pub(crate) buf_merge_b: DevicePtr,
    pub(crate) buf_merge_g: DevicePtr,
    pub(crate) buf_merge_u: DevicePtr,
    pub buf_out: DevicePtr,
}

pub struct GlmVit {
    pub(crate) patch_embed_w: DevicePtr,
    pub(crate) patch_embed_b: DevicePtr,
    pub(crate) blocks: Vec<GlmVitBlock>,
    pub(crate) merger: GlmVitMerger,

    // Shared kernels (module `gemm`).
    pub(crate) k_gemm: KernelHandle,
    pub(crate) k_gemm_f32: KernelHandle,
    // GLM-specific kernels (module `glm_vit`).
    pub(crate) k_add_bias: KernelHandle,
    pub(crate) k_rmsnorm: KernelHandle,
    pub(crate) k_layernorm: KernelHandle,
    pub(crate) k_gelu: KernelHandle,
    pub(crate) k_swiglu: KernelHandle,
    pub(crate) k_qknorm_rope: KernelHandle,
    pub(crate) k_softmax: KernelHandle,
    pub(crate) k_scatter_head: KernelHandle,
    pub(crate) k_im2col: KernelHandle,
    pub(crate) k_copy: KernelHandle,
    pub(crate) k_add: KernelHandle,
    pub(crate) k_f32_bf16: KernelHandle,

    // Geometry.
    pub(crate) hidden_size: usize,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub(crate) projection_intermediate_size: usize,
    /// `C × temporal_patch_size × patch_size²` — 1176 for GLM (3×2×14×14).
    /// A RUNTIME value, not the compiled `PATCH_DIM` constant the Qwen encoder
    /// carries: that constant is 1536 and is exactly why the Qwen tower cannot
    /// bind this checkpoint.
    pub(crate) patch_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) swiglu_limit: f32,
    /// Patch rows every buffer is sized for.
    pub(crate) p_max: usize,
    /// `inv_freq[k] = theta^(-2k/spatial_dim)`, `spatial_dim = head_dim/2`.
    /// Half the entries feed the h axis and half the w axis.
    pub(crate) rope_inv_freq: Vec<f32>,

    pub(crate) scratch: std::sync::OnceLock<GlmVitScratch>,
}

mod block;
mod forward;
pub mod init;
mod merge;
mod rope;

pub use init::GlmVitGeometry;
pub use rope::{block_major_position_ids, build_axial_rope_tables};
