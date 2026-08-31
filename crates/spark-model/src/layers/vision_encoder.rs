// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-VL vision encoder: 27-block ViT + DeepStack mergers.
//!
//! Processes patch embeddings (BF16) through a ViT backbone, extracts
//! intermediate hidden states at deepstack indices [8, 16, 24, 27], applies
//! 2×2 spatial merges + 2-layer MLPs, and concatenates the four outputs.
//! Result: [num_patches, out_hidden_size=2048] BF16 ready for LLM embedding.

use spark_runtime::gpu::{DevicePtr, KernelHandle};

pub(super) const IMAGE_PAD_TOKEN: u32 = 151_655;
pub const IMAGE_PAD_TOKEN_ID: u32 = IMAGE_PAD_TOKEN;

/// Fallback `<|video_pad|>` id, used when the checkpoint's config declares
/// none. Qwen3-VL's video token sits directly after its image token, and the
/// same holds for Qwen3.6/3.8 (248056 / 248057) — but a checkpoint that
/// declares its own always wins, exactly as for the image token.
pub const VIDEO_PAD_TOKEN_ID: u32 = IMAGE_PAD_TOKEN + 1;

/// The ViT's per-image scratch buffers, allocated as one group.
///
/// Sizes derive only from the encoder's geometry (`p_max` and the head/hidden
/// dims), so nothing here depends on the image itself — which is why it can be
/// built once, lazily, and reused for every image after.
pub struct VisionScratch {
    pub buf_f32: DevicePtr,
    pub buf_h1: DevicePtr,
    pub buf_h2: DevicePtr,
    pub buf_wide: DevicePtr,
    pub buf_merge_in: DevicePtr,
    pub buf_merge_fc1: DevicePtr,
    pub buf_out: DevicePtr,
    pub buf_pos_resampled: DevicePtr,
    pub buf_rope_cos: DevicePtr,
    pub buf_rope_sin: DevicePtr,
    pub buf_qr: DevicePtr,
    pub buf_kr: DevicePtr,
    pub buf_vt: DevicePtr,
    pub buf_scores: DevicePtr,
    pub buf_probs: DevicePtr,
    pub buf_o_stage: DevicePtr,
}

/// Flattened per-patch pixel dimension `C × temporal_patch_size × patch_size²`
/// = 3 × 2 × 16 × 16 for this ViT. It is baked into the encoder, not read from
/// config: `buf_f32` is allocated at `p_max × PATCH_DIM × 4` and the
/// patch-embed GEMM is issued with `K = PATCH_DIM`.
///
/// The host side computes the same quantity from `vision_config`
/// (`vision_preprocess::preprocess_image`), so a checkpoint declaring a
/// different `patch_size`/`temporal_patch_size` produces a pixel buffer of a
/// DIFFERENT length. Every use of this constant that touches a host slice must
/// therefore check the length rather than assume it — see `patch_embed`.
pub(crate) const PATCH_DIM: usize = 1536;

pub struct ViTBlock {
    pub norm1_w: DevicePtr,
    pub norm1_b: DevicePtr,
    pub qkv_w: DevicePtr,
    pub qkv_b: DevicePtr,
    pub proj_w: DevicePtr,
    pub proj_b: DevicePtr,
    pub norm2_w: DevicePtr,
    pub norm2_b: DevicePtr,
    pub fc1_w: DevicePtr,
    pub fc1_b: DevicePtr,
    pub fc2_w: DevicePtr,
    pub fc2_b: DevicePtr,
}

pub struct MergerLayer {
    pub norm_w: DevicePtr,
    pub norm_b: DevicePtr,
    pub fc1_w: DevicePtr,
    pub fc1_b: DevicePtr,
    pub fc2_w: DevicePtr,
    pub fc2_b: DevicePtr,
}

pub struct VisionEncoder {
    pub patch_embed_w: DevicePtr,      // [1152, 1536] BF16
    pub patch_embed_b: DevicePtr,      // [1152] BF16
    pub pos_embed: DevicePtr,          // [2304, 1152] BF16 (untouched, kept for reference)
    pub blocks: Vec<ViTBlock>,         // 27 blocks
    pub deepstack: Vec<MergerLayer>,   // 3 deepstack mergers
    pub deepstack_indexes: Vec<usize>, // [8, 16, 24] (1-indexed, after Nth block)
    pub merger: MergerLayer,           // final merger (after block 27)
    // kernel handles
    k_gemm: KernelHandle, // vision_gemm_bias: C[M,N] = A[M,K]@B[N,K]^T + bias
    k_gemm_pipelined: KernelHandle, // dense_gemm_bf16_pipelined (tensor-core, ~40×; no bias)
    k_add_bias: KernelHandle, // vision_add_bias: C += bias[n] (fuses bias for the TC path)
    k_norm: KernelHandle, // vision_layer_norm (biased, in-place)
    k_add: KernelHandle,  // vision_add_inplace
    k_gelu: KernelHandle, // vision_gelu (in-place)
    k_attn: KernelHandle, // vision_attention_rope (legacy SDPA — ATLAS_VISION_ATTN_LEGACY=1)
    k_rope_deint: KernelHandle, // vit_rope_deinterleave (rope + head-contig Qr/Kr + V transpose)
    k_softmax: KernelHandle, // vit_softmax_rows (parallel row softmax)
    k_scatter_head: KernelHandle, // vit_scatter_head (contig → interleaved O slot)
    k_gemm_f32: KernelHandle, // dense_gemm_bf16_f32out (raw QKᵀ scores, f32 out)
    k_merge: KernelHandle, // vision_spatial_merge (2×2)
    k_f32_bf16: KernelHandle, // vision_f32_to_bf16
    k_copy: KernelHandle, // vision_bf16_copy
    // config
    pub hidden_size: usize,        // 1152
    pub num_heads: usize,          // 16
    pub head_dim: usize,           // 72
    pub spatial_merge_size: usize, // 2
    pub out_hidden_size: usize,    // 2048
    pub intermediate_size: usize,  // 4304
    pub p_max: usize,              // 6400 (80×80 patches for 1280×1280 image)
    // num_grid_per_side = sqrt(num_position_embeddings) = 48 for Qwen3-VL/3.6.
    pub num_grid_per_side: usize,
    /// ViT scratch, allocated on the FIRST IMAGE rather than at load.
    ///
    /// ~2.2 GB at the 16384-patch rung on Qwen3.8-27B — the fourth-largest
    /// consumer in the process — and a text-only serve never touches a byte of
    /// it. Deferring hands that back to the KV budget on every text workload
    /// while costing an image request one allocation it used to pay at boot.
    ///
    /// `OnceLock` rather than a flag: the encoder's forward path takes `&self`,
    /// and the buffers must be filled exactly once even if two images race.
    scratch: std::sync::OnceLock<VisionScratch>,
    // host-side prep state
    pos_embed_host_f32: Vec<f32>, // [num_position_embeddings × hidden_size] row-major
    rope_inv_freq: Vec<f32>,      // [head_dim / 4] frequencies
}

mod enc_impl;
