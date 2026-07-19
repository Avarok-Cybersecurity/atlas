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
    // pre-allocated scratch buffers
    pub buf_f32: DevicePtr,           // [p_max × 1536] f32  — pixel upload
    pub buf_h1: DevicePtr,            // [p_max × 1152] bf16 — active hidden
    pub buf_h2: DevicePtr,            // [p_max × 1152] bf16 — residual
    pub buf_wide: DevicePtr,          // [p_max × 4304] bf16 — QKV/MLP scratch
    pub buf_merge_in: DevicePtr,      // [p_max/4 × 4608] bf16
    pub buf_merge_fc1: DevicePtr,     // [p_max/4 × 4608] bf16
    pub buf_out: DevicePtr,           // [p_max × 2048] bf16 — final output
    pub buf_pos_resampled: DevicePtr, // [p_max × 1152] bf16 — per-image interp pos_embed
    pub buf_rope_cos: DevicePtr,      // [p_max × head_dim] bf16 — per-image rotary cos
    pub buf_rope_sin: DevicePtr,      // [p_max × head_dim] bf16 — per-image rotary sin
    // GEMM-based ViT SDPA scratch (reused across the per-head loop within one image).
    pub buf_qr: DevicePtr, // [H × p_max × head_dim] bf16 — rotated Q, head-contiguous
    pub buf_kr: DevicePtr, // [H × p_max × head_dim] bf16 — rotated K, head-contiguous
    pub buf_vt: DevicePtr, // [H × head_dim × p_max] bf16 — transposed V, head-contiguous
    pub buf_scores: DevicePtr, // [attn_max × attn_max] f32  — per-head raw QKᵀ
    pub buf_probs: DevicePtr, // [attn_max × attn_max] bf16 — per-head softmax probs
    pub buf_o_stage: DevicePtr, // [p_max × head_dim] bf16 — per-head GEMM2 output staging
    // host-side prep state
    pos_embed_host_f32: Vec<f32>, // [num_position_embeddings × hidden_size] row-major
    rope_inv_freq: Vec<f32>,      // [head_dim / 4] frequencies
}

mod enc_impl;
