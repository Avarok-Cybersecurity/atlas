// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash vision tower (`vision_config.model_type = "glm5_next_vision"`).
//!
//! 24 pre-norm ViT blocks over a conv3d patch embed, then a conv2d 2×2
//! downsample (which IS the spatial merge) and a five-tensor merger whose
//! output width equals the LLM hidden size, so merged rows drop straight onto
//! the hidden state.
//!
//! Structural deltas from the Qwen tower next door, every one of them the
//! reason this is a separate type rather than a flag (see `vision_tower.rs`):
//!
//! | | Qwen3-VL | GLM-5.3 |
//! |---|---|---|
//! | block norm | LayerNorm + bias | RMSNorm, weight only |
//! | q/k norm | none | RMSNorm `[64]` per head, between QKV and RoPE |
//! | block MLP | fc1 → tanh-GELU → fc2 | gate/up → clamped SwiGLU → down |
//! | position | learned grid, bilinear-resampled | none, 2D axial RoPE only |
//! | DeepStack | 3 mergers | none |
//! | merge | `vision_spatial_merge` + 2-layer MLP | merge + conv2d downsample + 5-tensor merger |
//!
//! GLM-4.1V had a learned `embeddings.position_embedding` and a
//! `post_conv_layernorm`; 5.3 dropped BOTH (`glm4v.py:107-110` sets
//! `pos_embedding = False`, `post_conv_norm = False` for `glm5_next_vision`),
//! and the checkpoint carries neither tensor. Do not look for them.

use spark_runtime::gpu::{DevicePtr, KernelHandle};

/// Flattened per-patch pixel dimension `C × temporal_patch_size × patch_size²`
/// = 3 × 2 × 14 × 14. Baked in for the same reason [`super::vision_encoder::PATCH_DIM`]
/// is: `buf_f32` is allocated at `p_max × GLM_PATCH_DIM × 4` and the patch-embed
/// GEMM is issued with `K = GLM_PATCH_DIM`. The host preprocessor computes the
/// same quantity from `vision_config`, so a checkpoint that disagrees produces a
/// pixel buffer of a DIFFERENT length — which is what `check_pixel_len` is for.
pub const GLM_PATCH_DIM: usize = 3 * 2 * 14 * 14;

/// 🔴 ASSUMPTION A1. The vision RoPE base is NOT in `config.json`; it is an
/// exllamav3 hardcode (`architecture/glm4v.py:98`, `v.rope_theta = 10000.0`,
/// applied to every `glm4v*`/`glm5_next_vision` tower). Every other numeric in
/// this file is read from the checkpoint.
///
/// Blast radius if wrong: TOTAL and SILENT. Every block's attention is subtly
/// mis-positioned, nothing errors, and the model answers fluently about the
/// wrong picture. Verify against HF `modeling_glm5_next.py` if it ever lands
/// on the box.
pub const GLM_VISION_ROPE_THETA: f32 = 10_000.0;

/// `vision_config.rms_norm_eps` for GLM-5.3, applied INSIDE the sqrt by every
/// RMSNorm in the tower (block norm1/norm2, q/k norm, post_layernorm).
///
/// A named constant rather than a `VisionConfig` field because the config
/// struct carries no eps today and adding one is a cross-family change; the
/// checkpoint's value is pinned here with its source so a differing checkpoint
/// is a code change, not a silent mismatch.
pub const GLM_VISION_RMS_EPS: f32 = 1e-5;

/// 🔴 ASSUMPTION A2. The merger's `post_projection_norm` is a torch LayerNorm
/// (weight AND bias) whose eps upstream hardcodes to 1e-6, while
/// `vision_config.rms_norm_eps` is 1e-5 and torch's own LayerNorm default is
/// also 1e-5. The fork is deliberate: upstream picked 1e-6 and we match it.
///
/// Blast radius: tiny. The variance is taken over 4096 dimensions, which is far
/// above both epsilons, so the two choices differ in the last BF16 digit.
pub const GLM_MERGER_LN_EPS: f32 = 1e-6;

/// One GLM ViT block. All twelve pointers are BF16 and bound zero-copy from the
/// weight store — the tower is BF16 in BOTH shipped checkpoints (EXL3-K2 and
/// NVFP4), 347 tensors, all in shard 120/120.
pub struct GlmViTBlock {
    pub norm1_w: DevicePtr,  // [1024]
    pub qkv_w: DevicePtr,    // [3072, 1024]
    pub qkv_b: DevicePtr,    // [3072]
    pub q_norm_w: DevicePtr, // [64] — ONE vector broadcast across all 16 heads
    pub k_norm_w: DevicePtr, // [64]
    pub proj_w: DevicePtr,   // [1024, 1024]
    pub proj_b: DevicePtr,   // [1024]
    pub norm2_w: DevicePtr,  // [1024]
    pub gate_w: DevicePtr,   // [4096, 1024]
    pub gate_b: DevicePtr,   // [4096]
    pub up_w: DevicePtr,     // [4096, 1024]
    pub up_b: DevicePtr,     // [4096]
    pub down_w: DevicePtr,   // [1024, 4096]
    pub down_b: DevicePtr,   // [1024]
}

/// Everything after the last block: the tower norm, the conv2d downsample, and
/// the merger's five tensors.
///
/// 🪤 Bias map, verified against the checkpoint header: `downsample` HAS a bias;
/// `merger.proj`, `merger.gate_proj`, `merger.up_proj` and `merger.down_proj`
/// have NONE. `merger.post_projection_norm` has weight AND bias because it is a
/// LayerNorm, not an RMSNorm — the one place this tower still calls the shared
/// `vision_layer_norm` kernel.
pub struct GlmMergerWeights {
    /// `post_layernorm.weight` [1024] — RMSNorm over the last block's output.
    pub post_layernorm_w: DevicePtr,
    /// `downsample.weight` PERMUTED at load into row-major `[4096, 4096]`
    /// `(n, k)` form; see `glm_impl::init::permute_downsample_weight`.
    pub downsample_w: DevicePtr,
    /// `downsample.bias` [4096].
    pub downsample_b: DevicePtr,
    pub proj_w: DevicePtr, // [4096, 4096], no bias
    pub ln_w: DevicePtr,   // post_projection_norm.weight [4096]
    pub ln_b: DevicePtr,   // post_projection_norm.bias   [4096]
    pub gate_w: DevicePtr, // [10240, 4096], no bias
    pub up_w: DevicePtr,   // [10240, 4096], no bias
    pub down_w: DevicePtr, // [4096, 10240], no bias
}

/// Per-batch device scratch, allocated on the FIRST image and never at load —
/// same rationale as the Qwen tower's `VisionScratch`: a text-only serve must
/// not pay for buffers it never touches, and on GLM that is the difference
/// between a K=3 draft fitting at 32K and not.
pub struct GlmVisionScratch {
    /// `[p_max, GLM_PATCH_DIM]` f32 — the H2D landing zone for host pixels.
    pub buf_f32: DevicePtr,
    /// `[p_max, 1024]` BF16 — the residual stream.
    pub buf_h1: DevicePtr,
    /// `[p_max, 1024]` BF16 — the saved residual.
    pub buf_h2: DevicePtr,
    /// `[p_max, 3 * intermediate_size]` BF16. THREE intermediate blocks, not
    /// one: `[gate | up]` occupies `2 * intermediate` and the clamped SwiGLU's
    /// destination must not alias its own source (the kernel's threads are
    /// unordered, so writing `dst[r*N+n]` over `src[r'*2N+…]` corrupts rows
    /// that have not been read yet). The third block is that destination. It is
    /// also the widest use: `3*intermediate = 12288` beats QKV's `3*H*D = 3072`
    /// and `GLM_PATCH_DIM = 1176`.
    pub buf_wide: DevicePtr,
    /// `[p_max/4, 4 * 1024]` BF16 — post-spatial-merge rows.
    pub buf_merge_in: DevicePtr,
    /// `[p_max/4, 2 * projection_intermediate_size]` BF16 — merger `[gate | up]`.
    pub buf_merge_fc1: DevicePtr,
    /// `[p_max, out_hidden_size]` BF16 — the packed encoder output the splice
    /// reads. Deliberately NOT shrunk to `p_max/4` rows: `check_packed_rows`
    /// bounds every write against this capacity, and a tighter buffer would
    /// turn the conservative guard into an exact one for no memory that matters.
    pub buf_out: DevicePtr,
    pub buf_rope_cos: DevicePtr,
    pub buf_rope_sin: DevicePtr,
    pub buf_qr: DevicePtr,
    pub buf_kr: DevicePtr,
    pub buf_vt: DevicePtr,
    pub buf_scores: DevicePtr,
    pub buf_probs: DevicePtr,
    pub buf_o_stage: DevicePtr,
    /// A zero vector as wide as the widest GEMM output. Four of GLM's
    /// projections carry no bias, and both GEMM paths (`vision_gemm_bias` and
    /// pipelined + `vision_add_bias`) take a bias pointer unconditionally.
    /// Adding zero is cheaper than a second, bias-free GEMM helper that would
    /// have to duplicate the tensor-core/scalar fallback decision.
    pub buf_zero_bias: DevicePtr,
}

/// The GLM-5.3-Flash ViT.
pub struct Glm5NextVisionEncoder {
    pub patch_embed_w: DevicePtr, // [1024, 1176] — see C-T10, no permutation needed
    pub patch_embed_b: DevicePtr, // [1024]
    pub blocks: Vec<GlmViTBlock>, // 24
    pub merger: GlmMergerWeights,

    // ── kernel handles ──
    // Hard-required, shared with the Qwen tower's table.
    pub(crate) k_gemm: KernelHandle, // vision_gemm_bias (scalar fallback)
    pub(crate) k_add_bias: KernelHandle, // vision_add_bias
    pub(crate) k_layer_norm: KernelHandle, // vision_layer_norm (merger LN only)
    pub(crate) k_add: KernelHandle,  // vision_add_inplace
    pub(crate) k_merge: KernelHandle, // vision_spatial_merge
    pub(crate) k_f32_bf16: KernelHandle, // vision_f32_to_bf16
    pub(crate) k_copy: KernelHandle, // vision_bf16_copy
    // GEMMs from the shared `gemm` module in kernels/gb10/common.
    pub(crate) k_gemm_pipelined: KernelHandle, // dense_gemm_bf16_pipelined
    pub(crate) k_gemm_f32: KernelHandle,       // dense_gemm_bf16_f32out
    // GEMM-ViT SDPA quintet. HARD-required here, unlike the Qwen table's soft
    // resolve: GLM has exactly one kernel tree (kernels/gb10/glm-5.3-flash) and
    // that tree ships all of them, so a null handle means a broken build, and
    // this encoder has no legacy `vision_attention_rope` fallback path (the
    // warp kernel would need q/k-norm handling it does not have).
    pub(crate) k_rope_deint: KernelHandle,
    pub(crate) k_softmax: KernelHandle,
    pub(crate) k_scatter_head: KernelHandle,
    // GLM-only.
    pub(crate) k_rms_norm: KernelHandle,       // vision_rms_norm
    pub(crate) k_rms_norm_heads: KernelHandle, // vision_rms_norm_heads
    pub(crate) k_swiglu_clamp: KernelHandle,   // vision_swiglu_clamp
    pub(crate) k_gelu_exact: KernelHandle,     // vision_gelu_exact (erf, NOT tanh)

    // ── geometry ──
    pub hidden_size: usize,        // 1024
    pub num_heads: usize,          // 16
    pub head_dim: usize,           // 64
    pub spatial_merge_size: usize, // 2
    pub out_hidden_size: usize,    // 4096 == text hidden_size
    pub intermediate_size: usize,  // 4096
    /// `projection_intermediate_size` — the merger's SwiGLU width (10240).
    pub proj_intermediate_size: usize,
    /// SwiGLU clamp bound (`swiglu_limit`, 10.0). Numerics, not a hint.
    pub swiglu_limit: f32,
    pub p_max: usize,

    pub(crate) scratch: std::sync::OnceLock<GlmVisionScratch>,
    /// `[head_dim / 4]` = 16 rotary frequencies, shared by the row and column
    /// halves of the 2D axial table.
    pub(crate) rope_inv_freq: Vec<f32>,
}

// `pub(crate)` so the weight loader can reach `init::permute_downsample_weight`,
// the one weight this tower cannot bind in place.
pub(crate) mod glm_impl;
