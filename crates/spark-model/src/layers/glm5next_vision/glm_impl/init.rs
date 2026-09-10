// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextVisionEncoder::new`, its scratch group, and the one weight
//! permutation the tower needs at load.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{
    GLM_PATCH_DIM, GLM_VISION_ROPE_THETA, Glm5NextVisionEncoder, GlmMergerWeights, GlmViTBlock,
    GlmVisionScratch,
};
use crate::layers::vision_encoder::enc_impl::init::derive_max_patches;

/// Destination element index for the downsample weight, in row-major `(n, k)`
/// order — the layout `vit_gemm_bias`'s `B` argument expects.
///
/// The conv2d is `[out=4096, in=1024, kh=2, kw=2]` applied to a 2×2 block of
/// patches, and `vision_spatial_merge` already emits those blocks as
/// `channel-concat` rows: output element `j` of a merged row is
/// `j = (mh*merge + mw) * in_ch + c` (`vision_encoder.cu:413-437` gathers
/// `src_idx = (oh*m + p_local/m)*grid_w + (ow*m + p_local%m)` and emits
/// `p_local = mh*m + mw`). So the GEMM's `k` index runs `(kh*merge + kw)*in_ch + c`
/// and the weight has to be rewritten to match.
///
/// Getting this wrong has NO runtime symptom — the shapes are square, nothing
/// overruns, and the model simply describes a scrambled image fluently. Hence
/// the pinning test at the bottom of this file.
fn downsample_dst_index(
    o: usize,
    c: usize,
    kh: usize,
    kw: usize,
    in_ch: usize,
    merge: usize,
) -> usize {
    o * (merge * merge * in_ch) + (kh * merge + kw) * in_ch + c
}

/// Source element index into the checkpoint's `[out, in, kh, kw]` tensor.
fn downsample_src_index(
    o: usize,
    c: usize,
    kh: usize,
    kw: usize,
    in_ch: usize,
    merge: usize,
) -> usize {
    ((o * in_ch + c) * merge + kh) * merge + kw
}

/// Permute `downsample.weight` from the checkpoint's `[out, in, kh, kw]` into
/// the `[out, merge²·in]` row-major matrix the merger GEMM reads, returning a
/// NEW device allocation.
///
/// One D2H + permute + H2D of 33 MB at load. The alternative — permuting on the
/// device — would need a kernel that exists nowhere else in the tree for a
/// one-shot transform.
pub(crate) fn permute_downsample_weight(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    out_ch: usize,
    in_ch: usize,
    merge: usize,
) -> Result<DevicePtr> {
    let n = out_ch * in_ch * merge * merge;
    let mut host = vec![0u8; n * 2];
    gpu.copy_d2h(src, &mut host)?;
    let src_u16: Vec<u16> = host
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut dst = vec![0u16; n];
    for o in 0..out_ch {
        for c in 0..in_ch {
            for kh in 0..merge {
                for kw in 0..merge {
                    dst[downsample_dst_index(o, c, kh, kw, in_ch, merge)] =
                        src_u16[downsample_src_index(o, c, kh, kw, in_ch, merge)];
                }
            }
        }
    }
    let mut bytes = Vec::with_capacity(n * 2);
    for v in &dst {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let ptr = gpu.alloc(n * 2)?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

impl Glm5NextVisionEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patch_embed_w: DevicePtr,
        patch_embed_b: DevicePtr,
        blocks: Vec<GlmViTBlock>,
        merger: GlmMergerWeights,
        hidden_size: usize,
        num_heads: usize,
        spatial_merge_size: usize,
        out_hidden_size: usize,
        intermediate_size: usize,
        proj_intermediate_size: usize,
        swiglu_limit: f32,
        patch_size: usize,
        max_pixels: Option<usize>,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(num_heads > 0, "glm vision: num_heads is 0");
        anyhow::ensure!(
            hidden_size.is_multiple_of(num_heads),
            "glm vision: hidden_size {hidden_size} is not a multiple of num_heads {num_heads}"
        );
        let head_dim = hidden_size / num_heads;
        // Same derivation, same constants, same clamp as the Qwen tower: the
        // encoder's capacity and the CPU preprocessor's bound must come from
        // ONE number or they agree only by coincidence.
        let (p_max, asked_for) = derive_max_patches(max_pixels, patch_size);
        if let Some(wanted) = asked_for {
            tracing::warn!(
                "GLM vision encoder capacity {p_max} patches — the resolved area bound wanted \
                 {wanted}, clamped. GLM-5.3 declares max_image_tokens 8000, i.e. 32000 pre-merge \
                 patches, which is 2x the encoder ceiling; the full declared budget is not \
                 serveable. Pass --vision-max-pixels 3211264 to land exactly on the ceiling."
            );
        } else {
            tracing::info!("GLM vision encoder capacity {p_max} patches");
        }

        // 2D axial RoPE, NEOX pairing (d, d + head_dim/2). rotary_dim =
        // head_dim/2 = 32, so 16 frequencies feed the row half and the same 16
        // feed the column half. Identical derivation to the Qwen tower; only
        // head_dim differs.
        let rope_dim = head_dim / 2;
        let rope_half = rope_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..rope_half)
            .map(|k| 1.0 / GLM_VISION_ROPE_THETA.powf(2.0 * k as f32 / rope_dim as f32))
            .collect();

        Ok(Self {
            patch_embed_w,
            patch_embed_b,
            blocks,
            merger,
            k_gemm: gpu.kernel("vision_encoder", "vision_gemm_bias")?,
            k_add_bias: gpu.kernel("vision_encoder", "vision_add_bias")?,
            k_layer_norm: gpu.kernel("vision_encoder", "vision_layer_norm")?,
            k_add: gpu.kernel("vision_encoder", "vision_add_inplace")?,
            k_merge: gpu.kernel("vision_encoder", "vision_spatial_merge")?,
            k_f32_bf16: gpu.kernel("vision_encoder", "vision_f32_to_bf16")?,
            k_copy: gpu.kernel("vision_encoder", "vision_bf16_copy")?,
            k_gemm_pipelined: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            k_gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            k_rope_deint: gpu.kernel("vision_encoder", "vit_rope_deinterleave")?,
            k_softmax: gpu.kernel("vision_encoder", "vit_softmax_rows")?,
            k_scatter_head: gpu.kernel("vision_encoder", "vit_scatter_head")?,
            k_rms_norm: gpu.kernel("vision_encoder", "vision_rms_norm")?,
            k_rms_norm_heads: gpu.kernel("vision_encoder", "vision_rms_norm_heads")?,
            k_swiglu_clamp: gpu.kernel("vision_encoder", "vision_swiglu_clamp")?,
            k_gelu_exact: gpu.kernel("vision_encoder", "vision_gelu_exact")?,
            hidden_size,
            num_heads,
            head_dim,
            spatial_merge_size,
            out_hidden_size,
            intermediate_size,
            proj_intermediate_size,
            swiglu_limit,
            p_max,
            scratch: std::sync::OnceLock::new(),
            rope_inv_freq,
        })
    }

    /// Widest row this tower ever writes into `buf_wide`, in elements.
    ///
    /// `3 × intermediate` (12288) because the block MLP holds `[gate | up]`
    /// AND the clamped SwiGLU's non-aliasing destination at once; it dominates
    /// QKV's `3 × num_heads × head_dim` (3072) and `GLM_PATCH_DIM` (1176). The
    /// max is taken rather than assumed so a differently-proportioned
    /// checkpoint cannot silently overrun — the naive `p_max × intermediate`
    /// sizing overruns by 3× here, in release, with no symptom.
    pub(crate) fn wide_row_elems(&self) -> usize {
        (3 * self.intermediate_size)
            .max(3 * self.num_heads * self.head_dim)
            .max(GLM_PATCH_DIM)
    }

    fn build_scratch(&self, gpu: &dyn GpuBackend) -> Result<GlmVisionScratch> {
        let p_max = self.p_max;
        let h = self.hidden_size;
        let sms2 = (self.spatial_merge_size * self.spatial_merge_size).max(1);
        let mp_max = p_max / sms2;
        let merger_in_dim = sms2 * h;
        let qkv_head_elems = p_max * self.num_heads * self.head_dim;

        let buf_zero_bias = gpu.alloc(self.zero_bias_elems() * 2)?;
        gpu.copy_h2d(&vec![0u8; self.zero_bias_elems() * 2], buf_zero_bias)?;

        Ok(GlmVisionScratch {
            buf_f32: gpu.alloc(p_max * GLM_PATCH_DIM * 4)?,
            buf_h1: gpu.alloc(p_max * h * 2)?,
            buf_h2: gpu.alloc(p_max * h * 2)?,
            buf_wide: gpu.alloc(p_max * self.wide_row_elems() * 2)?,
            buf_merge_in: gpu.alloc(mp_max * merger_in_dim * 2)?,
            buf_merge_fc1: gpu.alloc(mp_max * 2 * self.proj_intermediate_size * 2)?,
            buf_out: gpu.alloc(p_max * self.out_hidden_size * 2)?,
            buf_rope_cos: gpu.alloc(p_max * self.head_dim * 2)?,
            buf_rope_sin: gpu.alloc(p_max * self.head_dim * 2)?,
            buf_qr: gpu.alloc(qkv_head_elems * 2)?,
            buf_kr: gpu.alloc(qkv_head_elems * 2)?,
            buf_vt: gpu.alloc(qkv_head_elems * 2)?,
            // O(patches²), and the reason CEILING_MAX_PATCHES exists.
            buf_scores: gpu.alloc(p_max * p_max * 4)?,
            buf_probs: gpu.alloc(p_max * p_max * 2)?,
            buf_o_stage: gpu.alloc(p_max * self.head_dim * 2)?,
            buf_zero_bias,
        })
    }

    /// Width of the shared zero-bias vector: the widest N of any bias-free
    /// GEMM in the tower (the merger's gate/up at `proj_intermediate_size`).
    pub(crate) fn zero_bias_elems(&self) -> usize {
        self.proj_intermediate_size.max(self.out_hidden_size)
    }

    /// Allocate the scratch group on first use. Racing callers converge via
    /// `OnceLock`; the loser's buffers are reclaimed by the backend ledger.
    pub(crate) fn scratch_init(&self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.scratch.get().is_none() {
            let s = self.build_scratch(gpu)?;
            let _ = self.scratch.set(s);
            tracing::info!(
                "GLM vision scratch allocated on first image: {} patches",
                self.p_max
            );
        }
        Ok(())
    }

    /// Scratch accessor for the encode path. Panics if `scratch_init` has not
    /// run — a wiring bug, not a runtime condition.
    pub fn scratch(&self) -> &GlmVisionScratch {
        self.scratch
            .get()
            .expect("glm vision scratch: encode entry must call scratch_init(gpu) first")
    }
}

#[cfg(test)]
mod tests {
    use super::{downsample_dst_index, downsample_src_index};

    /// C-T11. Hand-computed 2×2 case with `in_ch = 3`, `out_ch = 2`.
    ///
    /// `vision_spatial_merge` emits merged element `j = (kh*2 + kw)*in_ch + c`,
    /// so the permuted weight's `k` index must be exactly that. Written out by
    /// hand rather than derived from the functions under test, because the
    /// whole risk here is that a plausible-looking index formula is wrong and
    /// nothing at runtime says so.
    #[test]
    fn downsample_permutation_matches_the_merge_kernels_output_order() {
        let (in_ch, merge) = (3usize, 2usize);
        // k index of (c, kh, kw) as vision_spatial_merge lays it out.
        let k_of = |c: usize, kh: usize, kw: usize| (kh * merge + kw) * in_ch + c;
        for o in 0..2usize {
            for c in 0..in_ch {
                for kh in 0..merge {
                    for kw in 0..merge {
                        assert_eq!(
                            downsample_dst_index(o, c, kh, kw, in_ch, merge),
                            o * (merge * merge * in_ch) + k_of(c, kh, kw),
                        );
                    }
                }
            }
        }
        // A few spot values, fully hand-computed.
        assert_eq!(downsample_dst_index(0, 0, 0, 0, 3, 2), 0);
        assert_eq!(downsample_dst_index(0, 2, 0, 0, 3, 2), 2);
        assert_eq!(downsample_dst_index(0, 0, 0, 1, 3, 2), 3);
        assert_eq!(downsample_dst_index(0, 1, 1, 1, 3, 2), 10);
        assert_eq!(downsample_dst_index(1, 0, 0, 0, 3, 2), 12);
    }

    /// The source side is plain `[out, in, kh, kw]` row-major. If this drifts,
    /// the permutation reads the wrong element and the test above still passes.
    #[test]
    fn downsample_source_is_plain_row_major_out_in_kh_kw() {
        assert_eq!(downsample_src_index(0, 0, 0, 0, 3, 2), 0);
        assert_eq!(downsample_src_index(0, 0, 0, 1, 3, 2), 1);
        assert_eq!(downsample_src_index(0, 0, 1, 0, 3, 2), 2);
        assert_eq!(downsample_src_index(0, 1, 0, 0, 3, 2), 4);
        assert_eq!(downsample_src_index(1, 0, 0, 0, 3, 2), 12);
    }

    /// The permutation must be a bijection: every destination hit exactly once.
    #[test]
    fn the_permutation_is_a_bijection() {
        let (out_ch, in_ch, merge) = (4usize, 5usize, 2usize);
        let n = out_ch * in_ch * merge * merge;
        let mut seen = vec![0u8; n];
        for o in 0..out_ch {
            for c in 0..in_ch {
                for kh in 0..merge {
                    for kw in 0..merge {
                        seen[downsample_dst_index(o, c, kh, kw, in_ch, merge)] += 1;
                        assert!(downsample_src_index(o, c, kh, kw, in_ch, merge) < n);
                    }
                }
            }
        }
        assert!(seen.iter().all(|&v| v == 1));
    }
}
