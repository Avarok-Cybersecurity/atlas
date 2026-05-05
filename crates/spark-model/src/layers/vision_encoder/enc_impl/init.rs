// SPDX-License-Identifier: AGPL-3.0-only

//! `VisionEncoder::new` constructor.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{MergerLayer, ViTBlock, VisionEncoder};

impl VisionEncoder {
    pub fn new(
        patch_embed_w: DevicePtr,
        patch_embed_b: DevicePtr,
        pos_embed: DevicePtr,
        num_position_embeddings: usize,
        blocks: Vec<ViTBlock>,
        deepstack: Vec<MergerLayer>,
        deepstack_indexes: Vec<usize>,
        merger: MergerLayer,
        hidden_size: usize,
        num_heads: usize,
        spatial_merge_size: usize,
        out_hidden_size: usize,
        intermediate_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let p_max = 6400usize; // 80×80 patches for 1280×1280 image
        let merger_in_dim = spatial_merge_size * spatial_merge_size * hidden_size; // 4608

        // num_grid_per_side is the side length of the square pos_embed grid
        // (e.g. 48 for Qwen3-VL with 2304 position embeddings). Non-square
        // layouts are not seen in the wild for this family.
        let num_grid_per_side = (num_position_embeddings as f64).sqrt().round() as usize;
        anyhow::ensure!(
            num_grid_per_side * num_grid_per_side == num_position_embeddings,
            "non-square pos_embed: {num_position_embeddings} is not a perfect square"
        );

        let buf_f32 = gpu.alloc(p_max * 1536 * 4)?;
        let buf_h1 = gpu.alloc(p_max * hidden_size * 2)?;
        let buf_h2 = gpu.alloc(p_max * hidden_size * 2)?;
        let buf_wide = gpu.alloc(p_max * intermediate_size * 2)?;
        let buf_merge_in = gpu.alloc((p_max / 4) * merger_in_dim * 2)?;
        let buf_merge_fc1 = gpu.alloc((p_max / 4) * merger_in_dim * 2)?;
        let buf_out = gpu.alloc(p_max * out_hidden_size * 2)?;
        let buf_pos_resampled = gpu.alloc(p_max * hidden_size * 2)?;
        let buf_rope_cos = gpu.alloc(p_max * head_dim * 2)?;
        let buf_rope_sin = gpu.alloc(p_max * head_dim * 2)?;

        // Download pos_embed weight to host as f32 so we can bilinear-
        // interpolate it per image (HF: `fast_pos_embed_interpolate`).
        let pos_n = num_position_embeddings * hidden_size;
        let mut pe_bytes = vec![0u8; pos_n * 2];
        gpu.copy_d2h(pos_embed, &mut pe_bytes)?;
        let pos_embed_host_f32: Vec<f32> = pe_bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect();

        // RoPE inverse-frequency table. Qwen3-VL/3.6 vision RoPE uses
        // `rotary_dim = head_dim / 2`, with `inv_freq[k] = theta^(-2k/dim)`
        // for k in [0, dim/2). theta is fixed at 10000 for vision.
        let rope_dim = head_dim / 2; // e.g. 36
        let rope_half = rope_dim / 2; // e.g. 18
        let theta: f32 = 10_000.0;
        let rope_inv_freq: Vec<f32> = (0..rope_half)
            .map(|k| 1.0 / theta.powf(2.0 * k as f32 / rope_dim as f32))
            .collect();

        Ok(Self {
            patch_embed_w,
            patch_embed_b,
            pos_embed,
            blocks,
            deepstack,
            deepstack_indexes,
            merger,
            k_gemm: gpu.kernel("vision_encoder", "vision_gemm_bias")?,
            k_norm: gpu.kernel("vision_encoder", "vision_layer_norm")?,
            k_add: gpu.kernel("vision_encoder", "vision_add_inplace")?,
            k_gelu: gpu.kernel("vision_encoder", "vision_gelu")?,
            k_attn: gpu.kernel("vision_encoder", "vision_attention_rope")?,
            k_merge: gpu.kernel("vision_encoder", "vision_spatial_merge")?,
            k_f32_bf16: gpu.kernel("vision_encoder", "vision_f32_to_bf16")?,
            k_copy: gpu.kernel("vision_encoder", "vision_bf16_copy")?,
            hidden_size,
            num_heads,
            head_dim,
            spatial_merge_size,
            out_hidden_size,
            intermediate_size,
            p_max,
            num_grid_per_side,
            buf_f32,
            buf_h1,
            buf_h2,
            buf_wide,
            buf_merge_in,
            buf_merge_fc1,
            buf_out,
            buf_pos_resampled,
            buf_rope_cos,
            buf_rope_sin,
            pos_embed_host_f32,
            rope_inv_freq,
        })
    }
}
