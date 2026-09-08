// SPDX-License-Identifier: AGPL-3.0-only

//! Binds GLM-5.3-Flash's `model.visual.*` tower.
//!
//! 347 tensors, ALL BF16, all in shard 120/120, in BOTH shipped checkpoints
//! (vcruz305 EXL3-K2 and LibertAIDAI NVFP4) — so there is no EXL3 arm and no
//! NVFP4 arm here, only pointer binding from the store. The ONE exception is
//! `downsample.weight`, which is permuted on the host into the order
//! `vision_spatial_merge` emits and re-uploaded (33 MB, once, at load).
//!
//! A separate file from `glm5_next_load.rs` because the text loader is already
//! at its size waiver and this is a self-contained unit: it reads only `store`
//! and `config.vision`.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::glm5next_vision::glm_impl::init::permute_downsample_weight;
use crate::layers::{Glm5NextVisionEncoder, GlmMergerWeights, GlmViTBlock, VisionTower};

/// GLM-5.3's `swiglu_limit` when the config parser did not see one. The
/// checkpoint declares 10.0; a MISSING limit is not the same as "unclamped", so
/// falling back to the declared value is safer than falling back to infinity.
const DEFAULT_SWIGLU_LIMIT: f32 = 10.0;

fn ptr(store: &WeightStore, name: &str) -> Result<spark_runtime::gpu::DevicePtr> {
    Ok(store
        .get(name)
        .with_context(|| format!("glm vision: missing tensor {name}"))?
        .ptr)
}

/// Build the GLM-5.3 tower, or `Ok(None)` when this build has no vision config.
///
/// Returning `None` here MUST agree with `binds_vision_encoder` returning
/// `false`: both key off `config.vision.is_some()`, which `serve_load.rs` nulls
/// when the kernel target ships no `vision_encoder` PTX module. Disagreement is
/// bound-then-freed or skipped-then-bound, and both are CUDA-700 at the first
/// image rather than a clean refusal at load.
pub(super) fn load_glm5_next_vision(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<VisionTower>> {
    let Some(v) = config.vision.as_ref() else {
        return Ok(None);
    };
    anyhow::ensure!(
        v.attention_bias,
        "glm vision: vision_config.attention_bias is false, but every projection in this \
         tower is loaded WITH a bias. Refusing rather than binding a bias that the \
         checkpoint says should not exist."
    );
    let vp = "model.visual";

    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        let bp = format!("{vp}.blocks.{i}");
        blocks.push(GlmViTBlock {
            norm1_w: ptr(store, &format!("{bp}.norm1.weight"))?,
            qkv_w: ptr(store, &format!("{bp}.attn.qkv.weight"))?,
            qkv_b: ptr(store, &format!("{bp}.attn.qkv.bias"))?,
            q_norm_w: ptr(store, &format!("{bp}.attn.q_norm.weight"))?,
            k_norm_w: ptr(store, &format!("{bp}.attn.k_norm.weight"))?,
            proj_w: ptr(store, &format!("{bp}.attn.proj.weight"))?,
            proj_b: ptr(store, &format!("{bp}.attn.proj.bias"))?,
            norm2_w: ptr(store, &format!("{bp}.norm2.weight"))?,
            gate_w: ptr(store, &format!("{bp}.mlp.gate_proj.weight"))?,
            gate_b: ptr(store, &format!("{bp}.mlp.gate_proj.bias"))?,
            up_w: ptr(store, &format!("{bp}.mlp.up_proj.weight"))?,
            up_b: ptr(store, &format!("{bp}.mlp.up_proj.bias"))?,
            down_w: ptr(store, &format!("{bp}.mlp.down_proj.weight"))?,
            down_b: ptr(store, &format!("{bp}.mlp.down_proj.bias"))?,
        });
    }

    // The ONE weight that is not bound in place. `downsample.weight` is a conv2d
    // `[out, in, kh, kw]`; the merger GEMM needs `[out, (kh*2+kw)*in + c]`,
    // which is the element order `vision_spatial_merge` produces.
    let downsample_w = permute_downsample_weight(
        gpu,
        ptr(store, &format!("{vp}.downsample.weight"))?,
        v.out_hidden_size,
        v.hidden_size,
        v.spatial_merge_size,
    )?;

    let merger = GlmMergerWeights {
        post_layernorm_w: ptr(store, &format!("{vp}.post_layernorm.weight"))?,
        downsample_w,
        downsample_b: ptr(store, &format!("{vp}.downsample.bias"))?,
        proj_w: ptr(store, &format!("{vp}.merger.proj.weight"))?,
        ln_w: ptr(store, &format!("{vp}.merger.post_projection_norm.weight"))?,
        ln_b: ptr(store, &format!("{vp}.merger.post_projection_norm.bias"))?,
        gate_w: ptr(store, &format!("{vp}.merger.gate_proj.weight"))?,
        up_w: ptr(store, &format!("{vp}.merger.up_proj.weight"))?,
        down_w: ptr(store, &format!("{vp}.merger.down_proj.weight"))?,
    };

    // `projection_intermediate_size` is not optional in practice — the merger's
    // gate/up tensors are `[10240, 4096]` — so a missing one is a config-parse
    // regression, not a checkpoint variant. Fail rather than guess a width the
    // GEMM would then read past.
    let proj_interm = v.projection_intermediate_size.ok_or_else(|| {
        anyhow::anyhow!(
            "glm vision: vision_config.projection_intermediate_size is absent; the merger's \
             gate/up width cannot be guessed from the other fields"
        )
    })?;

    let ve = Glm5NextVisionEncoder::new(
        ptr(store, &format!("{vp}.patch_embed.proj.weight"))?,
        ptr(store, &format!("{vp}.patch_embed.proj.bias"))?,
        blocks,
        merger,
        v.hidden_size,
        v.num_heads,
        v.spatial_merge_size,
        v.out_hidden_size,
        v.intermediate_size,
        proj_interm,
        v.swiglu_limit.unwrap_or(DEFAULT_SWIGLU_LIMIT),
        v.patch_size,
        v.max_pixels,
        gpu,
    )?;
    tracing::info!(
        "GLM-5.3 vision tower loaded: depth={}, hidden={}, heads={}, merger_interm={}, \
         swiglu_limit={}",
        v.depth,
        v.hidden_size,
        v.num_heads,
        proj_interm,
        ve.swiglu_limit,
    );
    Ok(Some(VisionTower::Glm(ve)))
}
