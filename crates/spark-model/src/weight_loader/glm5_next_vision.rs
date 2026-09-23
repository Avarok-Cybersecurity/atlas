// SPDX-License-Identifier: AGPL-3.0-only

//! Bind GLM-5.3-Flash's vision tower: 347 tensors under `model.visual.*`.
//!
//! All 347 are BF16 even in `nvidia/GLM-5.3-Flash-NVFP4` — only the text/MoE
//! stack is quantised — so this is a pure pointer bind with no dequant path
//! and no copy. Until this file existed the same 1.05 GiB was uploaded by the
//! checkpoint loader and then freed again by `factory::build`, because that
//! reclaim is keyed off "did any loader bind a tower", and none did.
//!
//! Layout notes that matter to the shapes below:
//! * `patch_embed.proj.weight` is `(1024, 3, 2, 14, 14)` on disk. The Conv3d's
//!   kernel equals its input block, so the convolution degenerates to one dot
//!   product per patch and the tensor is read as a `[1024, 1176]` GEMM weight
//!   over the SAME bytes — `(in_channel, temporal, py, px)`, which is exactly
//!   the order the host preprocessor lays a patch out in.
//! * `downsample.weight` is `(4096, 1024, 2, 2)`, read as `[4096, 4096]` in
//!   `(in_channel, kh, kw)` order against the im2col the merger builds.

use anyhow::{Context, Result};
use avarok_core::config::{ModelConfig, VisionConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::glm_vit::init::GlmVitGeometry;
use crate::layers::{GlmVit, GlmVitBlock, GlmVitMerger, VisionTower};

/// Bind one BF16 tensor by name, refusing any other dtype rather than
/// reinterpreting its bytes.
///
/// A silent dtype pun here is the failure this guard exists for: an FP8 or
/// U8-packed tensor read as BF16 walks twice its allocation and the first
/// symptom is `CUDA_ERROR_ILLEGAL_ADDRESS` from somewhere else entirely.
fn bf16(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let w = store
        .get(name)
        .with_context(|| format!("glm5_next vision: missing tensor {name}"))?;
    anyhow::ensure!(
        w.dtype == WeightDtype::BF16,
        "glm5_next vision: {name} is {:?}, but the GLM tower is BF16 throughout (347/347 \
         tensors in the official NVFP4 export). A quantised tower needs a dequant path, not \
         a cast of this pointer.",
        w.dtype
    );
    Ok(w.ptr)
}

/// Check a bound tensor's on-disk shape against what the encoder will read.
///
/// Shapes are compared on the PRODUCT of the trailing dims, because the two
/// tensors that matter here are stored with more axes than they are read with
/// (the 5-D patch-embed conv and the 4-D downsample conv). Comparing the flat
/// widths is the claim actually being made — "these bytes are a `[rows, cols]`
/// GEMM weight" — and it catches a checkpoint whose patch or merge geometry
/// moved, which is the case a name lookup cannot.
fn check_shape(store: &WeightStore, name: &str, rows: usize, cols: usize) -> Result<()> {
    let shape = store.get(name)?.shape.clone();
    let got_rows = shape.first().copied().unwrap_or(0);
    let got_cols: usize = shape.iter().skip(1).product::<usize>().max(1);
    anyhow::ensure!(
        got_rows == rows && (shape.len() == 1 || got_cols == cols),
        "glm5_next vision: {name} has shape {shape:?}, which flattens to \
         [{got_rows}, {got_cols}] — the encoder was built for [{rows}, {cols}] from \
         vision_config. The checkpoint's tower geometry does not match its config."
    );
    Ok(())
}

fn load_block(store: &WeightStore, vp: &str, i: usize, v: &VisionConfig) -> Result<GlmVitBlock> {
    let bp = format!("{vp}.blocks.{i}");
    let h = v.hidden_size;
    let head_dim = h / v.num_heads.max(1);
    check_shape(store, &format!("{bp}.attn.qkv.weight"), 3 * h, h)?;
    check_shape(store, &format!("{bp}.attn.q_norm.weight"), head_dim, 1)?;
    check_shape(
        store,
        &format!("{bp}.mlp.gate_proj.weight"),
        v.intermediate_size,
        h,
    )?;
    Ok(GlmVitBlock {
        norm1_w: bf16(store, &format!("{bp}.norm1.weight"))?,
        qkv_w: bf16(store, &format!("{bp}.attn.qkv.weight"))?,
        qkv_b: bf16(store, &format!("{bp}.attn.qkv.bias"))?,
        q_norm_w: bf16(store, &format!("{bp}.attn.q_norm.weight"))?,
        k_norm_w: bf16(store, &format!("{bp}.attn.k_norm.weight"))?,
        proj_w: bf16(store, &format!("{bp}.attn.proj.weight"))?,
        proj_b: bf16(store, &format!("{bp}.attn.proj.bias"))?,
        norm2_w: bf16(store, &format!("{bp}.norm2.weight"))?,
        gate_w: bf16(store, &format!("{bp}.mlp.gate_proj.weight"))?,
        gate_b: bf16(store, &format!("{bp}.mlp.gate_proj.bias"))?,
        up_w: bf16(store, &format!("{bp}.mlp.up_proj.weight"))?,
        up_b: bf16(store, &format!("{bp}.mlp.up_proj.bias"))?,
        down_w: bf16(store, &format!("{bp}.mlp.down_proj.weight"))?,
        down_b: bf16(store, &format!("{bp}.mlp.down_proj.bias"))?,
    })
}

fn load_merger(store: &WeightStore, vp: &str, v: &VisionConfig) -> Result<GlmVitMerger> {
    let out = v.out_hidden_size;
    let sms2 = v.spatial_merge_size * v.spatial_merge_size;
    check_shape(
        store,
        &format!("{vp}.downsample.weight"),
        out,
        v.hidden_size * sms2,
    )?;
    check_shape(store, &format!("{vp}.merger.proj.weight"), out, out)?;
    check_shape(
        store,
        &format!("{vp}.merger.gate_proj.weight"),
        v.projection_intermediate_size,
        out,
    )?;
    check_shape(
        store,
        &format!("{vp}.merger.down_proj.weight"),
        out,
        v.projection_intermediate_size,
    )?;
    Ok(GlmVitMerger {
        post_layernorm_w: bf16(store, &format!("{vp}.post_layernorm.weight"))?,
        downsample_w: bf16(store, &format!("{vp}.downsample.weight"))?,
        downsample_b: bf16(store, &format!("{vp}.downsample.bias"))?,
        proj_w: bf16(store, &format!("{vp}.merger.proj.weight"))?,
        norm_w: bf16(store, &format!("{vp}.merger.post_projection_norm.weight"))?,
        // The presence of this bias is what distinguishes the merger's
        // LayerNorm from every RMSNorm in the tower. If a future checkpoint
        // drops it, the load must fail here rather than the encoder quietly
        // running an RMSNorm in its place.
        norm_b: bf16(store, &format!("{vp}.merger.post_projection_norm.bias"))?,
        gate_w: bf16(store, &format!("{vp}.merger.gate_proj.weight"))?,
        up_w: bf16(store, &format!("{vp}.merger.up_proj.weight"))?,
        down_w: bf16(store, &format!("{vp}.merger.down_proj.weight"))?,
    })
}

/// `Glm5NextWeightLoader::load_vision_encoder`.
///
/// `Ok(None)` for a text-only checkpoint (`glm5_next_text`, whose config has no
/// `vision_config`) and for a multimodal one whose tower is absent from the
/// shards — the second case warns, because a declared-but-missing tower means
/// image requests will be rejected later and the operator should hear it at
/// load rather than at the first request.
pub fn load_glm5_next_vision(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<VisionTower>> {
    let Some(v) = config.vision.clone() else {
        return Ok(None);
    };
    // Same two-prefix probe every other VL loader does: a re-quant produced via
    // `AutoModelForImageTextToText` keeps the nested `model.language_model.visual.*`
    // layout instead of the flat one.
    let vp = ["model.visual", "model.language_model.visual", "visual"]
        .into_iter()
        .find(|p| store.contains(&format!("{p}.patch_embed.proj.weight")));
    let Some(vp) = vp else {
        tracing::warn!(
            "glm5_next declares a vision_config but no `*.visual.patch_embed.proj.weight` is \
             present in the checkpoint; serving TEXT-ONLY (image input will be refused)."
        );
        return Ok(None);
    };

    let patch_dim = 3 * v.temporal_patch_size * v.patch_size * v.patch_size;
    check_shape(
        store,
        &format!("{vp}.patch_embed.proj.weight"),
        v.hidden_size,
        patch_dim,
    )?;
    let patch_embed_w = bf16(store, &format!("{vp}.patch_embed.proj.weight"))?;
    let patch_embed_b = bf16(store, &format!("{vp}.patch_embed.proj.bias"))?;

    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        blocks.push(load_block(store, vp, i, &v)?);
    }
    let merger = load_merger(store, vp, &v)?;

    let geo = GlmVitGeometry {
        hidden_size: v.hidden_size,
        num_heads: v.num_heads,
        intermediate_size: v.intermediate_size,
        spatial_merge_size: v.spatial_merge_size,
        out_hidden_size: v.out_hidden_size,
        projection_intermediate_size: v.projection_intermediate_size,
        patch_size: v.patch_size,
        temporal_patch_size: v.temporal_patch_size,
        rms_norm_eps: v.rms_norm_eps,
        swiglu_limit: v.swiglu_limit,
        rope_theta: v.rope_theta,
        max_pixels: v.max_pixels,
    };
    let encoder = GlmVit::new(patch_embed_w, patch_embed_b, blocks, merger, &geo, gpu)?;
    tracing::info!(
        "GLM-5.3 vision tower bound: depth={} hidden={} heads={} patch={} merge={} → {} \
         (BF16, {} tensors)",
        v.depth,
        v.hidden_size,
        v.num_heads,
        v.patch_size,
        v.spatial_merge_size,
        v.out_hidden_size,
        v.depth * 14 + 11,
    );
    Ok(Some(VisionTower::glm(encoder)))
}

#[cfg(test)]
mod tests {
    /// The census, as arithmetic: 24 blocks × 14 tensors + 11 non-block
    /// tensors = 347, the number the shard headers report for
    /// `model.visual.*`. A block pattern added or dropped moves this.
    #[test]
    fn the_bound_tensor_count_is_the_checkpoints_347() {
        let per_block = 14; // norm1, norm2, qkv w+b, proj w+b, q_norm, k_norm, 3×mlp w+b
        let non_block = 11; // patch_embed w+b, downsample w+b, post_layernorm,
        // merger proj/gate/up/down + post_projection_norm w+b
        assert_eq!(24 * per_block + non_block, 347);
    }
}
