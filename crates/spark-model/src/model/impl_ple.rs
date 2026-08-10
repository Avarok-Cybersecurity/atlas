// SPDX-License-Identifier: AGPL-3.0-only

//! Model-level Gemma-4 E2B per-layer-embedding (PLE) precompute.
//!
//! Once per forward pass (from the pass's token IDs + `inputs_embeds`):
//!
//! 1. token-identity: `embed_tokens_per_layer[token_ids] * 16` reshaped
//!    `[S, 35, 256]` — one `batched_embed` over the full 8960-wide table
//!    (all 35 slices in one row per token).
//! 2. context: `per_layer_model_projection @ inputs_embeds * (1/sqrt(h))`
//!    reshaped `[S, 35, 256]`, then RMSNorm(256) over the per-layer rows
//!    (`per_layer_projection_norm`).
//! 3. combined: `(context + identity) / sqrt(2)` -> `[S, 35, 256]`.
//!
//! The result is written to the caller's `dst` (`[S, num_layers*256]` BF16
//! row-major). Each decoder layer's PLE block reads its own 256-dim slice at
//! column `layer_idx*256` (strided) via `gemma4_ple_mul`. The combined
//! vectors are recomputed EVERY pass (never cached across steps).
//!
//! No-op when PLE is disabled (`self.ple_tables.is_none()`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::types::TransformerModel;
use crate::layer::TransformerLayer;
use crate::layers::ops;

/// Gemma-4 E2B PLE scratch + per-layer install (called from
/// `TransformerModel::new`). Allocates three row-major
/// `[max_batch_tokens, num_layers*256]` BF16 buffers and attaches each
/// layer's PLE weights + the KV-shared flag. All-NULL/0 when PLE is
/// disabled (non-E2B).
pub(crate) fn allocate_ple_scratch(
    layers: &mut [Box<dyn TransformerLayer>],
    ple_tables: Option<&crate::weight_loader::Gemma4PleTables>,
    config: &ModelConfig,
    max_batch_tokens: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, usize, KernelHandle)> {
    let Some(tables) = ple_tables else {
        return Ok((
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            0,
            KernelHandle(0),
        ));
    };
    let row_stride = config
        .num_hidden_layers
        .saturating_mul(config.hidden_size_per_layer_input.max(1));
    let per_buf = max_batch_tokens
        .saturating_mul(row_stride)
        .saturating_mul(2); // BF16
    tracing::info!(
        "Gemma-4 E2B PLE: allocating {per_buf} B x3 scratch \
         ({max_batch_tokens} tokens x {row_stride} per-layer dims)"
    );
    let combined = gpu.alloc(per_buf)?;
    let combined_b = gpu.alloc(per_buf)?;
    let identity = gpu.alloc(per_buf)?;
    let rk = gpu.kernel("residual_add", "bf16_residual_add")?;
    // Attach per-layer PLE weights + KV-shared flags.
    crate::layers::gemma4_ple::install_ple_on_layers(layers, tables, config);
    Ok((combined, combined_b, identity, max_batch_tokens, rk))
}

impl TransformerModel {
    /// Compute the combined `[num_tokens, num_layers*256]` BF16 PLE vectors
    /// for the current pass into `dst`, from the pass's `token_ids`
    /// (`[num_tokens]` u32 device) and `inputs_embeds` (`[num_tokens, h]`
    /// BF16). Recompute-every-step; never cached.
    pub(super) fn compute_ple(
        &self,
        token_ids: DevicePtr,
        inputs_embeds: DevicePtr,
        num_tokens: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some(ref tables) = self.ple_tables else {
            return Ok(());
        };
        let h = self.config.hidden_size;
        let per_layer_dim = self.config.hidden_size_per_layer_input.max(1);
        let num_layers = self.config.num_hidden_layers.max(1);
        let row_stride = num_layers * per_layer_dim;
        let n = num_tokens as u32;
        let n_elems = num_tokens as u32 * row_stride as u32;
        let gpu = self.gpu.as_ref();
        let identity = self.ple_identity;

        // 1. identity = embed_tokens_per_layer[token_ids] (full 8960-wide row)
        //    then * 16.
        ops::batched_embed(
            gpu,
            self.batched_embed_kernel,
            token_ids,
            tables.embed_tokens_per_layer[0].weight,
            identity,
            n,
            row_stride as u32,
            stream,
        )?;
        self.scale_bf16_inplace(identity, n_elems, 16.0, stream)?;

        // 2. context = per_layer_model_projection @ inputs_embeds, * 1/sqrt(h).
        ops::dense_gemm(
            gpu,
            self.dense_gemm_kernel,
            inputs_embeds,
            &tables.per_layer_model_projection,
            dst,
            n,
            row_stride as u32,
            h as u32,
            stream,
        )?;
        self.scale_bf16_inplace(dst, n_elems, 1.0 / (h as f32).sqrt(), stream)?;
        // RMSNorm(256) over the per-layer rows: [S*35, 256].
        ops::rms_norm(
            gpu,
            self.rms_norm_kernel,
            dst,
            &tables.per_layer_projection_norm,
            dst,
            n * num_layers as u32,
            per_layer_dim as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;

        // 3. combined = (context + identity) / sqrt(2) -> dst (in place).
        ops::residual_add(gpu, self.ple_residual_add_k, dst, identity, n_elems, stream)?;
        self.scale_bf16_inplace(dst, n_elems, 1.0 / 2.0f32.sqrt(), stream)?;
        Ok(())
    }

    /// Arm every layer's PLE slice for the current pass from `base` (the
    /// combined buffer the precompute just filled). No-op when PLE is
    /// disabled (all layers default to no-op `set_ple_base`).
    pub(super) fn ple_arm_layers(&self, base: DevicePtr) {
        for layer in &self.layers {
            layer.set_ple_base(base);
        }
    }

    /// Scale `num_elements` BF16 values in place via the model's
    /// `embed_scale::bf16_scale_inplace` kernel. E2B always ships an
    /// `embed_scale`, so the handle is non-zero on every PLE path.
    fn scale_bf16_inplace(
        &self,
        data: DevicePtr,
        num_elements: u32,
        scalar: f32,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        debug_assert!(
            self.embed_scale_kernel.0 != 0,
            "PLE scaling requires the embed_scale kernel (E2B always loads it)"
        );
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu.as_ref(), self.embed_scale_kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(num_elements)
            .arg_f32(scalar)
            .launch(stream)
    }
}
