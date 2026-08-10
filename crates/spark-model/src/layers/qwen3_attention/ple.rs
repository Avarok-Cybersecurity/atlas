// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer PLE block for [`super::Qwen3AttentionLayer`] (Gemma-4 E2B).
//!
//! Runs at the END of the decoder layer, immediately BEFORE the
//! `layer_scalar` multiply:
//!
//! ```text
//! h = input_gate(hidden)            # Linear hidden_size -> 256, no bias
//! h = gelu_pytorch_tanh(h)
//! h = h * ple_slice[i]              # [S, 256] elementwise (layer i's slice)
//! h = projection(h)                 # Linear 256 -> hidden_size, no bias
//! h = post_norm(h)                  # RMSNorm(hidden_size)
//! hidden = residual + h
//! ```
//!
//! Split from `helpers.rs` to keep every file under the 500-LoC cap and so
//! this PLE-only module can reach the layer's private kernel/weight fields.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Run the per-layer PLE block in place on `hidden` ([S, hidden_size]
    /// BF16). `hidden` is only READ by step 1 (input_gate) and WRITTEN by the
    /// final residual add on the same stream, so the in-place update is
    /// exactly `hidden = residual + h` (no separate residual copy needed).
    ///
    /// No-op when this layer has no PLE weights, or when the model-level
    /// precompute did not arm a combined buffer for this pass
    /// (`ple_slice_ptr()` is NULL).
    pub(crate) fn gemma4_ple_forward(
        &self,
        ctx: &ForwardContext<'_>,
        hidden: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(ref ple) = self.ple else {
            return Ok(());
        };
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size as u32;
        let per_layer_dim = ctx.config.hidden_size_per_layer_input.max(1) as u32;
        let num_layers = ctx.config.num_hidden_layers.max(1) as u32;
        let combined = self.ple_slice_ptr();
        if combined.0 == 0 {
            // Non-E2B pass (or a path that never ran the precompute) — no
            // model-level vectors this pass, so skip the whole block.
            return Ok(());
        }
        let n = num_tokens as u32;
        let eps = ctx.config.rms_norm_eps as f32;

        // Scratch: ple_h [S, 256] (ssm_deinterleaved is [S, >=5120] here) and
        // proj_out [S, hidden] (moe_output). Both are dead at this point in
        // the decode/prefill layer body (post-FFN-residual-add).
        let ple_h = ctx.buffers.ssm_deinterleaved();
        let proj_out = ctx.buffers.moe_output();

        // 1. input_gate: [S, h] x [256, h]^T -> [S, 256]
        ops::dense_gemm(
            gpu,
            self.dense_gemm_k,
            hidden,
            &ple.input_gate,
            ple_h,
            n,
            per_layer_dim,
            h,
            stream,
        )?;
        // 2. gelu_pytorch_tanh in place.
        Self::gelu_tanh_inplace(gpu, self.gelu_tanh_k, ple_h, n * per_layer_dim, stream)?;
        // 3. Multiply by this layer's slice of the combined buffer (strided).
        ops::gemma4_ple_mul(
            gpu,
            self.ple_mul_k,
            ple_h,
            combined,
            self.attn_layer_idx as u32 * per_layer_dim,
            num_layers * per_layer_dim,
            n,
            per_layer_dim,
            stream,
        )?;
        // 4. projection: [S, 256] x [h, 256]^T -> [S, h]
        ops::dense_gemm(
            gpu,
            self.dense_gemm_k,
            ple_h,
            &ple.projection,
            proj_out,
            n,
            h,
            per_layer_dim,
            stream,
        )?;
        // 5. post_norm: RMSNorm(hidden_size) in place.
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            proj_out,
            &ple.post_norm,
            proj_out,
            n,
            h,
            eps,
            stream,
        )?;
        // 6. hidden = residual + h.
        ops::residual_add(gpu, self.residual_add_k, hidden, proj_out, n * h, stream)?;
        Ok(())
    }

    /// Launch `gelu::gelu_tanh` in place over `num_elements` BF16 values.
    fn gelu_tanh_inplace(
        gpu: &dyn GpuBackend,
        kernel: spark_runtime::gpu::KernelHandle,
        data: DevicePtr,
        num_elements: u32,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        KernelLaunch::new(gpu, kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_ptr(data)
            .arg_u32(num_elements)
            .launch(stream)
    }
}
