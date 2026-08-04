// SPDX-License-Identifier: AGPL-3.0-only

//! `impl TransformerLayer for NemotronMamba2Layer`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::ops;

impl TransformerLayer for NemotronMamba2Layer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        // 1. RMS norm + save residual
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // 2. in_proj GEMV: normed[hidden_size] -> proj[in_proj_size]
        //    Layout: [z(d_inner) | xBC(d_xbc) | dt(num_heads)]
        let proj = ctx.buffers.ssm_qkvz();
        // Use FP8 GEMV if available (skips double-quantization lossy path)
        if let Some(ref fp8w) = self.in_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                normed,
                &self.ssm.in_proj,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        }

        // Pointers into projection output (BF16, 2 bytes per element)
        let z_ptr = proj;
        let xbc_ptr = proj.offset(self.d_inner * 2);
        let dt_ptr = proj.offset((self.d_inner + self.d_xbc) * 2);

        // 3. Conv1d update on xBC (with bias, fused SiLU)
        let xbc_out = ctx.buffers.ssm_deinterleaved();
        self.conv1d_update_biased(
            ctx.gpu,
            ssm_state.conv_state,
            xbc_ptr,
            xbc_out,
            self.d_xbc as u32,
            self.d_conv as u32,
            1,
            stream,
        )?;

        // 4. Split xBC_out into x, B, C (BF16 offsets)
        let x_ptr = xbc_out;
        let gs = self.n_groups * self.state_size;
        let b_ptr = xbc_out.offset(self.d_inner * 2);
        let c_ptr = xbc_out.offset((self.d_inner + gs) * 2);

        // 5. SSM decode: state update + y output
        let y_ptr = ctx.buffers.attn_output();
        self.ssm_decode(
            ctx.gpu,
            ssm_state.h_state,
            x_ptr,
            b_ptr,
            c_ptr,
            dt_ptr,
            y_ptr,
            1,
            stream,
        )?;

        // 6. Gated RMS norm: rms_norm(y, ssm_norm) * silu(z)
        //    y is [d_inner], z is [d_inner], gate_stride = in_proj_size (z at start of proj)
        let gated_out = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_ptr,
            z_ptr,
            &self.ssm.ssm_norm,
            gated_out,
            1,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        // 7. out_proj GEMV: gated_out[d_inner] -> out[hidden_size]
        // Use qkv_output (NOT ssm_qkvz) — ssm_qkvz still holds z_ptr being read
        // by gated_rms_norm above. Writing out_proj to the same buffer creates a
        // write-after-read race that corrupts the gate signal → all-zero output.
        let out = ctx.buffers.qkv_output();
        if let Some(ref fp8w) = self.out_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gated_out,
                fp8w.weight,
                fp8w.row_scale,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                gated_out,
                &self.ssm.out_proj,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        }

        // 8. Residual add: hidden += out_proj_result (hidden unchanged by rms_norm_residual)
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, out, h as u32, stream)?;

        Ok(())
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_ssm(hidden, residual, num_tokens, state, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
        }))
    }
}
