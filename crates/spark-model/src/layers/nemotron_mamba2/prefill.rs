// SPDX-License-Identifier: AGPL-3.0-only

//! `NemotronMamba2Layer` prefill: in_proj GEMM, causal conv1d, the Mamba-2 SSD
//! chunked scan (with the sequential fallback), gated RMS norm and out_proj.
//! Split from `trait_impl.rs` (500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl NemotronMamba2Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_ssm(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let bf16 = 2usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let gs = self.n_groups * self.state_size; // 8*128 = 1024

        // ── 1. RMS norm + residual (batched, all N tokens) ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2. in_proj GEMM: [N, h] × [h, in_proj_size] → [N, in_proj_size] ──
        //    Layout per token: [z(d_inner) | xBC(d_xbc) | dt(num_heads)]
        let proj = ctx.buffers.ssm_qkvz();
        // Pre-cast A to FP8 once per GEMM. The BF16-A kernel converts A to E4M3
        // inside every CTA, so each activation element was re-converted once per
        // n-block (in_proj_size/128 = ~141x for in_proj). The MMA consumes E4M3
        // either way -- numerically identical, strictly less work.
        // Only when the GEMMs are big enough to amortize the two extra dependent
        // launches per layer: at n<=256 the pre-cast measured ~+15 ms on short
        // prompts (80 serial kernel boundaries) while saving nothing, and at
        // n=1023 it saves ~15 ms. Crossover is between; 512 is safely past it.
        let fp8_a = n >= 512
            && self.fp8_fp8_gemm_t_k.0 != 0
            && self.bf16_to_fp8_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * self.d_inner.max(h);
        // Native FP4 tensor cores (mma.sync kind::mxf4nvf4, sm_121): weights stay in
        // their original NVFP4 form (no FP8 copies read) and activations are
        // dynamically quantized to NVFP4 (packed E2M1 + per-16 E4M3 scales) in one
        // pass. Halves B traffic vs the FP8 path and doubles per-MMA throughput.
        // W4A4 changes activation numerics -- ATLAS_NO_SSM_W4A4=1 falls back to the
        // FP8 path (same-binary A/B + quality escape hatch). Scratch: packed A at
        // fp8_act[0], scales at fp8_act[n*K/2]; total n*K*9/16 <= fp8_act's n*K.
        let w4a4 = n >= 512
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * self.d_inner.max(h)
            && std::env::var("ATLAS_NO_SSM_W4A4").is_err();
        if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * h / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                normed,
                a4,
                a4_sf,
                n,
                h as u32,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.ssm.in_proj,
                proj,
                n,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(w_fp8) = self.in_proj_pd_fp8 {
            // Weights already FP8: no per-K-step dequant, no M-block redundancy.
            if fp8_a {
                let a8 = ctx.buffers.fp8_act();
                ops::bf16_to_fp8(
                    ctx.gpu,
                    self.bf16_to_fp8_k,
                    normed,
                    a8,
                    n * h as u32,
                    stream,
                )?;
                ops::fp8_fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_fp8_gemm_t_k,
                    a8,
                    w_fp8,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_t_k,
                    normed,
                    w_fp8,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref wt) = self.in_proj_t {
            // Fast path: transposed weights + FP8 MMA (N128, K32, cp.async pipeline)
            if n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    wt,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    wt,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                &self.ssm.in_proj,
                proj,
                n,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        }

        // ── 3. Conv1d prefill on xBC (WITH bias, fused SiLU) ──
        //    Input: xBC at proj+d_inner, stride=in_proj_size between tokens
        //    Output: conv_out contiguous [N, d_xbc], stride=d_xbc
        let xbc_ptr = proj.offset(self.d_inner * bf16);
        let conv_out = ctx.buffers.ssm_deinterleaved();
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            ssm_state.conv_state,
            xbc_ptr,
            &self.ssm.conv1d_weight,
            self.ssm.conv1d_bias.weight,
            conv_out,
            self.d_xbc as u32,
            self.d_conv as u32,
            n,
            self.in_proj_size as u32,
            self.d_xbc as u32,
            stream,
        )?;

        // ── 4. Mamba-2 SSM prefill ──
        //    x at conv_out+0, B at conv_out+d_inner, C at conv_out+d_inner+gs
        //    dt at proj+(d_inner+d_xbc), stride=in_proj_size
        let x_ptr = conv_out;
        let b_ptr = conv_out.offset(self.d_inner * bf16);
        let c_ptr = conv_out.offset((self.d_inner + gs) * bf16);
        let dt_ptr = proj.offset((self.d_inner + self.d_xbc) * bf16);
        let y_out = ctx.buffers.attn_output();
        // Use persistent kernel (H in shared memory) when available — eliminates
        // ~64 KB global memory traffic per token per head for h_state reads/writes.
        // SSD chunked scan: the recurrence becomes tensor-core matmuls with only
        // ceil(T/64) sequential links instead of T. Falls back to the sequential
        // kernels if the SSD kernels are unavailable, the shapes do not divide, or
        // ATLAS_NO_SSD=1 (same-binary A/B + escape hatch).
        let ssd_ok = self.ssd_cumsum_k.0 != 0
            && self.ssd_bmm_k.0 != 0
            && self.ssd_scan_k.0 != 0
            && ctx.buffers.ssd_scratch() != spark_runtime::gpu::DevicePtr::NULL
            && self.head_dim.is_multiple_of(ops::SSD_PT as usize)
            && self.state_size.is_multiple_of(8)
            && (self.state_size / 8).is_multiple_of(4)
            && std::env::var("ATLAS_NO_SSD").is_err();

        if ssd_ok {
            let l = ops::SSD_L;
            let nchunks = n.div_ceil(l);
            let heads = self.num_heads as u32;
            let groups = self.n_groups as u32;
            let scratch = ctx.buffers.ssd_scratch();
            let dt_bytes = (heads * nchunks * l * 4) as usize;
            let dt_f32 = scratch;
            let da_cs = scratch.offset(dt_bytes);
            let cb = scratch.offset(2 * dt_bytes);

            ops::mamba2_ssd_cumsum(
                ctx.gpu,
                self.ssd_cumsum_k,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                dt_f32,
                da_cs,
                n,
                heads,
                nchunks,
                1,
                self.in_proj_size as u32,
                1e-9,
                1e9,
                stream,
            )?;
            ops::mamba2_ssd_bmm(
                ctx.gpu,
                self.ssd_bmm_k,
                b_ptr,
                c_ptr,
                cb,
                n,
                nchunks,
                groups,
                self.state_size as u32,
                1,
                self.d_xbc as u32,
                stream,
            )?;
            ops::mamba2_ssd_scan(
                ctx.gpu,
                self.ssd_scan_k,
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                self.ssm.d_param.weight,
                dt_f32,
                da_cs,
                cb,
                y_out,
                n,
                heads,
                self.head_dim as u32,
                self.state_size as u32,
                groups,
                nchunks,
                1,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if self.mamba2_ssm_prefill_persistent_k.0 != 0 {
            ops::mamba2_ssm_prefill_persistent(
                ctx.gpu,
                self.mamba2_ssm_prefill_persistent_k,
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.d_param.weight,
                self.ssm.dt_bias.weight,
                y_out,
                1,
                n,
                self.num_heads as u32,
                self.head_dim as u32,
                self.state_size as u32,
                self.n_groups as u32,
                1e-9,
                1e9,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.in_proj_size as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else {
            ops::mamba2_ssm_prefill(
                ctx.gpu,
                self.mamba2_ssm_prefill_k,
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.d_param.weight,
                self.ssm.dt_bias.weight,
                y_out,
                1,
                n,
                self.num_heads as u32,
                self.head_dim as u32,
                self.state_size as u32,
                self.n_groups as u32,
                1e-9,
                1e9,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.in_proj_size as u32,
                self.d_inner as u32,
                stream,
            )?;
        }

        // ── 5. Gated RMS norm (N tokens) ──
        //    input=y [N, d_inner], gate=z at proj+0 [stride=in_proj_size]
        let gated_out = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_out,
            proj,
            &self.ssm.ssm_norm,
            gated_out,
            n,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        // ── 6. out_proj GEMM: [N, d_inner] × [d_inner, h] → [N, h] ──
        let out = ctx.buffers.ssm_qkvz();
        if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * self.d_inner / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                gated_out,
                a4,
                a4_sf,
                n,
                self.d_inner as u32,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.ssm.out_proj,
                out,
                n,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if let Some(w_fp8) = self.out_proj_pd_fp8 {
            if fp8_a {
                let a8 = ctx.buffers.fp8_act();
                ops::bf16_to_fp8(
                    ctx.gpu,
                    self.bf16_to_fp8_k,
                    gated_out,
                    a8,
                    n * self.d_inner as u32,
                    stream,
                )?;
                ops::fp8_fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_fp8_gemm_t_k,
                    a8,
                    w_fp8,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_t_k,
                    gated_out,
                    w_fp8,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            }
        } else if let Some(ref wt) = self.out_proj_t {
            if n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    gated_out,
                    wt,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    gated_out,
                    wt,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                gated_out,
                &self.ssm.out_proj,
                out,
                n,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        }

        // ── 7. Residual add (N*h elements) ──
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            out,
            (num_tokens * h) as u32,
            stream,
        )?;

        Ok(())
    }
}
