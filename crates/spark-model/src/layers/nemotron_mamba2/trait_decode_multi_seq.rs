// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence batched decode for `NemotronMamba2Layer` (Milestone A).
//!
//! The default `decode_multi_seq` loop runs the full single-token `decode()`
//! per sequence, streaming the ~38.7M-param in_proj/out_proj weights N times
//! per layer per step. On bandwidth-bound LPDDR5X that weight traffic IS the
//! decode cost, so this override batches every stateless phase across all N
//! rows (one weight-DRAM pass) and keeps only the stateful conv+scan inner
//! as a per-seq loop:
//!
//!   1. batched input `rms_norm_residual`            [N, h]
//!   2. batched in_proj                              [N, h] → [N, in_proj_size]
//!   3. per-seq conv1d_update + mamba2_ssm_decode    (2 tiny launches/row)
//!   4. batched `gated_rms_norm`                     [N, d_inner]
//!   5. batched out_proj                             [N, d_inner] → [N, h]
//!   6. batched `residual_add`                       N*h elements
//!
//! Step 3 stays per-row because `causal_conv1d_update` hardcodes its input
//! stride to `dim` (batch>1 straight from the in_proj_size-strided proj
//! buffer would reproduce the d9b33f46 GDN input-stride corruption) and
//! `mamba2_ssm_decode` has no stride args at all; batch=1 with per-row
//! pointers is exact. Strided kernel twins are Milestone B.
//!
//! All batched launches cover ALL `num_seqs = padded_n` rows including pads
//! (stride-safe, graph-safe); pad rows carry the pool dummy slot and are
//! write-only sinks — the same convention as the GDN family.
//!
//! Arm-selection parity: `batched_in_proj`/`batched_out_proj` mirror
//! `decode()`'s arm ORDER (BF16 → native FP8 → NVFP4), NOT prefill's
//! `native_fp8_prefill` gating — the `ATLAS_NEMOTRON_NATIVE_FP8_SSM=decode`
//! bisect mode makes prefill and decode intentionally pick different
//! weights, and C=1 vs C=n outputs must not diverge by quantization arm.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight};

impl NemotronMamba2Layer {
    /// Batched N-sequence decode body. `hidden`/`residual` are `[N, h]` BF16
    /// contiguous; `states[i]` is row i's `SsmLayerState` (pad rows alias the
    /// pool dummy slot). Row contract: batch position i ↔ `states[i]` ↔
    /// hidden row i for EVERY phase — a single index drives all offsets.
    pub(super) fn decode_multi_seq_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let bf16 = 2usize;
        let gs = self.n_groups * self.state_size;

        // 1. Batched input norm + residual save (kernel batches natively,
        //    input/output row stride = h).
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 2. Batched in_proj: normed[N,h] → proj[N,in_proj_size]
        //    Row layout per seq: [z(d_inner) | xBC(d_xbc) | dt(num_heads)].
        let proj = ctx.buffers.ssm_qkvz();
        self.batched_in_proj(normed, proj, n as u32, h as u32, ctx, stream)?;

        // 3. Per-seq conv + SSM scan: 2 launches per row, per-row pointers.
        let xbc_base = ctx.buffers.ssm_deinterleaved();
        let y_base = ctx.buffers.attn_output();
        for (i, state) in states.iter_mut().take(n).enumerate() {
            let st = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
            let proj_i = proj.offset(i * self.in_proj_size * bf16);
            // Conv output rows are packed at d_xbc so step 4/5 inputs are
            // contiguous [N, d_inner]-prefixed rows.
            let xbc_out_i = xbc_base.offset(i * self.d_xbc * bf16);
            self.conv1d_update_biased(
                ctx.gpu,
                st.conv_state,
                proj_i.offset(self.d_inner * bf16), // xBC within this row
                xbc_out_i,
                self.d_xbc as u32,
                self.d_conv as u32,
                1,
                stream,
            )?;
            self.ssm_decode(
                ctx.gpu,
                st.h_state,
                xbc_out_i,                                         // x
                xbc_out_i.offset(self.d_inner * bf16),             // B
                xbc_out_i.offset((self.d_inner + gs) * bf16),      // C
                proj_i.offset((self.d_inner + self.d_xbc) * bf16), // dt
                y_base.offset(i * self.d_inner * bf16),            // y row i
                1,
                stream,
            )?;
        }

        // 4. Batched gated RMS norm: y rows are contiguous at d_inner (step 3
        //    wrote them that way); gate rows live at in_proj_size stride in
        //    proj (explicit gate_stride arg). Reusing norm_output is safe:
        //    `normed` was fully consumed by in_proj on this ordered stream.
        let gated = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_base,
            proj,
            &self.ssm.ssm_norm,
            gated,
            n as u32,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        // 5. Batched out_proj → qkv_output. MUST NOT target ssm_qkvz: proj
        //    still holds the z gate read by step 4 (documented WAR hazard,
        //    trait_impl.rs step 7).
        let out = ctx.buffers.qkv_output();
        self.batched_out_proj(gated, out, n as u32, h as u32, ctx, stream)?;

        // 6. Batched residual add: hidden[N,h] += out[N,h] (both contiguous).
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            out,
            (n * h) as u32,
            stream,
        )?;
        Ok(())
    }

    /// Batched in_proj: `[m, h] → [m, in_proj_size]`, arm order = `decode()`.
    fn batched_in_proj(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n_dim = self.in_proj_size as u32;
        if let Some(ref w) = self.in_proj_bf16 {
            self.batched_dense_proj(input, w, output, m, n_dim, k, ctx, stream)
        } else if let Some(ref fp8w) = self.in_proj_fp8 {
            self.batched_fp8_proj(input, fp8w, output, m, n_dim, k, ctx, stream)
        } else {
            self.batched_nvfp4_proj(input, &self.ssm.in_proj, output, m, n_dim, k, ctx, stream)
        }
    }

    /// Batched out_proj: `[m, d_inner] → [m, h]`, arm order = `decode()`.
    fn batched_out_proj(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let k = self.d_inner as u32;
        if let Some(ref w) = self.out_proj_bf16 {
            self.batched_dense_proj(input, w, output, m, h, k, ctx, stream)
        } else if let Some(ref fp8w) = self.out_proj_fp8 {
            self.batched_fp8_proj(input, fp8w, output, m, h, k, ctx, stream)
        } else {
            self.batched_nvfp4_proj(input, &self.ssm.out_proj, output, m, h, k, ctx, stream)
        }
    }

    /// Native-BF16 arm: one pipelined tensor-core GEMM; per-row GEMV loop if
    /// the GEMM kernel is not compiled for this target (weights re-read m×,
    /// still correct — the risk-register fallback for tiny-M misbehavior).
    #[allow(clippy::too_many_arguments)]
    fn batched_dense_proj(
        &self,
        input: DevicePtr,
        w: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.dense_gemm_bf16_k.0 != 0 {
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                w,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            anyhow::ensure!(
                self.dense_gemv_bf16_k.0 != 0,
                "native BF16 SSM batched decode: neither dense_gemm_bf16_pipelined \
                 nor dense_gemv_bf16 is loaded"
            );
            for i in 0..m as usize {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_bf16_k,
                    input.offset(i * k as usize * 2),
                    w,
                    output.offset(i * n as usize * 2),
                    n,
                    k,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// Native block-scaled FP8 arm. The batch4/batch16 GEMVs are documented
    /// bit-identical to m× `w8a16_gemv` (one weight-DRAM pass); above 16 the
    /// tile GEMM takes over unconditionally.
    #[allow(clippy::too_many_arguments)]
    fn batched_fp8_proj(
        &self,
        input: DevicePtr,
        w: &Fp8Weight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if m <= 4 && self.w8a16_gemv_batch4_k.0 != 0 {
            ops::w8a16_gemv_batch4(
                ctx.gpu,
                self.w8a16_gemv_batch4_k,
                input,
                w.weight,
                w.row_scale,
                output,
                m,
                n,
                k,
                stream,
            )
        } else if m <= 16 && self.w8a16_gemv_batch16_k.0 != 0 {
            // Same wrapper: w8a16_gemv_batch16 carries the identical argument
            // list (templated MAX_M=16 twin in w8a16_gemv_batch4.cu).
            ops::w8a16_gemv_batch4(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                input,
                w.weight,
                w.row_scale,
                output,
                m,
                n,
                k,
                stream,
            )
        } else if self.w8a16_gemm_pipelined_k.0 != 0 {
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                input,
                w.weight,
                w.row_scale,
                output,
                m,
                n,
                k,
                stream,
            )
        } else if self.w8a16_gemm_k.0 != 0 {
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                w.weight,
                w.row_scale,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            // `set_fp8_weights` guarantees w8a16_gemv at load; in the
            // decode-only bisect mode the GEMM twins may be absent.
            for i in 0..m as usize {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    input.offset(i * k as usize * 2),
                    w.weight,
                    w.row_scale,
                    output.offset(i * n as usize * 2),
                    n,
                    k,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// NVFP4 arm. `w4a16_gemv_batch16` SILENTLY truncates above M=16 (rows
    /// 16.. never written), so the ladder branches to the any-M tile GEMM at
    /// m>16 unconditionally — covering the padded_n rungs 24/32/48/64/96/128.
    #[allow(clippy::too_many_arguments)]
    fn batched_nvfp4_proj(
        &self,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rung: Option<KernelHandle> = [
            (4u32, self.w4a16_gemv_batch4_k),
            (8, self.w4a16_gemv_batch8_k),
            (16, self.w4a16_gemv_batch16_k),
        ]
        .iter()
        .find(|(cap, kh)| m <= *cap && kh.0 != 0)
        .map(|&(_, kh)| kh);
        if let Some(kh) = rung {
            ops::w4a16_gemv_batchm(ctx.gpu, kh, input, w, output, m, n, k, stream)
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                input,
                w,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }
}
