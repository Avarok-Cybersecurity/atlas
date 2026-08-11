// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-expert UP projection dispatch for `NemotronMoeLayer::prefill`
//! (native FP8 → W4A4 → pre-dequant FP8 → transposed NVFP4 → plain W4A16).
//! Split from `nemotron_moe.rs` (500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMoeLayer {
    /// Shared expert UP GEMM: `[N, h] × [h, shared_inter] → [N, shared_inter]`.
    ///
    /// Native FP4 tensor cores: shared_up consumed in its ORIGINAL NVFP4 form
    /// (no FP8 or transposed copies), activations quantized to NVFP4 in one
    /// pass. Same gates as the SSM W4A4 path; ATLAS_NO_SHARED_W4A4=1 disables.
    /// Native FP8 wins over w4a4. w4a4 quantizes the ACTIVATIONS to 4 bits as
    /// well as the weights, and it is the default for every prompt >= 512
    /// tokens — i.e. every real request — so leaving it ahead of the native
    /// arm would have kept the worst-precision path on exactly the traffic
    /// that matters while the native weights sat unused.
    pub(super) fn prefill_shared_up(
        &self,
        normed: DevicePtr,
        shared_up_out_base: DevicePtr,
        n: u32,
        h: usize,
        shared_inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let native_shared_up = self.weights.shared_up_fp8.is_some()
            && (self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0);
        let w4a4 = !native_shared_up
            && n >= 512
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (shared_inter as usize).max(h) * (n as usize)
            && std::env::var("ATLAS_NO_SHARED_W4A4").is_err();
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
                &self.weights.shared_up,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if let Some(fp8w) = self
            .weights
            .shared_up_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0)
        {
            // Native FP8 from the checkpoint — no NVFP4 requant, no derived copy.
            // `w4a4` is suppressed above when this is available (see there).
            let (kern, pipelined) = if self.w8a16_gemm_pipelined_k.0 != 0 {
                (self.w8a16_gemm_pipelined_k, true)
            } else {
                (self.w8a16_gemm_k, false)
            };
            let f = if pipelined {
                ops::w8a16_gemm_pipelined
            } else {
                ops::w8a16_gemm
            };
            f(
                ctx.gpu,
                kern,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if let Some(w_fp8) = self.shared_up_pd_fp8 {
            ops::fp8_gemm_m128_mfast(
                ctx.gpu,
                self.fp8_gemm_m128_k,
                normed,
                w_fp8,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        } else if let Some(ref sut) = self.shared_up_t {
            // Same NVFP4 weights, better kernel: w4a16_gemm_t_m128 tiles M at 128
            // (half the B panel passes of w4a16_gemm_t's 64) and puts M on the fast
            // grid axis so those passes hit L2. Costs nothing extra -- the transposed
            // copy already exists -- and needs no FP8 residency.
            if n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    sut,
                    shared_up_out_base,
                    n,
                    shared_inter,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    sut,
                    shared_up_out_base,
                    n,
                    shared_inter,
                    h as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                &self.weights.shared_up,
                shared_up_out_base,
                n,
                shared_inter,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
