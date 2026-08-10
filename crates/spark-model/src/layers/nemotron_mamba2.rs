// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H Mamba-2 SSM layer implementing TransformerLayer.
//!
//! Standalone SSM layer (no FFN component). Forward pass:
//!   1. RMS norm (standard weight*x scaling)
//!   2. in_proj GEMV → [z, xBC, dt]
//!   3. Conv1d update on xBC (WITH bias, fused SiLU)
//!   4. Split xBC_out → x, B, C
//!   5. Mamba-2 SSM decode (state update + output)
//!   6. Gated RMS norm: rms_norm(y, ssm_norm) * silu(z)
//!   7. out_proj GEMV
//!   8. Residual add

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8Weight, NemotronSsmWeights, QuantizedWeight};

mod prefill;
mod trait_impl;

#[allow(dead_code)]
pub struct NemotronMamba2Layer {
    input_norm: DenseWeight,
    ssm: NemotronSsmWeights,
    // FP8 native weights (skip double-quantization FP8→BF16→NVFP4)
    in_proj_fp8: Option<Fp8Weight>,
    out_proj_fp8: Option<Fp8Weight>,
    // Transposed NVFP4 weights for fast prefill GEMM (FP8 MMA, N128, cp.async)
    in_proj_t: Option<QuantizedWeight>,
    out_proj_t: Option<QuantizedWeight>,
    // Pre-dequantized FP8 E4M3 copies of the two SSM projections, [N, K].
    // Consumed by `fp8_gemm_t`, which has NO dequant phase at all.
    in_proj_pd_fp8: Option<DevicePtr>,
    out_proj_pd_fp8: Option<DevicePtr>,
    // Kernel handles — decode
    rms_norm_residual_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w8a16_gemv_k: KernelHandle,
    conv1d_update_k: KernelHandle,
    mamba2_ssm_k: KernelHandle,
    gated_rms_norm_k: KernelHandle,
    residual_add_k: KernelHandle,
    // Kernel handles — prefill (GEMM + batched kernels)
    w4a16_gemm_k: KernelHandle,
    w4a16_gemm_t_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    fp8_gemm_t_k: KernelHandle,
    fp8_fp8_gemm_t_k: KernelHandle,
    bf16_to_fp8_k: KernelHandle,
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
    conv1d_prefill_k: KernelHandle,
    mamba2_ssm_prefill_k: KernelHandle,
    mamba2_ssm_prefill_persistent_k: KernelHandle,
    // SSD chunked prefill scan (tensor-core; ceil(T/64) serial links instead of T).
    ssd_cumsum_k: KernelHandle,
    ssd_bmm_k: KernelHandle,
    ssd_scan_k: KernelHandle,
    // Pre-computed dimensions
    d_inner: usize,
    d_xbc: usize,
    in_proj_size: usize,
    num_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
    d_conv: usize,
    h_state_bytes: usize,
    conv_state_bytes: usize,
    layer_idx: usize,
}

impl NemotronMamba2Layer {
    pub fn new(
        input_norm: DenseWeight,
        ssm: NemotronSsmWeights,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        layer_idx: usize,
    ) -> Result<Self> {
        let num_heads = config.mamba_num_heads;
        let head_dim = config.mamba_head_dim;
        let state_size = config.ssm_state_size;
        let n_groups = config.n_groups;
        let d_conv = config.linear_conv_kernel_dim;
        let d_inner = config.mamba2_d_inner();
        let d_xbc = config.mamba2_d_xbc();
        let in_proj_size = config.mamba2_in_proj_size();

        Ok(Self {
            input_norm,
            ssm,
            in_proj_fp8: None,
            out_proj_fp8: None,
            in_proj_t: None,
            out_proj_t: None,
            in_proj_pd_fp8: None,
            out_proj_pd_fp8: None,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w8a16_gemv_k: super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            conv1d_update_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            mamba2_ssm_k: gpu.kernel("mamba2_ssm", "mamba2_ssm_decode")?,
            gated_rms_norm_k: gpu.kernel("norm", "gated_rms_norm")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            fp8_gemm_t_k: super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128_mfast"),
            fp8_fp8_gemm_t_k: super::try_kernel(gpu, "w4a16", "fp8_fp8_gemm_t_m128_mfast"),
            bf16_to_fp8_k: super::try_kernel(gpu, "w4a16", "bf16_to_fp8"),
            w4a4_gemm_k: super::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            conv1d_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            mamba2_ssm_prefill_k: gpu.kernel("mamba2_ssm", "mamba2_ssm_prefill")?,
            ssd_cumsum_k: super::try_kernel(gpu, "mamba2_ssd_chunk", "mamba2_ssd_cumsum"),
            ssd_bmm_k: super::try_kernel(gpu, "mamba2_ssd_chunk", "mamba2_ssd_bmm"),
            ssd_scan_k: super::try_kernel(gpu, "mamba2_ssd_chunk", "mamba2_ssd_scan"),
            mamba2_ssm_prefill_persistent_k: super::try_kernel(
                gpu,
                "mamba2_ssm",
                "mamba2_ssm_prefill_persistent",
            ),
            d_inner,
            d_xbc,
            in_proj_size,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            d_conv,
            h_state_bytes: num_heads * head_dim * state_size * 4, // FP32
            conv_state_bytes: d_xbc * d_conv * 4,                 // FP32
            layer_idx,
        })
    }

    /// Set native FP8 weights to skip double-quantization (FP8→BF16→NVFP4).
    /// When set, decode uses w8a16_gemv (FP8 LUT kernel) instead of w4a16_gemv.
    pub fn set_fp8_weights(&mut self, in_proj: Option<Fp8Weight>, out_proj: Option<Fp8Weight>) {
        self.in_proj_fp8 = in_proj;
        self.out_proj_fp8 = out_proj;
    }

    /// Access SSM weights (needed by weight loader for transpose).
    pub fn ssm_weights(&self) -> &NemotronSsmWeights {
        &self.ssm
    }

    /// Set transposed NVFP4 weights for fast prefill GEMM (FP8 MMA, N128, cp.async).
    /// Switches prefill from w4a16_gemm (M64,N64,K16 BF16) to w4a16_gemm_t
    /// (M64,N128,K32 FP8 MMA) — est. 3-4x TTFT improvement for SSM layers.
    pub fn set_prefill_weights(
        &mut self,
        in_proj_t: Option<QuantizedWeight>,
        out_proj_t: Option<QuantizedWeight>,
    ) {
        self.in_proj_t = in_proj_t;
        self.out_proj_t = out_proj_t;
    }

    /// Set pre-dequantized FP8 E4M3 copies of in_proj/out_proj for prefill.
    ///
    /// `w4a16_gemm_t_m128` dequantizes its NVFP4 B tile from FP4 to FP8 in
    /// shared memory on every K step, and that work is redone by every M-block:
    /// the cost is N*K*(M/M_TILE), so a 1k-token prefill pays for it 8x over.
    /// Measured on Puzzle: ablating just that dequant ALU cut a 1k prefill from
    /// 557 ms to 424 ms. Converting the weights once at load time removes it
    /// entirely and lets prefill use `fp8_gemm_t`, which has no dequant phase.
    pub fn set_fp8_prefill_weights(&mut self, in_proj: DevicePtr, out_proj: DevicePtr) {
        self.in_proj_pd_fp8 = Some(in_proj);
        self.out_proj_pd_fp8 = Some(out_proj);
    }

    /// Conv1d update with bias (Nemotron conv1d has learned bias, unlike Qwen3).
    ///
    /// Kernel: `causal_conv1d_update(conv_state, input, weight, bias, output,
    ///          batch, dim, d_conv)`
    fn conv1d_update_biased(
        &self,
        gpu: &dyn GpuBackend,
        conv_state: DevicePtr,
        input: DevicePtr,
        output: DevicePtr,
        d_inner: u32,
        d_conv: u32,
        batch_size: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.conv1d_update_k)
            .grid([div_ceil(d_inner, 256), batch_size, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_ptr(self.ssm.conv1d_weight.weight)
            .arg_ptr(self.ssm.conv1d_bias.weight)
            .arg_ptr(output)
            .arg_u32(batch_size)
            .arg_u32(d_inner)
            .arg_u32(d_conv)
            .launch(stream)
    }

    /// Launch Mamba-2 SSM decode kernel.
    ///
    /// Grid: (num_heads, batch, 1)  Block: (state_size, 1, 1)
    #[allow(clippy::too_many_arguments)]
    fn ssm_decode(
        &self,
        gpu: &dyn GpuBackend,
        h_state: DevicePtr,
        x: DevicePtr,
        b_proj: DevicePtr,
        c_proj: DevicePtr,
        dt_raw: DevicePtr,
        output: DevicePtr,
        batch_size: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.mamba2_ssm_k)
            .grid([self.num_heads as u32, batch_size, 1])
            .block([self.state_size as u32, 1, 1])
            .arg_ptr(h_state)
            .arg_ptr(x)
            .arg_ptr(b_proj)
            .arg_ptr(c_proj)
            .arg_ptr(dt_raw)
            .arg_ptr(self.ssm.a_log.weight)
            .arg_ptr(self.ssm.d_param.weight)
            .arg_ptr(self.ssm.dt_bias.weight)
            .arg_ptr(output)
            .arg_u32(batch_size)
            .arg_u32(self.num_heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_u32(self.state_size as u32)
            .arg_u32(self.n_groups as u32)
            .arg_f32(1e-9) // dt_min (no effective clamp — reference uses no clamping)
            .arg_f32(1e9) // dt_max (no effective clamp — reference uses no clamping)
            .launch(stream)
    }
}
