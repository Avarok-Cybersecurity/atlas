// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3AttentionLayer` prefill-side weight setup: transposed NVFP4 /
//! FP8 copies, FP8 weight installation, FP8 transpose for fast prefill,
//! and NVFP4→FP8 pre-dequant for zero-overhead prefill GEMMs. Also
//! hosts the W4A16 M=128 GEMM dispatcher (selects v1/v2/v3 by env).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::Qwen3AttentionLayer;
use crate::weight_map::{Fp8Weight, QuantWeight, QuantizedWeight};

impl Qwen3AttentionLayer {
    /// Dispatch the M=128 W4A16 prefill GEMM. Routes to the v2 shadow
    /// kernel when available (MiniMax-only), otherwise to the v1 kernel.
    /// Args mirror [`crate::layers::ops::w4a16_gemm_n128_m128`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w4a16_gemm_m128_dispatch(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        // ATLAS_W4A16_VARIANT env: "v1", "v2", "v3" — overrides auto.
        // Default: v2 (3 CTAs/SM, 8 warps). v3 (K_STEP=64, 1 CTA/SM) is
        // slower in practice; keep it available for A/B but don't default.
        static VARIANT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let v =
            *VARIANT.get_or_init(
                || match std::env::var("ATLAS_W4A16_VARIANT").ok().as_deref() {
                    Some("v1") => 1,
                    Some("v2") => 2,
                    Some("v3") => 3,
                    _ => 0, // auto (prefer v2)
                },
            );
        // LOSSLESS opt-in: route QKV/o projection prefill through the BF16-TC
        // kernel (FP4→BF16 dequant + BF16 MMA, bit-identical to base w4a16_gemm)
        // instead of the default t_m128 which crushes activations to FP8 E4M3.
        // Gated by ATLAS_BF16_TC_PROJ (default off → unchanged). Removes the
        // FP8 prefill perturbation on the attention projections.
        static BF16_PROJ: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let bf16_proj = *BF16_PROJ.get_or_init(|| std::env::var_os("ATLAS_BF16_TC_PROJ").is_some());
        if bf16_proj && self.w4a16_gemm_t_m128_bf16_k.0 != 0 {
            return crate::layers::ops::w4a16_gemm_n128_m128_bf16(
                gpu,
                self.w4a16_gemm_t_m128_bf16_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        if v == 3 && self.w4a16_gemm_t_m128_v3_k.0 != 0 {
            crate::layers::ops::w4a16_gemm_n128_m128_v3(
                gpu,
                self.w4a16_gemm_t_m128_v3_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else if v != 1 && self.w4a16_gemm_t_m128_v2_k.0 != 0 {
            crate::layers::ops::w4a16_gemm_n128_m128_v2(
                gpu,
                self.w4a16_gemm_t_m128_v2_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            crate::layers::ops::w4a16_gemm_n128_m128(
                gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Set transposed NVFP4 weight copies for prefill GEMM
    /// (`w4a16_gemm_t`, N_TILE=128).
    pub fn set_prefill_weights(
        &mut self,
        q_nvfp4_t: Option<QuantizedWeight>,
        k_nvfp4_t: Option<QuantizedWeight>,
        v_nvfp4_t: Option<QuantizedWeight>,
        o_nvfp4_t: Option<QuantizedWeight>,
    ) {
        self.q_nvfp4_t = q_nvfp4_t;
        self.k_nvfp4_t = k_nvfp4_t;
        self.v_nvfp4_t = v_nvfp4_t;
        self.o_nvfp4_t = o_nvfp4_t;
    }

    /// Set native FP8 checkpoint weights for the `w8a16_gemv` decode path.
    ///
    /// The block-scaled FP8 weights stored here (weight + per-128 `row_scale`)
    /// are ALSO consumed by block-scaled prefill: `fp8_gemm_t_blockscaled`
    /// folds both the per-token activation scale and the per-block weight
    /// scale in an FP32 epilogue. (Historical note: the older single-scale
    /// `fp8_gemm_t`/`fp8_gemm_n128` prefill could not apply block scales, so
    /// prefill used to fall through to the NVFP4/BF16 dequant path — that is
    /// no longer the case; block-scaled prefill is the default, see
    /// `ops::fp8_blockscaled_prefill_enabled`.)
    pub fn set_fp8_weights(
        &mut self,
        q: Option<Fp8Weight>,
        k: Option<Fp8Weight>,
        v: Option<Fp8Weight>,
        o: Option<Fp8Weight>,
    ) {
        // Overwrite decode weights with FP8 variant. Replaces any NVFP4
        // weights set during construction.
        if let Some(qw) = q {
            self.q_weight = Some(QuantWeight::Fp8(qw));
        }
        if let Some(kw) = k {
            self.k_weight = Some(QuantWeight::Fp8(kw));
        }
        if let Some(vw) = v {
            self.v_weight = Some(QuantWeight::Fp8(vw));
        }
        if let Some(ow) = o {
            self.o_weight = Some(QuantWeight::Fp8(ow));
        }
    }

    /// Install the startup-static LoRA adapter overlay (post-construction,
    /// mirroring [`Self::set_fp8_weights`]). `attn` carries the K/V/O pairs;
    /// `ffn` (when Some) is routed into this layer's dense FFN component —
    /// it lives here rather than on the model because `self.ffn` is
    /// `pub(super)`. M0: weights are stored only; compute reads land in M1.
    pub fn set_lora_weights(
        &mut self,
        attn: crate::layers::ops::lora_delta::LoraAttnWeights,
        ffn: Option<crate::layers::ops::lora_delta::LoraFfnWeights>,
    ) -> Result<()> {
        self.lora = Some(attn);
        if let Some(f) = ffn {
            match &mut self.ffn {
                crate::layers::FfnComponent::Dense(d) => d.set_lora_weights(f)?,
                _ => anyhow::bail!("LoRA: FFN targets on a non-dense FFN layer"),
            }
        }
        Ok(())
    }

    /// Transpose FP8 weights for fast prefill (`w8a16_gemm_t`: coalesced
    /// reads). Must be called after [`Self::set_fp8_weights`]. Allocates
    /// new GPU buffers.
    pub fn transpose_fp8_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        if crate::layers::ops::cutlass_nvfp4_gemm_enabled() {
            tracing::info!(
                "Skipping attention FP8 prefill transposes because ATLAS_CUTLASS_NVFP4_GEMM=1"
            );
            return Ok(());
        }
        if self.w8a16_gemm_t_k.0 == 0 {
            return Ok(()); // kernel not available
        }
        let transpose_k = gpu.kernel("w8a16_gemm_t", "transpose_fp8")?;
        let transpose_scale_k = gpu.kernel("w8a16_gemm_t", "transpose_block_scale")?;

        if let Some(w) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.q_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.k_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.k_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.v_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.v_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            self.o_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        Ok(())
    }

    /// Pre-dequant NVFP4 → FP8 for Q/K/V/O transposed weights.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        // Under native NVFP4 prefill (ATLAS_CUTLASS_NVFP4_GEMM=1) all of Q/K/V/O
        // take the CUTLASS NVFP4 path; the FP8 predequant outputs (q_fp8..o_fp8)
        // are read only by the legacy FP8 prefill path and decode never reads
        // them (decode attention uses its own weights), so they'd be allocated
        // at load and never used. Skip them — saves ~260MB and a wasted per-
        // prefill BF16->FP8 activation conversion. Mirrors transpose_fp8_for_prefill.
        if crate::layers::ops::cutlass_nvfp4_gemm_enabled() {
            tracing::info!(
                "Skipping attention FP8 prefill predequant because ATLAS_CUTLASS_NVFP4_GEMM=1"
            );
            return Ok(());
        }
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = nkv * hd;

        // Use NON-transposed weights for predequant.
        // `predequant_nvfp4_to_fp8` assumes [N, K/2] input layout.
        if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.q_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, q_proj_dim, h, stream)?);
        }
        if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.k_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, kv_dim, h, stream)?);
        }
        if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.v_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, kv_dim, h, stream)?);
        }
        // O proj: use attn.o_proj (non-transposed QuantizedWeight)
        if self.o_nvfp4_t.is_some() {
            self.o_fp8 =
                Some(
                    self.attn
                        .o_proj
                        .predequant_to_fp8(gpu, predequant_k, h, q_dim, stream)?,
                );
        }
        Ok(())
    }
}
