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
        dispatch: &crate::layers::ops::GemmDispatch,
        input: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        // ATLAS_W4A16_VARIANT: "v1"/"v2"/"v3" pin a kernel; 0 = auto (v2 — 3
        // CTAs/SM, 8 warps; v3 with K_STEP=64 is slower in practice, kept for
        // A/B). Resolved once per model into `GemmDispatch`, which the forward
        // pass already carries.
        let v = dispatch.w4a16_variant;
        // LOSSLESS opt-in: route QKV/o projection prefill through the BF16-TC
        // kernel (FP4→BF16 dequant + BF16 MMA, bit-identical to base w4a16_gemm)
        // instead of the default t_m128 which crushes activations to FP8 E4M3.
        // Gated by ATLAS_BF16_TC_PROJ (default off → unchanged). Removes the
        // FP8 prefill perturbation on the attention projections.
        // Load-time weight prep runs before any `TransformerModel` exists to
        // carry the levers, so this resolves at the point of use. The
        // interpretation stays SSOT in `ModelLevers`. Resolved ONCE: this is
        // called per attention layer per prefill chunk, and a full
        // `from_env()` here re-read a dozen env vars (a process-global lock)
        // and re-ran the drafter-context resolution — whose info line then
        // spammed ~4 lines per prefill chunk × layer (193 on one warm turn).
        // The env cannot change under a running process, so a OnceLock is
        // exact, not a staleness risk.
        static BF16_PROJ: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let bf16_proj =
            *BF16_PROJ.get_or_init(|| crate::layers::ops::ModelLevers::from_env().bf16_tc_proj);
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

    /// Install keep-packed ternary Q2_0 q/k/v/o weights (Tier-1c,
    /// `ATLAS_GGUF_NATIVE_Q2=1`). Decode dispatches `q2_0_gemv_vec` (2-bit
    /// resident, no NVFP4); prefill transient-dequants each to BF16 via
    /// `Self::q2_prefill_gemm`. Replaces the NVFP4 decode weights (which are
    /// NULL on this path — no NVFP4 was allocated).
    pub fn set_packed_q2_weights(
        &mut self,
        q: crate::weight_map::PackedQ2Weight,
        k: crate::weight_map::PackedQ2Weight,
        v: crate::weight_map::PackedQ2Weight,
        o: crate::weight_map::PackedQ2Weight,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) {
        self.q_weight = Some(QuantWeight::PackedQ2(q));
        self.k_weight = Some(QuantWeight::PackedQ2(k));
        self.v_weight = Some(QuantWeight::PackedQ2(v));
        self.o_weight = Some(QuantWeight::PackedQ2(o));
        // Resolved here, not in the constructor: these ship only in
        // GGUF-serving targets and the boot audit fails closed on an
        // unconditional probe everywhere else.
        self.q2_0_mmq_nc_k = crate::layers::try_kernel(gpu, "q2_0_mmq", "atlas_q2_0_mmq128_nc");
        self.q2_0_mmq_wc_k = crate::layers::try_kernel(gpu, "q2_0_mmq", "atlas_q2_0_mmq128_wc");
        self.q4k_quant_act_k =
            crate::layers::try_kernel(gpu, "q4k_mmq", "atlas_q8_1_quantize_ds4_bf16");
    }

    /// Transient-dequant prefill GEMM for a keep-packed Q2_0 projection: dequant
    /// the 2-bit weight `[n, k]` into the caller-provided PERSISTENT BF16
    /// `scratch` (the arena `q2_dequant_scratch`, sized to the largest packed
    /// projection), run the BF16 `dense_gemm` (`out[m,n] = in[m,k] @ w^T`).
    /// Mirrors `DenseFfnLayer`'s FFN prefill — the resident weight stays 2-bit.
    /// No per-matmul alloc/sync/free: the dequant is ordered before the GEMM on
    /// the same `stream`, and consecutive projections reuse `scratch` because
    /// each GEMM consumes it before the next dequant overwrites it. Returns an
    /// error if the dequant kernel is absent in this build.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn q2_prefill_gemm(
        &self,
        gpu: &dyn GpuBackend,
        w: &crate::weight_map::PackedQ2Weight,
        input: DevicePtr,
        out: DevicePtr,
        scratch: DevicePtr,
        act_q8: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let (n, k) = (w.n, w.k);

        // Tier-2 native MMQ (ATLAS_GGUF_NATIVE_Q2_MMQ=1): quantize `input` to q8_1
        // then run the packed 2-bit MMQ GEMM — no BF16 weight dequant, no shared
        // `q2_dequant_scratch` race. Group-128 only (else fall through).
        if self.q2_0_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && crate::layers::ops::native_q2_mmq_enabled()
            && w.group == 128
        {
            crate::layers::ops::quantize_act_q8_1(
                gpu,
                self.q4k_quant_act_k,
                input,
                act_q8,
                m,
                k,
                stream,
            )?;
            return crate::layers::ops::q2_0_mmq_gemm(
                gpu,
                self.q2_0_mmq_nc_k,
                self.q2_0_mmq_wc_k,
                act_q8,
                w.weight,
                out,
                m,
                n,
                k,
                stream,
            );
        }

        if self.dequant_q2_0_gn_k.0 == 0 {
            anyhow::bail!(
                "dequant_q2_0_gn_to_bf16 kernel missing — packed-Q2 attention prefill unavailable"
            );
        }
        crate::layers::ops::dequant_q2_0_gn_to_bf16(
            gpu,
            self.dequant_q2_0_gn_k,
            w.weight,
            scratch,
            n,
            k,
            w.group as u32,
            stream,
        )?;
        let dw = crate::weight_map::DenseWeight { weight: scratch };
        if self.dense_gemm_pipelined_k.0 != 0 {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        } else {
            crate::layers::ops::dense_gemm(
                gpu,
                self.dense_gemm_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        }
        Ok(())
    }

    /// Keep-packed Q2_0 (Tier-1c) prefill dispatch guard, shared by the QKV
    /// (`paged_qkv` / `cache_skip_qkv`) and o_proj call sites: when `weight`
    /// is the keep-packed variant, run [`Self::q2_prefill_gemm`] with the
    /// arena scratch buffers and return `Some(result)`. `None` = not packed
    /// Q2_0 — callers fall through to their NVFP4/FP8/dense arms. Must be
    /// checked FIRST: those fallbacks all read NULL pointers on this path.
    pub(crate) fn try_q2_prefill(
        &self,
        ctx: &crate::layer::ForwardContext,
        weight: Option<&QuantWeight>,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        let q2 = weight.and_then(|w| w.as_packed_q2())?;
        debug_assert!(
            (q2.n as usize) * (q2.k as usize) * 2 <= ctx.buffers.q2_dequant_scratch_bytes(),
            "packed-Q2 prefill dequant scratch too small"
        );
        let scratch = ctx.buffers.q2_dequant_scratch();
        let act_q8 = ctx.buffers.q2_act_q8();
        Some(self.q2_prefill_gemm(ctx.gpu, q2, input, out, scratch, act_q8, m, stream))
    }

    /// Install the fused [q|k|v] transposed twin. Separate from
    /// `set_prefill_weights` so the fused path is opt-in per loader and the
    /// separate twins stay available as the fallback.
    pub fn set_fused_qkv_prefill_weight(&mut self, qkv_nvfp4_t: Option<QuantizedWeight>) {
        self.qkv_nvfp4_t = qkv_nvfp4_t;
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

    /// Feature-1: install this layer's MoE router + routed-expert LoRA onto its
    /// `FfnComponent::Moe`. The MoE FFN lives in `self.ffn` or (some loaders)
    /// `self.moe_ffn` — try both, else the adapter targeted experts on a layer
    /// with no MoE FFN (hard reject). Scratch is allocated inside
    /// `crate::layers::MoeLayer::set_lora_weights`.
    pub fn set_moe_lora_weights(
        &mut self,
        router: Option<crate::layers::ops::lora_delta::LoraPair>,
        experts: crate::lora::ExpertLoraLayer,
        kernels: crate::layers::ops::lora_delta::LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if let crate::layers::FfnComponent::Moe(m) = &mut self.ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        if let Some(crate::layers::FfnComponent::Moe(m)) = &mut self.moe_ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        anyhow::bail!("LoRA: router/expert deltas installed on a layer with no MoE FFN component")
    }

    /// Transpose FP8 weights for fast prefill (`w8a16_gemm_t`: coalesced
    /// reads). Must be called after [`Self::set_fp8_weights`]. Allocates
    /// new GPU buffers.
    pub fn transpose_fp8_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        // Load-time decision, taken in the weight loader before any
        // `TransformerModel` exists to carry the config. Resolved at the point
        // of use rather than cached in a static: the resolution logic stays
        // SSOT in `GemmDispatch`, and one getenv per layer at load is free.
        if crate::layers::ops::GemmDispatch::from_env().cutlass_nvfp4_gemm {
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
        // Load-time decision, taken in the weight loader before any
        // `TransformerModel` exists to carry the config. Resolved at the point
        // of use rather than cached in a static: the resolution logic stays
        // SSOT in `GemmDispatch`, and one getenv per layer at load is free.
        if crate::layers::ops::GemmDispatch::from_env().cutlass_nvfp4_gemm {
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
