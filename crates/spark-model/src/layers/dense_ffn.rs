// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(SiLU(gate) * up)
//! 2 fused kernel launches per decode token (dual GEMV + SiLU-fused down GEMV).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight};

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Native FP8 E4M3 dense MLP weights with 2D block scales (`[N/128, K/128]`
/// BF16). Used when the checkpoint ships native FP8 (e.g. Qwen3.6-27B-FP8)
/// and we want to avoid the FP8 → BF16 → NVFP4 4-bit re-quantization that
/// the default `load_dense_ffn` path would otherwise perform — that
/// downgrade is what causes the documented "prose-attractor" failure mode
/// on the dense Qwen3.6 27B (see `kernels/gb10/qwen3.6-27b/MODEL.toml`).
/// When installed via `set_fp8_weights`, the forward paths dispatch to
/// `w8a16_gemv` (decode) and `w8a16_gemm` (prefill) — same kernels the
/// attention layer's native-FP8 path uses. Spec-decode batched paths
/// (`forward_k2`, `forward_k3`) bail explicitly: no fused FP8 batch2/3
/// kernels exist today, and Qwen3.6-27B ships with `mtp_layers = 0` so
/// the bail is unreachable in practice.
pub struct Fp8DenseFfnWeights {
    pub gate_proj: Fp8Weight,
    pub up_proj: Fp8Weight,
    pub down_proj: Fp8Weight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    w4a16_gemm: KernelHandle,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    /// Native FP8 dense MLP weights — when `Some`, all forward paths use
    /// the `w8a16_gemv` / `w8a16_gemm` kernels instead of w4a16 NVFP4.
    /// Set via `set_fp8_weights`. See [`Fp8DenseFfnWeights`].
    fp8_weights: Option<Fp8DenseFfnWeights>,
    w8a16_gemv_k: KernelHandle,
    w8a16_gemm_k: KernelHandle,
}

impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        // Native FP8 path kernels — optional. Set by `set_fp8_weights`
        // when the checkpoint ships native FP8 (Qwen3.6-27B-FP8).
        // `try_kernel` returns `KernelHandle(0)` on miss; the loader
        // checks for that and refuses to install FP8 weights when the
        // kernel module isn't built. Same `w8a16_gemv` / `w8a16_gemm`
        // modules the attention layer's native-FP8 path uses.
        let w8a16_gemv_k = super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv");
        let w8a16_gemm_k = super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm");

        Ok(Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            fp8_weights: None,
            w8a16_gemv_k,
            w8a16_gemm_k,
        })
    }

    /// Install native FP8 dense MLP weights. After this call, the
    /// non-spec-decode forward paths dispatch to `w8a16_gemv` /
    /// `w8a16_gemm` instead of w4a16 NVFP4. Returns an error if the
    /// `w8a16_gemv` / `w8a16_gemm` kernels are not present on this
    /// build of Atlas — the caller should fall back to NVFP4 in that
    /// case rather than silently mis-dispatching.
    pub fn set_fp8_weights(
        &mut self,
        gate: Fp8Weight,
        up: Fp8Weight,
        down: Fp8Weight,
    ) -> Result<()> {
        if self.w8a16_gemv_k.0 == 0 || self.w8a16_gemm_k.0 == 0 {
            anyhow::bail!(
                "DenseFfnLayer::set_fp8_weights called but w8a16_gemv / w8a16_gemm kernels \
                 are not registered for this target. Build with FP8 kernel support or fall \
                 back to NVFP4 / BF16."
            );
        }
        self.fp8_weights = Some(Fp8DenseFfnWeights {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
        Ok(())
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Native FP8 dispatch: per-projection w8a16_gemv. No fused
        // dual-FP8-GEMV kernel today; three sequential launches
        // (gate, up, down) plus the existing silu_mul. FP8 reads are
        // 2x smaller than BF16, so launch overhead is a larger relative
        // share — still faster than NVFP4 dequant-on-the-fly because
        // we avoid the BF16→NVFP4 precision loss this path was designed
        // to fix on Qwen3.6-27B-FP8.
        if let Some(ref fp8w) = self.fp8_weights {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                input,
                fp8w.gate_proj.weight,
                fp8w.gate_proj.row_scale,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                input,
                fp8w.up_proj.weight,
                fp8w.up_proj.row_scale,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gate_out,
                fp8w.down_proj.weight,
                fp8w.down_proj.row_scale,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2
        ops::w4a16_gemv_dual(
            ctx.gpu,
            self.w4a16_gemv_dual,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                ops::w4a16_gemv_silu_input(
                    ctx.gpu,
                    self.w4a16_gemv_silu_input,
                    gate_out,
                    up_out,
                    &self.weights.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    inter,
                    stream,
                )?;
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
            }
        }

        Ok(output)
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        // Native FP8 K=2: dispatch as 2× single-token w8a16_gemv with
        // per-token input/output offsets. The general w8a16_gemm has
        // an unverified numerical regression at M=2 on dense FFN
        // shapes (observed during 27B-FP8 MTP verify: drafts cascade
        // into 19\n\n\n19 / CSS-block repeats). Using the M=1 GEMV
        // — the same kernel the decode path uses — is guaranteed
        // bit-exact with decode-step verify. Cost: 6 launches instead
        // of 3, but K=2 verify happens at most once per accepted
        // token, so the absolute overhead is small.
        if let Some(ref fp8w) = self.fp8_weights {
            const BF16: usize = 2;
            let gate_out = ctx.buffers.expert_gate_out();
            let up_out = ctx.buffers.expert_up_out();
            let h_us = h as usize;
            let inter_us = inter as usize;
            for tok in 0..2 {
                let in_ptr = input.offset(tok * h_us * BF16);
                let g_ptr = gate_out.offset(tok * inter_us * BF16);
                let u_ptr = up_out.offset(tok * inter_us * BF16);
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    in_ptr,
                    fp8w.gate_proj.weight,
                    fp8w.gate_proj.row_scale,
                    g_ptr,
                    inter,
                    h,
                    stream,
                )?;
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    in_ptr,
                    fp8w.up_proj.weight,
                    fp8w.up_proj.row_scale,
                    u_ptr,
                    inter,
                    h,
                    stream,
                )?;
            }
            // silu_mul over the contiguous [2 * inter] BF16 stream.
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                2 * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            for tok in 0..2 {
                let g_ptr = gate_out.offset(tok * inter_us * BF16);
                let out_ptr = output.offset(tok * h_us * BF16);
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    g_ptr,
                    fp8w.down_proj.weight,
                    fp8w.down_proj.row_scale,
                    out_ptr,
                    h,
                    inter,
                    stream,
                )?;
            }
            return Ok(());
        }

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        // Native FP8 K=3: same per-token w8a16_gemv dispatch as
        // `forward_k2`. See that comment for rationale.
        if let Some(ref fp8w) = self.fp8_weights {
            const BF16: usize = 2;
            let gate_out = ctx.buffers.expert_gate_out();
            let up_out = ctx.buffers.expert_up_out();
            let h_us = h as usize;
            let inter_us = inter as usize;
            for tok in 0..3 {
                let in_ptr = input.offset(tok * h_us * BF16);
                let g_ptr = gate_out.offset(tok * inter_us * BF16);
                let u_ptr = up_out.offset(tok * inter_us * BF16);
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    in_ptr,
                    fp8w.gate_proj.weight,
                    fp8w.gate_proj.row_scale,
                    g_ptr,
                    inter,
                    h,
                    stream,
                )?;
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    in_ptr,
                    fp8w.up_proj.weight,
                    fp8w.up_proj.row_scale,
                    u_ptr,
                    inter,
                    h,
                    stream,
                )?;
            }
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                3 * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            for tok in 0..3 {
                let g_ptr = gate_out.offset(tok * inter_us * BF16);
                let out_ptr = output.offset(tok * h_us * BF16);
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    g_ptr,
                    fp8w.down_proj.weight,
                    fp8w.down_proj.row_scale,
                    out_ptr,
                    h,
                    inter,
                    stream,
                )?;
            }
            return Ok(());
        }

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens
        ops::w4a16_gemv_dual_batch3(
            ctx.gpu,
            self.w4a16_gemv_dual_batch3,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            3 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// N-token prefill: GEMM for all projections.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Native FP8 prefill dispatch: w8a16_gemm for all three
        // projections + silu_mul. Mirrors the attention layer's FP8
        // prefill path (`paged_qkv.rs::prefill_one_proj`). Uses the
        // 2D block-scale layout that ships in the FP8 checkpoint —
        // no transpose / predequant required for the M>1 kernel.
        if let Some(ref fp8w) = self.fp8_weights {
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                fp8w.gate_proj.weight,
                fp8w.gate_proj.row_scale,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                fp8w.up_proj.weight,
                fp8w.up_proj.row_scale,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                gate_out,
                fp8w.down_proj.weight,
                fp8w.down_proj.row_scale,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // BF16 prefill dispatch: dense_gemm_bf16 for all three projections.
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // gate_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;

        // up_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;

        // activation(gate) * up for all M tokens (SiLU or GELU)
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;

        // down_proj GEMM: [M, inter] → [M, H]
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// Batched forward (per-token loop). Used by forward_batched in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}
