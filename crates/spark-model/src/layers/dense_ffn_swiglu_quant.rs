// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: when the dense FFN's W8A8 prefill produces the down
//! projection's FP8 input in ONE launch (`swiglu_quant_fp8_ada`) instead of
//! `moe_silu_mul` then `per_token_group_quant_fp8`.
//!
//! WHY. Between the up and down GEMMs the W8A8 arm used to write the BF16
//! SwiGLU product over `gate_out` and read it straight back to quantize it:
//! 9 bytes of DRAM traffic per intermediate element where 5 suffice. On the 4B
//! Holo (intermediate 9,216) at a 3,921-token prefill that is 145 MB per layer
//! and 4.6 GB over 32 layers, about 16 ms at the RTX 4060 Ti's 288 GB/s, on a
//! kernel pair that is purely bandwidth-bound.
//!
//! Not a lever. The fused kernel's bytes and scales are bit-identical to the
//! pair's by construction (argument in the `.cu` header: same SiLU expression
//! under the same flags, the same BF16 round before the group max, the same
//! scale and encode), so it is the sm_89 default with no kill switch, like the
//! other drop-ins in `dense_ffn_ada_kernels.rs`; the A/B is a build without
//! this file. `AVAROK_FFN_W8A16_ONLY` still turns the whole W8A8 arm off.
//!
//! Clauses of [`swiglu_quant_selected`], each load-bearing:
//!
//! * the handle resolved: only `kernels/ada` compiles the module, so every
//!   other target keeps the pair and issues no lookup;
//! * `FfnActivation::SiLU`: the FP8 arm's `act_mul` is `gelu_mul` for GeLU
//!   models, and the fused kernel computes SiLU only;
//! * the down projection takes W8A8: the fused kernel writes ONLY the FP8
//!   bytes and scales, so a down projection on a W8A16 rung, which reads the
//!   BF16 product from `gate_out`, must keep the pair;
//! * `k % 128 == 0`: one scale per 128-wide group (also implied by the down
//!   projection's W8A8 rule; restated so this function stands alone).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// The selection rule as a pure function, so the CPU tests pin every clause.
pub(crate) fn swiglu_quant_selected(
    kernel: KernelHandle,
    activation: FfnActivation,
    down_w8a8: bool,
    k: u32,
) -> bool {
    kernel.0 != 0 && activation == FfnActivation::SiLU && down_w8a8 && k.is_multiple_of(128)
}

impl DenseFfnLayer {
    /// Whether this prefill's SwiGLU and down-input quantization run fused.
    pub(crate) fn swiglu_quant_fused(&self, down_w8a8: bool, k: u32) -> bool {
        swiglu_quant_selected(self.swiglu_quant_fp8_k, self.activation, down_w8a8, k)
    }

    /// `SiLU(gate) * up`, quantized per token and 128-group to E4M3, into the
    /// shared dense-FFN activation scratch: the same `(a_fp8, a_scale)`
    /// [`Self::w8a8_quant_act`] returns for the down projection, with the
    /// same bytes, and with `gate_out` left as the gate GEMM wrote it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a8_swiglu_quant(
        &self,
        ctx: &ForwardContext,
        gate: DevicePtr,
        up: DevicePtr,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let a_fp8 = ctx.buffers.ffn_act_a();
        let a_scale = ctx.buffers.ffn_act_scale();
        let rows = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(rows * k as usize <= ctx.buffers.ffn_act_a_bytes());
        debug_assert!(rows * (k as usize / 128) * 4 <= ctx.buffers.ffn_act_scale_bytes());
        if ctx.stats.once("log:ffn_swiglu_quant_fused") {
            tracing::info!(
                "[avarok] dense FFN prefill: SwiGLU + FP8 down-input quant fused \
                 (swiglu_quant_fp8_ada, bit-identical to moe_silu_mul + \
                 per_token_group_quant_fp8)"
            );
        }
        ops::swiglu_quant_fp8_ada(
            ctx.gpu,
            self.swiglu_quant_fp8_k,
            gate,
            up,
            a_fp8,
            a_scale,
            m,
            k,
            stream,
        )?;
        Ok((a_fp8, a_scale))
    }
}

#[cfg(test)]
#[path = "dense_ffn_swiglu_quant_tests.rs"]
mod tests;
