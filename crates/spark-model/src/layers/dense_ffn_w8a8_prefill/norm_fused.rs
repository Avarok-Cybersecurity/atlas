// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the prefill FFN-norm with the dense FFN's W8A8 gate/up
//! activation quantization fused into it (`kernels/ada/common/rmsnorm_quant_fp8_ada.cu`),
//! behind the `rmsnorm_fp8_quant_fused` row (off by default on every target),
//! and the hand-off that lets [`super::DenseFfnLayer::w8a8_quant_act`] skip the
//! quantizer launch whose bytes the norm already wrote.
//!
//! # The flow
//!
//! A Qwen3 / Qwen3.5 prefill layer ends with
//! `residual_add_rms_norm(hidden, src) -> norm_output`, then
//! `ffn.forward_prefill(norm_output)`, whose W8A8 arm starts with
//! `w8a8_quant_act(norm_output, m, hidden)`: a `per_token_group_quant_fp8`
//! launch that re-reads the normed rows. The call sites call
//! [`residual_add_rms_norm`] here instead of `ops::residual_add_rms_norm`:
//!
//! * row off, or the FFN would not take the W8A8 arm for gate and up: the
//!   SAME `ops::residual_add_rms_norm` launch as before, nothing else;
//! * row on and the FFN would: ONE `rmsnorm_quant_fp8_ada` launch that writes
//!   `hidden`, `residual` and `norm_output` exactly as the norm does, plus the
//!   E4M3 codes and 1x128 scales into the FFN's `ffn_act_a` / `ffn_act_scale`
//!   scratch, and a hand-off token naming them. `w8a8_quant_act` consumes the
//!   token when its call matches it field for field and returns the scratch
//!   without launching.
//!
//! # Why this cannot change an output
//!
//! * The fused kernel's bytes equal the pair's (its header gives the argument;
//!   `examples/rmsnorm_quant_fp8_microtest.rs` byte-compares all five outputs).
//! * The token is consumed only by a quantization of the SAME input pointer,
//!   row count, width, scratch and stream, and EVERY `w8a8_quant_act` call
//!   takes the token, matching or not, because every such call rewrites the
//!   scratch the token describes. A stale token can therefore never stand in
//!   for bytes that were overwritten.
//! * The call site holds the returned [`PrequantScope`] until the FFN call
//!   returns and drops it there; dropping clears the token, so nothing outside
//!   that one FFN call can see it (a later layer, a verify step, an error
//!   path).
//! * The prediction ([`DenseFfnLayer::prefill_gate_up_w8a8_planned`]) restates
//!   `forward_prefill_inner`'s `gate_up_w8a8` rule. If the two ever drift the
//!   token is simply not consumed: the FFN quantizes as before and the only
//!   cost is the fused kernel's extra stores.
//!
//! `sole_consumer` is the call site's statement that nothing but this FFN
//! runs between the norm and the FFN call (the attention layer's LongCat
//! shortcut MoE does), so no other kernel can touch the scratch in between.

use std::cell::Cell;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layer::ForwardContext;
use crate::layers::FfnComponent;
use crate::layers::dense_ffn::DenseFfnLayer;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Largest hidden size the fused kernel covers. SSOT with `RQ_MAX_HIDDEN` in
/// the `.cu` (a CPU test reads it there).
pub const MAX_HIDDEN: u32 = 16384;
/// Activation group width; SSOT with `RQ_GROUP_K`.
pub const GROUP_K: u32 = 128;
/// Threads per CTA the kernel's block reduction assumes at most.
pub const MAX_THREADS: u32 = 1024;

/// A quantized activation the norm wrote for the next `w8a8_quant_act`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Prequant {
    /// The BF16 input the quantization is OF (`norm_output`).
    pub act: DevicePtr,
    pub m: u32,
    pub k: u32,
    /// Where the codes and scales are (the FFN scratch).
    pub a_fp8: DevicePtr,
    pub a_scale: DevicePtr,
    pub stream: u64,
}

thread_local! {
    /// At most one pending hand-off per thread. Prefill runs a layer's norm
    /// and its FFN on one thread, synchronously.
    static PENDING: Cell<Option<Prequant>> = const { Cell::new(None) };
}

/// Take the pending hand-off; `true` when it is exactly `want`. Always
/// clears it: the caller is about to rewrite the scratch either way.
pub(crate) fn take_matching(want: Prequant) -> bool {
    PENDING.with(|p| p.take()) == Some(want)
}

fn arm(p: Prequant) {
    PENDING.with(|c| c.set(Some(p)));
}

fn clear() {
    PENDING.with(|c| c.set(None));
}

/// The hand-off pending on this thread, if any.
#[cfg(test)]
pub(crate) fn pending() -> Option<Prequant> {
    PENDING.with(|c| c.get())
}

/// Clears the hand-off when dropped. The call site drops it right after the
/// FFN call that may consume it.
#[must_use = "hold it across the FFN call and drop it right after"]
pub struct PrequantScope {
    _private: (),
}

impl Drop for PrequantScope {
    fn drop(&mut self) {
        clear();
    }
}

/// The FFN-norm of a prefill layer: `hidden += src`, `residual = hidden`,
/// `output = rms_norm(hidden) * (1 + weight)`, as
/// `ops::residual_add_rms_norm` with `norm_k` computes it, plus (row on, and
/// the FFN takes the W8A8 gate/up arm) the FFN's quantized input. Hold the
/// returned scope across `ffn.forward_prefill(output, ..)` and drop it after.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm(
    ffn: &FfnComponent,
    sole_consumer: bool,
    ctx: &ForwardContext,
    norm_k: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<PrequantScope> {
    clear();
    let scope = PrequantScope { _private: () };
    let row = ops::port_rows::rmsnorm_fp8_quant_fused::row();
    let plan = if row {
        fused_plan(ffn, sole_consumer, ctx, num_tokens, hidden_size)
    } else {
        None
    };
    let Some(FusedPlan {
        kernel,
        a_fp8,
        a_scale,
    }) = plan
    else {
        ops::residual_add_rms_norm(
            ctx.gpu,
            norm_k,
            hidden,
            src,
            weight,
            output,
            residual,
            num_tokens,
            hidden_size,
            eps,
            stream,
        )?;
        return Ok(scope);
    };
    if ctx.stats.once("log:ffn_rmsnorm_fp8_quant_fused") {
        tracing::info!(
            "[avarok] prefill FFN-norm: residual add + RMSNorm + W8A8 activation quant in one \
             launch (rmsnorm_quant_fp8_ada; bytes equal to the unfused pair). \
             AVAROK_RMSNORM_FP8_QUANT=0 restores the pair."
        );
    }
    rmsnorm_quant_fp8(
        ctx.gpu,
        kernel,
        hidden,
        src,
        weight,
        output,
        residual,
        a_fp8,
        a_scale,
        num_tokens,
        hidden_size,
        eps,
        stream,
    )?;
    arm(Prequant {
        act: output,
        m: num_tokens,
        k: hidden_size,
        a_fp8,
        a_scale,
        stream,
    });
    Ok(scope)
}

/// The fused launch's handle and where it writes the quantized input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedPlan {
    pub kernel: KernelHandle,
    pub a_fp8: DevicePtr,
    pub a_scale: DevicePtr,
}

/// Every clause of the fused path except the row, cheapest first, with the
/// handle read last (an op-cache hit primed at layer construction).
pub(crate) fn fused_plan(
    ffn: &FfnComponent,
    sole_consumer: bool,
    ctx: &ForwardContext,
    m: u32,
    h: u32,
) -> Option<FusedPlan> {
    fused_plan_with(ffn, sole_consumer, ctx, m, h, || {
        crate::layers::dense_ffn::ada_kernels::rmsnorm_quant_kernel(ctx.gpu)
    })
}

/// [`fused_plan`] with the handle injected (CPU tests).
pub(crate) fn fused_plan_with(
    ffn: &FfnComponent,
    sole_consumer: bool,
    ctx: &ForwardContext,
    m: u32,
    h: u32,
    kernel: impl FnOnce() -> KernelHandle,
) -> Option<FusedPlan> {
    let FfnComponent::Dense(dense) = ffn else {
        return None;
    };
    if !sole_consumer
        || crate::ships_vanilla_norm_weights(ctx.config)
        || !shape_ok(m, h, ctx.config.hidden_size)
        || !dense.prefill_gate_up_w8a8_planned(ctx, m)
    {
        return None;
    }
    let (a_fp8, a_scale) = (ctx.buffers.ffn_act_a(), ctx.buffers.ffn_act_scale());
    let fits = (m as usize) * (h as usize) <= ctx.buffers.ffn_act_a_bytes()
        && (m as usize) * (h / GROUP_K) as usize * 4 <= ctx.buffers.ffn_act_scale_bytes();
    if !fits {
        return None;
    }
    let kernel = kernel();
    (kernel.0 != 0).then_some(FusedPlan {
        kernel,
        a_fp8,
        a_scale,
    })
}

/// The kernel's own clauses: whole 128-groups, at most [`MAX_HIDDEN`], and the
/// width the FFN quantizes (`hidden_size`).
pub(crate) fn shape_ok(m: u32, h: u32, config_hidden: usize) -> bool {
    m >= 1 && h as usize == config_hidden && h.is_multiple_of(GROUP_K) && h <= MAX_HIDDEN
}

impl DenseFfnLayer {
    /// Whether `forward_prefill` over `m` rows takes the W8A8 arm for BOTH
    /// gate and up, i.e. starts with `w8a8_quant_act(input, m, hidden)`. A
    /// restatement of `gate_up_w8a8` in `forward_prefill_inner` (the FP8 arm,
    /// after the packed-Q2 arm that precedes it): rungs 2-5 claim the width
    /// first, then both projections must pass `prefill_w8a8_selected`.
    pub(crate) fn prefill_gate_up_w8a8_planned(&self, ctx: &ForwardContext, m: u32) -> bool {
        let Some(fp8w) = self.fp8_weights.as_ref() else {
            return false;
        };
        if self.q2_weights.is_some() {
            return false;
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let reachable = self.ffn_batch16_plan(m).is_none()
            && self.ffn_m16_tc_plan(m, h).is_none()
            && self.ffn_m16_tc_plan(m, inter).is_none();
        reachable
            && self.prefill_w8a8_selected(ctx, m, inter, h, &fp8w.gate_proj)
            && self.prefill_w8a8_selected(ctx, m, inter, h, &fp8w.up_proj)
    }
}

/// The fused residual add + RMSNorm + per-token-group FP8 quant.
///
/// Kernel: `rmsnorm_quant_fp8_ada(hidden, src, weight, output, residual,
/// out_fp8, a_scale, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1), the
/// block `ops::residual_add_rms_norm` launches (its reduction order depends
/// on it).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    out_fp8: DevicePtr,
    a_scale: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(MAX_THREADS), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_ptr(out_fp8)
        .arg_ptr(a_scale)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

#[cfg(test)]
#[path = "norm_fused_tests.rs"]
mod tests;
