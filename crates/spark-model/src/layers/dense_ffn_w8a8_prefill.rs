// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled dense-FFN PREFILL — the gate/up/down GEMM arm that the
//! native-FP8 dispatch in `dense_ffn.rs` reaches ahead of its W8A16 branches.
//!
//! # Reachability — read this before measuring anything here
//!
//! This whole module is **opt-in and unreached by default**. It runs only when
//! the dense FFN holds block-scaled FP8 weights, and
//! `qwen35_dense.rs::load_layers` installs those only when
//!
//! ```text
//! dense_fp8_enabled() && tp_world_size == 1
//!     && variant == Nvfp4Variant::Fp8Dequanted
//!     && proj_is_native_fp8(gate_proj)
//! ```
//!
//! and `dense_fp8_enabled()` is `ATLAS_DENSE_FP8 == "1"`. `git grep
//! ATLAS_DENSE_FP8` returns ONE hit — its own reader. No CI job, no
//! `BENCH.toml` entry and no gate sets it, so no certification record on any
//! branch has ever exercised this code. Without it `self.fp8_weights` is
//! `None`, `forward_prefill_inner` never reaches the selection below, and
//! NEITHER of the two route log lines is emitted. Three served A/B attempts
//! were spent discovering that, each reading as "the lever did not arm".
//!
//! It is off by default because on GB10 native dense FP8 LOSES to the NVFP4
//! autoquant fallback, and not narrowly. Same box, same binary, same
//! byte-identical 949-token prompt, `Qwen3.6-27B-FP8`, TTFT median of 5
//! (spark-256a, 2026-09-11):
//!
//! ```text
//! default (NVFP4)                       1437.2 ms   1.00x
//! ATLAS_DENSE_FP8=1, W8A16 (capped)     2555.8 ms   1.78x slower
//! ATLAS_DENSE_FP8=1, W8A8  (no cap)     3343.3 ms   2.33x slower
//! ```
//!
//! So the ceiling below is worth 23.4% *within* the dense-FP8 path, and the
//! dense-FP8 path is still the slower choice on this arch. GB10's FP8 W8A16
//! kernels are simply less tuned than its NVFP4 W4A16 ones (unfused
//! per-projection GEMV, non-transposed prefill GEMM); closing that is kernel
//! work, not loader wiring. On H100 the trade is the other way round, which is
//! why the ceiling is per-arch data in `kernels/<hw>/HARDWARE.toml` and not a
//! constant here.
//!
//! WHY (#917 / #928). On a native-FP8 checkpoint the dense FFN's prefill GEMMs
//! ran `w8a16_gemm_pipelined`: BF16 activations against E4M3 weights, so the
//! MMA is the BF16 tensor-core path and the FP8 bytes are pure memory savings.
//! Measured on H100 (2026-09-11, 1193-token prompt): TTFT 1075 ms against
//! vLLM's 287 ms for the same model and prompt, with the pipelined kernel
//! turning ~12 TFLOP/s on the FFN shapes. The attention Q/K/V/O projections had
//! already moved to the W8A8 block-scaled path (`paged_qkv.rs` /
//! `paged_oproj.rs`), and the MoE shared expert with them
//! (`moe/forward_prefill_fp8.rs`) — the dense FFN was the one large prefill
//! consumer still on W8A16, and on a dense model it is most of the FLOPs.
//!
//! This gives it the same arithmetic vLLM uses: per-token 1x128 FP32
//! activation scales (`per_token_group_quant_fp8`) multiplied against the
//! checkpoint's 128x128 FP32 weight scales in an FP32 epilogue, with the
//! product accumulated by `mma.sync.m16n8k32.e4m3` — native on sm_90a (H100)
//! and sm_121. Two GEMM implementations sit behind one selector:
//!
//!   * cuBLASLt `fp8_gemm_act_weight_t_blkscaled` (weight as A with
//!     BLK128x128 scales, activation as B with VEC128 scales — the DeepSeek
//!     block-FP8 scheme), when `ATLAS_CUBLAS_GEMM=1`. The Hopper fast path.
//!     Its VEC128 scales go through `fp8_act_scale_to_kmajor` first: cuBLASLt
//!     documents that operand's scales with the TOKEN index contiguous, which
//!     is the transpose of what the quantizer writes (see
//!     `spark_runtime::cublaslt::scale_layout`).
//!   * `ops::fp8_gemm_t_blockscaled`, the in-tree kernel, otherwise.
//!
//! Both consume the SAME quantized activation and the same FP32 epilogue, so
//! they are expected to agree to a BF16 ULP or two; the microtest
//! (`examples/native_fp8_ffn_w8a8_microtest.rs`) pins that.
//!
//! ACCURACY. W8A8 is lossier than W8A16 by construction — the activation is
//! quantized to E4M3 per 128-element group instead of kept in BF16. That is
//! vLLM's dynamic W8A8 arithmetic and a deliberate precision trade, not a bug:
//! the microtest gates it at cosine >= 0.999 / relative RMS <= 2% against the
//! W8A16 reference, and the serve logs the selected path once at INFO so which
//! arithmetic ran is visible in any TTFT report. `ATLAS_FFN_W8A16_ONLY=1`
//! restores the old path byte-for-byte.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// `ATLAS_FFN_W8A16_ONLY` kill switch: PRESENCE (any value, including empty)
/// keeps the dense-FFN prefill on today's W8A16 kernels. Presence rather than
/// `=1` because this is an escape hatch an operator reaches for while a serve
/// is misbehaving, and `ATLAS_FFN_W8A16_ONLY=0` meaning "on" is a trap.
///
/// `OnceLock`-cached: the selector runs per projection per layer per prefill
/// (3 x num_layers times), and `std::env::var_os` walks the environment block
/// on every call. Cached process-wide is correct here — the variable is read
/// once at first prefill and a serve never rewrites its own environment.
pub fn ffn_w8a16_only() -> bool {
    static ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONLY.get_or_init(|| std::env::var_os("ATLAS_FFN_W8A16_ONLY").is_some())
}

/// The ceiling that applies to a projection of shape `[n, k]`.
///
/// `n > k` is the WIDENING case (gate/up: N=17408, K=5120 on Qwen3.8-27B);
/// `n <= k` the NARROWING one (down: N=5120, K=17408). Two rows and not one
/// because the measured crossover differs by about 6x between the two shapes —
/// M~64-128 widening against M~384-512 narrowing.
///
/// Read from [`ops::target_defaults::resolved`] rather than carried on
/// `GemmDispatch`: that resolution is `OnceLock`-cached precisely so it can be
/// read per projection per layer per step, and it is where the other
/// target-declared levers already live. `ffn_w8a16_only` above is the same
/// shape of process-global, which is why `w8a8_prefill_selected` takes the
/// ceiling as an ARGUMENT — a `OnceLock` cannot be toggled per test.
pub(crate) fn max_m_for(n: u32, k: u32) -> u32 {
    let levers = ops::target_defaults::resolved();
    if n > k {
        levers.w8a8_prefill_max_m_widening.value
    } else {
        levers.w8a8_prefill_max_m_narrowing.value
    }
}

/// The whole W8A8 selection rule, as a pure function of shape + format +
/// handles. Split out from the layer method so the CPU tests can pin every
/// clause without a `ForwardContext` (`w8a16_only` is injected for the same
/// reason — a process-global `OnceLock` cannot be toggled per test).
///
/// Clauses, each load-bearing:
///
/// * `m > 4` — M<=4 stays on the batch4 GEMV, which streams each weight once
///   and beats any MMA tile at those shapes.
/// * `fp8_blockscaled_prefill` — the `ATLAS_FP8_SINGLE_SCALE` kill switch that
///   already governs the attention W8A8 path.
/// * `Fp8BlockScaled` — a per-ROW scale is a different `row_scale` layout; the
///   block-scaled GEMM would read it as `[N/128, K/128]`.
/// * `k % 128 == 0` — the activation quantizer emits one scale per 128-wide K
///   group and the GEMM folds per K-block.
/// * `n % 128 == 0` — the weight scale grid is `[N/128, K/128]`.
/// * `m <= max_m` — the per-arch upper bound. W8A8 beats W8A16 only while the
///   per-token activation quantization and its FP32 scale epilogue are small
///   against the GEMM; past that the quantization is the bill and W8A16's
///   larger MMA wins. Where that crosses is a property of the arch, so it is
///   declared in `kernels/<hw>/HARDWARE.toml` `[defaults]` rather than being a
///   constant here. `u32::MAX` (the baseline) is no cap.
/// * both handles loaded — a model shadow may not carry either entry point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn w8a8_prefill_selected(
    m: u32,
    n: u32,
    k: u32,
    scale_format: WeightQuantFormat,
    fp8_blockscaled_prefill: bool,
    quant_k: KernelHandle,
    gemm_k: KernelHandle,
    w8a16_only: bool,
    max_m: u32,
) -> bool {
    !w8a16_only
        && m > 4
        && m <= max_m
        && fp8_blockscaled_prefill
        && scale_format == WeightQuantFormat::Fp8BlockScaled
        && k.is_multiple_of(128)
        && n.is_multiple_of(128)
        && quant_k.0 != 0
        && gemm_k.0 != 0
}

impl DenseFfnLayer {
    /// Whether ONE dense-FFN prefill projection takes the W8A8 branch.
    ///
    /// Beyond [`w8a8_prefill_selected`] this also requires the shared
    /// dense-FFN activation scratch (`ffn_act_a` / `ffn_act_scale`, sized once
    /// for `max_batch_tokens x max(hidden, intermediate)` in
    /// `BufferSizes::from_config`). That scratch is NULL for MoE configs, which
    /// never take this path — but a null pointer here would be a kernel launch
    /// writing to address 0, so it is a gate and not an assert.
    pub(crate) fn prefill_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        m: u32,
        n: u32,
        k: u32,
        w: &Fp8Weight,
    ) -> bool {
        w8a8_prefill_selected(
            m,
            n,
            k,
            w.scale_format,
            ctx.dispatch.fp8_blockscaled_prefill,
            self.per_token_group_quant_fp8_k,
            self.fp8_gemm_t_blockscaled_k,
            ffn_w8a16_only(),
            max_m_for(n, k),
        ) && ctx.buffers.ffn_act_a().0 != 0
            && ctx.buffers.ffn_act_scale().0 != 0
    }

    /// Quantize `act[m, k]` BF16 into the shared dense-FFN scratch as FP8 E4M3
    /// plus per-token/128-group FP32 scales; returns `(a_fp8, a_scale)`.
    ///
    /// Callers must consume the result before the next call: there is ONE
    /// scratch pair per arena, deliberately (the previous per-call
    /// `alloc`/`synchronize`/`free` pattern — still used by the MoE shared
    /// expert — costs a full stream sync per projection). The dense-FFN
    /// ordering is safe because everything runs on one stream: gate and up read
    /// the `input` quantization, then `silu_mul` produces the down input, then
    /// down re-quantizes over the same bytes.
    ///
    /// `k` may be `hidden` (gate/up) or `intermediate` (down); the scratch is
    /// sized for `max(hidden, intermediate)` so neither overflows.
    pub(crate) fn w8a8_quant_act(
        &self,
        ctx: &ForwardContext,
        act: DevicePtr,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let a_fp8 = ctx.buffers.ffn_act_a();
        let a_scale = ctx.buffers.ffn_act_scale();
        // Padded extents, because the cuBLASLt arm READS `ceil16(m)` rows.
        let rows = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(rows * k as usize <= ctx.buffers.ffn_act_a_bytes());
        debug_assert!(rows * (k as usize / 128) * 4 <= ctx.buffers.ffn_act_scale_bytes());
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            act,
            a_fp8,
            a_scale,
            m,
            k,
            stream,
        )?;
        Ok((a_fp8, a_scale))
    }

    /// `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ` with both block-scale sets
    /// folded in FP32 — cuBLASLt when `ATLAS_CUBLAS_GEMM` is set and the output
    /// buffer has room for the padded M, else the in-tree kernel.
    ///
    /// `out_capacity_bytes` is the allocated size of `out`'s arena buffer. The
    /// cuBLASLt helper rounds M up to 16 and WRITES those phantom rows (their
    /// activation scales are zeroed, so the values are defined, but the stores
    /// happen); the arena sizes that headroom in, and this check is what keeps
    /// a future re-sizing from turning into a silent cross-buffer write.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m) as usize;
        let padded_out_bytes = m_pad * n as usize * 2;
        // Every clause the cuBLASLt arm needs beyond the shared W8A8 gate:
        //
        // * output room for the phantom rows the padded M writes;
        // * the VEC128 scale-layout adapter — kernel AND its scratch. cuBLASLt
        //   reads the activation scales token-contiguous, so without the
        //   transpose the GEMM is fast and WRONG (H100 2026-09-11: 1140 TFLOP/s
        //   at rel_rms 7.7e-2 vs this same in-tree kernel). Falling back is the
        //   only safe answer when either is missing;
        // * `k % 512 == 0` — the BLK128x128 weight scales are handed over as
        //   the checkpoint's `[N/128, K/128]` grid, and cuBLASLt requires that
        //   tensor's column stride (K/128) to be a multiple of 4.
        let scale_layout_ready = ctx.buffers.ffn_act_scale_kmajor().0 != 0
            && self.fp8_act_scale_kmajor_k.0 != 0
            && ctx.buffers.ffn_act_scale_kmajor_bytes() >= m_pad * (k as usize / 128) * 4;
        let cublas = ctx.dispatch.cublas.ffn
            && padded_out_bytes <= out_capacity_bytes
            && spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
            && (scale_layout_ready || !ops::cublas_scale_layout_kmajor());
        self.log_w8a8_prefill_route(ctx, cublas);
        if cublas {
            return ops::cublas_fp8_proj_prequant(
                ctx.gpu,
                self.fp8_act_scale_kmajor_k,
                a_fp8,
                a_scale,
                ctx.buffers.ffn_act_scale_kmajor(),
                w,
                out,
                m,
                n,
                k,
                stream,
            );
        }
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            a_fp8,
            a_scale,
            w.weight,
            w.row_scale,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// Log-once latch for the selected dense-FFN prefill arithmetic, matching
    /// the `log:ffn_*` lines the other prefill levers in `dense_ffn.rs` emit.
    /// The line matters beyond bookkeeping: W8A8 is a deliberate precision
    /// trade, so a TTFT or quality report has to be able to say which
    /// arithmetic produced it.
    fn log_w8a8_prefill_route(&self, ctx: &ForwardContext, cublas: bool) {
        if ctx.stats.once("log:ffn_w8a8_prefill") {
            let how = if cublas { "cuBLASLt" } else { "kernel" };
            tracing::info!(
                "[atlas] dense FFN prefill: W8A8 block-scaled via {how} \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                 vLLM-equivalent FP8 numerics). ATLAS_FFN_W8A16_ONLY=1 restores W8A16."
            );
        }
    }

    /// Counterpart log for the unchanged W8A16 path, so the absence of the
    /// W8A8 line is never ambiguous between "kill switch set" and "log lost".
    pub(crate) fn log_w8a16_prefill_route(&self, ctx: &ForwardContext) {
        if ctx.stats.once("log:ffn_w8a16_prefill") {
            tracing::info!(
                "[atlas] dense FFN prefill: W8A16 (BF16 act x FP8 weight). \
                 W8A8 not selected — see #917/#928."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_w8a8_prefill_tests.rs"]
mod tests;
