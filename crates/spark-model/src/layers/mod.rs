// SPDX-License-Identifier: AGPL-3.0-only

pub mod deepseek_v4_mtp;
pub mod dense_ffn;
pub mod dflash_head;
pub mod ep_dispatch;
pub mod exl3_dense;
pub mod fp8_calibration;
mod gemv_tier;
pub mod hc_ffn_plan;
pub mod moe;
pub mod mtp_head;
pub(crate) mod mtp_meta;
pub mod mtp_multi;
pub mod nemotron_mamba2;
pub mod nemotron_moe;
pub mod ngram_embed;
pub mod ops;
pub mod ple;
pub mod qsa;
pub mod qwen3_attention;
pub mod qwen3_ssm;
pub mod qwen4_exp_mtp;
pub mod qwen4_exp_mtp_proposer;
pub mod vision_encoder;
pub mod w4a16_gemv_tiers;

/// Minimum K at which the deep-K `w4a16_gemm_t_k64` (K_STEP_T=64) beats the
/// K_STEP_T=32 `w4a16_gemm_t`.
///
/// ★ 6144, not 4096. Measured with `w4a16_m17_bench` on the REAL decode shapes at
/// M=16 against the STREAM-measured 230 GB/s ceiling — `_k64` is the WORST tile
/// variant at K=5120 and the best only at K>=6144:
///
///   ssm_qkvz     N=16384 K=5120   _t 281.9us   _k64 341.6us   _m128 272.4us
///   attn qkv     N=14336 K=5120   _t 273.9us   _k64 328.5us   _m128 262.8us
///   ssm_out_proj N=5120  K=6144   _t 237.7us   _k64 163.3us   _m128 240.7us
///
/// The original 4096 threshold (this session) was derived from the ffn/out_proj
/// shapes and wrongly generalised to K=5120, sending 48 qkvz + 16 fused-qkv
/// launches per step to the slowest variant. Both variants accumulate K
/// sequentially, so moving between them is byte-identical.
///
/// `ATLAS_NO_W4A16_K64=1` restores the pre-session 8192 threshold.
pub(crate) fn w4a16_k64_min_k() -> u32 {
    static MIN_K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN_K.get_or_init(|| {
        // Explicit override so an A/B can pin a previous threshold exactly.
        if let Some(n) = std::env::var("ATLAS_W4A16_K64_MIN_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            return n;
        }
        if std::env::var("ATLAS_NO_W4A16_K64").ok().as_deref() == Some("1") {
            8192
        } else {
            6144
        }
    })
}

pub use deepseek_v4_mtp::{DeepseekV4MtpHead, DeepseekV4MtpProposerState};
pub use dense_ffn::{DenseFfnLayer, DenseFfnWeights, FfnActivation};
pub use dflash_head::{
    BlockDiffusionDraftHead, DflashLayer, DflashProposerState, DflashQuantization, dflash_ctx_cap,
};
pub use exl3_dense::{AttnProj, Exl3AttnWeights, Exl3GdnWeights};
pub use moe::MoeLayer;
pub use mtp_head::{MtpHead, MtpQuantization, mtp_drafter_prefill_enabled};
pub use nemotron_mamba2::NemotronMamba2Layer;
pub use nemotron_moe::NemotronMoeLayer;
pub use qwen3_attention::Qwen3AttentionLayer;
pub use qwen3_ssm::Qwen3SsmLayer;
pub use vision_encoder::{MergerLayer, ViTBlock, VisionEncoder};

use crate::layer::ForwardContext;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Try to load an optional kernel, logging at debug level if it's not found.
/// Returns `KernelHandle(0)` (null) on failure — callers must check before use.
///
/// Debug (not warn) because misses are expected when a model doesn't use a
/// given feature: e.g. Qwen3-Coder-Next (GDN+attention) never calls MLA
/// kernels, but the layer builder still probes them. Warning on expected
/// misses drowned out genuine problems in startup logs.
/// Resolve the `w4a16_gemm_t_m128_v2` handle honoring `ATLAS_W4A16_VARIANT`.
///
/// One resolver for the THREE sites that dispatch on this handle (attention
/// projections, dense-FFN prefill, SSM batched decode), so variant policy and
/// rollback live in exactly one place. Default (unset) resolves to a ZERO
/// handle — v1 everywhere — because the 27B port measured SLOWER than v1
/// (see body). `ATLAS_W4A16_VARIANT=v2` opts in on all three sites at once;
/// requesting it on a target without the kernel is a HARD startup error
/// (fail fast, not a silent fallback discovered in a perf regression).
#[track_caller]
pub fn w4a16_v2_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    let variant = std::env::var("ATLAS_W4A16_VARIANT").ok();
    // DEFAULT OFF on the qwen3 layer stack: the 27B port of the 8-warp v2
    // crush kernel is bit-identical to v1 (microtest 100% on 8 shapes) but
    // MEASURED SLOWER on the 27B FFN shapes — 0.78-0.82x of v1 standalone
    // (w4a16_bf16_v2_bench, 2026-07-30; v1 58-74 TFLOP/s). The kernel stays
    // in the PTX set for A/B and for shape regimes where the extra warps
    // might pay; nothing auto-activates it. `ATLAS_W4A16_VARIANT=v2` opts in
    // (hard error if the target lacks the kernel).
    if !matches!(variant.as_deref(), Some("v2") | Some("v3")) {
        if variant.as_deref() == Some("v1") {
            tracing::debug!("ATLAS_W4A16_VARIANT=v1: w4a16 m128 v2 suppressed (explicit)");
        }
        return KernelHandle(0);
    }
    let h = try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2");
    if h.0 == 0 {
        panic!(
            "ATLAS_W4A16_VARIANT={} requested but w4a16_v2::w4a16_gemm_t_m128_v2 is not in this \
             target's kernel set — refusing to start with a silently-degraded config",
            variant.unwrap()
        );
    }
    tracing::debug!(
        handle = h.0,
        "w4a16_gemm_t_m128_v2 resolution (explicit opt-in)"
    );
    h
}

/// Resolve the W4A16 m128 **v3** GEMM. Opt-in ONLY, same contract as
/// [`w4a16_v2_kernel`]: `ATLAS_W4A16_VARIANT=v3` selects it, anything else
/// resolves to a ZERO handle WITHOUT issuing a lookup.
///
/// Not issuing the lookup is the point. `prefill_weights` dispatches on
/// `v == 3 && handle != 0`, so on the default (`v1`) the probe could never be
/// used — it only ever added a permanently-failing row to the boot audit on
/// every target that does not ship `w4a16_v3`. Requesting the variant on such a
/// target is a HARD error, not a silent fallback discovered in a perf report.
#[track_caller]
pub fn w4a16_v3_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("ATLAS_W4A16_VARIANT").as_deref() != Ok("v3") {
        return KernelHandle(0);
    }
    let h = try_kernel(gpu, "w4a16_v3", "w4a16_gemm_t_m128_v3");
    if h.0 == 0 {
        panic!(
            "ATLAS_W4A16_VARIANT=v3 requested but w4a16_v3::w4a16_gemm_t_m128_v3 is not in this \
             target's kernel set — refusing to start with a silently-degraded config"
        );
    }
    h
}

/// Resolve the N128/M64 tile GEMM, preferring the 3-deep weight-pipeline variant.
/// **ON by default**; `ATLAS_NO_TGEMM_PIPELINE3` (presence — `=0` is NOT "off")
/// falls back to the 2-stage parent. Falls back automatically on any target that
/// does not ship `_p3`.
///
/// Same mechanism as [`k64_kernel`]: the parent drains its cp.async group before
/// the dequant phase, which only a co-resident CTA can cover. This kernel's live
/// shapes — ssm_qkvz (128 CTAs) and the fused QKV (112) — sit in the exposed
/// band of the grid.x-vs-efficiency curve. Bit-identical.
#[track_caller]
pub fn tgemm_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("ATLAS_NO_TGEMM_PIPELINE3").is_err() {
        let h = try_kernel(gpu, "w4a16", "w4a16_gemm_t_p3");
        if h.0 != 0 {
            return h;
        }
    }
    try_kernel(gpu, "w4a16", "w4a16_gemm_t")
}

/// Resolve the k64 deep-K tile GEMM, preferring the 3-deep weight-pipeline
/// variant. **ON by default**; `ATLAS_NO_K64_PIPELINE3` (presence — `=0` is NOT
/// "off") falls back to the 2-stage parent.
///
/// The parent issues one cp.async group then `wait_all`s it before the dequant
/// phase, so with a small grid there are ZERO outstanding loads across that
/// phase. The out_proj/o_proj shapes (N=5120, K=6144) launch 40 CTAs on 48 SMs —
/// exactly 1 CTA/SM — so nothing covers the drain, and they measure ~38% of
/// achievable while lm_head (1938 CTAs) reaches 83% on the identical loop.
/// `_p3` keeps step i+2's loads in flight across dequant(i+1). Bit-identical.
#[track_caller]
pub fn k64_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let want_p3 = std::env::var("ATLAS_NO_K64_PIPELINE3").is_err();
    if want_p3 {
        let h = try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_p3");
        if h.0 != 0 {
            return Ok(h);
        }
    }
    gpu.kernel("w4a16", "w4a16_gemm_t_k64")
}

/// Resolve the NARROW-N (N_TILE=64) deep-K twin. `KernelHandle(0)` when the
/// kernel is absent or the presence kill switch `ATLAS_NO_K64_N64` is set
/// (`=0` is NOT "off"). Callers must store the handle — `kernel()` is an
/// init-time lookup, not a per-launch one.
#[track_caller]
pub fn k64_n64_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("ATLAS_NO_K64_N64").is_ok() {
        return KernelHandle(0);
    }
    try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_n64_p3")
}

/// Wide-tile CTA count below which the N_TILE=64 deep-K twin wins.
///
/// `w4a16_gemm_t_k64_p3` owns a 128-wide N tile and a 64-row M tile, so a
/// launch is `ceil(n/128) * ceil(m/64)` CTAs of 128 threads. At the out_proj /
/// o_proj shape (N=5120, K=6144 — 64 launches per decode step at n=64) that is
/// **40 CTAs on a 48-SM device**: 8 SMs idle by construction and the other 40
/// hold ONE CTA each, i.e. 4 warps against a 48-warp SM budget.
///
/// That is NOT a bandwidth problem. A full-working-set replay (rotating over
/// enough distinct weight tensors that every launch sees a cold 24 MB L2)
/// decomposes the 167.8 us launch as: memory pipe alone 87.2 us = **95.6% of
/// the 229.6 GB/s row-strided ceiling** — already saturated — with the
/// remaining 80.6 us being barrier-serialized dequant that has no co-resident
/// warp to hide it. Halving the N tile doubles the grid to 80 CTAs, filling all
/// 48 SMs with 1-2 co-resident CTAs, and moves the SAME bytes: each CTA still
/// owns a disjoint N slice and the extra A re-reads are L2 hits (A is 786 KB).
///
/// Measured (replay, cold L2, bit-identical output at every point):
///   wide CTAs  20 -> 1.74x   40 -> 1.42x   48 -> 1.33x   64 -> 1.15x
///   wide CTAs  76 -> 0.77x   80 -> 0.80x   96 -> 0.79x  244 -> 0.95x
/// Above ~64 CTAs the wide tile already fills the machine and the narrow tile
/// only pays more epilogue and A traffic, so the gate is a hard `<= 64`.
const K64_N64_MAX_WIDE_CTAS: u32 = 64;

/// Should the narrow-N deep-K twin serve this shape? See
/// `K64_N64_MAX_WIDE_CTAS` for the derivation and the measured curve.
pub fn k64_n64_wins(m: u32, n: u32) -> bool {
    n.div_ceil(128) * m.div_ceil(64) <= K64_N64_MAX_WIDE_CTAS
}

/// Optional kernel lookup: `KernelHandle(0)` instead of an error.
///
/// `#[track_caller]` so the audit names the DISPATCH SITE — this helper stands
/// between ~500 call sites and `GpuBackend::kernel`, and without it every
/// optional lookup in the binary would be reported against this one line.
///
/// A zero handle is a SILENT slower path, so a lookup that lands here for a
/// model that genuinely needs the kernel is a bug. Either gate the call on the
/// model's config so it is never issued, or declare it in the target's
/// MODEL.toml `[expected_absent]` with a reason; the boot gate
/// (`kernel_audit::classify_failures`) fails closed on anything else.
/// Minimum rows in flight for the grouped-GEMM MoE decode arm. SSOT for the
/// SSM stack (`qwen3_ssm::trait_decode_multi_seq`) and the attention layers
/// (`qwen3_attention::…::multi_seq::ffn`), which must agree — they are the
/// same trade on the same weights.
///
/// The arm reads each routed expert ONCE instead of once per token, so it
/// wins when there are enough tokens to amortise the expert sort/permute
/// launch overhead, and loses when there are not. Both ends are measured:
///
/// | n | verdict | measurement |
/// |---|---|---|
/// | 4 | LOSS | 31 vs 56 tok/s on Holo — the fixed per-layer sort/permute dominates at small N |
/// | >=16 | WIN | SSM-side alone C=32 172.7 -> 216.2 tok/s (+25%); #415's attention-side extension +7.9% at C=32 / +9.7% at C=64 on Qwen3.6-35B-A3B-NVFP4, paired gsm8k n=200 strict 0.960 vs 0.900 baseline, zero regressions |
///
/// 16 is the smallest width measured on the winning side. n=5..15 is
/// UNMEASURED, not a known win — it sits on the losing side of this gate on
/// purpose, because the one thing we know about the gap is that the loss at
/// n=4 is large (-45%) and the win at n=16 is smaller (+25%).
pub fn moe_grouped_decode_min_rows() -> usize {
    16
}

/// Kill switch for the grouped-GEMM MoE decode arm. PRESENCE check per the
/// house convention (`ATLAS_NO_MOE_GROUPED_DECODE=0` is NOT off), read once
/// per process — this predicate sits in the decode path, and the `env::var`
/// it replaces ran on every dispatch for MoE models.
pub fn moe_grouped_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MOE_GROUPED_DECODE").is_none())
}

/// Force the grouped arm BELOW `moe_grouped_decode_min_rows()`. Diagnostic
/// only — it exists so the n=5..15 gap can be measured without a rebuild, and
/// it is the same var #415's measurements used, kept working on purpose.
/// Never a production setting: if forcing wins at a width, move the THRESHOLD.
pub fn moe_grouped_decode_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_GROUPED_DECODE").as_deref() == Ok("1"))
}

/// Whether the grouped-GEMM MoE decode arm should run for `n` rows —
/// PURE, so both polarities are testable without touching process env or the
/// `OnceLock`s below (which latch, and would make the tests order-dependent).
pub fn moe_grouped_decode_decide(n: usize, enabled: bool, forced: bool) -> bool {
    enabled && (n >= moe_grouped_decode_min_rows() || forced)
}

/// Whether the grouped-GEMM MoE decode arm should run for `n` rows.
pub fn moe_grouped_decode_for(n: usize) -> bool {
    moe_grouped_decode_decide(n, moe_grouped_decode_enabled(), moe_grouped_decode_forced())
}

#[cfg(test)]
#[path = "moe_grouped_decode_tests.rs"]
mod moe_grouped_decode_tests;

#[track_caller]
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}

/// FFN component: MoE (expert routing), dense SwiGLU, or None (standalone attention).
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Dense(DenseFfnLayer),
    /// No FFN — used by Nemotron-H standalone attention layers.
    None,
}

impl FfnComponent {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// True for a plain dense (SwiGLU) FFN. Wide-batch verify paths gate their
    /// `forward_prefill` fast path on this: batching reads dense weights once
    /// (big win at N=17), but on a 256-expert MoE the grouped-GEMM is a net
    /// loss at small batch (per-expert M~1 + sort/permute overhead), so MoE
    /// keeps its per-token loop.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// True when this MoE FFN can serve DECODE through the grouped read-once
    /// GEMM (forward_prefill) instead of the pairwise per-slot loop. The
    /// is_dense() comment above asserts grouped is "a net loss at small batch"
    /// on a 256-expert MoE, but that was never measured for decode CONCURRENCY
    /// (n=4) where the pairwise path re-reads ~14-20 distinct experts as 40
    /// per-slot CTAs. Native-NVFP4-routed only (forward_prefill's unconditional
    /// grouped path); dense/none are false.
    pub fn moe_grouped_decode_ok(&self) -> bool {
        match self {
            Self::Moe(m) => m.grouped_decode_ok(),
            _ => false,
        }
    }

    /// ATLAS_FP32_ROUTING active for this FFN (MoE only; false otherwise).
    pub fn fp32_routing_active(&self) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(),
            _ => false,
        }
    }

    /// True when this FFN's routed experts are served natively from EXL3
    /// trellis (`ATLAS_EXL3_NATIVE_MOE=1`). Every mgemm in that arm is a
    /// COOPERATIVE launch — not CUDA-graph-capturable — so each layer kind's
    /// `decode_graph_unsupported` (and the verify-path `use_graphs` terms)
    /// must include this, exactly like the `lm_head_exl3` veto.
    pub fn exl3_native_moe(&self) -> bool {
        match self {
            Self::Moe(m) => m.exl3_native_active(),
            _ => false,
        }
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward(input, ctx, stream),
            Self::Dense(d) => d.forward(input, ctx, stream),
            Self::None => Ok(input),
        }
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// Whether the K=m (m<=8) batched-GEMV verify FFN is available (dense
    /// only — MoE / missing batch4/batch8 kernel / non-NVFP4 weights →
    /// false). Lets callers gate branch entry BEFORE computing the pre-FFN
    /// norm, so there is no half-done fallthrough to `forward_prefill`.
    pub fn can_forward_km(&self, m: u32) -> bool {
        matches!(self, Self::Dense(d) if d.can_forward_km(m))
    }

    /// K=m (m=4..8) verify FFN via batched GEMV (dense only). Returns
    /// `false` when the path is unavailable (MoE / missing batchm kernel /
    /// non-NVFP4 weights) so the caller can fall back to `forward_prefill`.
    pub fn try_forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) if d.can_forward_km(m) => {
                d.forward_km(input, m, ctx, stream)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_token_major_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_atomic_c4_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}

pub(crate) use gemv_tier::batch8_kernel;
