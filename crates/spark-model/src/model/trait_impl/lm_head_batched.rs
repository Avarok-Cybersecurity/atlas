// SPDX-License-Identifier: AGPL-3.0-only

//! The decode LM head, batched — ONE source of truth for two call sites.
//!
//! `decode_a2.rs` (the pure-decode batch) and `decode_b2.rs`
//! (`mixed_final_norm_lm_head`, the prefill+decode co-dispatch head reached
//! from `decode_b.rs` via `mixed_forward_dispatch`) both finish a step with
//! RMS-norm then the vocab projection. They had drifted: `decode_a2` grew the
//! full ladder while `decode_b2` still looped `padded_n` times through
//! `ops::w4a16_gemv`, re-reading the whole vocab weight once per row —
//! ~N x 254 MB/step on live continuous-batching traffic.
//!
//! Credit for spotting the site: @rsafier in #332, which fixed it with a bare
//! default-OFF `batch16`. This lifts `decode_a2`'s ladder instead, so the two
//! heads cannot diverge NUMERICALLY at the same batch width — two
//! independently-maintained ladders would be a second source of truth for
//! which kernel a given `padded_n` lands on, and the first thing to go wrong
//! would be a silent accuracy difference between the pure-decode and
//! co-dispatch paths.
//!
//! HONESTY: this is a bandwidth-accounting argument, not a measurement. The
//! mixed path has never been A/B'd. Any throughput claim for it must be gated
//! spec-OFF or on reversed-order pairs (a single spec-ON pair drifts +/-2%).

use crate::weight_map::DenseWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::types::TransformerModel;
use crate::layers::ops;

/// Batched-GEMV decode lm_head: **ON by default**, disabled by
/// `ATLAS_NO_LM_HEAD_BATCH_GEMV=1`.
///
/// Strict `== "1"` on an `ATLAS_NO_*` name, not a presence check — presence
/// flags in this codebase are ENABLED by `=0`. Read once; this is a per-step
/// site.
pub(super) fn lm_head_batch_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_LM_HEAD_BATCH_GEMV").as_deref() != Ok("1"))
}

/// Legacy BF16-head switch: only the exact value "0" disables it.
fn bf16_batch_gemv_from_value(value: Option<&str>) -> bool {
    value != Some("0")
}

/// The BF16 decode head's batched-GEMV band, for THIS head only.
///
/// 🔴 Read `layers/ops/gemm_quant.rs` before touching the DEFAULT. That
/// constant is the FROZEN band, and it is frozen for a reason that still
/// holds: the MTP row dispatch (`layers/mtp_head/row_dispatch.rs`), the
/// verify-`k` workspace sizing (`weight_loader/glm5_next_load.rs`) and this
/// head all read it, the band's upper edge decides whether a width lands on
/// the batched GEMV or on a REASSOCIATING tile GEMM, and the A/B behind the
/// number measured the GEMV NEGATIVE above 8 on GB10 (-14.4% at C=16, commits
/// 84d5b763c / 78d276832).
///
/// ★ THE DEFAULT IS NO LONGER A LITERAL. It is the compiled target's
/// (`kernels/<hw>/HARDWARE.toml` `[defaults] lm_head_batchm_max`), so a target
/// that has measured a different edge declares it beside its arch facts
/// instead of exporting `ATLAS_LM_HEAD_BATCHM_MAX` from a launch script —
/// which is the arrangement the 2026-09-11 maintainer review called
/// "discipline rather than structure". Every target in the tree declares the
/// frozen 8 today, so this site's behaviour is unchanged. The variable still
/// overrides, and it is PER-SITE: it moves THIS head and nothing else.
///
/// Resolution, clamping and caching are `ops::target_defaults::resolve_batchm_max`
/// and `resolved()`; `OnceLock`-cached there because the route must be
/// CONSTANT across CUDA-graph replays.
fn lm_head_batchm_max() -> u32 {
    ops::target_defaults::resolved().lm_head_batchm_max.value
}

/// The BF16 head's TENSOR-CORE arm, resolved once at model construction:
/// which kernels this target actually carries, whether the operator asked for
/// them, and at what CTA width.
///
/// A struct rather than four parameters because the rule that reads them
/// ([`lm_head_m16_tc_route`]) is the thing worth testing, and a test that has
/// to spell out four positional flags grades the spelling as much as the rule.
#[derive(Debug, Clone, Copy)]
pub(super) struct LmHeadM16Tc {
    /// `dense_gemm_m16_bf16` (N_TILE=32). 0 when the kernel set lacks it.
    pub narrow: KernelHandle,
    /// `dense_gemm_m16_bf16_n64` (N_TILE=64). 0 when absent.
    pub wide: KernelHandle,
    /// `ATLAS_LM_HEAD_M16_TC` present.
    pub enabled: bool,
    /// Requested CTA width: 32 (default) or 64.
    pub n_tile: u32,
}

/// `ATLAS_LM_HEAD_M16_TC` — PRESENCE, default OFF.
///
/// Presence rather than `== "1"`, matching `ATLAS_FFN_M16_TC` /
/// `ATLAS_ATTN_M16_TC` next door: every A/B recipe stays a bare `VAR=1` prefix
/// with no "=0 means on" trap. Default OFF because this arm REASSOCIATES the K
/// reduction against `dense_gemv_bf16` (see [`lm_head_m16_tc_route`]) and the
/// LM head is the one layer where a near-tie argmax flip changes the emitted
/// token — so it ships dark until an H100 receipt says it wins.
fn m16_tc_enabled_from_presence(present: bool) -> bool {
    present
}

/// `ATLAS_LM_HEAD_M16_TC_NTILE` — 32 (default) or 64. An unrecognised value
/// falls back to 32 rather than failing the boot: the tile is a perf A/B knob,
/// and the route log names the tile that actually ran.
fn m16_tc_n_tile_from_value(value: Option<&str>) -> u32 {
    match value.map(str::trim) {
        Some("64") => ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE,
        _ => ops::DENSE_GEMM_M16_BF16_N_TILE,
    }
}

/// Process-wide resolution of both, `OnceLock`-cached for the reason the band
/// above is: the route must be CONSTANT across CUDA-graph replays, and a
/// per-call `env::var` could change the captured launch set between capture and
/// replay.
fn lm_head_m16_tc_env() -> (bool, u32) {
    static ENV: std::sync::OnceLock<(bool, u32)> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        let n_tile = std::env::var("ATLAS_LM_HEAD_M16_TC_NTILE").ok();
        (
            m16_tc_enabled_from_presence(std::env::var_os("ATLAS_LM_HEAD_M16_TC").is_some()),
            m16_tc_n_tile_from_value(n_tile.as_deref()),
        )
    })
}

/// The WHOLE selection rule for the tensor-core head arm, as a pure function of
/// the row count, the reduction depth and the resolved lever/handles.
///
/// 🔴 THE BAND IS 5..=16 AND NOTHING ELSE.
/// * `m <= 4` keeps today's path by construction: `dense_gemv_batchm` measures
///   at the memory roofline there (round 7: 798 us = 3.2 TB/s-class at C=1 on
///   the 2.54 GB head), so there is nothing to buy and a reassociating kernel
///   would only trade bits for noise. `m == 1` is the greedy single-sequence
///   decode path and is deliberately untouched.
/// * `m > 16` is past the kernel's M tile — rows above it are never computed,
///   which is stale output rather than a launch failure, so the tier DECLINES
///   and the batched GEMV (or the fallback GEMM) serves it.
///
/// `k` is part of the rule and not an `ensure!` at the call site: the kernel's
/// cp.async pipeline advances 64 elements per step and its 16-byte weight-row
/// chunks need `k % 8 == 0` anyway, so a K that is not a whole number of steps
/// has no correct route and the tier must decline rather than launch and be
/// wrong. Every Atlas BF16 head satisfies it (Qwen3.8-27B: K=5120).
///
/// Returns the launcher, its handle and the CTA width that will actually run —
/// a shadow built before the wide arm existed has no `_n64`, so a `=64` request
/// falls back to the 32-wide kernel rather than launching a zero handle.
fn lm_head_m16_tc_route(
    tc: LmHeadM16Tc,
    m: u32,
    k: u32,
) -> Option<(ops::DenseM16Bf16Gemm, KernelHandle, u32)> {
    if !tc.enabled
        || !(5..=ops::DENSE_GEMM_M16_BF16_MAX_M).contains(&m)
        || !k.is_multiple_of(ops::DENSE_GEMM_M16_BF16_K_STEP)
    {
        return None;
    }
    if tc.n_tile == ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE && tc.wide.0 != 0 {
        return Some((
            ops::dense_gemm_m16_bf16_n64,
            tc.wide,
            ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE,
        ));
    }
    if tc.narrow.0 != 0 {
        return Some((
            ops::dense_gemm_m16_bf16,
            tc.narrow,
            ops::DENSE_GEMM_M16_BF16_N_TILE,
        ));
    }
    None
}

fn lmhead_batch_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        bf16_batch_gemv_from_value(std::env::var("ATLAS_LMHEAD_BATCH_GEMV").ok().as_deref())
    })
}

/// Shared BF16-head dispatch for ordinary and mixed multi-sequence decode.
#[allow(clippy::too_many_arguments)]
fn project_bf16_lm_head(
    gpu: &dyn GpuBackend,
    fallback: KernelHandle,
    batch_gemv: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    [m, n, k]: [u32; 3],
    batch_enabled: bool,
    batchm_max: u32,
    m16_tc: LmHeadM16Tc,
    stream: u64,
) -> Result<()> {
    // The existing kernel shares one BF16 weight read across up to eight rows.
    // Its uint4 loads require each input/weight row to remain 16-byte aligned.
    //
    // 🔴 `batchm_max` is the DECODE band, NOT the kernel's `MAX_M`. The GEMV
    // tier was widened to 16 for PREFILL only; this is a decode head, and the
    // band's upper edge is what decides whether a width lands on the batched
    // GEMV or the reassociating tile GEMM — i.e. which bits a decode of that
    // width produces. When `MAX_M` was 8 the two names were the same number
    // and this site read the right one by accident; they are not the same
    // number any more. See `layers/ops/gemm_quant.rs` for the frozen band and
    // `lm_head_batchm_max` above for the per-target declaration that sets it.
    //
    // TENSOR-CORE ARM FIRST, and only at 5..=16 rows (#927/#928). nsys round 7
    // puts this one launch at 3,571 us = 8.19% of the 43.6 ms step at batch 16
    // — 2.54 GB of BF16 vocab weight at ~710 GB/s, where the SAME kernel reads
    // the SAME bytes at 3.2 TB/s-class for a single row. The batched GEMV is
    // FP32-FMA-bound, not bandwidth-bound, and an m16n8k16 MMA is what removes
    // the per-row scalar FFMA. Rule + band: `lm_head_m16_tc_route`.
    //
    // 🔴 It REASSOCIATES the K reduction, so it is NOT bit-identical to the
    // batched GEMV (which is bit-identical to M serial `dense_gemv_bf16`
    // calls). At the LM head that is a token-visible seam, which is why the arm
    // is behind `ATLAS_LM_HEAD_M16_TC` and defaults OFF. With the lever unset
    // this whole block vanishes and the ladder is exactly what round 7 measured.
    if let Some((gemm, kernel, n_tile)) = lm_head_m16_tc_route(m16_tc, m, k) {
        log_m16_tc_head_route(n_tile, m16_tc.n_tile);
        return gemm(gpu, kernel, input, weight, output, m, n, k, k, n, stream);
    }
    if batch_enabled && batch_gemv.0 != 0 && (1..=batchm_max).contains(&m) && k.is_multiple_of(8) {
        ops::dense_gemv_batchm(gpu, batch_gemv, input, weight, output, m, n, k, n, stream)
    } else {
        ops::dense_gemm(gpu, fallback, input, weight, output, m, n, k, stream)
    }
}

/// The route line's TEXT, split out from the `Once` latch below so a test can
/// pin the wording without tripping a process-global latch that would only
/// fire once across the whole test binary.
///
/// 🔴 The ULP parenthetical is NOT "<= 2 BF16 ULP" — that bound is what round
/// 7 assumed and it is wrong. The round-9 `native_bf16_lm_head_m16_microtest`
/// measured up to **100 ordinal BF16 ULP** at M=16 (`over_budget=37`,
/// `sign_flips=1`), all on logits that had catastrophically cancelled: every
/// violation's `|ref|/rms` fell in 4.9e-6..2.6e-4 of the row scale, where one
/// FP32 accumulation rounding spans hundreds of ordinal BF16 ULP. The
/// aggregate stayed inside budget the whole time (rel_rms 1.1e-4 against a
/// 1.0e-3 gate, a 9x margin) — it is a per-element tail, not a broken kernel.
/// The tier's actual contract is `layers::dense_ffn::m16_tc::within_m16_tc_budget`:
/// 2 ordinal BF16 ULP, OR the FP32 accumulation floor for an output that has
/// cancelled that far — never a bare 2-ULP bound on every element.
fn m16_tc_head_route_message(n_tile: u32, asked: u32) -> String {
    format!(
        "[atlas] BF16 lm_head decode: ATLAS_LM_HEAD_M16_TC — tensor-core \
         dense_gemm_m16_bf16 N_TILE={n_tile} (asked {asked}) for 5..=16 rows, ahead of \
         dense_gemv_bf16_batchm. One weight pass, m16n8k16 MMA, so logits are \
         REASSOCIATED vs the scalar dense_gemv_bf16 — within 2 ordinal BF16 ULP, OR the \
         FP32 accumulation floor for outputs that have catastrophically cancelled (the \
         contract is layers::dense_ffn::m16_tc::within_m16_tc_budget, not a bare 2-ULP \
         bound; H100 round 9 measured up to 100 ordinal ULP on logits cancelled to \
         4.9e-6..2.6e-4 of the row RMS) — unlike the batched GEMV. Unset it to restore the \
         bit-exact tier (#927/#928)."
    )
}

/// Log-once latch for the tensor-core head arm. Worth a line because this arm
/// is the one that is NOT bit-identical to the M=1 decode path: a TPOT report
/// or a parity complaint at 5..=16 rows needs to say which tier ran.
fn log_m16_tc_head_route(n_tile: u32, asked: u32) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!("{}", m16_tc_head_route_message(n_tile, asked));
    });
}

impl TransformerModel {
    /// This model's tensor-core head arm: the handles this target actually
    /// carries, plus the process-wide lever. Assembled here rather than stored
    /// so the env resolution stays in ONE `OnceLock` and the struct stays a
    /// pure description of what is available.
    fn lm_head_m16_tc(&self) -> LmHeadM16Tc {
        let (enabled, n_tile) = lm_head_m16_tc_env();
        LmHeadM16Tc {
            narrow: self.lm_head_m16_tc_kernel,
            wide: self.lm_head_m16_tc_n64_kernel,
            enabled,
            n_tile,
        }
    }

    /// Project `normed` [padded_n, H] into `logits` [padded_n, V].
    ///
    /// `v` is read from `self.config.vocab_size` rather than passed: it is the
    /// same number at both call sites and a parameter would be a second place
    /// for it to be wrong.
    pub(super) fn lm_head_project_batched(
        &self,
        normed: DevicePtr,
        padded_n: usize,
        h: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        if let Some(ref fp8) = self.lm_head_fp8 {
            for i in 0..padded_n {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    normed.offset(i * h * bf16),
                    fp8,
                    logits.offset(i * v * bf16),
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // Batched GEMV for the decode head. The base M64-tile
            // `w4a16_gemm` below wastes most of its MMA tile here: at
            // padded_n=16 only 16 of 64 tile-rows carry data, and the same
            // nsys note the verify path records (impl_a3.rs) measured it at
            // 19.3 ms on this [248320, 5120] NVFP4 head vs ~2.5 ms for the
            // batched GEMV streaming the same 636 MB once. That cost is
            // FLAT in n, so it sits in the fixed term at every batch size.
            //
            // Tier by padded_n exactly as the SSM mixer does: batch4 (M<=4)
            // / batch8 (M<=8) / batch16 (M<=16). A 0-handle on any tier
            // falls through to the GEMM, so targets lacking the kernel are
            // unaffected.
            // Tile GEMM at padded_n >= 5 over the PADDED transposed twin.
            // padded_n <= 4 stays on the GEMV, which measures 3174 us =
            // 226 GB/s = 98.3% of the memory roofline on this shape and is
            // therefore unimprovable; the tile GEMM LOSES there.
            if padded_n >= 5
                && self.w4a16_gemm_t_bf16_kernel.0 != 0
                && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
            {
                // LOSSLESS path: BF16 MMA, no activation downcast.
                ops::w4a16_gemm_n128_m128_bf16_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_bf16_kernel,
                    normed,
                    nvfp4_t,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    ldb,
                    stream,
                )?;
            } else if padded_n >= 5
                && self.w4a16_gemm_t_kernel.0 != 0
                && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
            {
                ops::w4a16_gemm_n128_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_kernel,
                    normed,
                    nvfp4_t,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    ldb,
                    stream,
                )?;
            } else {
                let narrow = self.w4a16_batchm.kernel(padded_n as u32);
                let gemv_k = if narrow.0 != 0 {
                    narrow
                } else if padded_n <= 16 {
                    self.w4a16_gemv_batch16_kernel
                } else {
                    spark_runtime::gpu::KernelHandle(0)
                };
                if gemv_k.0 != 0 && lm_head_batch_gemv_enabled() {
                    ops::w4a16_gemv_batchm(
                        self.gpu.as_ref(),
                        gemv_k,
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        self.gpu.as_ref(),
                        self.w4a16_gemm_kernel,
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else {
            project_bf16_lm_head(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                self.dense_gemv_batchm_kernel,
                normed,
                &self.lm_head_weight,
                logits,
                [padded_n as u32, v as u32, h as u32],
                lmhead_batch_gemv_enabled(),
                lm_head_batchm_max(),
                self.lm_head_m16_tc(),
                stream,
            )?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
#[path = "lm_head_bf16_tests.rs"]
mod bf16_tests;
