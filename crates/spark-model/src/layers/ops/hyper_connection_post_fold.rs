// SPDX-License-Identifier: AGPL-3.0-only

//! Folding a site's `hc_post` into the NEXT site's `hc_pre` stage kernel.
//!
//! `hc_post` is in place on the highway: `out == residual == streams`. Per
//! site it reads [T, hc*H] FP32 and writes it straight back, and then
//! `hc_pre_stage_bf16` reads the very same array to norm it. At the shipping
//! prefill chunk (T=7841, H=2560, hc=4) that is 321 MB read + 321 MB written,
//! then 321 MB read again for 161 MB of BF16 `normed`. Folding the residual
//! add into the stage kernel's RMS pass and keeping the updated value in
//! registers removes the second 321 MB read outright — 15.4 GB a chunk across
//! the 48 INTRA-LAYER sites this module serves.
//!
//! **Scope: intra-layer only.** 48 of the 96 prefill `hc_pre`/`hc_post` pairs
//! are the attention/GDN sublayer's `hc_post` followed by the FFN sublayer's
//! `hc_pre` inside ONE function with only debug taps between them
//! (`prefill_inner.rs` and `qwen3_ssm/trait_prefill_hc.rs`). Those need no
//! deferred state beyond a local. The other 48 — the FFN `hc_post` — cross the
//! layer loop, the attention/ssm file boundary, the PLE in-place highway add,
//! the `hc_expand` seed and the `hc_head` flush, and are deliberately left
//! alone: that is where the whole deferred-state hazard class lives.
//!
//! **One function owns the decision.** [`hc_post_folds_into_next_pre`] is
//! consulted by the site that SKIPS `hc_post` and, through
//! [`HcPreArm::stages_bf16`], by the launch that APPLIES it. A caller that
//! folds on one predicate while the callee computes on another is how a whole
//! sublayer's residual goes missing, so there is exactly one.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::hyper_connection_dispatch::HcVariant;
use super::hyper_connection_lowrank_rows::{
    HC_DEC_MAX_T, hc_decode_rows_enabled, hc_decode_rows_shape_ok, hc_pre_chunk_enabled,
};
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};

/// `AVAROK_HC_FUSE_POST=1`: fold the intra-layer `hc_post` into the next
/// `hc_pre` stage. Same opt-in convention as `AVAROK_HC_FUSE_UP_MIX` and
/// `AVAROK_HC_FUSE_DOWN_INJ`; default OFF until an nsys kernel-summary delta
/// says what it is worth. The predicted saving (~1.4-1.9% of prefill) sits
/// under the ~2.3% run-to-run band, so `measure_prefill` CANNOT settle it.
pub(crate) fn hc_fuse_post() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("AVAROK_HC_FUSE_POST").as_deref() == Ok("1"))
}

/// The stage kernel's launch geometry. `hc_pre_gemm` launches
/// `hc_pre_stage_bf16` at block 1024 and it MUST stay there: the per-thread
/// RMS accumulation order, and therefore the answer, is a function of
/// `blockDim.x`.
pub(crate) const HC_STAGE_BLOCK: u32 = 1024;
/// `QHC_STAGE_MULT` in `hyper_connection.cu`.
const HC_STAGE_MULT: u32 = 4;
/// `QHC_STAGE_SLOTS` in `hyper_connection.cu`.
const HC_STAGE_SLOTS: u32 = 3;

/// Whether `hc_pre_stage_bf16_post` can keep the folded value in registers at
/// this shape. Outside it the kernel is still CORRECT — it re-reads `streams`
/// — but the re-read is the entire win, so production is held to the
/// registered shape and the arm line says which one ran.
pub(crate) fn hc_stage_fold_shape_ok(hidden_size: u32, hc_mult: u32) -> bool {
    hc_mult <= HC_STAGE_MULT && hidden_size <= HC_STAGE_SLOTS * HC_STAGE_BLOCK
}

/// Which body `hc_pre_lowrank` runs. Extracted from that function's if-ladder
/// so the fold predicate and the dispatch cannot drift apart — the predicate
/// has to know whether the next `hc_pre` will reach a `hc_pre_stage_bf16`
/// launch at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HcPreArm {
    DecodeRows,
    DecodeRowsChunked,
    DecodeGemm,
    DecodeSplit,
    PrefillGemm,
    Fused,
}

impl HcPreArm {
    /// The arms that go through `hc_pre_gemm`, whose FIRST launch is
    /// `hc_pre_stage_bf16` — the only place a deferred `hc_post` can land.
    ///
    /// NOT "prefill vs decode", and NOT `use_cublas`: `DecodeGemm` sets
    /// `use_cublas = true` and `PrefillGemm` sets it to `hc_prefill_cublas()`,
    /// which is false by default. `use_cublas` picks how the three low-rank
    /// PROJECTIONS run and says nothing about the stage kernel, which both
    /// arms launch identically.
    pub(crate) fn stages_bf16(self) -> bool {
        matches!(self, Self::DecodeGemm | Self::PrefillGemm)
    }
}

/// The single source of truth for `hc_pre_lowrank`'s arm selection.
pub(crate) fn hc_pre_arm(
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    rank: u32,
) -> HcPreArm {
    let have_scratch = !scratch.is_null();
    if have_scratch
        && hc_decode_rows_enabled()
        && hc_decode_rows_shape_ok(num_tokens, hidden_size, hc_mult, rank)
    {
        return HcPreArm::DecodeRows;
    }
    if have_scratch
        && hc_decode_rows_enabled()
        && hc_pre_chunk_enabled()
        && num_tokens > HC_DEC_MAX_T
        && hc_decode_rows_shape_ok(HC_DEC_MAX_T, hidden_size, hc_mult, rank)
    {
        return HcPreArm::DecodeRowsChunked;
    }
    if num_tokens <= 64 && have_scratch {
        return if super::hyper_connection_lowrank::hc_decode_split_forced() {
            HcPreArm::DecodeSplit
        } else {
            HcPreArm::DecodeGemm
        };
    }
    if have_scratch && !super::hyper_connection_lowrank::hc_gemm_disabled() {
        return HcPreArm::PrefillGemm;
    }
    HcPreArm::Fused
}

/// A residual add the NEXT `hc_pre` owes the highway.
///
/// Not `Copy` and not `Clone`, so it can be applied at most once; the `Drop`
/// bomb below is what makes it at LEAST once. Together: exactly once.
#[derive(Debug)]
pub struct HcDeferredPost {
    /// `[T, H]` BF16 — the skipped site's block output. Must still be live
    /// when the next `hc_pre` runs; on both fused sites the only thing between
    /// them is a debug tap.
    pub(crate) block_out: DevicePtr,
    /// `[T, hc]` FP32 — the skipped site's injection vector.
    pub(crate) inj: DevicePtr,
}

impl HcDeferredPost {
    pub(crate) fn new(block_out: DevicePtr, inj: DevicePtr) -> Self {
        Self { block_out, inj }
    }

    /// Consume it. `mem::forget` is the point: it is how "applied" is spelled,
    /// and every other way out of scope trips the bomb.
    pub(crate) fn apply(self) -> (DevicePtr, DevicePtr) {
        let pair = (self.block_out, self.inj);
        std::mem::forget(self);
        pair
    }
}

impl Drop for HcDeferredPost {
    fn drop(&mut self) {
        // Reaching here means a folded `hc_post` was never applied: the
        // highway is missing an entire sublayer's residual and every token
        // after it is wrong. Loud in both profiles — a silent drop is exactly
        // the failure this type exists to prevent, and release builds serve.
        tracing::error!(
            "a folded hc_post was dropped without being applied — the mHC highway \
             is missing a sublayer's residual add"
        );
        debug_assert!(
            false,
            "HcDeferredPost dropped without apply(): a folded hc_post never reached \
             a stage launch"
        );
    }
}

/// **THE predicate.** True when this site's `hc_post` may be skipped because
/// the next `hc_pre` will apply it.
///
/// Every conjunct is load-bearing:
///
/// * `LowRank` — Sinkhorn's `hc_post` mixes the streams through a `[hc, hc]`
///   combine matrix. Different arithmetic, and its `hc_pre` does not stage
///   BF16 `normed` at all.
/// * `out == residual == streams` — the fold writes the highway in place,
///   which is what `hc_post` does at all 23 sites but is a contract, not an
///   accident.
/// * `taps_inert` — `tap_highway` and the `diag_norm_f32` probes between the
///   two sites READ the highway. Folding moves the residual add past them, so
///   with a tap armed they would report a stale highway. A bisect tap that
///   lies is worse than a slow one.
/// * `stages_bf16` — the next `hc_pre` must actually reach
///   `hc_pre_stage_bf16`. See [`HcPreArm::stages_bf16`] for why this is not
///   spelled "prefill" and not spelled `use_cublas`.
/// * shape and kernel presence — a target built without the fused entry point
///   degrades to the stock arm rather than refusing to serve, so the presence
///   check belongs here too.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_folds_into_next_pre(
    gpu: &dyn GpuBackend,
    hc: &HcWeights,
    next: &HcSiteWeights,
    residual: DevicePtr,
    out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    taps_inert: bool,
) -> bool {
    if !hc_fuse_post() || !taps_inert {
        return false;
    }
    if HcVariant::of(hc) != HcVariant::LowRank {
        return false;
    }
    // `inject_w` non-null is not cosmetic: `hc_pre_lowrank_folding` refuses a
    // site without it, and that refusal must not be reachable while a folded
    // residual is in flight — the bomb in `HcDeferredPost::drop` would fire on
    // top of the real error and bury it.
    let Some(nw) = next.lowrank.as_ref().filter(|w| !w.inject_w.is_null()) else {
        return false;
    };
    if residual != out {
        return false;
    }
    let hc_mult = hc.hc_mult as u32;
    if !hc_stage_fold_shape_ok(hidden_size, hc_mult) {
        return false;
    }
    if !hc_pre_arm(scratch, num_tokens, hidden_size, hc_mult, nw.rank as u32).stages_bf16() {
        return false;
    }
    // try_kernel, not kernel: same fail-soft shape as `hc_pre_up_mix`.
    if crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_stage_bf16_post").0 == 0 {
        return false;
    }
    {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(
                num_tokens,
                hidden_size,
                hc_mult,
                "hc_post arm: FOLDED into the next hc_pre stage (hc_post launch removed)"
            )
        });
    }
    true
}
