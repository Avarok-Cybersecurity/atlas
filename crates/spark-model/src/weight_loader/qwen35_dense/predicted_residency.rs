// SPDX-License-Identifier: AGPL-3.0-only

//! The derived-weight residency of the native-FP8 dense route, predicted from
//! `config.json` alone — BEFORE the checkpoint loads.
//!
//! **WHY (#915 second pass).** `fp8_residency::DerivedResidency` tallies the
//! derived copies as the loader builds them, which is four minutes too late
//! for the one decision that needs the number: preflight's SSM decode-ring
//! auto-fit runs before the first byte of the checkpoint is read, and its
//! first pass therefore fitted the ring against PRE-LOAD free memory (78.6 GB
//! on an 80 GB H100) instead of against the post-load KV headroom the KV
//! budget stage actually enforces. Measured on 1xH100 2026-09-11
//! (`h100-round2-report.md`, stage 3/5): at `--max-batch-size 16` and 32 the
//! fitter stayed silent — 8 slots, 18.94 / 37.88 GB of ring — and the serve
//! was refused minutes later by
//! `factory/build.rs`'s `No memory left for KV cache`. The fit needs the
//! post-load figure, and the only part of it that is not already known at
//! preflight is how many bytes the loader will derive on top of the
//! checkpoint.
//!
//! It IS knowable: every derived copy on this route is shape arithmetic over
//! the config, and which copies get built is
//! [`fp8_residency::DenseFp8Plan::resolve`], a pure function of the
//! environment plus one backend question (are the two W8A8 prefill kernels
//! loaded). This module evaluates both without a `WeightStore`.
//!
//! **Round-6 receipt** (`h100-round6-report.md`; serve I, Qwen3.8-27B-FP8,
//! `ATLAS_DENSE_FP8=1`, `--lm-head-dtype bf16`, tp 1): the loader's own
//! summary line reported **derived 4.24 GB** on top of a 28.75 GB checkpoint.
//! Reproduced here from `kernels/hopper/qwen3.8-27b/MODEL.toml`'s shapes:
//!
//! | term | per layer | layers | total |
//! |---|---|---|---|
//! | attn FP8 K twin + V twin | 10,488,320 B | 16 | 0.168 GB |
//! | SSM fused `[QKV\|Z]` FP8 weight | 83,886,080 B | 48 | 4.027 GB |
//! | SSM `[QKV\|Z]` + out_proj block scales | 28,160 B | 48 | 0.001 GB |
//! | SSM `in_proj_ba` interleaved BF16 | 983,040 B | 48 | 0.047 GB |
//! | **total** | | | **4.243 GB** |
//!
//! Q and O twins are absent from that total because the round-6 target ships
//! both W8A8 prefill kernels, so `DenseFp8Plan` declines them — which is why
//! [`Fp8RouteInputs::w8a8_prefill_kernels`] is an input and not an assumption.
//!
//! **What this module deliberately refuses to predict.** Anything that
//! re-opens an NVFP4 fallback (`ATLAS_DENSE_FP8_KEEP_NVFP4`, the
//! `ATLAS_CUTLASS_NVFP4_*` levers, `ATLAS_ATTN_W4A4`) returns
//! [`DerivedBytesEstimate::Unavailable`] rather than a guess: those paths
//! resurrect 18+ GiB of copies whose byte count the loader tallies through
//! `skip`, not `keep`, so a prediction built on `keep` would be wrong by more
//! than the quantity being predicted. The caller falls back to its pre-load
//! behaviour and says so in the log.

use atlas_core::config::ModelConfig;

use super::fp8_residency::{self, RouteEnv, TwinsBuilt};
use crate::layers::qwen3_attention::Fp8TwinSet;
use crate::weight_map::Nvfp4Variant;

/// The derived bytes the native-FP8 dense loader will allocate and KEEP,
/// broken out so the preflight log can name each term.
///
/// Mirrors `DerivedResidency::kept` term for term — see the module docs'
/// receipt table — so `predicted.total()` and the serve log's
/// `native FP8 dense residency: ... derived X GB` are the same arithmetic
/// evaluated at two different times.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredictedDerived {
    /// `Fp8Weight::transpose_for_gemm` twins for the projections
    /// [`fp8_residency::DenseFp8Plan`] selects, summed over full-attention
    /// layers.
    pub attn_fp8_twins: u64,
    /// The fused `[QKV|Z]` FP8 weight, its block-scale grid, the `out_proj`
    /// block-scale grid and the interleaved `in_proj_ba`, summed over
    /// linear-attention layers.
    pub ssm_fp8_concat: u64,
    /// Which twin families the prediction expects, for the log line.
    pub twins: TwinsBuilt,
    /// The twin set the attention term was priced at.
    pub attn_twin_set: Fp8TwinSet,
}

impl PredictedDerived {
    pub fn total(&self) -> u64 {
        self.attn_fp8_twins + self.ssm_fp8_concat
    }
}

/// The answer, or an honest refusal to answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivedBytesEstimate {
    /// The native-FP8 dense route will run and its derived bytes are this.
    NativeFp8Dense(PredictedDerived),
    /// No prediction. The `&'static str` is the reason, written to be
    /// readable in a serve log (`… — falling back to pre-load free memory
    /// (<reason>)`).
    Unavailable(&'static str),
}

impl DerivedBytesEstimate {
    /// Predicted bytes, or `None` when unavailable.
    pub fn bytes(&self) -> Option<u64> {
        match self {
            Self::NativeFp8Dense(p) => Some(p.total()),
            Self::Unavailable(_) => None,
        }
    }

    pub fn reason(&self) -> Option<&'static str> {
        match self {
            Self::NativeFp8Dense(_) => None,
            Self::Unavailable(why) => Some(*why),
        }
    }
}

/// The route gates `load_layers` applies, resolved from the things that exist
/// before the checkpoint does.
///
/// One field per clause of the loader's own `ffn_fp8` / `attn_fp8` /
/// `gdn_fp8_arm_selected` conditions, so a reader can diff this struct
/// against `qwen35_dense.rs:319`, `:578` and `:172` line by line.
#[derive(Clone, Copy, Debug)]
pub struct Fp8RouteInputs {
    /// `ATLAS_DENSE_FP8=1` — `qwen35_dense::dense_fp8_enabled`.
    pub dense_fp8: bool,
    /// `config.tp_world_size.max(1)`. The FP8 dense route is single-GPU only.
    pub tp_size: usize,
    /// The variant the CONFIG declares. `None` when config.json does not say
    /// — the loader would sniff the store, which does not exist yet.
    pub declared_variant: Option<Nvfp4Variant>,
    /// `ATLAS_NO_GDN_FP8` is unset — `qwen35_dense::gdn_fp8_arm_selected`.
    pub gdn_fp8: bool,
    /// Both `qwen3_attention::W8A8_PREFILL_KERNELS` are loaded for this
    /// target. Decides whether the Q and O FP8 prefill twins get built.
    pub w8a8_prefill_kernels: bool,
    /// The environment-resolved dispatch route the loader itself reads.
    pub route: RouteEnv,
}

impl Fp8RouteInputs {
    /// Resolve every gate from the process environment and the config,
    /// through the SAME predicates the loader uses.
    ///
    /// `w8a8_prefill_kernels` is the caller's, because it is a property of
    /// the loaded kernel set rather than of the environment — ask
    /// [`crate::layers::qwen3_attention::w8a8_prefill_kernels_loaded`].
    pub fn from_env(config: &ModelConfig, w8a8_prefill_kernels: bool) -> Self {
        Self {
            // Character for character `qwen35_dense::dense_fp8_enabled`:
            // `== Ok("1")`, NOT presence. `ATLAS_DENSE_FP8=0` is off.
            dense_fp8: std::env::var("ATLAS_DENSE_FP8").as_deref() == Ok("1"),
            tp_size: config.tp_world_size.max(1),
            declared_variant: crate::weight_map::config_declared_variant(config),
            // Presence, matching `gdn_fp8_arm_selected`'s `is_none()`.
            gdn_fp8: std::env::var_os("ATLAS_NO_GDN_FP8").is_none(),
            w8a8_prefill_kernels,
            route: RouteEnv::from_env(),
        }
    }
}

/// Bytes the native-FP8 dense loader will derive on top of the checkpoint.
///
/// Pure: no environment reads, no allocation, no device access. The `route`
/// argument carries everything impure (see [`Fp8RouteInputs::from_env`]), so
/// the decision table below is testable at exact integers.
pub fn predicted_derived_bytes(
    config: &ModelConfig,
    route: &Fp8RouteInputs,
) -> DerivedBytesEstimate {
    // ── Gate 1: is this even the Qwen3.5-dense loader? ────────────────────
    // `factory::loader_for_config` picks `Qwen35DenseWeightLoader` on
    // `is_qwen35_dense()` and nothing else reaches this arithmetic.
    if !config.is_qwen35_dense() {
        return DerivedBytesEstimate::Unavailable("not the Qwen3.5-dense loader");
    }
    if !route.dense_fp8 {
        return DerivedBytesEstimate::Unavailable("ATLAS_DENSE_FP8 is not 1");
    }
    if route.tp_size != 1 {
        return DerivedBytesEstimate::Unavailable("--tp-size > 1 takes the NVFP4 route");
    }
    if route.declared_variant != Some(Nvfp4Variant::Fp8Dequanted) {
        // Either the config declares a non-FP8 scheme, or it declares nothing
        // and the loader will sniff the store. Both are "ask again after the
        // load", never "assume FP8".
        return DerivedBytesEstimate::Unavailable(
            "config.json does not declare a block-scaled FP8 checkpoint",
        );
    }
    if route.route.keep_nvfp4 {
        return DerivedBytesEstimate::Unavailable(
            "ATLAS_DENSE_FP8_KEEP_NVFP4 restores the pre-#915 fallback copies",
        );
    }

    // ── Gate 2: does the plan keep any NVFP4 copy alive? ──────────────────
    // `DerivedResidency` tallies NVFP4 builds through `skip`, never `keep`,
    // so a prediction of `kept` cannot price them. Rather than guess at
    // 18+ GiB, decline. Asked at `attn_fp8 = true` because that is the route
    // under test; `ffn_nvfp4` is `!ffn_fp8` and is false by construction here.
    let plan = fp8_residency::DenseFp8Plan::resolve(fp8_residency::DenseFp8Inputs {
        ffn_fp8: true,
        attn_fp8: true,
        keep_nvfp4: false,
        dispatch: route.route.dispatch,
        w8a8_kernels: route.w8a8_prefill_kernels,
        attn_w4a4: route.route.attn_w4a4,
        attn_prefill_q_t: route.route.attn_prefill_q_t,
    });
    if plan.ffn_nvfp4 || plan.attn_nvfp4 {
        return DerivedBytesEstimate::Unavailable(
            "an NVFP4 fallback lever (ATLAS_CUTLASS_NVFP4_* / ATLAS_ATTN_W4A4) is set",
        );
    }

    let hidden = config.hidden_size;
    let (nh, hd) = (config.num_attention_heads, config.head_dim);
    let nkv = config.num_key_value_heads;
    // Same three lines as `qwen35_dense.rs:953`-`:956`.
    let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
    let attn_layers = config.num_attention_layers() as u64;
    let attn_fp8_twins = attn_layers
        * fp8_residency::attn_fp8_twin_bytes(plan.attn_fp8_twins, q_n, nkv * hd, nh * hd, hidden)
            as u64;

    let ssm_layers = config.num_ssm_layers() as u64;
    let ssm_fp8_concat = if route.gdn_fp8 && ssm_layers > 0 {
        ssm_layers * ssm_concat_bytes(config) as u64
    } else {
        0
    };

    DerivedBytesEstimate::NativeFp8Dense(PredictedDerived {
        attn_fp8_twins,
        ssm_fp8_concat,
        twins: TwinsBuilt {
            ffn_nvfp4: false,
            attn_nvfp4: false,
            attn_fp8: plan.attn_fp8_twins.any() && attn_layers > 0,
            ssm_fp8_concat: ssm_fp8_concat > 0,
        },
        attn_twin_set: plan.attn_fp8_twins,
    })
}

/// What ONE native-FP8 GDN layer keeps beyond the checkpoint bytes.
///
/// Mirrors `qwen35_dense.rs:1163`-`:1181` term for term:
///
/// * the fused `[QKV|Z]` E4M3 weight `concat_fp8_block_scaled` builds —
///   `ssm_qkvz_size() x hidden` bytes, and there is no un-fused dispatch to
///   fall back to, so it stays resident;
/// * the concatenated `[N/128, K/128]` FP32 block-scale grid, which is the
///   two source grids copied side by side (hence the sum, not one grid over
///   the fused N — `ceil` of a sum is not the sum of the `ceil`s);
/// * the `out_proj` block-scale grid, adopted for the same reason;
/// * `in_proj_ba`, the `[2*nv, hidden]` BF16 interleave of `in_proj_a` and
///   `in_proj_b`.
///
/// The per-projection source scale allocations are FREED right after the
/// concat (`residency.free`), so they are transient and correctly absent.
fn ssm_concat_bytes(config: &ModelConfig) -> usize {
    let hidden = config.hidden_size;
    let block_grid = |n: usize, k: usize| n.div_ceil(128) * k.div_ceil(128) * 4;
    let qkv_n = config.ssm_qkv_size();
    let z_n = config.ssm_z_size();
    let qkvz_bytes = config.ssm_qkvz_size() * hidden;
    let qkvz_scale_bytes = block_grid(qkv_n, hidden) + block_grid(z_n, hidden);
    // `out_proj` is `[hidden, value_dim]`.
    let out_scale_bytes = block_grid(hidden, config.ssm_z_size());
    let ba_bytes = config.linear_num_value_heads * 2 * hidden * 2;
    qkvz_bytes + qkvz_scale_bytes + out_scale_bytes + ba_bytes
}

#[cfg(test)]
#[path = "predicted_residency_tests.rs"]
mod tests;
