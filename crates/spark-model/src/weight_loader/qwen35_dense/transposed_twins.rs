// SPDX-License-Identifier: AGPL-3.0-only

//! Whether this loader builds the transposed second copy of every quantised
//! weight, and what declining costs.
//!
//! **WHY (`docs/porting/r9700-residency.md`).** Atlas keeps each NVFP4
//! projection in TWO layouts: the packed `[N, K/2]` original that decode reads,
//! and a transposed `[K, N/2]` twin that the fast prefill GEMMs
//! (`w4a16_gemm_t_m128` and its v2/k64 siblings) consume. On an AMD Radeon AI
//! PRO R9700 (gfx1201, 31.9 GB) serving `unsloth/Qwen3.8-27B-NVFP4` the twins
//! are **12.74 GiB** — 8.96 GiB dense FFN, 2.90 GiB SSM, 0.88 GiB attention.
//! The residency ledger for that serve puts the whole model at 47.07 GiB;
//! release-on-consume and the attention dequant-leak fix take it to 35.18 GiB,
//! which still does not load on a 32 GB board. Dropping the twins takes it to
//! 21.25 GiB, which does.
//!
//! **What it costs, stated before the knob rather than after it.** With the
//! twins absent every fast arm of `dense_ffn.rs`'s `w4_gemm!` is skipped and
//! prefill lands on the plain `w4a16_gemm`. The Gemma-4-31B measurement that
//! `gemma4/loader_a.rs:30-52` was written against puts that at **~7.0 TFLOP/s
//! against ~51 TFLOP/s for `t_m128`**, so call it a 7x slower FFN prefill; the
//! SSM and attention sides are the same shape of trade and are unmeasured.
//! Decode is untouched — it reads the packed original either way. This is a
//! trade that is only worth taking on a board that otherwise cannot load the
//! model at all, which is why the default everywhere is "build them".
//!
//! **The precedent.** `weight_loader/gemma4/loader_a.rs::ffn_transpose_fits`
//! already does exactly this for the Gemma-4 dense FFN, under
//! `ATLAS_GEMMA4_FFN_TRANSPOSE=0`, and for the same reason. This module is that
//! mechanism generalised to the three families the Qwen3.5-class dense loader
//! builds, and it keeps gemma4's most important property: **the decision is
//! made ONCE, before any layer allocates.** `free_memory()` shrinks as layers
//! load, so a per-layer probe transposes the early layers and skips the late
//! ones, leaving prefill straddling two dispatch arms with no way to read the
//! serve log and know which.
//!
//! **Every consumer tolerates `None`**, which is what makes this a knob rather
//! than a rewrite. Verified site by site:
//!
//! | twin | consumers | fallback when `None` |
//! |---|---|---|
//! | `DenseFfnWeights::{gate,up,down}_proj_t` | `dense_ffn.rs`'s `w4_gemm!` (`_ =>` arm), `forward`'s `decode_ffn_via_gemm` arm (`wt_alive` guard), `finalize_q4k_load` / `finalize_nvfp4_mmq_load` (`if let Some`) | `ops::w4a16_gemm` on the packed original |
//! | `Qwen3AttentionLayer::{q,k,v,o}_nvfp4_t` | `prefill/paged_qkv.rs`, `prefill/cache_skip_qkv.rs`, `prefill/paged_oproj.rs`, `multi_seq/qkv.rs::wide_verify_gemm`, `multi_seq/attn/o_proj.rs` | `ops::w4a16_gemm` on `weight_opt.as_nvfp4()` / `attn.o_proj` |
//! | `Qwen3AttentionLayer::qkv_nvfp4_t` (fused) | `multi_seq/qkv.rs:764` (`is_some()` gate) | three separate GEMMs |
//! | `Qwen3SsmLayer::{qkvz,out_proj}_nvfp4_t` | `trait_prefill_proj.rs:120/:327`, `trait_prefill_helper.rs:97/:261`, `trait_decode_batched.rs:427/:483/:1069/:1159`, `ssm_batched*.rs` (`is_some()` gates) | `ops::w4a16_gemm` on `ssm.out_proj` / the packed qkvz |
//!
//! One consumer keys off a twin to decide something ELSE, and dropping the twin
//! therefore drops that too: `Qwen3SsmLayer::predequant_for_prefill`
//! (`qwen3_ssm/init_fp8.rs:110`) builds the `out_proj` FP8 predequant only
//! `if self.out_proj_nvfp4_t.is_some()`. That is a further 1.41 GiB saved on
//! Qwen3.8-27B (30 MiB x48) and it is deliberate, not incidental — the FP8
//! predequant exists to feed the same transposed prefill GEMM.
//!
//! **What this lever does NOT govern, and must not.** The attention
//! `Fp8WeightTransposed` twins that `transpose_fp8_for_prefill_selected` builds
//! are a different family on a different route: they exist only under the
//! native-FP8 overlay (`ATLAS_DENSE_FP8=1` on a block-scaled FP8 checkpoint,
//! which is not the layout this work is about), they are already selected per
//! projection by `Fp8TwinSet` and the #915 plan, and `fp8_residency.rs` records
//! that the K and V members are dereferenced UNCONDITIONALLY on the first
//! prefill chunk of every request (`prefill/cache_skip_qkv.rs:218`/`:235`, a
//! chain with no W8A8 arm). Declining to build those is a null-pointer kernel
//! launch on the first token, not a slower GEMM. `fp8_residency.rs` owns that
//! family; this module owns the NVFP4 one, and the two must not be merged.
//!
//! **What a proper fix looks like**, so this knob is not mistaken for one: a
//! prefill GEMM that reads the PACKED `[N, K/2]` layout directly with a
//! transposed tile walk, the way `w4a16_gemm` already does at scalar speed but
//! with the 128x128 cp.async tiling the `_t` kernels have. Then there is no
//! second layout to build or skip, on any target. This module buys a serve
//! tonight; it does not buy the kernel.

use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;

/// VRAM the `auto` probe refuses to spend on twins.
///
/// `docs/porting/r9700-residency.md` prices the non-weight serve at 4 GB on
/// this board: the KV cache, the buffer arena and the vision encoder's working
/// set. gemma4's `ffn_transpose_fits` reserves 2 GiB for the same job and is
/// asked about ONE family; this is asked about all three at once, on a board
/// whose whole failure mode is running out during the layer loop, so it
/// reserves the number the ledger actually names.
const SERVE_RESERVE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// What `ATLAS_LOAD_TRANSPOSED_TWINS` selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TwinPolicy {
    /// Build every twin. Today's behaviour, byte for byte.
    Always,
    /// Build none of them. Prefill runs on the plain untransposed GEMM.
    Never,
    /// Build them only if the board has room after the checkpoint is resident.
    Auto,
}

impl TwinPolicy {
    /// The string the serve log prints for this policy, in the form an
    /// operator would type to pin it.
    fn lever(self) -> &'static str {
        match self {
            Self::Always => "ATLAS_LOAD_TRANSPOSED_TWINS=1",
            Self::Never => "ATLAS_LOAD_TRANSPOSED_TWINS=0",
            Self::Auto => "ATLAS_LOAD_TRANSPOSED_TWINS=auto",
        }
    }
}

/// The policy on its own, so CPU-only CI can pin the table without touching the
/// process environment.
///
/// EXPLICIT values, not presence: the interesting operator actions here are
/// both "force them off to fit" and "force them back on to bisect a slow
/// prefill", and a presence test can only express one of those.
///
/// A value that is none of the three is treated as UNSET rather than guessed
/// at. An operator who typed `=true` gets the target default, and the load line
/// below tells them which one they got.
pub(super) fn decide(env: Option<&str>, is_scale: bool) -> TwinPolicy {
    match env {
        Some("1") => TwinPolicy::Always,
        Some("0") => TwinPolicy::Never,
        Some("auto") => TwinPolicy::Auto,
        _ if is_scale => TwinPolicy::Auto,
        _ => TwinPolicy::Always,
    }
}

/// Bytes one NVFP4 `QuantizedWeight` costs: packed `[N, K/2]` E2M1 nibbles plus
/// the `[N, K/16]` per-group scale byte. Same arithmetic as
/// `fp8_residency::nvfp4_bytes`, and a transposed twin costs exactly this
/// again — `transpose_for_gemm_gs` (`weight_map/quantized.rs:261`-`262`)
/// allocates the same two sizes.
fn nvfp4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / 16
}

/// One dense-FFN layer's twins: gate, up and down, each `[inter, hidden]` or
/// `[hidden, inter]` and therefore the same element count.
pub(super) fn ffn_twin_bytes(hidden: usize, inter: usize) -> usize {
    3 * nvfp4_bytes(inter, hidden)
}

/// One GDN layer's twins: the fused `[Q|K|V|Z]` in-projection and `out_proj`.
pub(super) fn ssm_twin_bytes(hidden: usize, qkvz_size: usize, value_dim: usize) -> usize {
    nvfp4_bytes(qkvz_size, hidden) + nvfp4_bytes(hidden, value_dim)
}

/// One full-attention layer's twins: q, k, v and o.
///
/// The fused `[q|k|v]` twin is DELIBERATELY not counted. It is built only when
/// the three projections share one `weight_scale_2`
/// (`qwen35_dense.rs`'s `scales_equal` test), and on the checkpoint this
/// projection is calibrated against they do not: `quantize_to_nvfp4` derives
/// `scale2` from each projection's own absmax. Counting a copy that is usually
/// not built would make the `auto` probe refuse room it does not need.
pub(super) fn attn_twin_bytes(q_n: usize, kv_n: usize, o_k: usize, hidden: usize) -> usize {
    nvfp4_bytes(q_n, hidden) + 2 * nvfp4_bytes(kv_n, hidden) + nvfp4_bytes(hidden, o_k)
}

/// The three families, priced from the model's own dimensions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TwinBytes {
    pub ffn: usize,
    pub ssm: usize,
    pub attn: usize,
}

impl TwinBytes {
    pub(super) fn total(self) -> usize {
        self.ffn + self.ssm + self.attn
    }
}

/// What the twins WOULD cost for this model, from the same shape arithmetic
/// `docs/porting/r9700-residency.md` reproduces the measured ledger with.
///
/// Takes the resolved `layer_types` rather than re-deriving them, because
/// `load_layers` resolves an explicit `config.layer_types` list ahead of the
/// computed pattern and a projection made from the other one would be the wrong
/// model.
pub(super) fn projected_bytes(config: &ModelConfig, layer_types: &[LayerType]) -> TwinBytes {
    let hidden = config.hidden_size;
    let inter = if config.intermediate_size > 0 {
        config.intermediate_size
    } else {
        config.moe_intermediate_size
    };
    // Every layer of this architecture carries a dense FFN, whatever its mixer.
    let ffn = ffn_twin_bytes(hidden, inter) * layer_types.len();

    let qkvz_size = config.ssm_qkvz_size();
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let ssm_layers = layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::LinearAttention))
        .count();
    let ssm = ssm_twin_bytes(hidden, qkvz_size, value_dim) * ssm_layers;

    let (nh, hd) = (config.num_attention_heads, config.head_dim);
    let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
    let kv_n = config.num_key_value_heads * hd;
    let attn_layers = layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let attn = attn_twin_bytes(q_n, kv_n, nh * hd, hidden) * attn_layers;

    TwinBytes { ffn, ssm, attn }
}

/// The decision, made once for the whole load.
///
/// One field: the policy and the projected bytes are reported on the load line
/// and then thrown away, because nothing downstream may re-derive the decision
/// — a second answer that disagreed with the first would leave prefill
/// straddling two dispatch arms, which is the failure this type exists to make
/// impossible.
#[derive(Clone, Copy, Debug)]
pub(super) struct TwinPlan {
    /// Build the transposed copies.
    pub build: bool,
}

impl TwinPlan {
    /// Resolve the policy, price the twins, probe the board if asked, and say
    /// all three on one line.
    ///
    /// Called from `load_layers` BEFORE the layer loop: see the module docs on
    /// why a per-layer probe is the wrong shape.
    pub(super) fn resolve(
        config: &ModelConfig,
        layer_types: &[LayerType],
        gpu: &dyn GpuBackend,
    ) -> Self {
        let policy = decide(
            std::env::var("ATLAS_LOAD_TRANSPOSED_TWINS").ok().as_deref(),
            cfg!(atlas_scale),
        );
        let projected = projected_bytes(config, layer_types);
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let build = match policy {
            TwinPolicy::Always => true,
            TwinPolicy::Never => false,
            TwinPolicy::Auto => {
                // `free_memory()` here is the board AFTER the checkpoint is
                // resident and BEFORE the first layer allocates, which is the
                // only moment the question has a stable answer.
                let free = gpu.free_memory().unwrap_or(0);
                let need = projected.total().saturating_add(SERVE_RESERVE_BYTES);
                let fits = free > need;
                tracing::info!(
                    "transposed prefill twins (auto): {:.2} GiB free after the checkpoint, \
                     twins need {:.2} GiB + {:.2} GiB reserved for KV, arena and vision \
                     = {:.2} GiB -> {}",
                    gib(free),
                    gib(projected.total()),
                    gib(SERVE_RESERVE_BYTES),
                    gib(need),
                    if fits { "build" } else { "skip" },
                );
                fits
            }
        };
        tracing::info!(
            "transposed prefill twins: {} ({}), projected {:.2} GiB \
             (ffn {:.2} + ssm {:.2} + attn {:.2}), prefill fast arms {}",
            if build { "built" } else { "skipped" },
            policy.lever(),
            gib(projected.total()),
            gib(projected.ffn),
            gib(projected.ssm),
            gib(projected.attn),
            if build { "on" } else { "off" },
        );
        if !build {
            tracing::warn!(
                "transposed prefill twins skipped: FFN prefill falls back to the \
                 untransposed w4a16_gemm (~7 TFLOP/s against ~51 for w4a16_gemm_t_m128 \
                 on the Gemma-4-31B measurement). Decode is unaffected. \
                 ATLAS_LOAD_TRANSPOSED_TWINS=1 restores them. \
                 See docs/porting/r9700-residency.md."
            );
        }
        Self { build }
    }
}

#[cfg(test)]
#[path = "transposed_twins_tests.rs"]
mod tests;
