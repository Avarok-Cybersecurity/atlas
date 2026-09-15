// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model sampling presets from MODEL.toml `[sampling.*]`, split out of
//! `lib.rs` at the 500-LoC cap. Exact piecewise move — no logic changed; the
//! types are re-exported from the crate root so every `atlas_kernels::Sampling*`
//! path is unchanged.

/// Per-category sampling defaults from MODEL.toml.
#[derive(Debug, Clone, Copy)]
pub struct SamplingCategory {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// Multiplicative penalty on already-seen tokens (1.0 = disabled).
    /// Populated from MODEL.toml `[sampling.*].repetition_penalty` via build.rs.
    pub repetition_penalty: f32,
    /// DRY (Don't-Repeat-Yourself) sampler parameters. Penalises tokens
    /// that extend repeated n-grams past `dry_allowed_length` with an
    /// exponential `dry_multiplier * dry_base^(match_len - allowed)` —
    /// the targeted fix for phrase-level attractors (e.g. the
    /// ```` ```bash cd … cargo test ``` ```` fence-narration loop
    /// observed in Qwen3.5-35B-A3B-FP8 opencode sessions at turn ≥ 8).
    ///
    /// `presence_penalty` on its own is a FLAT per-unique-token hit
    /// (does not scale with repetition count), so it can't break a
    /// phrase attractor where individual tokens already paid their
    /// penalty once. DRY scales with the repeat-length and is the
    /// published remedy (oobabooga/text-generation-webui#5677, used in
    /// llama.cpp / Aphrodite / TabbyAPI).
    ///
    /// `dry_multiplier = 0.0` disables DRY for this category (default
    /// for every preset unless MODEL.toml sets it explicitly).
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    /// LZ penalty (arXiv:2504.20131). Per-extension n-gram penalty
    /// over a 256-token rolling window. Frequency-weighted and length-
    /// scaled, so it correctly distinguishes "phrase loop" from
    /// "legitimate vocabulary reuse" without the flat-per-token
    /// `presence_penalty` regression. 0.0 = disabled. SGLang reference
    /// strength = 0.2 (lossless on AIME/GPQA).
    pub lz_penalty: f32,
    /// Model-declared min-p, or `None` when MODEL.toml is silent.
    ///
    /// `Option`, not `f32`, because absence and `0.0` mean opposite things
    /// here. The server ships `--default-min-p 0.08` and every request that
    /// does not name min_p takes it, so a model whose card specifies
    /// `min_p = 0` had no way to say so: `[behavior].min_p_floor` only ever
    /// RAISES min_p (`min_p.max(floor)`), and the preset did not carry the
    /// value at all. A plain `f32` defaulting to 0.0 would silently strip the
    /// 0.08 floor from every model that has a `[sampling.*]` table, which is
    /// the opposite regression.
    ///
    /// `Some(x)` outranks the CLI default and is outranked by
    /// `generation_config.json` — the same precedence temperature/top_k/top_p
    /// already follow. `None` preserves the CLI-owned behaviour exactly.
    pub min_p: Option<f32>,
    /// Model-declared top-n-sigma, or `None` when MODEL.toml is silent.
    ///
    /// Same absence-vs-zero problem as `min_p`: the server ships
    /// `--default-top-n-sigma 1.0`, so a model whose card asks for NO sigma
    /// filter had no way to say so. `Some(0.0)` disables it; `None` leaves the
    /// CLI default owning the field.
    pub top_n_sigma: Option<f32>,
}

/// Model-specific sampling presets loaded from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingPresets {
    pub thinking_text: SamplingCategory,
    pub thinking_coding: SamplingCategory,
    pub non_thinking: SamplingCategory,
    /// Tool-calling preset: model-recommended sampling for agentic tasks.
    /// Qwen3.5 recommends temperature=0.6 (NOT greedy) to avoid repetition loops.
    pub tools: SamplingCategory,
}

impl Default for SamplingPresets {
    fn default() -> Self {
        let default_cat = SamplingCategory {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            // DRY defaults = disabled (multiplier 0.0). Per-MODEL.toml
            // tools presets opt in when the model needs it.
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        };
        let tools_cat = SamplingCategory {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        };
        Self {
            thinking_text: default_cat,
            thinking_coding: default_cat,
            non_thinking: default_cat,
            tools: tools_cat,
        }
    }
}
