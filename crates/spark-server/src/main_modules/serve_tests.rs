// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `serve.rs`.
//!
//! A sibling file rather than an inline `mod tests`, matching
//! `serve_load_tests.rs` and the rest of this directory: `serve.rs` is not on
//! the file-size-cap allow-list, and the vision-bound cases alone would have
//! carried it past 500 lines.

use super::*;

fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    for (name, body) in files {
        std::fs::write(d.path().join(name), body).expect("write");
    }
    d
}

/// The HF `save_pretrained` shape: image fields at the top level of
/// `preprocessor_config.json`. Qwen/Qwen3.6-35B-A3B-FP8 ships this.
#[test]
fn reads_the_flat_preprocessor_config() {
    let d = dir_with(&[(
        "preprocessor_config.json",
        r#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536}}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

/// The combined-processor shape: each modality under its own key in
/// `processor_config.json`. unsloth/Qwen3.6-27B-NVFP4 ships this, and
/// reading only the other filename is why it ran the 1280 fallback
/// while declaring 4096² of permitted area.
#[test]
fn reads_the_nested_processor_config() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

/// THE TRAP. `processor_config.json` carries a second, LARGER bound
/// under `video_processor`, so any implementation that scans for the
/// first (or largest) `longest_edge` in the document admits still
/// images at half again their permitted area. Uses the real numbers
/// from the shipped checkpoint.
#[test]
fn the_video_bound_never_wins_over_the_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{
            "video_processor": {"size": {"longest_edge": 25165824}, "fps": 2},
            "image_processor": {"size": {"longest_edge": 16777216}}
        }"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

/// A video-only processor config declares nothing about stills, so the
/// image path must fall back rather than borrow the video bound.
#[test]
fn a_video_only_config_yields_no_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"video_processor": {"size": {"longest_edge": 25165824}}}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), None);
}

/// Older processors write the count directly instead of under `size`.
#[test]
fn accepts_the_direct_max_pixels_spelling() {
    let d = dir_with(&[("preprocessor_config.json", r#"{"max_pixels": 1048576}"#)]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(1_048_576));
}

/// Precedence when a checkpoint ships both: the dedicated image file
/// wins, so a stale combined config cannot override it.
#[test]
fn the_dedicated_file_outranks_the_combined_one() {
    let d = dir_with(&[
        ("preprocessor_config.json", r#"{"max_pixels": 1048576}"#),
        (
            "processor_config.json",
            r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
        ),
    ]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(1_048_576));
}

/// A checkpoint we cannot read must still SERVE, at the historical
/// behaviour — never fail to boot over a preprocessing hint.
#[test]
fn unreadable_or_absent_config_falls_back_rather_than_failing() {
    let empty = tempfile::tempdir().expect("tempdir");
    assert_eq!(read_preprocessor_max_pixels(empty.path()), None);

    let broken = dir_with(&[("preprocessor_config.json", "{not json")]);
    assert_eq!(read_preprocessor_max_pixels(broken.path()), None);

    let zero = dir_with(&[("preprocessor_config.json", r#"{"max_pixels": 0}"#)]);
    assert_eq!(read_preprocessor_max_pixels(zero.path()), None);
}

/// A malformed FIRST source must not shadow a good later one — the
/// loop continues rather than committing to the file it opened.
#[test]
fn a_broken_first_source_does_not_mask_a_good_second() {
    let d = dir_with(&[
        ("preprocessor_config.json", r#"{"size": {}}"#),
        (
            "processor_config.json",
            r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
        ),
    ]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

// canonicalize_model_quant is exercised via integration through
// the server boot path; unit-testing it requires building
// ModelConfig which has no `Default` impl (it's intentionally
// bound to a loaded model). The pair-compatibility table is a
// pure function and worth a unit test.

#[test]
fn compat_self_pair() {
    assert!(quant_pair_compatible("nvfp4", "nvfp4"));
    assert!(quant_pair_compatible("fp8", "fp8"));
    assert!(quant_pair_compatible("bf16", "bf16"));
}

#[test]
fn compat_nvfp4_handles_fp8_and_bf16() {
    assert!(quant_pair_compatible("nvfp4", "fp8"));
    assert!(quant_pair_compatible("nvfp4", "bf16"));
}

#[test]
fn incompat_unknown_rejected() {
    assert!(!quant_pair_compatible("nvfp4", "gptq-4bit"));
    assert!(!quant_pair_compatible("fp8", "nvfp4"));
}

// ── --default-chat-template-kwargs parsing ─────────────────────────────
// The operator knob for the served default reasoning_effort (2026-08-15).
// Fail-fast contract: bad JSON, unknown keys, and unknown effort values
// abort startup — a typo'd operator default must never boot a server
// that silently serves a different tier (the pre-change parser warned
// and IGNORED).

#[test]
fn default_kwargs_reasoning_effort_sets_both_halves() {
    use crate::ir::{EffortLevel, ReasoningEffort, ThinkingDirective};
    let kw = parse_default_chat_template_kwargs(r#"{"reasoning_effort":"xhigh"}"#).unwrap();
    // The template string AND the budget directive come from one parse,
    // so effort-silent requests get the same tier on both paths.
    assert_eq!(kw.reasoning_effort, Some(ReasoningEffort::Max));
    assert_eq!(kw.thinking, ThinkingDirective::OnEffort(EffortLevel::XHigh));
    assert_eq!(kw.preserve_thinking, None);

    // "none" as the server default = thinking off by default.
    let kw = parse_default_chat_template_kwargs(r#"{"reasoning_effort":"none"}"#).unwrap();
    assert_eq!(kw.reasoning_effort, None);
    assert_eq!(kw.thinking, ThinkingDirective::Off);
}

#[test]
fn default_kwargs_explicit_thinking_keys_outrank_effort_directive() {
    use crate::ir::{ReasoningEffort, ThinkingDirective};
    let kw =
        parse_default_chat_template_kwargs(r#"{"thinking_budget":512,"reasoning_effort":"low"}"#)
            .unwrap();
    // Budget rung wins for the directive; the effort string still sets
    // the template default.
    assert_eq!(kw.thinking, ThinkingDirective::On { budget: Some(512) });
    assert_eq!(kw.reasoning_effort, Some(ReasoningEffort::Low));
}

#[test]
fn default_kwargs_legacy_shapes_still_parse() {
    use crate::ir::ThinkingDirective;
    let kw = parse_default_chat_template_kwargs(r#"{"enable_thinking":true}"#).unwrap();
    assert_eq!(kw.thinking, ThinkingDirective::On { budget: None });
    let kw = parse_default_chat_template_kwargs(r#"{"enable_thinking":false}"#).unwrap();
    assert_eq!(kw.thinking, ThinkingDirective::Off);
    let kw = parse_default_chat_template_kwargs("").unwrap();
    assert_eq!(kw, DefaultChatTemplateKwargs::default());
    let kw = parse_default_chat_template_kwargs(r#"{"preserve_thinking":false}"#).unwrap();
    assert_eq!(kw.preserve_thinking, Some(false));
}

#[test]
fn default_kwargs_fail_fast_on_typos() {
    // Unknown effort value.
    assert!(
        parse_default_chat_template_kwargs(r#"{"reasoning_effort":"hgih"}"#)
            .unwrap_err()
            .to_string()
            .contains("hgih")
    );
    // Unknown key (deny_unknown_fields): the old parser silently ignored
    // it, which is exactly how "--default-chat-template-kwargs
    // reasoning_effort=..." appeared to work while doing nothing.
    // ("reasoning_efforts" — plural — is the unknown-key stand-in; a true
    // misspelling here trips the typos CI lint.)
    assert!(parse_default_chat_template_kwargs(r#"{"reasoning_efforts":"low"}"#).is_err());
    // Invalid JSON.
    assert!(parse_default_chat_template_kwargs("not json").is_err());
}

/// The nvfp4-labeled bundle carries the EXL3 dispatch (exl3_matmul/exl3_moe/
/// exl3_reconstruct compile into it). A `quant_method: "exl3"` pack must reach
/// the loader rather than be refused at the gate — and the reverse pair must
/// still be refused, since a bundle built WITHOUT the EXL3 kernels cannot
/// decode a trellis.
#[test]
fn nvfp4_bundle_accepts_exl3_but_not_the_reverse() {
    assert!(super::quant_pair_compatible("nvfp4", "exl3"));
    assert!(!super::quant_pair_compatible("exl3", "nvfp4"));
    assert!(!super::quant_pair_compatible("bf16", "exl3"));
}

// ── GLM-5.3's token-budget spelling ──────────────────────────────────────

/// GLM-5.3-Flash's real `processor_config.json[image_processor]` shape: a token
/// budget, a patch size and a merge size, and NO `size` / `max_pixels`. Before
/// the token arm this returned `None` and every image took the 1280px clamp.
///
/// 8000 tokens x (14*2)^2 = 6_272_000 px, which is 32000 pre-merge patches —
/// twice the encoder's 16384 ceiling — so the clamp bites and the resolved
/// bound is 16384 * 14^2 = 3_211_264.
#[test]
fn a_token_budget_resolves_and_clamps_to_the_encoder_ceiling() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"patch_size": 14, "merge_size": 2,
             "min_image_tokens": 16, "max_image_tokens": 8000}}"#,
    )]);
    assert_eq!(
        read_preprocessor_max_tokens_as_pixels(d.path()),
        Some(3_211_264)
    );
}

/// Under the ceiling the declared budget passes through unclamped.
#[test]
fn a_small_token_budget_passes_through_unclamped() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"patch_size": 14, "merge_size": 2, "max_image_tokens": 1024}}"#,
    )]);
    // 1024 * 28^2 = 802_816, and 802_816 / 14^2 = 4096 patches < 16384.
    assert_eq!(
        read_preprocessor_max_tokens_as_pixels(d.path()),
        Some(802_816)
    );
}

/// An AREA bound is exact where a token budget has to be converted, so the
/// area wins whenever a checkpoint states both.
#[test]
fn an_area_bound_outranks_a_token_budget() {
    let d = dir_with(&[(
        "preprocessor_config.json",
        r#"{"max_pixels": 1048576, "max_image_tokens": 8000,
            "patch_size": 14, "merge_size": 2}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(1_048_576));
}

/// ⚠ The video budget is 30x the image budget on GLM-5.3 (240000 vs 8000).
/// Explicit `[image_processor]` addressing is what keeps it out; a recursive
/// key search would over-admit every still image by that factor.
#[test]
fn a_video_only_token_budget_yields_no_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"video_processor": {"patch_size": 14, "merge_size": 2,
             "max_image_tokens": 240000}}"#,
    )]);
    assert_eq!(read_preprocessor_max_tokens_as_pixels(d.path()), None);
}

/// The mean/std pair is all-or-nothing, and a zero std is rejected because it
/// divides in the preprocessor's patch loop.
#[test]
fn image_stats_are_all_or_nothing() {
    let glm = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"image_mean": [0.48145466, 0.4578275, 0.40821073],
             "image_std": [0.26862954, 0.26130258, 0.27577711],
             "min_image_tokens": 16, "max_image_tokens": 8000}}"#,
    )]);
    let (mean, std, min_t, max_t) = read_preprocessor_image_stats(glm.path());
    assert_eq!(mean.expect("mean")[0], 0.481_454_66);
    assert_eq!(std.expect("std")[2], 0.275_777_1);
    assert_eq!((min_t, max_t), (Some(16), Some(8000)));

    let zero = dir_with(&[(
        "preprocessor_config.json",
        r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.0, 0.5]}"#,
    )]);
    assert_eq!(
        read_preprocessor_image_stats(zero.path()),
        (None, None, None, None)
    );

    let half = dir_with(&[(
        "preprocessor_config.json",
        r#"{"image_mean": [0.5, 0.5, 0.5]}"#,
    )]);
    assert_eq!(
        read_preprocessor_image_stats(half.path()),
        (None, None, None, None)
    );
}
