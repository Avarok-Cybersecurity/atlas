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
