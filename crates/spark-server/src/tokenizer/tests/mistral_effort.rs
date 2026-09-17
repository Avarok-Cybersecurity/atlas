// SPDX-License-Identifier: AGPL-3.0-only

//! Mistral-override `reasoning_effort` renders through the PRODUCTION
//! `render_chat` path (the same core the golden Qwen tests prove).
//!
//! Mistral is the only other template family that consumes the
//! `reasoning_effort` Jinja variable, so it pins the OTHER half of the
//! 2026-08-15 unset-default change: Atlas's cross-template fallback moved
//! from `"high"` (which Qwen3.8 escalated to its most expensive `xhigh`
//! directive) to the neutral `"medium"`. Mistral's ladder is binary
//! (`none|high`), so its Atlas-owned override maps `medium` → `high` —
//! keeping the unset Mistral render byte-identical to the pre-change
//! behavior while Qwen3.8 drops to its neutral tier.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers;
use serde_json::json;

fn render_mistral(flags: RenderFlags<'_>) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/mistral.jinja"
    ))
    .expect("bundled Mistral override template must be present");
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let messages = [json!({"role": "user", "content": "Hello"})];
    render_chat(&env, &messages, None, flags)
}

const SETTINGS_HIGH: &str = r#"[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]"#;
const SETTINGS_NONE: &str = r#"[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]"#;

/// Unset + thinking on: the "medium" fallback must land on Mistral's
/// standard thinking tier ("high") via the override's mapping — the same
/// bytes the old "high" fallback produced. If this breaks, the fallback
/// in chat_render.rs and the mapping in mistral.jinja have drifted apart.
#[test]
fn unset_effort_thinking_on_renders_high_settings() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        ..Default::default()
    })
    .unwrap();
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}

/// Explicit "medium" (a client or `--default-chat-template-kwargs`
/// choosing the neutral tier) maps identically to "high" — Mistral has no
/// medium; the neutral tier IS its standard thinking mode.
#[test]
fn explicit_medium_maps_to_high_settings() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some("medium"),
        ..Default::default()
    })
    .unwrap();
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}

/// Thinking off must still emit the explicit "none" settings — the
/// medium fallback applies ONLY when thinking is on.
#[test]
fn unset_effort_thinking_off_renders_none_settings() {
    let r = render_mistral(RenderFlags::default()).unwrap();
    assert!(r.contains(SETTINGS_NONE), "render:\n{r}");
}

/// Tiers Mistral does not have stay rejected: an explicit "low"/"xhigh"
/// raises in-template (a 400 at the API). Only the neutral fallback is
/// mapped — supported-tier vocabulary remains per-model.
#[test]
fn unsupported_explicit_tiers_still_raise() {
    for effort in ["low", "xhigh"] {
        let err = render_mistral(RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some(effort),
            ..Default::default()
        })
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("reasoning_effort must be either"),
            "effort={effort}: expected the Mistral validator to fire, got: {err:#}"
        );
    }
}
