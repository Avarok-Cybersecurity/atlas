// SPDX-License-Identifier: AGPL-3.0-only

//! Golden-render tests for the dense Qwen 27B family (Qwen3.6 / Qwen3.8).
//!
//! These render through the PRODUCTION path — `chat_render::render_chat`
//! (preprocessing + context construction) against a `build_jinja_env` of the
//! checkpoint's own template, post `convert_python_jinja_to_minijinja` — so
//! the asserted bytes are exactly what prefill sees. Fixture templates are
//! byte-copies of the checkpoints' shipped `chat_template.jinja`:
//!   * `qwen3.6-27b-unsloth.jinja` — unsloth/Qwen3.6-27B-NVFP4
//!     (md5 a7f294a5f0be5f1903214304f259f87f)
//!   * `qwen3.8-27b-unsloth.jinja` — unsloth/Qwen3.8-27B-NVFP4
//!     (md5 2a79880b328d0e0387c8ecb62c4c0c80)
//!   * `retired-qwen3_5-override-2026-04.jinja` — the byte-frozen
//!     `jinja-templates/qwen3_5.jinja` override retired 2026-08-14, kept
//!     only as the parity reference for the MLPerf-edge prompt bytes.
//!
//! Every variant golden is composed from [`Q36_GOLDEN`] by the exact byte
//! delta the flag is supposed to cause — so a test failure shows precisely
//! which bytes moved.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers;
use serde_json::json;

pub(crate) fn render_fixture(
    fixture: &str,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> anyhow::Result<String> {
    let path = format!(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/chat_templates/{}.jinja"
        ),
        fixture
    );
    let raw = std::fs::read_to_string(&path).expect("checkpoint template fixture present");
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    render_chat(&env, messages, tools, flags)
}

/// Multi-turn agentic conversation: system prompt, a tool round-trip with
/// `reasoning_content` on the historical assistant turns, and a fresh user
/// query. This is the shape whose `<think>` retention differs between the
/// Qwen3.6 and Qwen3.8 template defaults.
pub(crate) fn fixture_messages() -> Vec<serde_json::Value> {
    vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "Check the weather in Paris, then summarize."}),
        json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "User wants live weather. I should call get_weather.",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
            }]
        }),
        json!({"role": "tool", "content": "18C, sunny"}),
        json!({
            "role": "assistant",
            "content": "It is 18C and sunny in Paris.",
            "reasoning_content": "Tool says 18C sunny; summarize."
        }),
        json!({"role": "user", "content": "Thanks - now in one word?"}),
    ]
}

pub(crate) fn fixture_tools() -> Vec<serde_json::Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    })]
}

pub(crate) fn thinking_on() -> RenderFlags<'static> {
    RenderFlags {
        enable_thinking: true,
        ..Default::default()
    }
}

/// Qwen3.6 baseline: thinking on, tools active, `preserve_thinking` unset.
/// Verified byte-identical to the retired `jinja-templates/qwen3_5.jinja`
/// override's render (see `q36_render_matches_retired_override_bytes`), i.e.
/// these are the MLPerf-edge reference prompt bytes. Note the COMPACT tool
/// JSON (ST-995) and the historical `<think>` blocks STRIPPED (the Qwen3.6
/// template default when `preserve_thinking` is undefined).
const Q36_GOLDEN: &str = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nCheck the weather in Paris, then summarize.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response><|im_end|>\n<|im_start|>assistant\nIt is 18C and sunny in Paris.<|im_end|>\n<|im_start|>user\nThanks - now in one word?<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// Qwen3.8's injected instruction for the `xhigh` tier. Reached by explicit
/// `xhigh`/`max`, and by explicit `high` (the template remaps `high`→`xhigh`
/// — checkpoint-owned, so `high` shares this sentence while keeping its own
/// smaller 2E budget rung in `api/chat/thinking.rs`). NOT reached by an
/// effort-silent request: since 2026-08-15 the unset fallback is the neutral
/// `"medium"` (no sentence), not `"high"` — the old fallback silently bought
/// every silent client this most-expensive directive.
const XHIGH_SENTENCE: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW_SENTENCE: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// The two historical `<think>` blocks as the templates rehydrate them.
const THINK1: &str = "<think>\nUser wants live weather. I should call get_weather.\n</think>\n\n";
const THINK2: &str = "<think>\nTool says 18C sunny; summarize.\n</think>\n\n";

/// Byte-delta helpers over the baseline. Each corresponds to exactly one
/// template behavior; goldens for variants are composed from these so the
/// intended difference is explicit.
fn with_history_think(base: &str) -> String {
    base.replacen(
        "assistant\n<tool_call>\n<function=get_weather>",
        &format!("assistant\n{THINK1}<tool_call>\n<function=get_weather>"),
        1,
    )
    .replacen(
        "assistant\nIt is 18C",
        &format!("assistant\n{THINK2}It is 18C"),
        1,
    )
}

fn with_effort_sentence(base: &str, sentence: &str) -> String {
    base.replacen(
        "<|im_start|>system\n# Tools",
        &format!("<|im_start|>system\n{sentence}\n\n# Tools"),
        1,
    )
}

fn with_closed_think_tail(base: &str) -> String {
    let open = "<|im_start|>assistant\n<think>\n";
    assert!(base.ends_with(open));
    format!(
        "{}<|im_start|>assistant\n<think>\n\n</think>\n\n",
        base.strip_suffix(open).unwrap()
    )
}

/// Qwen3.8 render at the XHIGH tier (explicit `xhigh`/`max`/`high`):
/// Qwen3.6 baseline + xhigh instruction sentence + historical `<think>`
/// blocks KEPT (3.8 inverts the preserve default). Before 2026-08-15 this
/// was ALSO the unset-default render; see [`q38_unset_golden`].
fn q38_golden() -> String {
    with_history_think(&with_effort_sentence(Q36_GOLDEN, XHIGH_SENTENCE))
}

/// Qwen3.8 render for an effort-SILENT request (the unset default):
/// the `"medium"` fallback injects NO instruction sentence, so the system
/// block is byte-identical to the Qwen3.6-era prompt shape; only the
/// preserve-thinking default (history `<think>` kept) differs. This is
/// the 2026-08-15 contract change: unset must buy the NEUTRAL tier, not
/// the most expensive directive (`high`→`xhigh` under the old fallback).
fn q38_unset_golden() -> String {
    with_history_think(Q36_GOLDEN)
}

#[test]
fn q36_default_strips_history_think() {
    let r = render_fixture(
        "qwen3.6-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        thinking_on(),
    )
    .unwrap();
    assert_eq!(r, Q36_GOLDEN);
}

/// PROMPT-STABILITY GATE for the override retirement: the checkpoint's own
/// template must keep producing the exact bytes the retired April-2026
/// `qwen3_5.jinja` override produced (the MLPerf-edge reference prompts).
/// Both renders go through today's converter + preprocessing, so this also
/// catches a `convert_python_jinja_to_minijinja` change that breaks one
/// template's Python-isms but not the other's.
#[test]
fn q36_render_matches_retired_override_bytes() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    for (name, tools, flags) in [
        ("tools+thinking", Some(&tools[..]), thinking_on()),
        ("tools+nothink", Some(&tools[..]), RenderFlags::default()),
        ("plain+thinking", None, thinking_on()),
    ] {
        let retired =
            render_fixture("retired-qwen3_5-override-2026-04", &msgs, tools, flags).unwrap();
        let own = render_fixture("qwen3.6-27b-unsloth", &msgs, tools, flags).unwrap();
        assert_eq!(own, retired, "parity with retired override broke: {name}");
    }
}

#[test]
fn q36_preserve_thinking_true_rehydrates_history_think() {
    let r = render_fixture(
        "qwen3.6-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags {
            enable_thinking: true,
            preserve_thinking: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r, with_history_think(Q36_GOLDEN));
}

#[test]
fn q36_thinking_off_closes_think_tail_only() {
    let r = render_fixture(
        "qwen3.6-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags::default(),
    )
    .unwrap();
    // The internal "none" effort fallback is inert on Qwen3.6 (the template
    // never reads reasoning_effort); only the generation tail changes.
    assert_eq!(r, with_closed_think_tail(Q36_GOLDEN));
}

#[test]
fn q38_default_keeps_history_think_and_injects_no_effort_sentence() {
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        thinking_on(),
    )
    .unwrap();
    // CONTRACT CHANGE 2026-08-15: an effort-silent request renders the
    // NEUTRAL "medium" fallback — no instruction sentence — instead of the
    // old "high" fallback that the template escalated to the most expensive
    // xhigh directive. The template's own `default('xhigh')` must stay
    // unreachable (Atlas always passes an explicit string), and
    // `preserve_thinking` unset must leave the variable UNDEFINED — Jinja
    // `none` would flip the default to strip.
    assert_eq!(r, q38_unset_golden());
}

/// The unset default IS the medium tier: an effort-silent render and an
/// explicit `"medium"` render must be byte-identical, matching the budget
/// side where unset and medium both resolve to `max_thinking_budget` (E).
/// If these ever diverge, the directive and budget paths have split.
#[test]
fn q38_unset_equals_explicit_medium() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    let unset = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), thinking_on()).unwrap();
    let medium = render_fixture(
        "qwen3.8-27b-unsloth",
        &msgs,
        Some(&tools),
        RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some("medium"),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(unset, medium);
}

#[test]
fn q38_preserve_thinking_false_strips_history_think() {
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags {
            enable_thinking: true,
            preserve_thinking: Some(false),
            ..Default::default()
        },
    )
    .unwrap();
    // preserve=false restores the Qwen3.6-shaped history (prefix-cache
    // friendly). Effort-silent since 2026-08-15 means the neutral medium
    // fallback (no sentence), so this render is exactly the Q36 baseline.
    assert_eq!(r, Q36_GOLDEN);
}

#[test]
fn q38_thinking_off_skips_effort_validator_and_sentence() {
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags::default(),
    )
    .unwrap();
    // PROOF the internal "none" fallback can never reach Qwen3.8's effort
    // validator: with thinking off the render must SUCCEED (the template
    // gates the whole effort block on enable_thinking), emit no instruction
    // sentence, keep history think blocks (preserve default is independent
    // of enable_thinking), and close the generation tail.
    assert_eq!(r, with_closed_think_tail(&with_history_think(Q36_GOLDEN)));
}

#[test]
fn q38_explicit_efforts_render_their_sentences() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    let flags = |effort: &'static str| RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some(effort),
        ..Default::default()
    };
    // "xhigh" is what ir::ReasoningEffort::Max renders as. Since the
    // 2026-08-15 default change this sentence appears ONLY on explicit
    // request — never for an effort-silent client.
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("xhigh")).unwrap();
    assert_eq!(r, q38_golden());
    // Explicit "high": the checkpoint template remaps high→xhigh, so the
    // DIRECTIVE bytes equal xhigh's. This escalation is deliberate and
    // documented (not silent): the sentence is template-owned, while the
    // budget side keeps high its own smaller rung (2E vs xhigh's 4E in
    // api/chat/thinking.rs::effort_budget) — "high" = xhigh's steering
    // text with half the thinking allowance.
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("high")).unwrap();
    assert_eq!(r, q38_golden());
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("low")).unwrap();
    assert_eq!(
        r,
        with_history_think(&with_effort_sentence(Q36_GOLDEN, LOW_SENTENCE))
    );
    // "medium" is accepted and injects NO sentence — ir::ReasoningEffort::
    // Medium must not be demoted to "low" (different bytes).
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("medium")).unwrap();
    assert_eq!(r, with_history_think(Q36_GOLDEN));
}

/// Negative test: the raw string "max" (the OLD `ReasoningEffort::Max`
/// spelling) is REJECTED by Qwen3.8's validator. This is what makes the
/// `Max → "xhigh"` mapping in `ir::ReasoningEffort::as_str` load-bearing —
/// without it every `reasoning_effort: "max" | "xhigh"` request would 400.
#[test]
fn q38_raw_max_effort_string_raises() {
    let err = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some("max"),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("Unexpected reasoning effort"),
        "expected the template's effort validator to fire, got: {err:#}"
    );
}

/// Qwen3.8 raises on tool-call `arguments` passed as a non-empty JSON
/// STRING (the shape OpenAI-compatible clients send back on turn 2). The
/// F76 preprocessing inside `render_chat` must keep parsing them into
/// mappings so the render stays byte-identical to structured input.
#[test]
fn q38_string_tool_args_are_normalized_before_strict_validation() {
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json!("{\"location\":\"Paris\"}");
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    )
    .unwrap();
    // Effort-silent render, so the 2026-08-15 unset golden (no sentence).
    assert_eq!(r, q38_unset_golden());
}
