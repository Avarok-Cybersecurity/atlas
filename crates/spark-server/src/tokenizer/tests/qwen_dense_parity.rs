// SPDX-License-Identifier: AGPL-3.0-only

//! Fleet-wide parity for the retired `qwen3_5.jinja` override.
//!
//! The override was served for EVERY `model_type = "qwen3_5"` checkpoint by
//! file presence alone, so retiring it moved four different shipped templates
//! into production, not one:
//!   * `qwen3.6-27b-unsloth.jinja` — unsloth/Qwen3.6-27B-NVFP4
//!   * `qwen3.6-27b-official.jinja` — Qwen/Qwen3.6-27B, Qwen/Qwen3.6-27B-FP8,
//!     nvidia/Qwen3.6-27B-NVFP4 AND centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf (the
//!     MLPerf-edge W4A4 reference checkpoint) all ship this byte-identical
//!     template (md5 52b6d51ae5b203cb67e64b648494dad2, verified against the
//!     four HF snapshots on-box 2026-08-14)
//!   * `qwen3.5-27b-kbenkhaled.jinja` — Kbenkhaled/Qwen3.5-27B-NVFP4
//!     (md5 94f89e03284d911fc65d06422439fd79)
//!
//! `qwen_dense.rs` proves override parity for the unsloth template over one
//! fixture conversation. This file (a) extends the parity claim to the other
//! two shipped templates, (b) sweeps the conversation shapes that fixture
//! does not reach, and (c) pins the KNOWN divergences — shapes where a
//! shipped template renders different bytes (or errors) than the override
//! did — so none of them can drift in silently.

use super::qwen_dense::{fixture_messages, fixture_tools, render_fixture, thinking_on};
use crate::tokenizer::chat_render::RenderFlags;
use serde_json::json;

/// The retired override plus every checkpoint template it was retired in
/// favor of. Parity shapes must render byte-identically across ALL of them.
const ALL_TEMPLATES: [&str; 4] = [
    "retired-qwen3_5-override-2026-04",
    "qwen3.6-27b-unsloth",
    "qwen3.6-27b-official",
    "qwen3.5-27b-kbenkhaled",
];

/// Render `messages` through every template and assert byte equality with
/// the retired override, i.e. the override retirement changed nothing for
/// this conversation shape on any qwen3_5-family checkpoint.
fn assert_fleet_parity(
    label: &str,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> String {
    let reference = render_fixture(ALL_TEMPLATES[0], messages, tools, flags)
        .unwrap_or_else(|e| panic!("{label}: retired override failed to render: {e:#}"));
    for fixture in &ALL_TEMPLATES[1..] {
        let got = render_fixture(fixture, messages, tools, flags)
            .unwrap_or_else(|e| panic!("{label}: {fixture} failed to render: {e:#}"));
        assert_eq!(got, reference, "{label}: {fixture} diverged from override");
    }
    reference
}

#[test]
fn fleet_parity_on_the_baseline_fixture_conversation() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    for (label, tools, flags) in [
        ("tools+thinking", Some(&tools[..]), thinking_on()),
        ("tools+nothink", Some(&tools[..]), RenderFlags::default()),
        ("plain+thinking", None, thinking_on()),
    ] {
        assert_fleet_parity(label, &msgs, tools, flags);
    }
}

#[test]
fn fleet_parity_assistant_content_and_tool_calls_together() {
    // Non-empty content AND tool_calls in one assistant turn exercises the
    // `content|trim` branch that inserts `\n\n` before the first call.
    let mut msgs = fixture_messages();
    msgs[2]["content"] = json!("Let me check the live weather for you.");
    let r = assert_fleet_parity(
        "content+tool_calls",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.contains("Let me check the live weather for you.\n\n<tool_call>\n"),
        "content must precede the call with a blank line:\n{r}"
    );
}

#[test]
fn fleet_parity_multiple_tool_calls_and_consecutive_results() {
    // Two calls in one turn, answered by two consecutive `tool` messages —
    // exercises the not-loop.first call separator and the previtem/nextitem
    // `<tool_response>` merging into ONE user turn.
    let msgs = vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "Weather in Paris and Lyon?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"location": "Paris"}}},
                {"id": "c2", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"location": "Lyon"}}}
            ]
        }),
        json!({"role": "tool", "content": "18C, sunny"}),
        json!({"role": "tool", "content": "16C, cloudy"}),
        json!({"role": "user", "content": "Summarize."}),
    ];
    let r = assert_fleet_parity(
        "two calls, two results",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.contains("</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLyon"),
        "second call separated by a single newline:\n{r}"
    );
    assert!(
        r.contains(
            "<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response>\n<tool_response>\n16C, cloudy\n</tool_response><|im_end|>\n"
        ),
        "consecutive results share one user turn:\n{r}"
    );
}

#[test]
fn fleet_parity_tool_result_as_final_message() {
    // History replayed up to (and ending on) the tool result — the
    // `loop.last` arm of the tool branch, plus the generation prompt after.
    let msgs = fixture_messages()[..4].to_vec();
    let r = assert_fleet_parity(
        "trailing tool result",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.ends_with("</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"),
        "tool tail closes before the generation prompt:\n{r}"
    );
}

#[test]
fn fleet_parity_inline_think_split_without_reasoning_content() {
    // No `reasoning_content` field: the templates split `<think>` out of the
    // content string instead (`content.split('</think>')…` — the Python-ism
    // path through `convert_python_jinja_to_minijinja`).
    let msgs = vec![
        json!({"role": "user", "content": "hi"}),
        json!({
            "role": "assistant",
            "content": "<think>\nsome hidden reasoning\n</think>\n\nHello there!"
        }),
        json!({"role": "user", "content": "again?"}),
    ];
    let r = assert_fleet_parity("inline think split", &msgs, None, thinking_on());
    assert!(
        r.contains("<|im_start|>assistant\nHello there!<|im_end|>\n"),
        "historical inline think must be stripped with the content kept:\n{r}"
    );
}

#[test]
fn fleet_parity_null_content_list_content_and_no_system() {
    // Three shapes in one conversation: no system message at all, a user
    // turn with OpenAI content-parts (list) form, and an assistant turn
    // whose content is JSON null next to its tool_calls.
    let msgs = vec![
        json!({"role": "user", "content": [
            {"type": "text", "text": "Check the weather in "},
            {"type": "text", "text": "Paris."}
        ]}),
        json!({
            "role": "assistant",
            "content": serde_json::Value::Null,
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}]
        }),
        json!({"role": "tool", "content": "18C"}),
        json!({"role": "user", "content": "  Thanks!  \n\n"}),
    ];
    let r = assert_fleet_parity(
        "null/list content, no system",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.contains("<|im_start|>user\nCheck the weather in Paris.<|im_end|>\n"),
        "content parts concatenate:\n{r}"
    );
    assert!(
        r.contains("<|im_start|>user\nThanks!<|im_end|>\n"),
        "user content is trimmed:\n{r}"
    );
}

#[test]
fn fleet_parity_scalar_and_structured_tool_args() {
    // Number, bool, nested-mapping and array argument values. The override
    // renders non-string scalars with `|string` while the official template
    // uses `|tojson` — for numbers and bools those agree byte-for-byte
    // (minijinja renders true/42 identically both ways), and this pins that.
    // (`null` is the one scalar where they disagree — see the divergence
    // test below.)
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json!({
        "location": "Paris",
        "days": 3,
        "metric": true,
        "filters": {"wind": true},
        "hours": [6, 12, 18]
    });
    let r = assert_fleet_parity(
        "scalar+structured args",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    for expect in [
        "<parameter=days>\n3\n</parameter>\n",
        "<parameter=metric>\ntrue\n</parameter>\n",
        "<parameter=filters>\n{\"wind\":true}\n</parameter>\n",
        "<parameter=hours>\n[6,12,18]\n</parameter>\n",
    ] {
        assert!(r.contains(expect), "missing {expect:?} in:\n{r}");
    }
}

#[test]
fn fleet_parity_empty_tools_slice_matches_absent_tools() {
    // `tools: []` and no tools at all must both skip the tools header —
    // `if tools and tools is iterable` treats the empty list as falsy.
    let msgs = fixture_messages();
    let with_empty = assert_fleet_parity("tools=[]", &msgs, Some(&[]), thinking_on());
    let with_none = assert_fleet_parity("tools absent", &msgs, None, thinking_on());
    assert_eq!(with_empty, with_none, "empty tool list must equal absent");
    assert!(
        !with_empty.contains("# Tools"),
        "no tools header:\n{with_empty}"
    );
}

#[test]
fn fleet_parity_multiple_tools_and_thinking_off() {
    let mut tools = fixture_tools();
    tools.push(json!({
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time for a location",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                           "required": ["location"]}
        }
    }));
    let r = assert_fleet_parity(
        "two tools, thinking off",
        &fixture_messages(),
        Some(&tools),
        RenderFlags::default(),
    );
    assert!(
        r.contains("\"get_weather\"") && r.contains("\"get_time\""),
        "both tools serialized:\n{r}"
    );
    assert!(
        r.ends_with("<think>\n\n</think>\n\n"),
        "closed think tail:\n{r}"
    );
}

#[test]
fn fleet_parity_continue_final_assistant_message() {
    // Diagnostic prefill mode: last message is an assistant turn and
    // `allow_continue_final` strips the generation prompt + trailing EOT.
    let msgs = vec![
        json!({"role": "user", "content": "Say exactly: banana"}),
        json!({"role": "assistant", "content": "ban"}),
    ];
    let flags = RenderFlags {
        enable_thinking: true,
        allow_continue_final: true,
        ..Default::default()
    };
    let r = assert_fleet_parity("continue-final", &msgs, None, flags);
    // The trailing assistant turn sits after `last_query_index`, so every
    // qwen3_5-family template rehydrates an (empty) think block before the
    // prefill content; `render_chat` then strips only the trailing EOT.
    assert!(
        r.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\nban"),
        "prefill tail:\n{r}"
    );
}

// ───────────────────────── known divergences ─────────────────────────
//
// Shapes where a shipped checkpoint template does NOT reproduce the retired
// override's bytes. Each is pinned exactly so a change in either direction
// (a fix upstream, or a converter change widening the blast radius) fails a
// test instead of drifting silently. All of them require a `tool_calls[*].
// function.arguments` value that survives F76 as a non-mapping — i.e. a
// malformed (non-JSON-object) arguments payload replayed from history — or
// an explicit JSON `null` argument value. Well-formed traffic (arguments
// parseable as an object, values string/number/bool/object/array) renders
// byte-identically fleet-wide, per the parity tests above.

/// JSON `null` as an argument VALUE: override/unsloth/Kbenkhaled render
/// minijinja's `|string` spelling "none"; the official template (Qwen org,
/// nvidia, AND the MLPerf centml W4A4 checkpoint) renders `|tojson` "null".
#[test]
fn divergence_null_arg_value_official_renders_null_not_none() {
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] =
        json!({"location": "Paris", "days": serde_json::Value::Null});
    let tools = fixture_tools();
    for (fixture, expect) in [
        (
            "retired-qwen3_5-override-2026-04",
            "<parameter=days>\nnone\n</parameter>\n",
        ),
        (
            "qwen3.6-27b-unsloth",
            "<parameter=days>\nnone\n</parameter>\n",
        ),
        (
            "qwen3.5-27b-kbenkhaled",
            "<parameter=days>\nnone\n</parameter>\n",
        ),
        (
            "qwen3.6-27b-official",
            "<parameter=days>\nnull\n</parameter>\n",
        ),
    ] {
        let r = render_fixture(fixture, &msgs, Some(&tools), thinking_on()).unwrap();
        assert!(r.contains(expect), "{fixture}: wanted {expect:?} in:\n{r}");
    }
}

/// Arguments that F76 cannot parse into a mapping (malformed JSON — e.g. a
/// truncated tool call replayed verbatim from history). The override
/// rendered the raw string inside the function block; unsloth's template
/// silently DROPS the arguments; the official and Kbenkhaled templates
/// ERROR the whole render (`|items` on a string), turning the request into
/// a 500. This is the one realistic shape where the override retirement
/// changed observable behavior — pinned here, called out in review.
#[test]
fn divergence_unparseable_string_args_raw_vs_dropped_vs_error() {
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json!("{\"location\": \"Par");
    let tools = fixture_tools();

    let retired = render_fixture(
        "retired-qwen3_5-override-2026-04",
        &msgs,
        Some(&tools),
        thinking_on(),
    )
    .unwrap();
    assert!(
        retired.contains("<function=get_weather>\n{\"location\": \"Par</function>\n"),
        "override emitted the raw string:\n{retired}"
    );

    let unsloth =
        render_fixture("qwen3.6-27b-unsloth", &msgs, Some(&tools), thinking_on()).unwrap();
    assert!(
        unsloth.contains("<function=get_weather>\n</function>\n"),
        "unsloth drops the malformed arguments entirely:\n{unsloth}"
    );

    for fixture in ["qwen3.6-27b-official", "qwen3.5-27b-kbenkhaled"] {
        let err = render_fixture(fixture, &msgs, Some(&tools), thinking_on())
            .expect_err("official-style templates raise on string arguments");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("render"),
            "{fixture}: expected a render error, got: {msg}"
        );
    }
}

/// A SECOND system message (no developer role anywhere, so
/// `remap_developer_role` does not touch it). The override and the
/// official/Kbenkhaled templates raise — the request failed before the
/// retirement and still fails on those checkpoints — while unsloth's
/// template now MERGES both into one system block (a previously-failing
/// request that now succeeds; strictly more permissive, never different
/// bytes for a previously-working conversation).
#[test]
fn divergence_two_system_messages_error_before_merge_on_unsloth_now() {
    let msgs = vec![
        json!({"role": "system", "content": "Rule A."}),
        json!({"role": "system", "content": "Rule B."}),
        json!({"role": "user", "content": "hi"}),
    ];
    for fixture in [
        "retired-qwen3_5-override-2026-04",
        "qwen3.6-27b-official",
        "qwen3.5-27b-kbenkhaled",
    ] {
        render_fixture(fixture, &msgs, None, thinking_on())
            .expect_err("second system message must raise");
    }
    let r = render_fixture("qwen3.6-27b-unsloth", &msgs, None, thinking_on()).unwrap();
    assert!(
        r.contains("<|im_start|>system\nRule A.\nRule B.<|im_end|>\n"),
        "unsloth merges the leading system pair:\n{r}"
    );
}
