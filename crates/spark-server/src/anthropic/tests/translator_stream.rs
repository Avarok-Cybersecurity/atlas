// SPDX-License-Identifier: AGPL-3.0-only
//
// Golden tests for the Anthropic streaming translator's event framing.
// These characterize the wire output Claude Code depends on; the
// translator now consumes neutral `ir::StreamDelta`s instead of
// re-parsed OpenAI chunk JSON, but the emitted event sequences are
// unchanged.

use super::super::translator::{AnthropicTranslator, SseEvent};
use crate::ir::response::{FinishReason, Usage};
use crate::ir::stream::StreamDelta;

fn usage(prompt: usize, completion: usize) -> Usage {
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        cached_prompt_tokens: 0,
        reasoning_tokens: 0,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    }
}

fn drive(deltas: Vec<StreamDelta>) -> Vec<SseEvent> {
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    for d in &deltas {
        t.on_delta(d, &mut out);
    }
    t.finalize(&mut out);
    out
}

fn names(evs: &[SseEvent]) -> Vec<&str> {
    evs.iter().map(|e| e.event.as_str()).collect()
}

#[test]
fn text_stream_framing() {
    let evs = drive(vec![
        StreamDelta::Content {
            text: "Hi".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(3, 1),
            token_ids: Vec::new(),
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "text");
    assert_eq!(evs[2].data["delta"]["type"], "text_delta");
    assert_eq!(evs[2].data["delta"]["text"], "Hi");
    assert_eq!(evs[4].data["delta"]["stop_reason"], "end_turn");
    assert_eq!(evs[4].data["usage"]["output_tokens"], 1);
    // B1: the final message_delta patches input_tokens (usage arrives on
    // the terminal delta, after message_start already reported 0).
    assert_eq!(evs[4].data["usage"]["input_tokens"], 3);
    assert_eq!(evs[4].data["usage"]["cache_read_input_tokens"], 0);
    // message_start still opens with zero (usage unknown at that point)
    // and a minted msg_ id.
    assert_eq!(evs[0].data["message"]["usage"]["input_tokens"], 0);
    let id = evs[0].data["message"]["id"].as_str().unwrap();
    assert!(id.starts_with("msg_"), "unexpected id shape: {id}");
}

#[test]
fn thinking_then_text_framing() {
    let evs = drive(vec![
        StreamDelta::Reasoning {
            text: "think".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Content {
            text: "answer".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(0, 0),
            token_ids: Vec::new(),
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "thinking");
    assert_eq!(evs[2].data["delta"]["type"], "thinking_delta");
    assert_eq!(evs[2].data["delta"]["thinking"], "think");
    assert_eq!(evs[5].data["delta"]["type"], "text_delta");
    assert_eq!(evs[5].data["delta"]["text"], "answer");
}

#[test]
fn tool_call_stream_framing() {
    let evs = drive(vec![
        StreamDelta::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "get_weather".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 0,
            fragment: "{\"city\":\"SF\"}".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::ToolCalls,
            usage: usage(0, 0),
            token_ids: Vec::new(),
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "tool_use");
    assert_eq!(evs[1].data["content_block"]["name"], "get_weather");
    assert_eq!(evs[1].data["content_block"]["id"], "call_1");
    assert_eq!(evs[2].data["delta"]["type"], "input_json_delta");
    assert_eq!(evs[2].data["delta"]["partial_json"], "{\"city\":\"SF\"}");
    assert_eq!(evs[4].data["delta"]["stop_reason"], "tool_use");
}

#[test]
fn multi_tool_calls_close_and_reopen_blocks() {
    // Two tool calls in one turn: the second start must close the first
    // block and open a fresh one at the next index (Claude Code executes
    // each block once).
    let evs = drive(vec![
        StreamDelta::ToolCallStart {
            index: 0,
            id: "c1".into(),
            name: "read".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 0,
            fragment: "{}".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::ToolCallStart {
            index: 1,
            id: "c2".into(),
            name: "bash".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 1,
            fragment: "{\"command\":\"ls\"}".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::ToolCalls,
            usage: usage(0, 2),
            token_ids: Vec::new(),
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start", // tool 0
            "content_block_delta",
            "content_block_stop",  // tool 0 closed by tool 1 start
            "content_block_start", // tool 1
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["id"], "c1");
    assert_eq!(evs[1].data["index"], 0);
    assert_eq!(evs[4].data["content_block"]["id"], "c2");
    assert_eq!(evs[4].data["index"], 1);
}

#[test]
fn finalize_without_finish_emits_end_turn() {
    // Upstream died before the Finish delta: finalize must still close
    // the block and end the message coherently.
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    t.on_delta(
        &StreamDelta::Content {
            text: "partial".into(),
            token_ids: Vec::new(),
        },
        &mut out,
    );
    t.finalize(&mut out);
    assert_eq!(
        names(&out),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(out[4].data["delta"]["stop_reason"], "end_turn");
}
