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
        accepted_prediction_tokens: 0,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        expert_activation: None,
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
fn finalize_without_finish_reports_truncation_not_end_turn() {
    // Upstream died before the Finish delta: finalize must still close
    // the block and end the message coherently — but the message it ends
    // is INCOMPLETE, and `end_turn` would tell the client the model
    // finished saying what it wanted. Every client action keyed on that
    // (accept, commit, end the agent run) is then wrong, and unlike a
    // truncation reason it gives nothing to retry on. `max_tokens` is
    // Anthropic's only "output was cut short" slot and is the reason
    // `convert_stop_reason` already maps the server deadline to.
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
    assert_eq!(
        out[4].data["delta"]["stop_reason"], "max_tokens",
        "an abrupt end must not be reported as a completed turn: {:?}",
        out[4].data
    );
}

/// The wire-ready envelope the generation core puts on
/// `StreamDelta::Error` (see `api::chat_stream::handle_error`) — kept
/// byte-identical here so a change to that shape breaks this test rather
/// than silently reverting to a passthrough of raw JSON.
fn core_error_envelope(msg: &str) -> StreamDelta {
    StreamDelta::Error {
        message: serde_json::json!({
            "error": {"message": msg, "type": "server_error", "code": 500}
        })
        .to_string(),
    }
}

#[test]
fn stream_error_terminates_the_turn_as_an_error_not_a_success() {
    // A server-side inference failure mid-response. The client is already
    // on a committed HTTP 200 with partial content, so the ONLY way it can
    // tell failure from success is the terminal event. Swallowing the
    // error and letting `finalize` append message_delta+message_stop made
    // a truncated answer indistinguishable from a finished one.
    let evs = drive(vec![
        StreamDelta::Content {
            text: "partial".into(),
            token_ids: Vec::new(),
        },
        core_error_envelope("KV cache exhausted"),
    ]);

    // `drive` calls finalize after the deltas: the error must have closed
    // the turn, so no success framing may follow it.
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "error",
        ],
        "terminal framing after a stream error"
    );
    let err = &evs[4].data;
    assert_eq!(err["type"], "error");
    assert_eq!(
        err["error"]["message"], "KV cache exhausted",
        "the core's message must reach the client, unwrapped from the \
         OpenAI envelope: {err:?}"
    );
    assert!(
        !names(&evs).contains(&"message_stop"),
        "a failed turn must not be closed with the success terminator: {:?}",
        names(&evs)
    );
    assert!(
        !names(&evs).contains(&"message_delta"),
        "a failed turn must not carry a stop_reason: {:?}",
        names(&evs)
    );
}

#[test]
fn stream_error_before_any_content_still_reaches_the_client() {
    // Failure before the first token: nothing has been emitted, so an
    // empty-but-well-formed message is the most convincing lie of all.
    let evs = drive(vec![core_error_envelope("model load failed")]);
    assert_eq!(names(&evs), vec!["message_start", "error"]);
    assert_eq!(evs[1].data["error"]["message"], "model load failed");
}

#[test]
fn non_envelope_error_payload_is_passed_through() {
    // A payload that is not the OpenAI envelope must still reach the
    // client verbatim rather than being dropped for failing to parse.
    let evs = drive(vec![StreamDelta::Error {
        message: "raw failure text".into(),
    }]);
    assert_eq!(evs[1].data["error"]["message"], "raw failure text");
}
