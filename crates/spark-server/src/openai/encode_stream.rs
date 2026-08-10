// SPDX-License-Identifier: AGPL-3.0-only
//
// Encoder: neutral streaming deltas (`ir::StreamDelta`) → OpenAI
// `chat.completion.chunk` SSE events. The single seam where the
// streaming chat core's provider-neutral output becomes OpenAI wire
// format, built with the exact same `ChatCompletionChunk` constructors
// the emission sites used to call — the wire bytes are unchanged.

use axum::response::sse::Event;

use crate::ir::StreamDelta;

use super::{ChatCompletionChunk, CompletionTokensDetails, PromptTokensDetails, Usage};

/// Encode one neutral delta into its OpenAI SSE event(s).
///
/// `include_usage` is the request's `stream_options.include_usage`
/// echo: the terminal framing decision (separate usage-only chunk
/// before a usage-less finish chunk vs a single finish chunk carrying
/// usage) belongs to this wire format, not to the generation core.
pub(crate) fn delta_to_chunk_events(
    d: &StreamDelta,
    model: &str,
    id: &str,
    include_usage: bool,
) -> Vec<Event> {
    delta_to_payloads(d, model, id, include_usage)
        .into_iter()
        .map(|payload| Event::default().data(payload))
        .collect()
}

/// The `data:` payload strings for one delta, in emission order. Split
/// from [`delta_to_chunk_events`] so tests can assert the exact chunk
/// JSON (axum's `Event` is write-only).
fn delta_to_payloads(d: &StreamDelta, model: &str, id: &str, include_usage: bool) -> Vec<String> {
    match d {
        StreamDelta::Content { text, token_ids } => vec![chunk_json(
            ChatCompletionChunk::content_chunk(model, id, text.clone())
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::Reasoning { text, token_ids } => vec![chunk_json(
            ChatCompletionChunk::reasoning_chunk(model, id, text.clone())
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::ToolCallStart {
            index,
            id: call_id,
            name,
        } => {
            // Rebuild the argument shape `tool_call_start_chunk` reads:
            // `tc.id`, `tc.call_type`, `tc.function.name` (arguments are
            // always emitted empty on the start chunk). Every core
            // emission site produces `call_type == "function"` — the
            // parser pipeline hardcodes it — so the delta carries only
            // id + name.
            let tc = crate::tool_parser::ToolCall {
                id: call_id.clone(),
                call_type: "function".to_string(),
                function: crate::tool_parser::FunctionCall {
                    name: name.clone(),
                    arguments: String::new(),
                },
            };
            vec![chunk_json(ChatCompletionChunk::tool_call_start_chunk(
                model, id, &tc, *index,
            ))]
        }
        StreamDelta::ToolCallArgs {
            index,
            fragment,
            token_ids,
        } => vec![chunk_json(
            ChatCompletionChunk::tool_call_args_fragment(model, id, *index, fragment)
                .with_token_ids(token_ids.clone()),
        )],
        StreamDelta::Refusal { text } => vec![chunk_json(ChatCompletionChunk::refusal_chunk(
            model,
            id,
            text.clone(),
        ))],
        StreamDelta::Finish {
            reason,
            usage,
            token_ids,
        } => {
            let wire_usage = wire_usage(usage);
            if include_usage {
                // OpenAI `stream_options.include_usage=true` framing:
                // the usage-only chunk (`choices: []`) precedes the
                // final `finish_reason` chunk (usage omitted). Residual
                // token ids ride the final chunk, keeping
                // Σ token_ids == completion_tokens.
                vec![
                    chunk_json(ChatCompletionChunk::usage_only_chunk(model, id, wire_usage)),
                    chunk_json(
                        ChatCompletionChunk::final_chunk_no_usage(model, id, reason.as_wire())
                            .with_token_ids(token_ids.clone()),
                    ),
                ]
            } else {
                vec![chunk_json(
                    ChatCompletionChunk::done_chunk(model, id, reason.as_wire(), wire_usage)
                        .with_token_ids(token_ids.clone()),
                )]
            }
        }
        // The core hands over a wire-ready error envelope; forward it
        // verbatim as SSE data (matching the historical error arm).
        StreamDelta::Error { message } => vec![message.clone()],
    }
}

fn chunk_json(chunk: ChatCompletionChunk) -> String {
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// `ir::Usage` → OpenAI wire usage, field for field the construction
/// the streaming Done arm performed historically (`total_tokens` is
/// prompt + completion; audio/prediction counters pinned to 0).
fn wire_usage(u: &crate::ir::Usage) -> Usage {
    Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.prompt_tokens + u.completion_tokens,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: u.cached_prompt_tokens,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: u.reasoning_tokens,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms: u.time_to_first_token_ms,
        response_tokens_per_second: u.response_tokens_per_second,
    }
}

/// Full OpenAI SSE response for the `/v1/chat/completions` surface:
/// role prologue + encoded deltas + `[DONE]`, with keep-alive. Mints
/// the wire chunk id (all chunks of one response share it).
pub(crate) fn encode_sse_response(
    deltas: crate::ir::DeltaStream,
    model: String,
    include_usage: bool,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures::StreamExt;

    let chunk_id = super::new_chunk_id();
    let role_chunk = super::ChatCompletionChunk::role_chunk(&model, &chunk_id);
    let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();
    let role_event = futures::stream::once(async move {
        Ok::<_, std::convert::Infallible>(Event::default().data(role_json))
    });
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let token_stream = deltas.flat_map(move |d| {
        let events: Vec<Result<Event, std::convert::Infallible>> =
            delta_to_chunk_events(&d, &model, &chunk_id, include_usage)
                .into_iter()
                .map(Ok)
                .collect();
        futures::stream::iter(events)
    });
    let full_stream = role_event.chain(token_stream).chain(done_event);
    Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FinishReason, StreamDelta};

    /// Parse a chunk payload and zero the only wall-clock field so a
    /// payload and its expected constructor twin compare equal even
    /// when their `unix_timestamp()` calls straddle a second boundary.
    fn norm(payload: &str) -> serde_json::Value {
        let mut v: serde_json::Value = serde_json::from_str(payload).expect("valid chunk JSON");
        v["created"] = serde_json::json!(0);
        v
    }

    #[test]
    fn content_delta_encodes_via_content_chunk() {
        let d = StreamDelta::Content {
            text: "hi".into(),
            token_ids: Vec::new(),
        };
        let p = delta_to_payloads(&d, "m", "chatcmpl-x", false);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("\"content\":\"hi\""), "payload: {}", p[0]);
        assert!(
            !p[0].contains("token_ids"),
            "opt-out payload must omit token_ids: {}",
            p[0]
        );
        let expected = serde_json::to_string(&ChatCompletionChunk::content_chunk(
            "m",
            "chatcmpl-x",
            "hi".to_string(),
        ))
        .unwrap();
        assert_eq!(norm(&p[0]), norm(&expected));

        // Opted-in ids are stamped onto the chunk's first choice.
        let d = StreamDelta::Content {
            text: "hi".into(),
            token_ids: vec![1, 2],
        };
        let p = delta_to_payloads(&d, "m", "chatcmpl-x", false);
        assert!(p[0].contains("\"token_ids\":[1,2]"), "payload: {}", p[0]);
    }

    #[test]
    fn reasoning_delta_encodes_via_reasoning_chunk() {
        let d = StreamDelta::Reasoning {
            text: "mull".into(),
            token_ids: vec![9],
        };
        let p = delta_to_payloads(&d, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        assert!(
            p[0].contains("\"reasoning_content\":\"mull\""),
            "payload: {}",
            p[0]
        );
        let expected = serde_json::to_string(
            &ChatCompletionChunk::reasoning_chunk("m", "id-1", "mull".to_string())
                .with_token_ids(vec![9]),
        )
        .unwrap();
        assert_eq!(norm(&p[0]), norm(&expected));
    }

    #[test]
    fn tool_call_start_and_args_deltas_encode_openai_tool_chunks() {
        let start = StreamDelta::ToolCallStart {
            index: 3,
            id: "call_abc".into(),
            name: "get_weather".into(),
        };
        let p = delta_to_payloads(&start, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        // Per OpenAI streaming spec the start chunk carries role +
        // id/type/name with empty arguments, at the delta's tool slot.
        assert!(p[0].contains("\"role\":\"assistant\""), "payload: {}", p[0]);
        assert!(
            p[0].contains(
                "\"tool_calls\":[{\"index\":3,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]"
            ),
            "payload: {}",
            p[0]
        );

        let args = StreamDelta::ToolCallArgs {
            index: 3,
            fragment: "{\"city\":".into(),
            token_ids: Vec::new(),
        };
        let p = delta_to_payloads(&args, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        let expected = serde_json::to_string(&ChatCompletionChunk::tool_call_args_fragment(
            "m",
            "id-1",
            3,
            "{\"city\":",
        ))
        .unwrap();
        assert_eq!(norm(&p[0]), norm(&expected));
    }

    #[test]
    fn finish_framing_matches_include_usage_modes() {
        let usage = crate::ir::Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            cached_prompt_tokens: 2,
            reasoning_tokens: 3,
            time_to_first_token_ms: 12.5,
            response_tokens_per_second: 40.0,
        };
        let d = StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage,
            token_ids: vec![7],
        };

        // include_usage=true: usage-only chunk (`choices:[]`) FIRST,
        // then the finish chunk with usage omitted; residual ids ride
        // the final chunk.
        let p = delta_to_payloads(&d, "m", "id-1", true);
        assert_eq!(p.len(), 2);
        assert!(p[0].contains("\"choices\":[]"), "payload: {}", p[0]);
        assert!(p[0].contains("\"prompt_tokens\":10"), "payload: {}", p[0]);
        assert!(
            p[0].contains("\"completion_tokens\":5"),
            "payload: {}",
            p[0]
        );
        assert!(p[0].contains("\"total_tokens\":15"), "payload: {}", p[0]);
        assert!(p[0].contains("\"cached_tokens\":2"), "payload: {}", p[0]);
        assert!(p[0].contains("\"reasoning_tokens\":3"), "payload: {}", p[0]);
        assert!(
            p[0].contains("\"time_to_first_token_ms\":12.5"),
            "payload: {}",
            p[0]
        );
        assert!(
            p[0].contains("\"response_token/s\":40.0"),
            "payload: {}",
            p[0]
        );
        assert!(
            p[1].contains("\"finish_reason\":\"stop\""),
            "payload: {}",
            p[1]
        );
        assert!(
            !p[1].contains("\"usage\""),
            "final chunk must omit usage when include_usage=true: {}",
            p[1]
        );
        assert!(p[1].contains("\"token_ids\":[7]"), "payload: {}", p[1]);

        // include_usage=false: one done chunk carrying finish_reason
        // AND usage, with the residual ids stamped on it.
        let p = delta_to_payloads(&d, "m", "id-1", false);
        assert_eq!(p.len(), 1);
        assert!(
            p[0].contains("\"finish_reason\":\"stop\""),
            "payload: {}",
            p[0]
        );
        assert!(p[0].contains("\"usage\":{"), "payload: {}", p[0]);
        assert!(p[0].contains("\"total_tokens\":15"), "payload: {}", p[0]);
        assert!(p[0].contains("\"token_ids\":[7]"), "payload: {}", p[0]);
    }

    #[test]
    fn error_delta_payload_is_verbatim() {
        let msg = r#"{"error":{"message":"boom","type":"server_error","code":500}}"#;
        let d = StreamDelta::Error {
            message: msg.to_string(),
        };
        let p = delta_to_payloads(&d, "m", "id-1", false);
        assert_eq!(p, vec![msg.to_string()]);
    }
}
