// SPDX-License-Identifier: AGPL-3.0-only

use axum::response::sse::Event;

use super::helpers::*;

/// A typed Anthropic SSE event (event name + JSON data). Testable, unlike
/// axum's opaque `Event`; converted to an axum event at the handler edge
/// by [`Self::to_axum_event`].
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SseEvent {
    pub(super) event: String,
    pub(super) data: serde_json::Value,
}

impl SseEvent {
    /// Convert a reference to self to an axum SSE event for the wire.
    pub(super) fn to_axum_event(&self) -> Event {
        Event::default()
            .event(&self.event)
            .data(serde_json::to_string(&self.data).unwrap_or_default())
    }
}

/// Open-block tracker for the streaming translator's state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenBlock {
    /// No block currently open. Either before the first delta arrives
    /// or right after a `content_block_stop`.
    None,
    /// A text content block is open at index `block_idx`.
    Text,
    /// A `thinking` content block is open at index `block_idx`. Atlas
    /// emits its reasoning trace as `delta.reasoning_content` chunks
    /// inside the OpenAI stream; we map those to Anthropic's
    /// `content_block_start{type:"thinking"}` + `content_block_delta
    /// {delta.type:"thinking_delta"}` events so Claude Code can show
    /// the model's thinking progress in real time. Without this,
    /// Claude Code displayed only "Brewed for Ns" with no progress
    /// for the entire thinking phase, and users cancelled because
    /// they thought the server had stalled (2026-04-25 incident).
    Thinking,
    /// A tool_use block is open at index `block_idx` for the given
    /// OpenAI tool-call index (the `index` field on
    /// `delta.tool_calls[]`). We track it so subsequent argument
    /// fragments for the same OpenAI index get routed into the same
    /// Anthropic block, and so we know to close+reopen if a new
    /// OpenAI index appears (multi-tool turn).
    ToolUse(usize),
}

/// State machine kept alive for the lifetime of one /v1/messages
/// streaming response. Each incoming OpenAI chunk JSON is fed in via
/// `process_openai_chunk`, which pushes zero or more Anthropic
/// `Event`s into the shared sender. The state ensures correct
/// `content_block_{start,stop}` framing across:
/// - text blocks (single per turn, but produced over many OpenAI
///   `delta.content` chunks)
/// - tool_use blocks (one per OpenAI tool-call index, args streamed
///   as `input_json_delta` partials)
/// - mixed turns (text first, then tool_use; or vice versa)
/// - `finish_reason` chunk (closes any still-open block, emits
///   `message_delta` + `message_stop`)
pub struct AnthropicTranslator {
    model: String,
    msg_started: bool,
    /// Anthropic wire message id (`msg_<uuid>`), minted at construction
    /// — the delta stream is wire-id-free.
    msg_id: String,
    block_idx: u32,
    open_block: OpenBlock,
    /// Per-OpenAI-tool-index: have we emitted `content_block_start`
    /// for this tool yet? Needed because the OpenAI stream may emit
    /// the `id`+`name` (start fragment) once and then many
    /// `arguments` fragments — each subsequent `delta.tool_calls[i]`
    /// shouldn't re-trigger another `content_block_start`.
    tool_started: std::collections::HashMap<usize, ()>,
    completion_tokens: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    finished: bool,
}

/// Pull the human-readable text out of the wire-ready error envelope the
/// generation core puts on `StreamDelta::Error`.
///
/// That payload is an OpenAI envelope (`{"error":{"message":..}}`) because the
/// OpenAI surface forwards it verbatim as SSE data; Anthropic's error event
/// carries its own envelope, so the message has to be lifted out rather than
/// nested. Anything that is not that shape is passed through unchanged, so a
/// future non-JSON payload still reaches the client instead of vanishing.
fn error_detail(payload: &str) -> String {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| payload.to_string())
}

impl AnthropicTranslator {
    pub(super) fn new(model: String) -> Self {
        Self {
            model,
            msg_started: false,
            msg_id: format!("msg_{}", crate::ids::uuid_v4()),
            block_idx: 0,
            open_block: OpenBlock::None,
            tool_started: std::collections::HashMap::new(),
            completion_tokens: 0,
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            finished: false,
        }
    }

    /// Final `message_delta.usage`. Carries `input_tokens` (and cache
    /// hits) as well as `output_tokens`: the upstream usage arrives on
    /// the LAST chunk, after `message_start` already went out with
    /// `input_tokens: 0` — without patching it here, streaming clients
    /// billed every request at zero input tokens. Anthropic's
    /// MessageDeltaUsage carries all three fields (2025-05 API), and
    /// clients merge message_delta.usage over message_start's.
    fn final_usage(&self) -> serde_json::Value {
        serde_json::json!({
            "input_tokens": self.prompt_tokens,
            "cache_read_input_tokens": self.cached_prompt_tokens,
            "output_tokens": self.completion_tokens,
        })
    }

    fn make_event(ev_type: &str, data: serde_json::Value) -> SseEvent {
        SseEvent {
            event: ev_type.to_string(),
            data,
        }
    }

    /// Close the currently open block (if any) and increment the
    /// block index. Returns the matching `content_block_stop` event,
    /// or `None` if no block was open.
    ///
    /// When the closed block is a `ToolUse(oa_idx)`, also remove
    /// `oa_idx` from `tool_started` so a subsequent argument fragment
    /// for the *same* OpenAI tool index can re-trigger a fresh
    /// `content_block_start` cleanly. Without this cleanup,
    /// late-arriving arguments fall into the (formerly) defensive
    /// re-open branch below and emit a duplicate `tool_use` block at
    /// a new index — Claude Code interprets each duplicate as a
    /// separate tool execution. Root-caused 2026-04-26 (8-agent
    /// sweep, F3).
    pub(super) fn close_open_block(&mut self) -> Option<SseEvent> {
        match self.open_block {
            OpenBlock::None => None,
            OpenBlock::Text | OpenBlock::Thinking => {
                let ev = Self::make_event(
                    "content_block_stop",
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": self.block_idx,
                    }),
                );
                self.open_block = OpenBlock::None;
                self.block_idx += 1;
                Some(ev)
            }
            OpenBlock::ToolUse(oa_idx) => {
                self.tool_started.remove(&oa_idx);
                let ev = Self::make_event(
                    "content_block_stop",
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": self.block_idx,
                    }),
                );
                self.open_block = OpenBlock::None;
                self.block_idx += 1;
                Some(ev)
            }
        }
    }

    pub(super) fn ensure_message_start(&mut self, out: &mut Vec<SseEvent>) {
        if self.msg_started {
            return;
        }
        let id = self.msg_id.clone();
        out.push(Self::make_event(
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": serde_json::Value::Null,
                    "stop_sequence": serde_json::Value::Null,
                    "usage": {
                        "input_tokens": self.prompt_tokens,
                        "output_tokens": 0,
                    },
                },
            }),
        ));
        self.msg_started = true;
    }

    /// Process one neutral streaming delta, pushing the resulting
    /// Anthropic events into `out`. The block state machine is the same
    /// one that used to reverse-engineer OpenAI chunk JSON — the inputs
    /// are just typed now.
    pub(super) fn on_delta(&mut self, d: &crate::ir::StreamDelta, out: &mut Vec<SseEvent>) {
        use crate::ir::StreamDelta;
        match d {
            StreamDelta::Reasoning { text, .. } if !text.is_empty() => {
                self.ensure_message_start(out);
                if !matches!(self.open_block, OpenBlock::Thinking) {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }),
                    ));
                    self.open_block = OpenBlock::Thinking;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    }),
                ));
            }
            StreamDelta::Content { text, .. } if !text.is_empty() => {
                self.ensure_message_start(out);
                if !matches!(self.open_block, OpenBlock::Text) {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    self.open_block = OpenBlock::Text;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {"type": "text_delta", "text": text},
                    }),
                ));
            }
            StreamDelta::ToolCallStart { index, id, name } => {
                self.ensure_message_start(out);
                // The delta stream guarantees the name on the start event
                // (the old chunk protocol could stream pre-name argument
                // fragments, which had to be dropped).
                let need_start = !self.tool_started.contains_key(index);
                if need_start
                    && !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == *index)
                {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {},
                            },
                        }),
                    ));
                    self.open_block = OpenBlock::ToolUse(*index);
                    self.tool_started.insert(*index, ());
                }
            }
            StreamDelta::ToolCallArgs {
                index, fragment, ..
            } => {
                if fragment.is_empty() {
                    return;
                }
                self.ensure_message_start(out);
                if !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == *index) {
                    // Argument fragment for a tool block that is not the
                    // open one — upstream protocol violation. Drop with a
                    // warning rather than emit a duplicate tool_use block
                    // (which Claude Code would execute twice).
                    tracing::warn!(
                        target: "anthropic_translator",
                        oa_idx = index,
                        current_block = ?self.open_block,
                        arg_fragment_len = fragment.len(),
                        "dropping tool-call argument fragment for non-open tool block"
                    );
                    return;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": fragment,
                        },
                    }),
                ));
            }
            StreamDelta::Finish { reason, usage, .. } => {
                if self.finished {
                    return;
                }
                self.prompt_tokens = usage.prompt_tokens;
                self.completion_tokens = usage.completion_tokens;
                self.cached_prompt_tokens = usage.cached_prompt_tokens;
                self.ensure_message_start(out);
                if let Some(stop) = self.close_open_block() {
                    out.push(stop);
                }
                out.push(Self::make_event(
                    "message_delta",
                    serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": convert_stop_reason(reason.as_wire()),
                            "stop_sequence": serde_json::Value::Null,
                        },
                        "usage": self.final_usage(),
                    }),
                ));
                out.push(Self::make_event(
                    "message_stop",
                    serde_json::json!({"type": "message_stop"}),
                ));
                self.finished = true;
            }
            // Refusals have no Anthropic streaming representation (the
            // old translator ignored `delta.refusal` too).
            StreamDelta::Refusal { .. } => {}
            // A stream-level failure. The premise of the arm this replaced
            // — "errors abort upstream and carry no renderable event here"
            // — was false on both halves. Anthropic's streaming spec DOES
            // define a terminal `error` event, and the error does NOT abort
            // the stream: `handle_error` turns it into a `StreamDelta::Error`
            // that arrives here like any other delta, after which
            // `anthropic_sse_from_deltas` runs `finalize`. Swallowing it
            // therefore did not merely lose the message — it converted a
            // server-side failure into a well-formed SUCCESS: the client saw
            // `message_delta{stop_reason}` + `message_stop` over the HTTP 200
            // that was committed with the first token, and banked the partial
            // answer as the model's finished turn.
            //
            // `finished` is set so `finalize` cannot append that fake
            // terminal pair after the error — the error event IS the end of
            // the message, which is what the SDKs raise on.
            StreamDelta::Error { message } => {
                self.ensure_message_start(out);
                if let Some(stop) = self.close_open_block() {
                    out.push(stop);
                }
                out.push(Self::make_event(
                    "error",
                    serde_json::json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": error_detail(message)},
                    }),
                ));
                self.finished = true;
            }
            // Empty text/reasoning fragments.
            StreamDelta::Content { .. } | StreamDelta::Reasoning { .. } => {}
        }
    }

    /// Stream ended without an explicit `finish_reason` — the generation
    /// core stopped mid-turn (shutdown drain, a dropped sink, a scheduler
    /// error that never reached a terminal frame). Close the message so the
    /// framing stays legal, but do NOT claim the turn completed.
    ///
    /// `stop_reason` here is the whole point. `"end_turn"` asserts "the model
    /// said what it wanted"; for a server-side truncation that is false, and
    /// every client behaviour keyed on it is then the wrong action — accept,
    /// commit, end the agent run — with nothing anywhere to retry on.
    /// `"max_tokens"` asserts only "the output is cut short", which is true in
    /// every part clients act on. That is not a new judgement call: it is the
    /// rule `convert_stop_reason` (`helpers.rs`) already applies to the
    /// deadline cut, in a comment naming this exact failure — "mapping a
    /// deadline cut to the default `end_turn` would tell the client the turn
    /// is COMPLETE, which is the exact silent-truncation bug this reason
    /// exists to prevent". An abrupt stream end is the same event with less
    /// information, so it takes the same slot.
    ///
    /// A `StreamDelta::Error` sets `finished`, so this is unreachable after a
    /// reported failure — the arms do not overlap.
    pub(super) fn finalize(&mut self, out: &mut Vec<SseEvent>) {
        if self.finished {
            return;
        }
        self.ensure_message_start(out);
        if let Some(stop) = self.close_open_block() {
            out.push(stop);
        }
        out.push(Self::make_event(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "max_tokens",
                    "stop_sequence": serde_json::Value::Null,
                },
                "usage": self.final_usage(),
            }),
        ));
        out.push(Self::make_event(
            "message_stop",
            serde_json::json!({"type": "message_stop"}),
        ));
        self.finished = true;
    }
}
