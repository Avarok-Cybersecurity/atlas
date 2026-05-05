// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused_imports)]
use std::sync::Arc;

#[allow(unused_imports)]
use axum::extract::State;
#[allow(unused_imports)]
use axum::extract::rejection::JsonRejection;
#[allow(unused_imports)]
use axum::http::StatusCode;
#[allow(unused_imports)]
use axum::response::sse::{Event, KeepAlive};
#[allow(unused_imports)]
use axum::response::{IntoResponse, Json, Response, Sse};
#[allow(unused_imports)]
use futures::StreamExt;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::AppState;
#[allow(unused_imports)]
use crate::openai;
#[allow(unused_imports)]
use crate::tool_parser;

#[allow(unused_imports)]
use super::convert::*;
#[allow(unused_imports)]
use super::handlers_stream::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use super::translate::*;
#[allow(unused_imports)]
use super::types::*;

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
    finished: bool,
}

impl AnthropicTranslator {
    pub(super) fn new(model: String) -> Self {
        Self {
            model,
            msg_started: false,
            msg_id: String::new(),
            block_idx: 0,
            open_block: OpenBlock::None,
            tool_started: std::collections::HashMap::new(),
            completion_tokens: 0,
            prompt_tokens: 0,
            finished: false,
        }
    }

    fn make_event(ev_type: &str, data: serde_json::Value) -> Event {
        let json_str = serde_json::to_string(&data).unwrap_or_default();
        Event::default().event(ev_type).data(json_str)
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
    pub(super) fn close_open_block(&mut self) -> Option<Event> {
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

    pub(super) fn ensure_message_start(&mut self, out: &mut Vec<Event>) {
        if self.msg_started {
            return;
        }
        let id = if self.msg_id.is_empty() {
            "msg_unknown".to_string()
        } else {
            format!("msg_{}", self.msg_id.trim_start_matches("chatcmpl-"))
        };
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

    /// Process one OpenAI chat-completion chunk JSON value, push
    /// resulting Anthropic events into `out`.
    pub(super) fn process_openai_chunk(&mut self, val: &serde_json::Value, out: &mut Vec<Event>) {
        // Pick up the chat-completion id on the first chunk that
        // carries it (typically the role chunk).
        if self.msg_id.is_empty()
            && let Some(s) = val.get("id").and_then(|v| v.as_str())
        {
            self.msg_id = s.to_string();
        }

        let choice = match val.get("choices").and_then(|c| c.get(0)) {
            Some(c) => c,
            None => {
                // Some OpenAI chunks (the separate-usage chunk) carry
                // no `choices`. Pick up usage if present.
                if let Some(u) = val.get("usage") {
                    if let Some(p) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                        self.prompt_tokens = p as usize;
                    }
                    if let Some(c) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                        self.completion_tokens = c as usize;
                    }
                }
                return;
            }
        };

        let delta = choice.get("delta");

        // Capture `usage` whenever it appears (the final chunk
        // typically carries it).
        if let Some(u) = val.get("usage") {
            if let Some(p) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.prompt_tokens = p as usize;
            }
            if let Some(c) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.completion_tokens = c as usize;
            }
        }

        // Lazy message_start. We delay it until we have something
        // meaningful (the role chunk) so the emitted `prompt_tokens`
        // reflect the chat-completion's reported input usage if the
        // first chunk happens to carry it. Most servers send role +
        // usage in the first / penultimate chunks respectively.
        let has_delta_signal = delta
            .and_then(|d| d.as_object())
            .map(|o| {
                o.contains_key("role")
                    || o.contains_key("content")
                    || o.contains_key("tool_calls")
                    || o.contains_key("reasoning_content")
            })
            .unwrap_or(false);
        if has_delta_signal {
            self.ensure_message_start(out);
        }

        // Reasoning / thinking delta — Atlas emits these as
        // `delta.reasoning_content` chunks during the model's
        // thinking phase. Anthropic's streaming spec wraps thinking
        // in its own content block (`content_block.type="thinking"`)
        // with `delta.type="thinking_delta"`. Without this mapping
        // Claude Code shows "Brewed for Ns" with no visible progress
        // for the entire thinking phase, and users cancel thinking
        // the server has stalled (2026-04-25 incident).
        if let Some(d) = delta
            && let Some(text) = d.get("reasoning_content").and_then(|v| v.as_str())
            && !text.is_empty()
        {
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

        // Text content delta.
        if let Some(d) = delta {
            if let Some(text) = d.get("content").and_then(|v| v.as_str())
                && !text.is_empty()
            {
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

            // Tool-call deltas. OpenAI emits these as an array; each
            // entry has `index` (which OpenAI tool-call slot it
            // refers to) plus optionally `id`, `function.name`,
            // `function.arguments` (a string fragment).
            if let Some(tcs) = d.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let oa_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let function = tc.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments_fragment = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let need_start = !self.tool_started.contains_key(&oa_idx);
                    if need_start {
                        // Only open a new block once we have at least
                        // a name (the model's first emit per tool
                        // typically carries `id`+`name` together).
                        if name.is_empty() {
                            // Skip this fragment — wait for the start
                            // chunk to land. Subsequent argument
                            // fragments stay queued.
                            continue;
                        }
                        if !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == oa_idx) {
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
                            self.open_block = OpenBlock::ToolUse(oa_idx);
                            self.tool_started.insert(oa_idx, ());
                        }
                    } else if !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == oa_idx) {
                        // Should be unreachable now that
                        // `close_open_block` clears `tool_started`
                        // for `ToolUse` (F3, 2026-04-26): if the
                        // block were closed, `need_start` would be
                        // true and we'd have taken the branch above.
                        // The only way to reach here is the upstream
                        // OpenAI stream interleaving args for tool
                        // index `oa_idx` while a *different* tool
                        // block is open without the prior block ever
                        // being closed — protocol-level upstream
                        // bug, not something we can synthesise a
                        // safe fix for. Drop the late fragment with
                        // a warning rather than emit a duplicate
                        // `tool_use` block (which Claude Code would
                        // execute twice).
                        tracing::warn!(
                            target: "anthropic_translator",
                            oa_idx,
                            current_block = ?self.open_block,
                            arg_fragment_len = arguments_fragment.len(),
                            "dropping interleaved tool-call argument fragment for already-open different tool block"
                        );
                        continue;
                    }

                    if !arguments_fragment.is_empty() {
                        out.push(Self::make_event(
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": self.block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments_fragment,
                                },
                            }),
                        ));
                    }
                }
            }
        }

        // Finish reason: close the current block (if any) and emit
        // message_delta + message_stop. The OpenAI stream may follow
        // up with a [DONE] sentinel after this; we ignore it.
        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str())
            && !self.finished
        {
            self.ensure_message_start(out);
            if let Some(stop) = self.close_open_block() {
                out.push(stop);
            }
            out.push(Self::make_event(
                "message_delta",
                serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": convert_stop_reason(fr),
                        "stop_sequence": serde_json::Value::Null,
                    },
                    "usage": {"output_tokens": self.completion_tokens},
                }),
            ));
            out.push(Self::make_event(
                "message_stop",
                serde_json::json!({"type": "message_stop"}),
            ));
            self.finished = true;
        }
    }

    /// Stream ended without an explicit `finish_reason`. Best-effort
    /// flush so the client sees a coherent end of message.
    pub(super) fn finalize(&mut self, out: &mut Vec<Event>) {
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
                    "stop_reason": "end_turn",
                    "stop_sequence": serde_json::Value::Null,
                },
                "usage": {"output_tokens": self.completion_tokens},
            }),
        ));
        out.push(Self::make_event(
            "message_stop",
            serde_json::json!({"type": "message_stop"}),
        ));
        self.finished = true;
    }
}
