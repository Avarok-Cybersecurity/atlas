// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

use super::*;

// ── Responses API (2026 stable) — adapter types ──
//
// The Responses API is OpenAI's newer agentic-first surface. Atlas
// implements a thin adapter: requests are translated into a
// ChatCompletionRequest, run through the existing pipeline, and the
// result re-serialized in the Responses shape.
//
// Stateful resume (`previous_response_id`) is supported via the in-memory
// [`crate::response_store`]: the prior turn's transcript is prepended to
// the current input. Built-in tools (web_search, file_search, computer_
// use) are **not** supported — those require NSFW tool integrations Atlas
// does not ship.

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ResponsesRequest {
    pub model: String,
    /// String or array of Responses "input items".
    pub input: serde_json::Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    /// Raw tool list. Parsed lazily in `lower_responses_to_chat` so
    /// built-in tools (`web_search_preview`, `file_search`,
    /// `computer_use_preview`, `code_interpreter`, `image_generation`,
    /// `mcp`) can be rejected with an informative 400 instead of
    /// failing `ToolDefinition` deserialization.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Raw tool_choice. Parsed in `lower_responses_to_chat` so the
    /// Responses-flat form `{type:"function", name:"X"}` can be re-shaped
    /// into the chat-format `{function:{name:"X"}}` the downstream
    /// `ToolChoice` enum expects.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Prior-turn id for stateful conversation resume. Resolved via the
    /// in-memory `ResponseStore`; the stored transcript is prepended
    /// to the current `input`.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Store the response for later retrieval (`true` = default per
    /// OpenAI's 2026 Responses API). `false` opts out of storage.
    #[serde(default)]
    pub store: Option<bool>,
    /// Run the response asynchronously. Atlas completes responses
    /// synchronously (no background queue), so this flag is **accepted
    /// and ignored** — the call returns the finished response directly.
    /// Cancel via `POST /v1/responses/{id}/cancel` therefore returns a
    /// 400 `response_not_cancellable` error.
    #[serde(default)]
    pub background: Option<bool>,
    /// `include: ["reasoning.encrypted_content", ...]` — additional
    /// payloads to embed in the response. Atlas accepts and ignores
    /// this; the base response already carries reasoning and tool data.
    #[serde(default)]
    pub include: Option<Vec<String>>,
    /// `truncation: "auto" | "disabled"` — OpenAI's auto-compaction
    /// hint. Atlas has its own auto-compaction (`--auto-compact` CLI
    /// flag) so this is accepted and ignored.
    #[serde(default)]
    pub truncation: Option<String>,
    /// `conversation: <conv_id>` — new in 2026: link this response to
    /// a Conversations API object. When set, items from the conversation
    /// are prepended to `input`, and the new turn's items are appended
    /// to the conversation after completion.
    #[serde(default)]
    pub conversation: Option<serde_json::Value>,
    /// `parallel_tool_calls: bool` — Atlas emits one tool call per
    /// turn regardless; accepted for compat.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Top-level `max_tool_calls` cap. Accepted for compat; Atlas
    /// already bounds tool calls by the scheduler.
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    /// Advanced text-output config (`{format: {...}, verbosity: ...}`)
    /// per the 2026 Responses spec. Accepted; verbosity is ignored
    /// (Atlas honors `max_output_tokens` only).
    #[serde(default)]
    pub text: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    pub output: Vec<ResponsesOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    pub usage: ResponsesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<ResponsesContentPart>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<Annotation>>,
    },
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<PromptTokensDetails>,
    pub output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<CompletionTokensDetails>,
    pub total_tokens: usize,
}

// ── Streaming Responses API event types ──
//
// The 2026-stable Responses streaming model emits typed events wrapped in
// SSE frames. Atlas maps the existing chat-completions `StreamEvent`
// channel into these events as it flushes tokens:
//
//   response.created                → once, on admission
//   response.in_progress            → once, after the first token
//   response.output_item.added      → at the start of each output item
//                                     (message or function_call)
//   response.output_text.delta      → per decoded text chunk
//   response.function_call.arguments.delta → per tool-arg fragment
//   response.output_item.done       → at the end of each output item
//   response.completed              → terminal envelope with usage
//   response.failed                 → emitted on generation error
//
// `sequence_number` increments by 1 per emitted event (OpenAI behavior).
// The handler layer is responsible for serializing these in SSE frames
// (`event: <type>\ndata: <json>\n\n`).
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: ResponsesStreamEnvelope,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        sequence_number: u64,
        response: ResponsesStreamEnvelope,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: usize,
        item: ResponsesOutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ResponsesContentPart,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        arguments: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: usize,
        item: ResponsesOutputItem,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        refusal: String,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: ResponsesResponse,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: ResponsesStreamEnvelope,
        error: serde_json::Value,
    },
}

/// Envelope shape used by the `created`, `in_progress`, and `failed`
/// events — a trimmed subset of `ResponsesResponse` (no output items yet
/// / no final usage).
#[derive(Debug, Serialize)]
pub struct ResponsesStreamEnvelope {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// The SSE `event:` name for a given stream event (used by the handler's
/// SSE writer).
pub fn responses_event_name(ev: &ResponsesStreamEvent) -> &'static str {
    match ev {
        ResponsesStreamEvent::Created { .. } => "response.created",
        ResponsesStreamEvent::InProgress { .. } => "response.in_progress",
        ResponsesStreamEvent::OutputItemAdded { .. } => "response.output_item.added",
        ResponsesStreamEvent::ContentPartAdded { .. } => "response.content_part.added",
        ResponsesStreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
        ResponsesStreamEvent::OutputTextDone { .. } => "response.output_text.done",
        ResponsesStreamEvent::FunctionCallArgumentsDelta { .. } => {
            "response.function_call_arguments.delta"
        }
        ResponsesStreamEvent::FunctionCallArgumentsDone { .. } => {
            "response.function_call_arguments.done"
        }
        ResponsesStreamEvent::OutputItemDone { .. } => "response.output_item.done",
        ResponsesStreamEvent::RefusalDelta { .. } => "response.refusal.delta",
        ResponsesStreamEvent::RefusalDone { .. } => "response.refusal.done",
        ResponsesStreamEvent::Completed { .. } => "response.completed",
        ResponsesStreamEvent::Failed { .. } => "response.failed",
    }
}
