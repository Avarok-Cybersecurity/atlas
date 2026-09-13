// SPDX-License-Identifier: AGPL-3.0-only
//
// Canonical chat IR (streaming direction). The streaming generation
// core emits a flat sequence of these provider-neutral deltas; each API
// surface owns a thin encoder that turns them into its SSE wire format
// (OpenAI chat: `openai::delta_to_chunk_events`). Emission sites never
// build wire chunks directly, so surfaces cannot drift on framing.

/// One streaming output event from the generation core. Each API
/// surface encodes these into its own SSE wire format.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamDelta {
    /// Assistant text fragment. `token_ids` are the exact sampled ids
    /// for this fragment (empty unless the client opted in).
    Content { text: String, token_ids: Vec<u32> },
    /// Reasoning/thinking fragment.
    Reasoning { text: String, token_ids: Vec<u32> },
    /// A tool call opens: OpenAI-slot `index`, call id, tool name.
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// Argument JSON fragment for the open tool call at `index`.
    ToolCallArgs {
        index: usize,
        fragment: String,
        token_ids: Vec<u32>,
    },
    /// Refusal message (safety classifier).
    Refusal { text: String },
    /// Terminal event: finish reason + final usage. `token_ids` are the
    /// residual sampled ids not yet attached to any earlier delta
    /// (tokens whose decoded text was buffered or suppressed, tool-call
    /// body tokens); carrying them on the terminal event keeps
    /// Σ token_ids == `usage.completion_tokens` for clients that opted
    /// into `return_token_ids`.
    Finish {
        reason: super::response::FinishReason,
        usage: super::response::Usage,
        token_ids: Vec<u32>,
        /// The server-side degeneration guard that cut this response,
        /// when one did (`"content_loop_watchdog"`,
        /// `"fuzzy_repetition"`, `"simhash_semantic_loop"`,
        /// `"token_loop_watchdog"`, …); `None` for every ordinary
        /// stop. Deliberately SEPARATE from `reason`: the wire
        /// `finish_reason` for a guard cut stays `"length"` (see
        /// `scheduler::lifecycle::guard_stop_wire_reason` and
        /// `api::chat_stream::handle_done::resolve_wire_finish_reason`
        /// for the measured reason that value must not move), so this
        /// is the only channel that says WHICH guard fired.
        ///
        /// #927 / #1000 / #1002, round-13 cell V
        /// (`--prefill-varlen-batch`): 6 of 16 responses were cut at 49
        /// tokens by the content-loop / fuzzy / SimHash watchdogs and
        /// every one of them reported `finish_reason: "length"` on the
        /// wire. A client could not tell a quality cut from a budget
        /// stop — which is exactly the retry decision it needs to make.
        stop_reason: Option<&'static str>,
    },
    /// Stream-level error payload, sent verbatim as SSE data.
    Error { message: String },
}

/// Boxed stream of deltas — the streaming counterpart of
/// [`super::ChatResponse`]: the pipeline produces it, each surface
/// encoder consumes it.
pub type DeltaStream = std::pin::Pin<Box<dyn futures::Stream<Item = StreamDelta> + Send>>;
