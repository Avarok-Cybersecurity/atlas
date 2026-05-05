// SPDX-License-Identifier: AGPL-3.0-only
//
// Mutable per-stream state captured by the `flat_map` closure in
// `chat_stream.rs`. Lifted out of that closure so each `StreamEvent`
// arm can be extracted to a free function (`handle_token`,
// `handle_done`, `handle_error`) that takes `&mut StreamState` plus
// any additional non-state arguments.
//
// Read-only context (`Arc<AppState>`, model name, tool defs, ...) is
// passed via `StreamCtx` (see `ctx.rs`) so the helpers don't need to
// duplicate two dozen function-parameter slots.

use std::collections::HashMap;

use crate::tool_parser;

pub(super) struct StreamState {
    /// Token IDs accumulated since the last reset (cleared at the
    /// `</think>` boundary so post-thinking content decodes cleanly).
    pub(super) all_toks: Vec<u32>,
    /// Byte offset into the thinking-phase decoded text already
    /// emitted as `reasoning_chunk` deltas.
    pub(super) emitted: usize,
    /// Lazy streaming-decoder over the content phase (post-thinking).
    pub(super) content_decoder: Option<crate::tokenizer::StreamingDecoder<'static>>,
    /// Buffer used for stop-string matching across delta boundaries.
    pub(super) accumulated_content: String,
    /// Mirror of the post-sanitizer content stream; used by the
    /// post-stream refusal classifier and the `--dump` synthesiser.
    pub(super) refusal_scan_buf: String,
    /// Flips true on first stop-string match or on watchdog/dedup
    /// trip; suppresses further content emissions.
    pub(super) stop_string_triggered: bool,
    /// Sanitiser state: suppressing content while waiting for a
    /// matching `</parameter>` close after an orphan `<parameter=`.
    pub(super) suppressing_param_leak: bool,
    /// Sanitiser state: currently inside a tool-call envelope opener
    /// (e.g. `<minimax:tool_call>`); inner `<invoke ...>` etc. are
    /// legitimate content while this is true.
    pub(super) inside_envelope: bool,
    /// Mirror of `inside_envelope` for the reasoning sanitiser.
    pub(super) reasoning_inside_envelope: bool,
    /// Tag-scan buffer for the content sanitiser.
    pub(super) tag_scan_buf: String,
    /// Sanitiser state for reasoning content (parallel to
    /// `suppressing_param_leak` above).
    pub(super) reasoning_suppressing_leak: bool,
    /// Tag-scan buffer for the reasoning sanitiser.
    pub(super) reasoning_tag_scan_buf: String,
    /// Repetition-loop watchdog: tail buffer for line-level
    /// duplicate detection.
    pub(super) loop_scan_buf: String,
    /// Set true when the watchdog or SimHash guard fires.
    pub(super) loop_watchdog_triggered: bool,
    /// Set true when the watchdog salvages a fenced/XML tool intent
    /// into a synthetic `tool_call` so the Done arm picks the right
    /// `finish_reason`.
    pub(super) salvaged_tool_call: bool,
    /// F4: SimHash semantic-loop guard for paraphrased restarts.
    pub(super) simhash_guard: crate::loop_simhash::SimHashLoopGuard,
    /// F4: pending bytes accumulated until a sentence-boundary or
    /// 1KB force-flush triggers a `simhash_guard.check()`.
    pub(super) simhash_pending: String,
    /// F5: cross-flush tool-arg dedup (default thresholds).
    pub(super) tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup,
    /// F11: tighter within-response tool-arg dedup for the
    /// streaming `ToolCallEnd` path.
    pub(super) tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup,
    /// F11: per-streaming-toolcall accumulator keyed by `oa_idx`.
    /// Holds (name, args_so_far) until `ToolCallEnd` runs the dedup.
    pub(super) streaming_tool_args: HashMap<usize, (String, String)>,
    /// F12: per-response total tool-call count.
    pub(super) tool_calls_emitted_count: usize,
    /// Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,
    /// True iff the reasoning/`<think>` phase has finished. Starts
    /// `true` when the request did not enable thinking.
    pub(super) thinking_done: bool,
}

impl StreamState {
    pub(super) fn new(tools_active: bool, enable_thinking: bool) -> Self {
        Self {
            all_toks: Vec::new(),
            emitted: 0,
            content_decoder: None,
            accumulated_content: String::new(),
            refusal_scan_buf: String::new(),
            stop_string_triggered: false,
            suppressing_param_leak: false,
            inside_envelope: false,
            reasoning_inside_envelope: false,
            tag_scan_buf: String::new(),
            reasoning_suppressing_leak: false,
            reasoning_tag_scan_buf: String::new(),
            loop_scan_buf: String::new(),
            loop_watchdog_triggered: false,
            salvaged_tool_call: false,
            simhash_guard: crate::loop_simhash::SimHashLoopGuard::new(),
            simhash_pending: String::new(),
            tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup::new(),
            tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup::with_params(4, 2, 3),
            streaming_tool_args: HashMap::new(),
            tool_calls_emitted_count: 0,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new())
            } else {
                None
            },
            thinking_done: !enable_thinking,
        }
    }
}
