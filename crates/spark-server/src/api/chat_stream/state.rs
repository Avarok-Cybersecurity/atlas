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
    /// Byte offset into the decoded text (`content_decoded`) already
    /// emitted as `reasoning_chunk` / content deltas.
    pub(super) emitted: usize,
    /// Cumulative STABLE decoded text of `all_toks` for the current phase,
    /// byte-identical to `decode(all_toks)` with any trailing incomplete
    /// multibyte token trimmed. Grown incrementally (see `detok_incremental`)
    /// so streaming a response is O(n) rather than re-decoding the whole
    /// history every token (O(n²)). Reset alongside `all_toks`/`emitted`.
    pub(super) content_decoded: String,
    /// vLLM-style incremental-detokenizer offsets into `all_toks`: the decode
    /// window is `all_toks[prefix_offset..]`, with `[prefix_offset..read_offset]`
    /// the already-emitted prefix used for left context.
    pub(super) detok_prefix_offset: usize,
    pub(super) detok_read_offset: usize,
    /// Lazy streaming-decoder over the content phase (post-thinking).
    pub(super) content_decoder: Option<crate::tokenizer::StreamingDecoder<'static>>,
    /// Buffer used for stop-string matching across delta boundaries.
    pub(super) accumulated_content: String,
    /// Number of bytes of `accumulated_content` already forwarded to
    /// the client. The vLLM-style hold-back (see `handle_token`) keeps
    /// the last `max(stop_string_len) - 1` bytes back until either a
    /// match completes or the stream finalises, so the emitted prefix
    /// can lag behind the accumulator. Used to compute the next delta
    /// slice without re-emitting bytes.
    pub(super) stop_string_emitted_len: usize,
    /// Mirror of the post-sanitizer content stream; used by the
    /// post-stream refusal classifier and the `--dump` synthesiser.
    pub(super) refusal_scan_buf: String,
    /// Flips true on first stop-string match or on watchdog/dedup
    /// trip; suppresses further content emissions.
    pub(super) stop_string_triggered: bool,
    /// Sanitiser state: suppressing content while waiting for a
    /// matching `</parameter>` close after an orphan `<parameter=`.
    pub(super) suppressing_param_leak: bool,
    /// Consecutive tokens spent in `suppressing_param_leak=true`
    /// without a matching close arriving. When this exceeds
    /// `MAX_SUPPRESS_STREAK_TOKENS` (handle_token.rs), the stream is
    /// killed — the model is in an orphan-tool-call doom loop
    /// emitting partial envelopes that never close (observed
    /// opencode-hotfix.jsonl 2026-05-24 seq=10: 8192 tokens of
    /// suppressed content until max_tokens, no watchdog fire
    /// because the partial-envelope period exceeded 64).
    pub(super) suppress_streak_tokens: u32,
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
    /// Bug-2 (OpenClaw 2026-05-08): per-tool-name consecutive-call
    /// guard. F11 keys on `(name, canonical_args)` and is defeated by
    /// runaway loops where the model varies args slightly each
    /// iteration (e.g. timestamps, sequence numbers, IDs). This
    /// counter trips whenever the same tool name fires in N
    /// successive `ToolCallEnd` events regardless of args drift,
    /// catching the `cron`+`exec` alternation pattern observed when
    /// the streaming detector did successfully classify the calls.
    /// `(last_name, run_length)`. `last_name = None` means the run
    /// was just broken by a different tool name.
    pub(super) name_run: Option<(String, u32)>,
    /// Set true when ANY tool-call loop guard forcibly ends the
    /// response: the Bug-2 name-run cap, F11 within-response dedup,
    /// F5 cross-flush dedup, or F44 permanent-failure circuit-breaker.
    /// `handle_done` reads this and overrides `finish_reason` to
    /// `"length"` — without the override the response otherwise looks
    /// like a normal `"tool_calls"` completion (because tool calls
    /// were emitted), and agent clients (opencode, etc.) cheerfully
    /// run the tools and send the next request, perpetuating the loop
    /// from the outside. `"length"` is the OpenAI-spec slot for
    /// "response was forcibly truncated" and gives every agent a
    /// clean hook to break its outer retry loop.
    pub(super) tool_loop_capped: bool,
    /// Scheduler-side guard that force-finished the sequence (from
    /// `StreamEvent::Done.guard_stop`, e.g. "fuzzy_repetition") — surfaced
    /// in the synthesized --dump body only.
    pub(super) guard_stop: Option<&'static str>,
    /// P0-3: one corrective empty-required-args content chunk per response.
    pub(super) corrective_hint_sent: bool,
    /// Cooperative cancellation flag shared with the scheduler. Flipped
    /// true on any forced-stop condition (`tool_loop_capped`, loop-
    /// watchdog fire, …); the scheduler reads it in
    /// `emit_step::emit_token` and finalises the sequence. Without
    /// this, `stop_string_triggered` only suppresses output and the
    /// scheduler keeps generating until natural EOS / max_tokens.
    pub(super) cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Rolling tail (≤256 chars) of decoded reasoning text, used by the
    /// in-think tool-call leak scanner. Accumulated cross-delta so a
    /// boundary-split opener (e.g. `<too` in one delta + `l_call>` in
    /// the next) is still visible when the buffer is scanned. Only
    /// populated during the thinking phase.
    pub(super) reasoning_xml_scan_buf: String,
    /// One-shot flag: true once the scanner has detected a literal
    /// `<tool_call>` / `<function=` / `<parameter=` / `<invoke ` opener
    /// inside the reasoning stream. After it flips, subsequent
    /// thinking-phase tokens short-circuit with empty SSE output until
    /// the scheduler picks up the cancel_flag and finalises.
    pub(super) reasoning_xml_leak_detected: bool,
    /// Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,
    /// True iff the reasoning/`<think>` phase has finished. Starts
    /// `true` when the request did not enable thinking.
    pub(super) thinking_done: bool,
    /// Dead after the tool-call retry stack was removed (`tool_retry_enabled`
    /// is now constant `false`, so chunks are always streamed in real time
    /// and this map stays empty). Retained so the buffering helpers in
    /// `tool_handlers.rs` still type-check.
    pub(super) buffered_tool_chunks: std::collections::HashMap<usize, Vec<String>>,
    /// Dead after the tool-call retry stack was removed; never set now that
    /// `tool_retry_enabled` is constant `false`.
    pub(super) pending_retry: Option<PendingRetry>,
    /// `return_token_ids`: sampled token IDs not yet attached to an
    /// emitted chunk. One ID is pushed per `handle_token` call (== one
    /// sampled token == one increment of `usage.completion_tokens`),
    /// then drained onto the next client-visible chunk. The sum of all
    /// drained IDs across the stream therefore equals
    /// `completion_tokens` exactly. Stays empty unless the request
    /// opted in, so it costs nothing on the default path.
    pub(super) pending_token_ids: Vec<u32>,
}

/// Carrier for the (now-removed) tool-call retry path. Never constructed
/// anymore, but retained so `pending_retry`'s type still resolves.
pub(super) struct PendingRetry {
    pub(super) errors_summary: String,
    pub(super) failed_idx: usize,
}

impl StreamState {
    pub(super) fn new(
        tools_active: bool,
        enable_thinking: bool,
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        tool_defs: Vec<tool_parser::ToolDefinition>,
    ) -> Self {
        Self {
            all_toks: Vec::new(),
            emitted: 0,
            content_decoded: String::new(),
            detok_prefix_offset: 0,
            detok_read_offset: 0,
            content_decoder: None,
            accumulated_content: String::new(),
            stop_string_emitted_len: 0,
            refusal_scan_buf: String::new(),
            stop_string_triggered: false,
            suppressing_param_leak: false,
            suppress_streak_tokens: 0,
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
            name_run: None,
            tool_loop_capped: false,
            guard_stop: None,
            corrective_hint_sent: false,
            cancel_flag,
            reasoning_xml_scan_buf: String::new(),
            reasoning_xml_leak_detected: false,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new_with_tools(
                    tool_defs,
                ))
            } else {
                None
            },
            thinking_done: !enable_thinking,
            buffered_tool_chunks: HashMap::new(),
            pending_retry: None,
            pending_token_ids: Vec::new(),
        }
    }
}
