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
    /// Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,
    /// True iff the reasoning/`<think>` phase has finished. Starts
    /// `true` when the request did not enable thinking.
    pub(super) thinking_done: bool,
    /// True iff the streaming content is currently inside a fenced
    /// code block (between ``` markers). Toggled per delta in
    /// `process_detector_content`. The SimHash semantic-loop
    /// watchdog is *skipped* while this is true — CSS/HTML/JS has
    /// no sentence boundaries, so SimHash flushes via its 1024-byte
    /// fallback and consecutive code chunks share enough common
    /// tokens (`{`, `;`, identifiers) to false-positive on
    /// similarity even when the code is structurally distinct.
    /// The line-level watchdog at `check_loop_watchdog` already
    /// applies its own code-aware tolerance; this flag only
    /// suppresses the SimHash detector.
    pub(super) inside_code_block: bool,
    /// Sticky hint that this response has entered a structured
    /// HTML/CSS/JS/code payload. Raw HTML can exceed the watchdog's
    /// rolling tail buffer, so relying only on the latest 8 KB to
    /// infer code context can false-positive on repeated layout
    /// phrases after the opening tags have rolled out.
    pub(super) structured_content_seen: bool,
    /// True when we've seen a complete HTML document (`<!doctype html>
    /// ...</html>`) with only whitespace/fences after the closing
    /// tag. In this state, the next content is held back briefly:
    /// whitespace/fences are emitted, but alphabetic prose (self-
    /// critique like "Wait, the above...") triggers the watchdog.
    /// Without this hold, we'd either fire prematurely on the
    /// clean document end (stopping multi-part responses) or let
    /// self-critique fragments leak through before detection.
    pub(super) html_complete_seen: bool,
    /// Sticky state for long generated HTML documents. The rolling
    /// loop-scan buffer is capped, so by the time a large code block
    /// closes its ``` fence, the original `<!DOCTYPE html>` may have
    /// rolled out. These fields preserve enough document state to
    /// auto-close substantial incomplete HTML before markdown
    /// explanation loops.
    pub(super) html_doc_started: bool,
    pub(super) html_doc_closed: bool,
    pub(super) html_body_closed: bool,
    pub(super) html_script_closed: bool,
    pub(super) html_open_script_tags: usize,
    pub(super) html_doc_bytes: usize,
    /// Short hold buffer for possible closing Markdown fences after a
    /// substantial incomplete HTML document. Some token streams split
    /// ``` across three chunks, so scanning only the current delta
    /// misses the last chance to auto-close the HTML before prose.
    pub(super) pending_incomplete_html_fence: String,
    /// Component P (think→content carry-forward). True when this stream
    /// is eligible: thinking enabled, model flag on, no tools. While
    /// eligible, the thinking-phase emitter watches for an artifact
    /// start (HTML doc / language-tagged code fence).
    pub(super) cf_active: bool,
    /// Flipped true once the artifact start is detected inside `<think>`.
    /// From that point the thinking-buffer bytes stream as `content`
    /// (not `reasoning`), and the eventual `</think>` ends the response
    /// so the degenerate post-`</think>` restart is dropped.
    pub(super) cf_content_mode: bool,
    /// Bytes streamed as content since the carry-forward flip. At
    /// `</think>` we only END the response (dropping the post-`</think>`
    /// restart) if this is substantial — otherwise the flip was on a
    /// plan-illustration snippet and we must fall through to the normal
    /// content phase so the real implementation is not truncated.
    pub(super) cf_content_bytes: usize,
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
            name_run: None,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new())
            } else {
                None
            },
            thinking_done: !enable_thinking,
            inside_code_block: false,
            structured_content_seen: false,
            html_complete_seen: false,
            html_doc_started: false,
            html_doc_closed: false,
            html_body_closed: false,
            html_script_closed: false,
            html_open_script_tags: 0,
            html_doc_bytes: 0,
            pending_incomplete_html_fence: String::new(),
            cf_active: enable_thinking
                && !tools_active
                && crate::scheduler::enable_think_content_carry_forward(),
            cf_content_mode: false,
            cf_content_bytes: 0,
        }
    }
}
