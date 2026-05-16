// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Token` / `StreamEvent::TokenWithLogprobs` arm of the
// streaming `flat_map` closure (originally ~672 LoC at the top of the
// `chat_stream::chat_completions_stream` body).
//
// Returns the SSE events produced for this single token. Callers
// invoke `futures::stream::iter(...)` on the result to feed the
// `flat_map` output stream.

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::failures::{
    HTML_RESTART_GUARD_MIN_BYTES, bump_f12_tool_call_count, check_loop_watchdog_with_context,
};
use super::super::sanitizer::sanitize_content_chunk;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_delta, handle_tool_call_end, handle_tool_call_start,
};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;
const HTML_AUTOCLOSE_MIN_BYTES: usize = 4_000;
/// Minimum content bytes carried forward from `<think>` for Component P
/// to END the response (dropping the post-`</think>` restart). Below
/// this, the flip was on a plan-illustration snippet — fall through to
/// the normal content phase instead of truncating the real answer.
/// ~2 KB is far above any plan sketch, far below a real single-file app.
const CF_MIN_ARTIFACT_BYTES: usize = 2_000;

/// Whether the carried thinking text holds a STRUCTURALLY COMPLETE
/// artifact (not just a substantial draft). An HTML doc must have a
/// `</html>` after its last start; otherwise a code fence must be
/// balanced (even, non-zero ``` count). A draft the model abandons
/// ("…Let's write the actual code now.") has no close → returns false
/// so Component P does NOT truncate; the real post-`</think>`
/// implementation is allowed to stream instead.
fn cf_artifact_is_complete_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let html_start = ["<!doctype html", "<html>", "<html ", "<html\n"]
        .iter()
        .filter_map(|m| lower.rfind(m))
        .max();
    if let Some(start) = html_start {
        return lower[start..].contains("</html>");
    }
    let fences = lower.matches("```").count();
    fences >= 2 && fences % 2 == 0
}

/// Minimum byte span of a CURRENTLY-OPEN artifact inside `<think>`
/// before the streaming channel flips to `content`. Reasoning plans
/// contain tiny illustrative fences (a ~150 B `<!DOCTYPE>…</html>`
/// skeleton, a 5-line sample) that OPEN and CLOSE quickly — they never
/// stay open this long, so they keep streaming as `reasoning` (clean).
/// Only a genuine large artifact stays open and crosses this, flipping
/// to `content`.
const CF_OPEN_FLIP_BYTES: usize = 1_200;

/// If a substantial code/HTML artifact is currently OPEN (start marker
/// seen, matching close NOT yet seen) in the decoded thinking text and
/// its open span ≥ [`CF_OPEN_FLIP_BYTES`], return the byte offset of
/// the artifact start (the flip point). Otherwise `None`. Offsets are
/// byte-length-preserving under ASCII lowercasing and land on ASCII
/// marker starts → valid `text` char boundaries.
fn cf_open_artifact_flip_byte(text: &str) -> Option<usize> {
    let lower = text.to_ascii_lowercase();
    // HTML doc takes precedence (unambiguous `</html>` close).
    let html_start = ["<!doctype html", "<html>", "<html ", "<html\n"]
        .iter()
        .filter_map(|m| lower.rfind(m))
        .max();
    if let Some(start) = html_start {
        let closed = lower[start..].contains("</html>");
        if !closed && text.len().saturating_sub(start) >= CF_OPEN_FLIP_BYTES {
            return Some(start);
        }
        if closed {
            return None; // skeleton closed → stay reasoning
        }
    }
    // Language-tagged code fence (bare ``` excluded — plans use it).
    const FENCE_OPENS: &[&str] = &[
        "```html",
        "```javascript",
        "```js\n",
        "```js ",
        "```jsx",
        "```typescript",
        "```ts\n",
        "```ts ",
        "```tsx",
        "```python",
        "```py\n",
        "```css",
        "```json",
        "```rust",
        "```cpp",
        "```c++",
        "```go\n",
        "```java",
        "```bash",
        "```sh\n",
        "```svg",
        "```xml",
    ];
    let fence_open = FENCE_OPENS.iter().filter_map(|m| lower.rfind(m)).max()?;
    // A fence is OPEN iff there is no closing ``` after the opener.
    let after = &lower[fence_open + 3..];
    let closed = after.contains("```");
    if !closed && text.len().saturating_sub(fence_open) >= CF_OPEN_FLIP_BYTES {
        Some(fence_open)
    } else {
        None
    }
}

enum IncompleteHtmlFenceAction {
    Hold,
    Release(String),
    AutoClose(String),
}

/// Process one token. Returns the SSE events to forward to the
/// client (empty `Vec` is valid).
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> SseVec {
    let mut sse_events: SseVec = Vec::new();
    state.all_toks.push(tok);

    // ── Thinking-phase: token-ID based </think> detection ────────────
    if !state.thinking_done {
        if let Some(end_id) = ctx.state.think_end_token_id
            && tok == end_id
        {
            // Component P (streaming): the artifact was written inside
            // `<think>` and has been streaming as `content` since its
            // start was detected. Component G guarantees `</think>` is
            // only reached AFTER the artifact is complete. Emit the
            // residual artifact bytes as content, then END the response
            // so the degenerate post-`</think>` restart is dropped.
            if state.cf_content_mode {
                let mut residual = String::new();
                let mut full = String::new();
                if state.all_toks.len() > 1 {
                    full = ctx
                        .state
                        .tokenizer
                        .decode(&state.all_toks[..state.all_toks.len() - 1])
                        .unwrap_or_default();
                    let stable = full.trim_end_matches('\u{FFFD}');
                    if stable.len() > state.emitted {
                        residual = stable[state.emitted..].to_string();
                        state.emitted = stable.len();
                    }
                }
                let total = state.cf_content_bytes + residual.len();
                // Only END the response (dropping the post-`</think>`
                // restart) when the in-`<think>` artifact is BOTH
                // structurally complete (closed) AND substantial. An
                // incomplete draft ("…Let's write the actual code now.")
                // means the real implementation is post-`</think>` — do
                // NOT truncate it; fall through to the normal content
                // phase (matches the no-carry-forward baseline, which
                // for this model produces the real code post-`</think>`).
                let complete = cf_artifact_is_complete_text(&full);
                if !residual.is_empty() {
                    let chunk =
                        ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, residual);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    sse_events.push(Ok(Event::default().data(json)));
                }
                if complete && total >= CF_MIN_ARTIFACT_BYTES {
                    state.thinking_done = true;
                    state.stop_string_triggered = true;
                    ctx.cancel_flag
                        .store(true, std::sync::atomic::Ordering::Release);
                    tracing::info!(
                        "Component P (streaming): COMPLETE substantial artifact ({total} B) \
                         carried forward from <think>; ending response, dropping post-</think> \
                         restart"
                    );
                    return sse_events;
                }
                // Incomplete draft (or sub-threshold): do NOT truncate.
                // Fall through to the normal post-`</think>` content
                // phase so the model's real implementation streams.
                state.thinking_done = true;
                state.cf_content_mode = false;
                state.emitted = 0;
                state.all_toks.clear();
                if let Some(ref mut det) = state.detector {
                    det.reset();
                }
                tracing::info!(
                    "Component P (streaming): in-<think> artifact NOT complete-and-substantial \
                     (complete={complete}, {total} B); not truncating — continuing into normal \
                     content phase for the real post-</think> implementation"
                );
                return sse_events;
            }
            state.thinking_done = true;
            // Emit only the residual reasoning delta not yet sent
            // by incremental streaming (e.g. trailing bytes held
            // back due to incomplete UTF-8 at prior token boundary).
            // The full reasoning has already been streamed
            // incrementally via reasoning_chunk deltas above —
            // re-emitting the full text here would double it.
            if ctx.enable_thinking && state.all_toks.len() > 1 {
                let full = ctx
                    .state
                    .tokenizer
                    .decode(&state.all_toks[..state.all_toks.len() - 1])
                    .unwrap_or_default();
                let stable = full.trim_end_matches('\u{FFFD}');
                if stable.len() > state.emitted {
                    let residual = &stable[state.emitted..];
                    if !residual.trim().is_empty() {
                        let chunk = ChatCompletionChunk::reasoning_chunk(
                            &ctx.model,
                            &ctx.id,
                            residual.to_string(),
                        );
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        sse_events.push(Ok(Event::default().data(json)));
                    }
                }
            }
            // Reset tool detector to clear any thinking-era tag fragments.
            if let Some(ref mut det) = state.detector {
                det.reset();
            }
            state.emitted = 0; // Reset — next decode will be content-only
            state.all_toks.clear(); // Clear thinking tokens from accumulator
            return sse_events;
        }
        // Still in thinking — accumulate but don't emit as content
        if ctx.enable_thinking {
            // Open thinking: emit as reasoning_content
            let full = ctx
                .state
                .tokenizer
                .decode(&state.all_toks)
                .unwrap_or_default();
            let stable_end = full.trim_end_matches('\u{FFFD}').len();
            // Component P (streaming eager-flip): a SUBSTANTIAL artifact
            // is currently OPEN in `<think>` (≥CF_OPEN_FLIP_BYTES, no
            // close yet) — plan-illustration snippets close long before
            // this and never flip. On the flip token, re-emit the WHOLE
            // artifact-so-far `[start..stable_end]` as `content` so the
            // content channel holds the COMPLETE artifact (the
            // `[start..emitted]` prefix already sent as reasoning is
            // harmless — OpenWebUI shows it collapsed). Subsequent tokens
            // append via the `cf_content_mode` branch.
            if state.cf_active
                && !state.cf_content_mode
                && let Some(start) = cf_open_artifact_flip_byte(&full[..stable_end])
                && stable_end > start
            {
                let delta = full[start..stable_end].to_string();
                state.emitted = stable_end;
                state.cf_content_bytes += delta.len();
                state.cf_content_mode = true;
                let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, delta);
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                sse_events.push(Ok(Event::default().data(json)));
                return sse_events;
            }
            if state.cf_content_mode {
                // Artifact bytes → content verbatim (no reasoning cleanups).
                if stable_end > state.emitted {
                    let delta = full[state.emitted..stable_end].to_string();
                    state.emitted = stable_end;
                    if !delta.is_empty() {
                        state.cf_content_bytes += delta.len();
                        let chunk =
                            ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, delta);
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        sse_events.push(Ok(Event::default().data(json)));
                    }
                }
                return sse_events;
            }
            if stable_end > state.emitted {
                let mut cleaned = full[state.emitted..stable_end].to_string();
                state.emitted = stable_end;
                // Strip format tokens that shouldn't appear in thinking
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                while let Some(start) = cleaned.find("<tool_call>") {
                    if let Some(end) = cleaned[start..].find("</tool_call>") {
                        cleaned = format!(
                            "{}{}",
                            &cleaned[..start],
                            &cleaned[start + end + "</tool_call>".len()..]
                        );
                    } else {
                        cleaned = cleaned[..start].to_string();
                        break;
                    }
                }
                if let Some(start) = cleaned.find("<function=") {
                    cleaned = cleaned[..start].to_string();
                }
                // Strip leaked tool-call closing tags from reasoning
                // (observed pattern: `</parameter></function>` right
                // before a role-word repetition loop — the model
                // emits them as BPE tokens after the real tool call
                // has already been structured by the detector).
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    cleaned = cleaned.replace(tag, "");
                }
                // Collapse role-word repetition loops in reasoning
                // (Qwen3.5/3.6 post-tool-call hallucination). Pair-
                // collapse `userX...userX` → "" until no adjacent
                // pairs remain; then strip surviving line-bounded
                // standalones (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    while cleaned.contains(&pair) {
                        cleaned = cleaned.replace(&pair, "");
                    }
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                // F19: final structured sanitisation pass catches
                // any leak markers the hand-rolled cleanups missed.
                let cleaned = sanitize_content_chunk(
                    &cleaned,
                    &mut state.reasoning_tag_scan_buf,
                    &mut state.reasoning_suppressing_leak,
                    &mut state.reasoning_inside_envelope,
                    &ctx.leak_markers,
                );
                if !cleaned.trim().is_empty() {
                    let chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, cleaned);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    sse_events.push(Ok(Event::default().data(json)));
                }
            }
        }
        return sse_events;
    }

    // ── Content phase: incremental decode via DecodeStream ───────────
    let decoder = state.content_decoder.get_or_insert_with(|| {
        // SAFETY: ctx.state (Arc<AppState>) is owned by the closure
        // and lives for its entire duration. The DecodeStream borrows
        // &Tokenizer from it. We extend the lifetime because the Arc
        // guarantees the tokenizer outlives the closure (and thus
        // the DecodeStream).
        let tokenizer_ref: &'static crate::tokenizer::ChatTokenizer =
            unsafe { &*(&ctx.state.tokenizer as *const crate::tokenizer::ChatTokenizer) };
        tokenizer_ref.streaming_decoder(true)
    });
    let mut delta = match decoder.step(tok) {
        Ok(Some(chunk)) => chunk,
        Ok(None) => return sse_events,
        Err(e) => {
            tracing::warn!("Streaming decoder error: {e:?}");
            return sse_events;
        }
    };

    // Strip residual think tags from content after thinking is done.
    if state.thinking_done {
        for tag in &[
            "</think>",
            "</thinking>",
            "<thinking>",
            "</analysis>",
            "<analysis>",
        ] {
            while let Some(pos) = delta.find(tag) {
                delta = format!("{}{}", &delta[..pos], delta[pos + tag.len()..].trim_start());
            }
        }
        // If model re-opens <think>, suppress content from <think> onward.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
        }
    }

    // Bare role-literal leak (Qwen3.5/3.6) — companion to the
    // scheduler-side <|im_start|> hard-stop.
    {
        let trimmed = delta.trim();
        if delta.len() < 20 && matches!(trimmed, "user" | "assistant" | "tool") {
            tracing::debug!("role-literal strip: dropped bare '{trimmed}' delta");
            delta.clear();
        }
    }

    if delta.is_empty() {
        return sse_events;
    }

    // Multi-token stop sequences via string matching.
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        state.accumulated_content.push_str(&delta);
        for stop_str in &ctx.stop_strings {
            if let Some(pos) = state.accumulated_content.find(stop_str.as_str()) {
                let content_before_stop = &state.accumulated_content[..pos];
                let already_emitted = state.accumulated_content.len() - delta.len();
                if pos > already_emitted {
                    delta = content_before_stop[already_emitted..].to_string();
                } else {
                    delta = String::new();
                }
                state.stop_string_triggered = true;
                break;
            }
        }
        if state.stop_string_triggered && delta.is_empty() {
            return sse_events;
        }
    }

    if state.stop_string_triggered {
        if state.loop_watchdog_triggered {
            return sse_events;
        }
        if !delta.is_empty() {
            let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, delta);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            sse_events.push(Ok(Event::default().data(json)));
        }
        return sse_events;
    }

    // Fork: detector-active vs pure-content path.
    if state.detector.is_some() {
        // Drain the detector outputs into a local Vec so we can drop
        // the &mut borrow on `state.detector` before the helpers below
        // (which take other &mut state fields) run.
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        sse_events.extend(events_out);
                        return sse_events;
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    } else {
        let sanitized = sanitize_content_chunk(
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            sse_events.extend(events_out);
            return sse_events;
        }
        // process_detector_content does NOT pre-sanitize when called
        // from the no-detector branch — but the sanitizer was already
        // run above, so the helper's branch handling matches.
    }

    sse_events
}

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `content_chunk`. Returns `Some(events)` when the watchdog
/// fired (caller must short-circuit), else `None` (caller continues).
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<SseVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized_cow = match stage_incomplete_html_fence_hold(state, sanitized_or_raw) {
        Some(IncompleteHtmlFenceAction::Hold) => return Some(Vec::new()),
        Some(IncompleteHtmlFenceAction::Release(released)) => std::borrow::Cow::Owned(released),
        Some(IncompleteHtmlFenceAction::AutoClose(final_delta)) => {
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            ctx.cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
            tracing::warn!(
                "loop watchdog fired — auto-closed substantial HTML document at split markdown fence"
            );
            if !final_delta.is_empty() {
                let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, final_delta);
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                return Some(vec![Ok(Event::default().data(json))]);
            }
            return Some(Vec::new());
        }
        None => std::borrow::Cow::Borrowed(sanitized_or_raw),
    };
    let sanitized = sanitized_cow.as_ref();
    let sanitized_lower = sanitized.to_ascii_lowercase();
    if !state.structured_content_seen && looks_like_structured_content(&sanitized_lower) {
        state.structured_content_seen = true;
    }

// ── Post-HTML hold: suppress self-critique after `</html>` ──────
    //
    // When we've seen a complete HTML document (<!doctype html>...
    // </html>) with only whitespace/fences after it, we enter
    // `html_complete_seen` mode. On each subsequent chunk:
    //   - whitespace / backticks / newlines → emit normally
    //   - closing ``` fence → emit it, EXIT hold (downstream
    //     SimHash/watchdog will catch degeneration loops)
    //   - structural content (<tag, ```, {, [) → exit hold, process
    //   - any other alphabetic prose → self-critique, fire watchdog
    //
    // The prose detection is language-agnostic: any Unicode alphabetic
    // character starting a line after </html> signals natural-language
    // self-critique, whether English, Chinese, Arabic, etc. Valid
    // structural continuations (new code blocks, HTML tags, JSON) are
    // distinguished from prose by their opening characters.
    if state.html_complete_seen {
        let only_whitespace_or_fences = sanitized
            .trim()
            .chars()
            .all(|c| c.is_ascii_whitespace() || c == '`');
        if only_whitespace_or_fences {
            // Closing markdown fence in the whitespace: the code block
            // has ended.  Exit hold mode so that downstream watchdogs
            // (SimHash, line-level loop check) handle any later prose.
            if sanitized.contains("```") {
                state.html_complete_seen = false;
            }
            // Emit whitespace/fences regardless.
            if !sanitized.is_empty() {
                let chunk =
                    ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, sanitized.to_string());
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                return Some(vec![Ok(Event::default().data(json))]);
            }
            return None;
        }
        // Non-whitespace content after a complete HTML document.
        // Use language-agnostic structural classification: any
        // Unicode alphabetic character starting a line is natural-
        // language prose (self-critique), unless it's a structural
        // continuation (code fence, HTML tag, JSON, list).
        let trimmed = sanitized.trim();
        let first_char = trimmed.chars().next();
        let is_structural = first_char.is_some_and(|c| {
            matches!(c, '<' | '`' | '{' | '[' | '#' | '-' | '*' | '|')
        });
        let starts_with_prose = first_char.is_some_and(|c| c.is_alphabetic());
        if starts_with_prose && !is_structural {
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            ctx.cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
            tracing::warn!(
                "loop watchdog fired — self-critique prose after complete HTML document (post-HTML hold)"
            );
            return Some(Vec::new());
        }
        // Structural continuation (code fence, HTML tag, etc.).
        // Exit hold mode and process normally.
        state.html_complete_seen = false;
        // Fall through to normal processing below.
    }

    // HTML document boundary detection. Dense Qwen3.6 sometimes
    // closes a complete HTML document, then starts self-critique
    // ("Wait, the above..."). When we see a complete document, we
    // classify the trailing content:
    //   - alphabetic prose immediately after </html> → fire watchdog
    //   - only whitespace/fences after </html> → enter hold mode
    //     (see html_complete_seen above)
    let prior_lower = state.loop_scan_buf.to_ascii_lowercase();
    let combined_lower = format!("{prior_lower}{sanitized_lower}");
    if !state.structured_content_seen && looks_like_structured_content(&combined_lower) {
        state.structured_content_seen = true;
    }
    update_html_doc_state(state, sanitized, &sanitized_lower, &combined_lower);
    if combined_lower.contains("<!doctype html")
        && let Some(html_end_start) = combined_lower.rfind("</html>")
    {
        let prior_len = prior_lower.len();
        let html_end = html_end_start + "</html>".len();
        let html_start = combined_lower[..html_end_start]
            .rfind("<!doctype html")
            .or_else(|| combined_lower[..html_end_start].rfind("<html"));
        if let Some(html_start) = html_start
            && html_end.saturating_sub(html_start) >= HTML_RESTART_GUARD_MIN_BYTES
        {
            let emit_start = html_end.saturating_sub(prior_len).min(sanitized.len());
            let mut trailing_fence_len = 0usize;
            for (rel, ch) in sanitized[emit_start..].char_indices() {
                if ch.is_ascii_whitespace() || ch == '`' {
                    trailing_fence_len = rel + ch.len_utf8();
                    continue;
                }
                break;
            }
            let after_html = &combined_lower[html_end..];
            let after_fence =
                after_html.trim_start_matches(|c: char| c.is_ascii_whitespace() || c == '`');
            // Language-agnostic prose detection: any Unicode alphabetic
            // character after </html> signals self-critique, unless
            // it starts with a structural continuation character.
            let after_fence_first = after_fence.chars().next();
            let is_structural_continuation = after_fence_first.is_some_and(|c| {
                matches!(c, '<' | '`' | '{' | '[' | '#' | '-' | '*' | '|')
            });
            let starts_with_prose = after_fence_first.is_some_and(|c| c.is_alphabetic());
            if starts_with_prose && !is_structural_continuation {
                // Self-critique prose detected immediately after </html>.
                // Emit everything up to the HTML end (plus fences/whitespace)
                // and fire the watchdog.
                state.loop_watchdog_triggered = true;
                state.stop_string_triggered = true;
                ctx.cancel_flag
                    .store(true, std::sync::atomic::Ordering::Release);
                let emit_end = emit_start + trailing_fence_len;
                let final_delta = sanitized[..emit_end].to_string();
                if !final_delta.is_empty() {
                    let chunk =
                        ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, final_delta);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    return Some(vec![Ok(Event::default().data(json))]);
                }
                return Some(Vec::new());
            }
            // Only whitespace/fences after </html> — enter post-HTML
            // hold mode instead of immediately firing the watchdog.
            // Subsequent tokens will be held back until we can determine
            // whether they're self-critique or a legitimate continuation.
            if after_fence.is_empty() {
                state.html_complete_seen = true;
            }
        }
    }
    if state.html_doc_started
        && !state.html_doc_closed
        && state.html_doc_bytes >= HTML_AUTOCLOSE_MIN_BYTES
        && let Some(final_delta) = maybe_autoclose_incomplete_html_at_code_fence(
            sanitized,
            state.html_body_closed,
            html_autoclose_needs_script_close(state),
        )
    {
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;
        ctx.cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
        tracing::warn!(
            "loop watchdog fired — auto-closed substantial HTML document before markdown explanation"
        );
        if !final_delta.is_empty() {
            let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, final_delta);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            return Some(vec![Ok(Event::default().data(json))]);
        }
        return Some(Vec::new());
    }

    // Track ``` fence toggles. Odd count in this chunk flips the
    // inside_code_block state; even count (e.g. open+close in one
    // delta) nets to no toggle. Used to suppress the SimHash
    // semantic-loop guard inside fenced code — see the field doc
    // on `StreamState::inside_code_block` for rationale.
    let fence_count = sanitized.matches("```").count();
    if fence_count % 2 == 1 {
        state.inside_code_block = !state.inside_code_block;
    }

    // F4 SimHash guard. Skipped inside code blocks because code
    // chunks (CSS rules, JS function bodies) share enough common
    // tokens to false-positive on similarity, prematurely
    // terminating long code generations. The line-level
    // `check_loop_watchdog` still runs with its code-aware
    // threshold; F27 logit-fingerprint guard at the scheduler
    // also remains active.
    let semantic_trip = if !state.loop_watchdog_triggered && !state.inside_code_block {
        state.simhash_pending.push_str(sanitized);
        let mut dup = false;
        if crate::loop_simhash::ends_at_sentence_boundary(&state.simhash_pending).is_some()
            || state.simhash_pending.len() >= 1024
        {
            dup = state.simhash_guard.check(&state.simhash_pending);
            state.simhash_pending.clear();
        }
        if state.simhash_pending.len() > 4096 {
            let drop_to = state.simhash_pending.len() / 2;
            state.simhash_pending.drain(..drop_to);
        }
        dup
    } else {
        // Inside a code block — drop any partial buffer so we don't
        // cross-compare pre-block prose against post-block prose
        // when we eventually exit the code block.
        if state.inside_code_block {
            state.simhash_pending.clear();
        }
        false
    };

    let token_trip = check_loop_watchdog_with_context(
        sanitized,
        &mut state.loop_scan_buf,
        state.loop_watchdog_triggered,
        state.structured_content_seen || state.inside_code_block,
    );

    if semantic_trip || token_trip {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
        }
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;
        // Signal the scheduler to finish this seq instead of running to
        // `max_tokens`. Pre-fix the watchdog only set
        // `stop_string_triggered`, which suppressed SSE deltas but did
        // not propagate to scheduling — under MTP (greedy verify
        // bypasses DRY/LZ/presence_penalty) the model kept emitting
        // the loop tokens until the cap, wasting decode cycles and
        // surfacing `finish_reason="length"` to the client. See
        // `inference_types::InferenceRequest::Streaming::cancel_flag`.
        ctx.cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);

        let salvaged =
            crate::tool_salvage::salvage(&state.loop_scan_buf, &ctx.tool_defs_for_backfill);
        let mut events: SseVec = Vec::new();
        for (idx, tc) in salvaged.iter().enumerate() {
            tracing::warn!(
                tool = %tc.function.name,
                block_index = idx,
                "watchdog salvage: emitting synthetic tool_call",
            );
            bump_f12_tool_call_count(
                &mut state.tool_calls_emitted_count,
                ctx.max_tool_calls_per_response,
                &mut state.stop_string_triggered,
            );
            let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, idx);
            events.push(Ok(
                Event::default().data(serde_json::to_string(&start).unwrap_or_default())
            ));
            let frag = ChatCompletionChunk::tool_call_args_fragment(
                &ctx.model,
                &ctx.id,
                idx,
                &tc.function.arguments,
            );
            events.push(Ok(
                Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
            ));
        }
        if !salvaged.is_empty() {
            state.salvaged_tool_call = true;
        }
        return Some(events);
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, sanitized.to_string());
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        let events: SseVec = vec![Ok(Event::default().data(json))];
        return Some(events);
    }
    None
}

fn looks_like_structured_content(lower: &str) -> bool {
    lower.contains("<!doctype html")
        || lower.contains("<html")
        || lower.contains("<style")
        || lower.contains("<script")
        || lower.contains("</div>")
        || lower.contains("function ")
        || lower.contains("const ")
        || lower.contains("class ")
}

fn update_html_doc_state(
    state: &mut StreamState,
    sanitized: &str,
    sanitized_lower: &str,
    combined_lower: &str,
) {
    if !state.html_doc_started && looks_like_html_document(combined_lower) {
        state.html_doc_started = true;
    }
    if !state.html_doc_started || state.html_doc_closed {
        return;
    }
    state.html_doc_bytes = state.html_doc_bytes.saturating_add(sanitized.len());
    let script_opens = sanitized_lower.matches("<script").count();
    let script_closes = sanitized_lower.matches("</script>").count();
    if script_opens > 0 {
        state.html_open_script_tags = state.html_open_script_tags.saturating_add(script_opens);
    }
    if script_closes > 0 {
        state.html_open_script_tags = state.html_open_script_tags.saturating_sub(script_closes);
    }
    if combined_lower.contains("</script>") {
        state.html_script_closed = true;
    }
    if combined_lower.contains("</body>") {
        state.html_body_closed = true;
    }
    if combined_lower.contains("</html>") {
        state.html_doc_closed = true;
    }
}

fn looks_like_html_document(lower: &str) -> bool {
    lower.contains("<!doctype html")
        || lower.contains("<html")
        || lower.contains("<head")
        || lower.contains("<body")
        || lower.contains("<style")
        || lower.contains("</style>")
        || lower.contains("<script")
        || lower.contains("</script>")
}

fn html_autoclose_eligible(state: &StreamState) -> bool {
    state.html_doc_started
        && !state.html_doc_closed
        && state.html_doc_bytes >= HTML_AUTOCLOSE_MIN_BYTES
}

fn html_autoclose_needs_script_close(state: &StreamState) -> bool {
    state.html_open_script_tags > 0 || (!state.html_body_closed && state.html_script_closed)
}

fn stage_incomplete_html_fence_hold(
    state: &mut StreamState,
    sanitized: &str,
) -> Option<IncompleteHtmlFenceAction> {
    if !html_autoclose_eligible(state) {
        if state.pending_incomplete_html_fence.is_empty() {
            return None;
        }
        let mut released = std::mem::take(&mut state.pending_incomplete_html_fence);
        released.push_str(sanitized);
        return Some(IncompleteHtmlFenceAction::Release(released));
    }

    if state.pending_incomplete_html_fence.is_empty() && !sanitized.contains('`') {
        return None;
    }

    if state.pending_incomplete_html_fence.is_empty() {
        if let Some(final_delta) = maybe_autoclose_incomplete_html_at_code_fence(
            sanitized,
            state.html_body_closed,
            html_autoclose_needs_script_close(state),
        ) {
            return Some(IncompleteHtmlFenceAction::AutoClose(final_delta));
        }
        if let Some(suffix_start) = trailing_partial_markdown_fence_start(sanitized) {
            state
                .pending_incomplete_html_fence
                .push_str(&sanitized[suffix_start..]);
            let prefix = &sanitized[..suffix_start];
            if prefix.is_empty() {
                return Some(IncompleteHtmlFenceAction::Hold);
            }
            return Some(IncompleteHtmlFenceAction::Release(prefix.to_string()));
        }
        return None;
    }

    state.pending_incomplete_html_fence.push_str(sanitized);
    if let Some(final_delta) = maybe_autoclose_incomplete_html_at_code_fence(
        &state.pending_incomplete_html_fence,
        state.html_body_closed,
        html_autoclose_needs_script_close(state),
    ) {
        state.pending_incomplete_html_fence.clear();
        return Some(IncompleteHtmlFenceAction::AutoClose(final_delta));
    }

    let only_possible_fence = state
        .pending_incomplete_html_fence
        .chars()
        .all(|c| c.is_ascii_whitespace() || c == '`');
    if only_possible_fence {
        return Some(IncompleteHtmlFenceAction::Hold);
    }

    let released = std::mem::take(&mut state.pending_incomplete_html_fence);
    Some(IncompleteHtmlFenceAction::Release(released))
}

fn trailing_partial_markdown_fence_start(sanitized: &str) -> Option<usize> {
    let trimmed_end = sanitized
        .trim_end_matches(|c: char| c.is_ascii_whitespace())
        .len();
    let trimmed = &sanitized[..trimmed_end];
    let mut tick_count = 0usize;
    let mut suffix_start = trimmed.len();
    for (idx, ch) in trimmed.char_indices().rev() {
        if ch != '`' {
            break;
        }
        tick_count += 1;
        suffix_start = idx;
    }
    if matches!(tick_count, 1 | 2) {
        Some(suffix_start)
    } else {
        None
    }
}

fn maybe_autoclose_incomplete_html_at_code_fence(
    sanitized: &str,
    body_closed: bool,
    script_open: bool,
) -> Option<String> {
    let fence_start = sanitized.find("```")?;
    let mut injection = String::new();
    if script_open {
        injection.push_str("\n</script>");
    }
    if !body_closed {
        injection.push_str("\n</body>");
    }
    injection.push_str("\n</html>\n");
    let mut emit_end = fence_start;
    for (rel, ch) in sanitized[fence_start..].char_indices() {
        if ch.is_ascii_whitespace() || ch == '`' {
            emit_end = fence_start + rel + ch.len_utf8();
            continue;
        }
        break;
    }
    Some(format!(
        "{}{}{}",
        &sanitized[..fence_start],
        injection,
        &sanitized[fence_start..emit_end]
    ))
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
fn detector_content_arm(state: &mut StreamState, ctx: &StreamCtx, text: &str) -> Option<SseVec> {
    let sanitized = sanitize_content_chunk(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoclose_incomplete_html_before_markdown_fence() {
        let prior = format!(
            "```html\n<!DOCTYPE html><html><body><script>{}</script>\n",
            "x".repeat(HTML_AUTOCLOSE_MIN_BYTES)
        );
        let sanitized = "```\n\n### Explanation";
        let mut state = StreamState::new(false, false);
        update_html_doc_state(
            &mut state,
            &prior,
            &prior.to_ascii_lowercase(),
            &prior.to_ascii_lowercase(),
        );
        let out = maybe_autoclose_incomplete_html_at_code_fence(
            sanitized,
            state.html_body_closed,
            state.html_open_script_tags > 0,
        )
        .expect("expected auto-close delta");
        assert!(out.contains("</body>\n</html>\n```"));
        assert!(!out.contains("Explanation"));
    }

    #[test]
    fn autoclose_gate_ignores_short_html_drafts() {
        let prior = "<!DOCTYPE html><html><body><script>let x = 1;</script>\n";
        let sanitized = "```\n\n### Explanation";
        let mut state = StreamState::new(false, false);
        update_html_doc_state(
            &mut state,
            prior,
            &prior.to_ascii_lowercase(),
            &prior.to_ascii_lowercase(),
        );
        assert!(state.html_doc_bytes < HTML_AUTOCLOSE_MIN_BYTES);
        assert!(maybe_autoclose_incomplete_html_at_code_fence(sanitized, false, false).is_some());
    }

    #[test]
    fn autoclose_adds_script_close_for_open_main_script() {
        let prior = format!(
            "<!DOCTYPE html><html><head><script src=\"three.js\"></script></head><body><script>{}",
            "x".repeat(HTML_AUTOCLOSE_MIN_BYTES)
        );
        let sanitized = "```\n\n### Explanation";
        let mut state = StreamState::new(false, false);
        update_html_doc_state(
            &mut state,
            &prior,
            &prior.to_ascii_lowercase(),
            &prior.to_ascii_lowercase(),
        );
        assert_eq!(state.html_open_script_tags, 1);
        let out = maybe_autoclose_incomplete_html_at_code_fence(
            sanitized,
            state.html_body_closed,
            state.html_open_script_tags > 0,
        )
        .expect("expected auto-close delta");
        assert!(out.contains("</script>\n</body>\n</html>\n```"));
    }

    #[test]
    fn autoclose_adds_script_close_after_seen_script_tags() {
        let mut state = StreamState::new(false, false);
        state.html_doc_started = true;
        state.html_doc_bytes = HTML_AUTOCLOSE_MIN_BYTES;
        state.html_script_closed = true;

        match stage_incomplete_html_fence_hold(&mut state, "```\n\n### Explanation") {
            Some(IncompleteHtmlFenceAction::AutoClose(out)) => {
                assert!(out.contains("</script>\n</body>\n</html>\n```"));
                assert!(!out.contains("Explanation"));
            }
            _ => panic!("expected heuristic script close to auto-close"),
        }
    }

    #[test]
    fn autoclose_html_state_detects_split_script_close() {
        let mut state = StreamState::new(false, false);
        let first = "<!DOCTYPE html><html><head><script src=\"three.js\"></scr";
        update_html_doc_state(
            &mut state,
            first,
            &first.to_ascii_lowercase(),
            &first.to_ascii_lowercase(),
        );
        assert!(!state.html_script_closed);

        let second = "ipt></head><body><script>";
        update_html_doc_state(
            &mut state,
            second,
            &second.to_ascii_lowercase(),
            &format!(
                "{}{}",
                first.to_ascii_lowercase(),
                second.to_ascii_lowercase()
            ),
        );
        assert!(state.html_script_closed);
    }

    #[test]
    fn autoclose_stages_split_markdown_fence() {
        let mut state = StreamState::new(false, false);
        state.html_doc_started = true;
        state.html_doc_bytes = HTML_AUTOCLOSE_MIN_BYTES;

        assert!(matches!(
            stage_incomplete_html_fence_hold(&mut state, "`"),
            Some(IncompleteHtmlFenceAction::Hold)
        ));
        assert_eq!(state.pending_incomplete_html_fence, "`");
        assert!(matches!(
            stage_incomplete_html_fence_hold(&mut state, "`"),
            Some(IncompleteHtmlFenceAction::Hold)
        ));

        match stage_incomplete_html_fence_hold(&mut state, "`\n\n### Explanation") {
            Some(IncompleteHtmlFenceAction::AutoClose(out)) => {
                assert!(out.contains("</body>\n</html>\n```"));
                assert!(!out.contains("Explanation"));
            }
            _ => panic!("expected split fence to auto-close"),
        }
        assert!(state.pending_incomplete_html_fence.is_empty());
    }

    #[test]
    fn autoclose_holds_trailing_partial_fence_after_code_prefix() {
        let mut state = StreamState::new(false, false);
        state.html_doc_started = true;
        state.html_doc_bytes = HTML_AUTOCLOSE_MIN_BYTES;
        state.html_open_script_tags = 1;

        match stage_incomplete_html_fence_hold(&mut state, "buildings=[];\n``") {
            Some(IncompleteHtmlFenceAction::Release(out)) => {
                assert_eq!(out, "buildings=[];\n");
            }
            _ => panic!("expected code prefix to release while holding partial fence"),
        }
        assert_eq!(state.pending_incomplete_html_fence, "``");

        match stage_incomplete_html_fence_hold(&mut state, "`\n\n### Explanation") {
            Some(IncompleteHtmlFenceAction::AutoClose(out)) => {
                assert!(out.contains("</script>\n</body>\n</html>\n```"));
                assert!(!out.contains("Explanation"));
            }
            _ => panic!("expected trailing split fence to auto-close"),
        }
        assert!(state.pending_incomplete_html_fence.is_empty());
    }

    #[test]
    fn autoclose_hold_releases_template_backtick() {
        let mut state = StreamState::new(false, false);
        state.html_doc_started = true;
        state.html_doc_bytes = HTML_AUTOCLOSE_MIN_BYTES;

        assert!(matches!(
            stage_incomplete_html_fence_hold(&mut state, "`"),
            Some(IncompleteHtmlFenceAction::Hold)
        ));
        match stage_incomplete_html_fence_hold(&mut state, "altitude`") {
            Some(IncompleteHtmlFenceAction::Release(out)) => {
                assert_eq!(out, "`altitude`");
            }
            _ => panic!("expected non-fence backtick to release"),
        }
        assert!(state.pending_incomplete_html_fence.is_empty());
    }
}
