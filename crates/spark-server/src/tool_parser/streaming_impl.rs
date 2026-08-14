// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

impl StreamingToolDetector {
    pub fn new() -> Self {
        Self::new_with_tools(Vec::new())
    }

    /// Build a detector with the request's tool schemas, enabling per-parameter
    /// type coercion during live argument streaming. The `ATLAS_BUFFER_TOOL_ARGS`
    /// env var (set to `1`/`true`) restores the legacy buffer-until-close path.
    pub fn new_with_tools(tools: Vec<ToolDefinition>) -> Self {
        let buffer_args = matches!(
            std::env::var("ATLAS_BUFFER_TOOL_ARGS").as_deref(),
            Ok("1") | Ok("true")
        );
        Self {
            buffer: String::new(),
            inside_tag: false,
            inside_dsml: false,
            promote_bare_names: false,
            call_counter: 0,
            emitted_tool_calls: false,
            current_tc_name: None,
            current_tc_id: None,
            current_tc_emitted: 0,
            tools,
            buffer_args,
            args_open: false,
            emitted_keys: Vec::new(),
            incremental_emitted: false,
        }
    }

    /// Opt this detector into Poolside v1's bare-name zero-argument calls.
    /// Wired from `ToolCallParser::promotes_bare_call_names` at stream setup;
    /// request-scoped, so `reset()` deliberately leaves it alone.
    pub fn set_promote_bare_names(&mut self, on: bool) {
        self.promote_bare_names = on;
    }

    /// Reset the detector state. Called when thinking→content transition occurs
    /// to prevent thinking-era tag fragments from corrupting tool detection.
    /// Preserves `tools` / `buffer_args` (request-scoped config, not per-call).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.inside_tag = false;
        self.inside_dsml = false;
        self.reset_call_state();
    }

    /// Clear the per-tool-call incremental-streaming bookkeeping (called after
    /// each call closes and on `reset`). Does NOT touch `tools`/`buffer_args`.
    pub(super) fn reset_call_state(&mut self) {
        self.current_tc_name = None;
        self.current_tc_id = None;
        self.current_tc_emitted = 0;
        self.args_open = false;
        self.emitted_keys.clear();
        self.incremental_emitted = false;
    }

    /// Feed a text delta. Returns events to emit (content or tool calls).
    /// Emits incremental ToolCallStart/ToolCallDelta/ToolCallEnd events
    /// so clients see tool call arguments stream in real-time.
    pub fn process(&mut self, new_text: &str) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        self.buffer.push_str(new_text);
        loop {
            match self.process_dsml(&mut outputs) {
                DsmlStreamAction::Continue => continue,
                DsmlStreamAction::Wait => break,
                DsmlStreamAction::NotDsml => {}
            }
            if self.inside_tag {
                // Check for closing tag. Recognised forms:
                //   - `</tool_call>` (hermes / qwen3-coder, 12 chars)
                //   - `<tool_call|>` (gemma-4, 12 chars)
                //   - `</minimax:tool_call>` (MiniMax canonical, 20 chars)
                //   - `</minimax:_call>` (MiniMax BPE-broken — F73 / fix42)
                let close_pos = self
                    .buffer
                    .find("</tool_call>")
                    .map(|p| (p, 12usize))
                    .or_else(|| self.buffer.find("<tool_call|>").map(|p| (p, 12usize)))
                    .or_else(|| {
                        self.buffer
                            .find("</minimax:tool_call>")
                            .map(|p| (p, "</minimax:tool_call>".len()))
                    })
                    .or_else(|| {
                        self.buffer
                            .find("</minimax:_call>")
                            .map(|p| (p, "</minimax:_call>".len()))
                    });
                if let Some((end, close_len)) = close_pos {
                    let idx = self.call_counter as usize;

                    if self.current_tc_name.is_some() {
                        // Name was already emitted via ToolCallStart.
                        // Live path: if we have streamed fragments incrementally
                        // (`!buffer_args` && `incremental_emitted`), emit only the
                        // residual (remaining complete params + backfill + closing
                        // `}` for XML, or the JSON tail). Compute it BEFORE the
                        // buffer is truncated, since `stream_ready_fragments`
                        // reads `self.buffer[..end]`.
                        if !self.buffer_args && self.incremental_emitted {
                            let frags = self.stream_ready_fragments(end, true);
                            outputs.extend(frags);
                            outputs.push(DetectorOutput::ToolCallEnd { idx });
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                            self.buffer = self.buffer[end + close_len..].to_string();
                            self.inside_tag = false;
                            self.reset_call_state();
                            continue;
                        }
                        let inner = self.buffer[..end].to_string();
                        self.buffer = self.buffer[end + close_len..].to_string();
                        self.inside_tag = false;
                        // Buffered mode OR never-streamed fallback: emit the full
                        // canonical args once (unchanged legacy behaviour).
                        // Parse the complete inner content to extract JSON arguments.
                        if let Some(tc) =
                            parse_complete_call(&inner, self.call_counter, self.promote_bare_names)
                        {
                            // Always emit when the parser produced a named call,
                            // even if arguments are `{}`. Argument-less tools
                            // (e.g. get_current_time) are legitimate. The bare-
                            // narration case is caught by the else branch below
                            // where parse_one_call returns None or current_tc_name
                            // is unset.
                            outputs.push(DetectorOutput::ToolCallDelta {
                                args: tc.function.arguments,
                                idx,
                            });
                            outputs.push(DetectorOutput::ToolCallEnd { idx });
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                        } else {
                            tracing::warn!("Failed to parse tool call body, dropping");
                        }
                        // Reset incremental state for next tool call.
                        self.reset_call_state();
                        continue;
                    } else {
                        let inner = self.buffer[..end].to_string();
                        self.buffer = self.buffer[end + close_len..].to_string();
                        self.inside_tag = false;
                        // Name was never extracted — fall back to complete
                        // ToolCall(s). F75 (2026-04-29): MiniMax envelopes
                        // can contain MULTIPLE `<invoke>` blocks (the
                        // canonical multi-tool form). `parse_one_call`
                        // returns only the first; the rest get dropped
                        // and the response shows `has_tool_calls=false`
                        // because higher layers see no completed call.
                        // Iterate all `<invoke>` blocks for MiniMax
                        // shape; fall back to single-call parse for the
                        // other formats.
                        let trimmed = inner.trim();
                        if trimmed.contains("<invoke name=") {
                            for tc in parse_minimax_xml_calls_all(trimmed) {
                                let call_idx = self.call_counter as usize;
                                self.call_counter += 1;
                                self.emitted_tool_calls = true;
                                outputs.push(DetectorOutput::ToolCall(tc, call_idx));
                            }
                        } else if let Some(tc) =
                            parse_complete_call(trimmed, self.call_counter, self.promote_bare_names)
                        {
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                            outputs.push(DetectorOutput::ToolCall(tc, idx));
                        }
                    }
                    // Reset incremental state for next tool call
                    self.reset_call_state();
                    continue;
                }

                // No closing tag yet — try to extract function name for early header emission.
                // Arguments are NOT streamed incrementally because Qwen3-Coder XML format
                // (`<parameter=key>value</parameter>`) must be converted to JSON before
                // emission. The name header is emitted immediately so clients get instant
                // feedback that a tool call started.
                if self.current_tc_name.is_none()
                    && let Some(name) = extract_streaming_name(&self.buffer)
                {
                    let id = next_tool_call_id();
                    let idx = self.call_counter as usize;
                    outputs.push(DetectorOutput::ToolCallStart {
                        id: id.clone(),
                        name: name.clone(),
                        idx,
                    });
                    self.current_tc_name = Some(name);
                    self.current_tc_id = Some(id);
                }
                // Live-streaming (default): emit any newly-complete argument
                // fragments seen so far so the client gets `function.arguments`
                // incrementally instead of buffered until `</tool_call>`. The
                // legacy buffer-until-close path runs when `buffer_args` is set.
                if !self.buffer_args && self.current_tc_name.is_some() {
                    let frags = self.stream_ready_fragments(self.buffer.len(), false);
                    outputs.extend(frags);
                }
                break; // Wait for more tokens (closing tag not yet seen)
            } else if let Some(mistral_start) = self.buffer.find(MISTRAL_TOOL_CALLS_TAG) {
                // Mistral native: [TOOL_CALLS]name[ARGS]{json}
                // No wrapping tag — emit content before the tag, then try to
                // parse a complete segment when both [ARGS] and a balanced
                // JSON object are present. If not yet complete, break and
                // wait for more tokens.
                if mistral_start > 0 {
                    let before = self.buffer[..mistral_start].to_string();
                    outputs.push(DetectorOutput::Content(before));
                    self.buffer = self.buffer[mistral_start..].to_string();
                }
                // Must have [ARGS] before we can extract a name.
                let after_tag = &self.buffer[MISTRAL_TOOL_CALLS_TAG.len()..];
                let args_rel = match after_tag.find(MISTRAL_ARGS_TAG) {
                    Some(p) => p,
                    None => break, // wait for more tokens
                };
                let name = after_tag[..args_rel].trim().to_string();
                let json_abs_start =
                    MISTRAL_TOOL_CALLS_TAG.len() + args_rel + MISTRAL_ARGS_TAG.len();
                // Skip leading whitespace before the JSON object.
                let mut json_rel = json_abs_start;
                while json_rel < self.buffer.len()
                    && self.buffer.as_bytes()[json_rel].is_ascii_whitespace()
                {
                    json_rel += 1;
                }
                if json_rel >= self.buffer.len() || self.buffer.as_bytes()[json_rel] != b'{' {
                    break; // wait for {
                }
                // Look for a balanced JSON object; if not complete, break.
                let json_tail = &self.buffer[json_rel..];
                let Some(json_end_rel) = find_balanced_json_end(json_tail) else {
                    break; // wait for more tokens to close the JSON
                };
                // Emit ToolCallStart now (name is known).
                let id = next_tool_call_id();
                let idx = self.call_counter as usize;
                if !name.is_empty() {
                    outputs.push(DetectorOutput::ToolCallStart {
                        id: id.clone(),
                        name: name.clone(),
                        idx,
                    });
                }
                // Extract and canonicalize the JSON arguments, then emit delta + end.
                let raw_args = &json_tail[..json_end_rel];
                let canonical = serde_json::from_str::<serde_json::Value>(raw_args)
                    .ok()
                    .and_then(|v| serde_json::to_string(&v).ok())
                    .unwrap_or_else(|| "{}".to_string());
                let args_empty = canonical == "{}" || canonical.is_empty();
                if !name.is_empty() && !args_empty {
                    outputs.push(DetectorOutput::ToolCallDelta {
                        args: canonical,
                        idx,
                    });
                    outputs.push(DetectorOutput::ToolCallEnd { idx });
                    self.call_counter += 1;
                    self.emitted_tool_calls = true;
                } else if !name.is_empty() {
                    tracing::warn!("Dropping empty Mistral tool call '{name}' — args were empty");
                }
                // Advance the buffer past the parsed JSON.
                let consumed = json_rel + json_end_rel;
                self.buffer = self.buffer[consumed..].to_string();
                continue;
            } else if let Some((start, tag_len)) = self
                .buffer
                .find("<tool_call>")
                .map(|p| (p, 11usize))
                .or_else(|| self.buffer.find("<|tool_call>").map(|p| (p, 12usize)))
                .or_else(|| {
                    self.buffer
                        .find("<minimax:tool_call>")
                        .map(|p| (p, "<minimax:tool_call>".len()))
                })
                .or_else(|| {
                    self.buffer
                        .find("<minimax:_call>")
                        .map(|p| (p, "<minimax:_call>".len()))
                })
            {
                // Recognised opener forms:
                //   - `<tool_call>` (hermes / qwen3-coder, 11 chars)
                //   - `<|tool_call>` (gemma-4, 12 chars)
                //   - `<minimax:tool_call>` (MiniMax canonical, 19 chars)
                //   - `<minimax:_call>` (MiniMax BPE-broken — F73 / fix42)
                let before = self.buffer[..start].to_string();
                self.buffer = self.buffer[start + tag_len..].to_string();
                self.inside_tag = true;
                if !before.is_empty() {
                    outputs.push(DetectorOutput::Content(before));
                }
                continue;
            } else if self.buffer.contains("<function") {
                // Bare `<function>` / `<function=` without a `<tool_call>` wrapper.
                // Body lives in `streaming_emit.rs` (≤500 LoC cap).
                if self.process_bare_function(&mut outputs) {
                    continue;
                }
                break; // Wait for more tokens (closing `</function>` not yet seen)
            } else {
                if self.buffer.trim().is_empty() {
                    break;
                }
                let safe = self.safe_emit_len();
                if safe > 0 {
                    let content = self.buffer[..safe].to_string();
                    let remainder = self.buffer[safe..].to_string();
                    let dsml_leading_whitespace = !remainder.is_empty()
                        && content.trim().is_empty()
                        && DSML_OPEN.starts_with(&remainder);
                    self.buffer = remainder;
                    if !content.is_empty() && !dsml_leading_whitespace {
                        outputs.push(DetectorOutput::Content(content));
                    }
                }
                break;
            }
        }
        outputs
    }
}
