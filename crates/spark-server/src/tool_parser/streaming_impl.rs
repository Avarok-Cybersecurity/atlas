// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

impl StreamingToolDetector {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            inside_tag: false,
            call_counter: 0,
            emitted_tool_calls: false,
            current_tc_name: None,
            current_tc_id: None,
            current_tc_emitted: 0,
        }
    }

    /// Reset the detector state. Called when thinking→content transition occurs
    /// to prevent thinking-era tag fragments from corrupting tool detection.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.inside_tag = false;
        self.current_tc_name = None;
        self.current_tc_id = None;
        self.current_tc_emitted = 0;
    }

    /// Feed a text delta. Returns events to emit (content or tool calls).
    /// Emits incremental ToolCallStart/ToolCallDelta/ToolCallEnd events
    /// so clients see tool call arguments stream in real-time.
    pub fn process(&mut self, new_text: &str) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        self.buffer.push_str(new_text);
        loop {
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
                    let inner = self.buffer[..end].to_string();
                    self.buffer = self.buffer[end + close_len..].to_string();
                    self.inside_tag = false;

                    let idx = self.call_counter as usize;

                    if self.current_tc_name.is_some() {
                        // Name was already emitted via ToolCallStart.
                        // Parse the complete inner content to extract JSON arguments.
                        if let Some(tc) = parse_one_call(inner.trim(), self.call_counter) {
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
                    } else {
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
                        } else if let Some(tc) = parse_one_call(trimmed, self.call_counter) {
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                            outputs.push(DetectorOutput::ToolCall(tc, idx));
                        }
                    }
                    // Reset incremental state for next tool call
                    self.current_tc_name = None;
                    self.current_tc_id = None;
                    self.current_tc_emitted = 0;
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
                break; // Wait for more tokens (args buffered until </tool_call>)
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
            } else if let Some(func_pos) = self.buffer.find("<function") {
                // Bare <function> or <function= without <tool_call> wrapper.
                // Emit content before it, then try to parse the function block.
                if func_pos > 0 {
                    let before = self.buffer[..func_pos].to_string();
                    self.buffer = self.buffer[func_pos..].to_string();
                    outputs.push(DetectorOutput::Content(before));
                }
                // Check if we have a complete bare function block
                if let Some(end) = bare_function_end(&self.buffer) {
                    let block = self.buffer[..end].to_string();
                    self.buffer = self.buffer[end..].to_string();
                    let (_, calls) = parse_bare_function_calls(&block);
                    for tc in calls {
                        let idx = self.call_counter as usize;
                        self.call_counter += 1;
                        self.emitted_tool_calls = true;
                        outputs.push(DetectorOutput::ToolCall(tc, idx));
                    }
                    continue;
                }
                break; // Keep buffering until function block is complete
            } else {
                let safe = self.safe_emit_len();
                if safe > 0 {
                    let content = self.buffer[..safe].to_string();
                    self.buffer = self.buffer[safe..].to_string();
                    if !content.is_empty() {
                        outputs.push(DetectorOutput::Content(content));
                    }
                }
                break;
            }
        }
        outputs
    }

    /// Flush remaining buffer (call at stream end).
    /// Also attempts bare `<function>` detection as a last resort.
    pub fn flush(&mut self) -> Vec<DetectorOutput> {
        if self.buffer.is_empty() {
            return vec![];
        }
        let text = std::mem::take(&mut self.buffer);
        let was_inside_tag = self.inside_tag;
        self.inside_tag = false;

        // When inside_tag was true, we have the raw content between
        // <tool_call> and end-of-stream (</tool_call> was a stop token
        // and wasn't streamed). Try to parse the tool call directly.
        //
        // Issue #33: if the incremental path already emitted ToolCallStart
        // for this call (current_tc_name is Some), the downstream consumer
        // has already sent a `tool_calls[0].id=A,name=…,args=""` chunk to
        // the client. Emitting a fresh `ToolCall(tc, idx)` here makes
        // `handle_complete_tool_call` send ANOTHER `tool_call_start_chunk`
        // with a brand-new id (parse_one_call generates one), so the client
        // sees two distinct `id`s for the same `index:0` and either drops
        // one or dispatches the wrong one with empty args. Mirror the
        // in-stream close path: emit ToolCallDelta + ToolCallEnd against
        // the already-streamed header, not a full ToolCall.
        if was_inside_tag && let Some(tc) = parse_one_call(text.trim(), self.call_counter) {
            let idx = self.call_counter as usize;
            self.call_counter += 1;
            self.emitted_tool_calls = true;
            if self.current_tc_name.is_some() {
                self.current_tc_name = None;
                self.current_tc_id = None;
                self.current_tc_emitted = 0;
                return vec![
                    DetectorOutput::ToolCallDelta {
                        args: tc.function.arguments,
                        idx,
                    },
                    DetectorOutput::ToolCallEnd { idx },
                ];
            }
            return vec![DetectorOutput::ToolCall(tc, idx)];
        }

        let text = if was_inside_tag {
            format!("<tool_call>{text}")
        } else {
            text
        };

        // Try bare function detection on the remaining buffer.
        // Only if no tool calls were already found (avoid duplicate extraction).
        if !self.has_tool_calls() && !self.emitted_tool_calls {
            let (content, calls) = parse_bare_function_calls(&text);
            if !calls.is_empty() {
                let mut out = Vec::new();
                if let Some(c) = content {
                    out.push(DetectorOutput::Content(c));
                }
                for tc in calls {
                    let idx = self.call_counter as usize;
                    self.call_counter += 1;
                    out.push(DetectorOutput::ToolCall(tc, idx));
                }
                return out;
            }
        }

        // Fallback: JSON tool calls without any XML wrapper.
        // Nemotron-H Super 120B sometimes outputs Hermes-style JSON or JSON
        // in code blocks instead of <tool_call> XML. Catch those here.
        if !self.has_tool_calls() && !self.emitted_tool_calls {
            let json_calls = parse_json_fallback_calls(&text);
            if !json_calls.is_empty() {
                let mut out = Vec::new();
                // Strip matched JSON from content
                let mut clean = text.clone();
                for pattern in extract_json_code_blocks(&text) {
                    clean = clean.replace(&pattern, "");
                }
                let clean = clean.trim().to_string();
                if !clean.is_empty() {
                    out.push(DetectorOutput::Content(clean));
                }
                for tc in json_calls {
                    let idx = self.call_counter as usize;
                    self.call_counter += 1;
                    out.push(DetectorOutput::ToolCall(tc, idx));
                }
                return out;
            }
        }

        vec![DetectorOutput::Content(text)]
    }

    pub fn has_tool_calls(&self) -> bool {
        self.call_counter > 0
    }

    /// Returns safe byte length to emit without splitting a partial tag.
    /// Holds back content that could be the start of `<tool_call>` or bare `<function`.
    fn safe_emit_len(&self) -> usize {
        let buf = self.buffer.as_bytes();
        // Check all tag prefixes — don't emit partial matches for any of
        // them. F75 (2026-04-29): include the MiniMax envelope opens
        // (canonical + BPE-broken). Without these in the list, a
        // `<minimax:` trailing prefix (split across stream chunks)
        // gets emitted as content and the detector never sees the
        // complete open tag — exactly the failure shape captured in
        // opencode-session.md `ses_224cc79f4ffeUtq7NFV9YMTVMH` where
        // `has_tool_calls=false` and the full envelope leaked into
        // `content`. Close tags don't need preserving here — close
        // matching only runs when `inside_tag=true`, where the buffer
        // accumulates the entire inner block until close lands.
        for tag in [
            b"<tool_call>" as &[u8],
            b"<|tool_call>",
            b"<minimax:tool_call>",
            b"<minimax:_call>",
            b"<function",
            b"call:",
            MISTRAL_TOOL_CALLS_TAG.as_bytes(),
        ] {
            for i in (buf.len().saturating_sub(tag.len() - 1))..buf.len() {
                if tag.starts_with(&buf[i..]) {
                    return i;
                }
            }
        }
        buf.len()
    }
}
