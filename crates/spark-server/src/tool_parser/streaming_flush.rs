// SPDX-License-Identifier: AGPL-3.0-only

//! End-of-stream detector behaviour, split from `streaming_impl.rs` at the
//! 500-line cap. `process` handles the incremental path; everything here runs
//! when the stream is finished or is asked what it already saw, so the split
//! follows an existing seam rather than cutting one.

use super::*;

impl StreamingToolDetector {
    /// Flush remaining buffer (call at stream end).
    /// Also attempts bare `<function>` detection as a last resort.
    pub fn flush(&mut self) -> Vec<DetectorOutput> {
        if let Some(outputs) = self.flush_dsml() {
            return outputs;
        }
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
        // #192 containment (parity with `parse_tool_calls`): this buffer has NO
        // `</tool_call>` close, so an unterminated trailing `<parameter=…>`
        // value is unbounded — cut at the last complete `</parameter>` (else
        // drop the param section) before salvaging, so drifted tail garbage
        // is never swallowed into an argument string.
        if was_inside_tag
            && !text.contains("<arg_key>")
            && let Some(tc) = parse_one_call(
                contain_unterminated_call_tail(text.trim()),
                self.call_counter,
            )
        {
            let idx = self.call_counter as usize;
            if self.current_tc_name.is_some() {
                // Live path: if we already streamed fragments, emit only the
                // residual (remaining complete params + backfill + closing `}`,
                // or the JSON tail) instead of the full args. `flush()` already
                // took the buffer, so restore it for `stream_ready_fragments` to
                // scan, then clear it again. IMPORTANT: call_counter is bumped
                // AFTER `stream_ready_fragments` — it reads `self.call_counter`
                // for the fragment `idx`, so bumping first would emit the
                // closing `}` under the wrong index (handler drops it).
                if !self.buffer_args && self.incremental_emitted {
                    self.buffer = text;
                    let limit = self.buffer.len();
                    let mut out = self.stream_ready_fragments(limit, true);
                    out.push(DetectorOutput::ToolCallEnd { idx });
                    self.call_counter += 1;
                    self.emitted_tool_calls = true;
                    self.buffer.clear();
                    self.reset_call_state();
                    return out;
                }
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                self.reset_call_state();
                return vec![
                    DetectorOutput::ToolCallDelta {
                        args: tc.function.arguments,
                        idx,
                    },
                    DetectorOutput::ToolCallEnd { idx },
                ];
            }
            self.call_counter += 1;
            self.emitted_tool_calls = true;
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

    /// True while the detector is between a `<tool_call>` opener and its
    /// matching close — i.e. accumulating a tool-call body. Callers use this
    /// to suppress content-level scrubbing (e.g. the bare role-literal strip
    /// in `handle_token`) that would otherwise eat a legitimate name/argument
    /// fragment. A standalone `tool` BPE token inside the body is the leading
    /// fragment of a `tool_*`-prefixed NAME (`tool_search`, `tool_call`,
    /// `tool_describe`) being reassembled across token boundaries — dropping
    /// it truncates the streamed name by exactly `len("tool")` == 4 chars.
    pub fn inside_tool_call(&self) -> bool {
        self.inside_tag || self.inside_dsml || self.has_partial_tool_opener()
    }

    /// Returns safe byte length to emit without splitting a partial tag.
    /// Holds back content that could be the start of `<tool_call>` or bare `<function`.
    pub(super) fn safe_emit_len(&self) -> usize {
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
            DSML_OPEN.as_bytes(),
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
