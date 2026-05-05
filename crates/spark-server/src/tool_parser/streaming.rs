// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Buffers streaming text and detects `<tool_call>` tags in real-time.
/// Emits incremental tool call events as tokens arrive:
/// - `ToolCallStart` when the function name is extracted
/// - `ToolCallDelta` for each argument fragment
/// - `ToolCallEnd` when `</tool_call>` is found
pub struct StreamingToolDetector {
    pub(super) buffer: String,
    pub(super) inside_tag: bool,
    pub(super) call_counter: u32,
    /// Track if any tool calls were emitted during process() to prevent
    /// flush() from re-emitting them (causes duplicate arguments in stream).
    pub(super) emitted_tool_calls: bool,
    /// For incremental streaming: name already emitted for current tool call.
    pub(super) current_tc_name: Option<String>,
    /// For incremental streaming: ID of the current in-progress tool call.
    pub(super) current_tc_id: Option<String>,
    /// Bytes already emitted as ToolCallDelta for current tool call.
    pub(super) current_tc_emitted: usize,
}

pub enum DetectorOutput {
    /// Plain text content (not a tool call).
    Content(String),
    /// Complete tool call (used by flush/blocking path).
    ToolCall(ToolCall, usize),
    /// Incremental: tool call header (name + id). Emitted once when name is known.
    ToolCallStart {
        id: String,
        name: String,
        idx: usize,
    },
    /// Incremental: argument fragment. Emitted per-token as arguments arrive.
    ToolCallDelta { args: String, idx: usize },
    /// Incremental: tool call complete. `</tool_call>` seen.
    ToolCallEnd { idx: usize },
}

/// Extract function name from partial tool call buffer for incremental streaming.
/// Handles Hermes JSON, Qwen3-Coder XML, Gemma-4, and Mistral native formats.
pub(super) fn extract_streaming_name(buffer: &str) -> Option<String> {
    // Mistral native: [TOOL_CALLS]NAME[ARGS]
    if let Some(start) = buffer.find(MISTRAL_TOOL_CALLS_TAG) {
        let after = &buffer[start + MISTRAL_TOOL_CALLS_TAG.len()..];
        if let Some(end) = after.find(MISTRAL_ARGS_TAG) {
            let name = after[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    // Gemma-4 native: call:NAME{
    if let Some(start) = buffer.find("call:") {
        let after = &buffer[start + 5..]; // len("call:") = 5
        if let Some(end) = after.find('{') {
            let name = after[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    // Qwen3-Coder: <function=NAME>
    if let Some(start) = buffer.find("<function=") {
        let after = &buffer[start + "<function=".len()..];
        if let Some(end) = after.find(['>', '\n', '<']) {
            let mut name = after[..end].trim().to_string();
            // Sanitize: model may generate "Bash=bashash" or "Bash=Bash" at long context.
            if let Some(eq_pos) = name.find('=') {
                name = name[..eq_pos].trim().to_string();
            }
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    // Hermes JSON: "name":"X"
    if let Some(start) = buffer.find("\"name\"") {
        let after = &buffer[start + "\"name\"".len()..];
        // Skip optional whitespace and colon
        let after = after
            .trim_start()
            .strip_prefix(':')
            .unwrap_or(after)
            .trim_start();
        if let Some(after) = after.strip_prefix('"')
            && let Some(end) = after.find('"')
        {
            let name = &after[..end];
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Find where arguments start in the buffer, after the name header.
/// Returns byte offset into buffer where argument content begins.
fn find_args_start(buffer: &str) -> usize {
    // Qwen3-Coder: after <function=NAME>\n
    if let Some(pos) = buffer.find("<function=")
        && let Some(gt) = buffer[pos..].find('>')
    {
        let after_gt = pos + gt + 1;
        // Skip leading newline after >
        if after_gt < buffer.len() && buffer.as_bytes().get(after_gt) == Some(&b'\n') {
            return after_gt + 1;
        }
        return after_gt;
    }
    // Hermes JSON: after "arguments":
    if let Some(pos) = buffer.find("\"arguments\"") {
        let after = &buffer[pos + "\"arguments\"".len()..];
        let after = after.trim_start();
        if let Some(rest) = after.strip_prefix(':') {
            return buffer.len() - rest.len();
        }
    }
    buffer.len()
}

impl Default for StreamingToolDetector {
    fn default() -> Self {
        Self::new()
    }
}
