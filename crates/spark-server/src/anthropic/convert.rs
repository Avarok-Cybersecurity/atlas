// SPDX-License-Identifier: AGPL-3.0-only

use crate::tool_parser;

use super::types::{AnthropicContent, ContentBlock};

/// Flatten Anthropic message content blocks into a single text string.
/// For assistant messages with tool_use blocks, also extracts tool calls
/// so they can be formatted by the tool parser for multi-turn.
pub(super) fn flatten_content(
    content: &AnthropicContent,
) -> (String, Vec<tool_parser::IncomingToolCall>) {
    match content {
        AnthropicContent::Text(s) => (s.clone(), Vec::new()),
        AnthropicContent::Blocks(blocks) => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => text_parts.push(text.clone()),
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(tool_parser::IncomingToolCall {
                            id: Some(id.clone()),
                            function: tool_parser::IncomingFunction {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        });
                    }
                    ContentBlock::ToolResult { content, .. } => {
                        if let Some(c) = content {
                            text_parts.push(c.to_text());
                        }
                    }
                    ContentBlock::Thinking { .. } | ContentBlock::Unknown => {}
                }
            }
            (text_parts.join(""), tool_calls)
        }
    }
}
