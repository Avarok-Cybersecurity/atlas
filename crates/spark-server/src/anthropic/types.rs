// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

// ── Anthropic request types ──

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<SystemContent>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

/// System content: either a plain string or an array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl SystemContent {
    pub(super) fn to_text(&self) -> String {
        match self {
            SystemContent::Text(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| {
                    if b.block_type == "text" {
                        b.text.clone()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Message content: string or array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
        /// Anthropic tool_result error flag. F6 (2026-04-26): when
        /// `true`, the tool execution failed (e.g. `Exit code 127`).
        /// Without surfacing this to the model, the model treats the
        /// error text as flavour rather than a structural signal —
        /// observed pattern: cargo command not found → model emits
        /// "I see the basic echo functionality is working" (pure
        /// hallucination). The conversion to OpenAI format prepends
        /// `[tool error]\n` to the content text when this flag is
        /// set, so any chat-tuned model (which has no `is_error`
        /// concept in its training data — there's no OpenAI tool
        /// schema field for this) sees an explicit, ASCII,
        /// token-aligned error marker.
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

/// Tool result content: string or nested blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl ToolResultContent {
    pub(super) fn to_text(&self) -> String {
        match self {
            ToolResultContent::Text(s) => s.clone(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    /// Only present when type = "tool".
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
}

// ── Anthropic response types ──

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}
