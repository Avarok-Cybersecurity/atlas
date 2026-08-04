// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// Model reasoning trace (from <think>...</think> tags).
    /// Only populated when enable_thinking=true. Both Cline and Roo Code
    /// check for this field. DeepSeek-originated, vLLM/LiteLLM standard.
    /// This is the single canonical reasoning field on the response wire —
    /// Atlas deliberately does NOT also emit a `reasoning` mirror, because
    /// strict OpenAI-compatible clients reject a message that carries both
    /// (they expect exactly one). Requests may still send either name; see
    /// the `alias = "reasoning"` on the input-side message type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ToolCall>>,
    /// OpenAI-compatible annotations (URL citations, etc.).
    /// Populated post-hoc from URLs found in `content` so web-search /
    /// retrieval clients see a familiar shape. Omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Annotation>>,
    /// Assistant refusal message. When set, `content` should be treated
    /// as null by the client. Atlas does not currently emit refusals; the
    /// field is present so safety-aware clients stay compatible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// OpenAI `message.annotations[]` entry. Only `url_citation` is populated
/// today — the tagged variant keeps the wire format forward-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    UrlCitation { url_citation: UrlCitation },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlCitation {
    pub start_index: usize,
    pub end_index: usize,
    pub url: String,
    pub title: String,
}

impl From<crate::citation::Citation> for Annotation {
    fn from(c: crate::citation::Citation) -> Self {
        Annotation::UrlCitation {
            url_citation: UrlCitation {
                start_index: c.start_index,
                end_index: c.end_index,
                url: c.url,
                title: c.title,
            },
        }
    }
}

/// Bare/markdown URL scan (`citation::extract_url_citations`) in the
/// OpenAI wire shape. `None` when no URLs remain so the wire format
/// stays identical for non-web-search responses.
pub fn extract_url_annotations(content: &str) -> Option<Vec<Annotation>> {
    let cits = crate::citation::extract_url_citations(content);
    if cits.is_empty() {
        None
    } else {
        Some(cits.into_iter().map(Annotation::from).collect())
    }
}

/// Combined bare + structured citation extraction
/// (`citation::merged_citations`) in the OpenAI wire shape.
pub fn merged_annotations(content: &str) -> Option<Vec<Annotation>> {
    crate::citation::merged_citations(content)
        .map(|cits| cits.into_iter().map(Annotation::from).collect())
}
