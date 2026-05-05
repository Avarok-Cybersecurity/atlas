// SPDX-License-Identifier: AGPL-3.0-only

//! Tool call parsing for OpenAI-compatible function calling.
//!
//! All supported models use `<tool_call></tool_call>` outer tags.
//! The inner content format differs by parser:
//! - **hermes**: JSON `{"name":"fn","arguments":{...}}`
//! - **qwen3_coder**: XML `<function=fn><parameter=key>value</parameter></function>`

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::grammar::{GrammarEngine, GrammarError};
use xgrammar::CompiledGrammar;

/// Global counter for unique tool call IDs across all requests.
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a globally unique tool call ID.
fn next_tool_call_id() -> String {
    let id = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{id:016x}")
}

// ── Request types (from OpenAI-compatible clients) ──

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    /// OpenAI format: {"type": "function", "function": {"name": "X"}}
    /// or simplified: {"function": {"name": "X"}}
    Specific {
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

impl ToolChoice {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::Mode(s) if s == "none")
    }
}

/// Tool call from a previous assistant message (multi-turn conversations).
///
/// `Serialize` is derived so the response_store filesystem backend can
/// round-trip historical tool calls through disk. The emitted shape
/// matches the OpenAI inbound schema (the same one used to parse it),
/// so replay feeds correctly back into the pipeline.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IncomingToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: IncomingFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IncomingFunction {
    pub name: String,
    pub arguments: String,
}

// ── Response types (to clients) ──

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool call delta for streaming chunks.
///
/// Per OpenAI spec, the first delta carries `id`, `type`, and `function.name`
/// with empty `arguments`. Subsequent deltas carry only `arguments` fragments
/// (all other fields are omitted/null).
#[derive(Debug, Clone, Serialize)]
pub struct ChunkToolCall {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ChunkFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}

// ── ToolCallParser trait ──

/// Trait for tool call parser implementations.
///
/// Each parser defines how to:
/// 1. Generate the system prompt that teaches the model the tool calling format
/// 2. Format assistant tool_calls for multi-turn conversations
/// 3. Format tool response messages
///
/// Output parsing (extracting tool calls from model text) and streaming
/// detection are shared infrastructure — both formats use `<tool_call>` outer
/// tags, and `parse_one_call()` auto-detects JSON vs XML inner content.
/// Leak markers a streaming content sanitizer uses to suppress malformed
/// tool-call fragments that escape into the assistant content channel.
///
/// Each `ToolCallParser` declares its own markers via
/// `ToolCallParser::leak_markers`. The sanitizer runs a generic
/// state-machine keyed on these symbols; no tool-format-specific logic
/// lives in the streaming layer. Parsers that don't need sanitization
/// return `LeakMarkers::EMPTY`, which short-circuits the scanner to a
/// pass-through.
///
/// Semantics:
/// - When any string in `orphan_open` is found in the content stream
///   AND we're not currently inside a tool-call envelope, the
///   sanitizer enters suppression and drops bytes until it finds
///   ANY string from `close`.
/// - When any string in `envelope_open` is found in the content
///   stream, the sanitizer enters "inside envelope" mode. While in
///   that mode, `orphan_open` matches do NOT trigger suppression —
///   the inner content is expected. Matching `envelope_close`
///   exits the mode. F73 (2026-04-29).
/// - Stray `close` occurrences outside suppression are silently dropped
///   (dangling XML tags from malformed output).
/// - Slices are `'static` so the sanitizer can hold references across a
///   stream's lifetime with zero allocation.
#[derive(Copy, Clone)]
pub struct LeakMarkers {
    pub orphan_open: &'static [&'static str],
    pub close: &'static [&'static str],
    /// F73 (2026-04-29): sanctioned outer-envelope openers (e.g.
    /// `<minimax:tool_call>`). When matched, sanitizer enters
    /// "inside envelope" mode and stops treating `orphan_open` as
    /// orphan. Empty default keeps existing parsers behaviour-
    /// identical.
    pub envelope_open: &'static [&'static str],
    /// Matching closers for `envelope_open` (e.g.
    /// `</minimax:tool_call>`). Match exits envelope mode.
    pub envelope_close: &'static [&'static str],
}

impl LeakMarkers {
    /// No markers — sanitizer is a pass-through. Use for parsers whose
    /// leaks can't be caught by text matching (e.g. Mistral's special
    /// tokens, bare JSON, Hermes JSON body).
    pub const EMPTY: Self = Self {
        orphan_open: &[],
        close: &[],
        envelope_open: &[],
        envelope_close: &[],
    };
}

pub trait ToolCallParser: Send + Sync {
    /// Parser name for logging (e.g. "hermes", "qwen3_coder").
    fn name(&self) -> &str;

    /// Generate the system prompt that teaches the model how to make tool calls.
    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String;

    /// Format assistant tool_calls as text for multi-turn chat template injection.
    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String;

    /// Format a tool response message for the chat template.
    /// Default wraps in `<tool_response>` tags (works for all current formats).
    fn format_tool_response(&self, content: &str) -> String {
        format!("<tool_response>\n{content}\n</tool_response>")
    }

    /// Declare the tag vocabulary the streaming content sanitizer uses to
    /// suppress malformed tool-call leaks. Default: `LeakMarkers::EMPTY`
    /// (sanitizer passes text through untouched). Override per-parser to
    /// opt in — see `Qwen3CoderParser::leak_markers` for the canonical
    /// example.
    fn leak_markers(&self) -> LeakMarkers {
        LeakMarkers::EMPTY
    }

    /// F69 (2026-04-29): symmetric grammar dispatch. Compile the
    /// XGrammar structural-tag grammar that constrains decoding for
    /// this parser's tool-call envelope.
    ///
    /// Default: `None` — no constrained decoding. Parsers that want
    /// XGrammar enforcement override this to call the matching
    /// [`GrammarEngine`] entry point. The previous design lived as a
    /// `match parser_name.as_str()` in `scheduler.rs::compile_grammar_state`,
    /// which kept two sources of truth (parser name in `ToolCallFormat`
    /// vs. parser name string in scheduler) that could drift — F66
    /// added a startup audit precisely to catch that drift. With this
    /// trait method the parser is the single source of truth for both
    /// scanning AND grammar.
    ///
    /// `tools` are the request's tool definitions; `use_triggers` is
    /// `false` for `tool_choice="required"` (forces a tool call) and
    /// `true` for `tool_choice="auto"` (model may also emit free text).
    ///
    /// Return value:
    /// - `None`        — this parser doesn't support constrained decoding
    ///                   (currently only `MistralNativeParser`).
    /// - `Some(Ok(g))` — compiled grammar; runtime applies it.
    /// - `Some(Err(_))` — compile failed; runtime logs and skips
    ///                    constraint (model can still emit unconstrained).
    fn compile_tool_grammar(
        &self,
        _engine: &mut GrammarEngine,
        _tools: &[ToolDefinition],
        _use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        None
    }

    /// F69 (2026-04-29): cheap query — does this parser produce a
    /// non-`None` result from [`Self::compile_tool_grammar`] with at
    /// least one tool? Used for startup-time validation logs and the
    /// `ToolCallFormat::has_grammar` derived discriminator.
    ///
    /// Default `false` — parsers that override `compile_tool_grammar`
    /// should also override this to return `true`.
    fn has_tool_grammar(&self) -> bool {
        false
    }

    /// F71 (2026-04-29): "broken opener" stop strings — patterns that,
    /// if they appear in the model's content stream, indicate the
    /// model has fallen into a known BPE-merge attractor that produces
    /// a *corrupted* tool-call envelope. No parser currently overrides;
    /// F73's envelope-aware sanitizer (and `parse_tool_calls`'s
    /// `<minimax:_call>` → `<tool_call>` normalisation) recover the
    /// broken-envelope case at the streaming + parser layer instead.
    fn broken_opener_stop_strings(&self) -> &'static [&'static str] {
        &[]
    }
}

impl std::fmt::Display for dyn ToolCallParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── CLI format enum → parser resolution ──

/// Maps CLI `--tool-call-parser` string to a concrete parser.
#[derive(Debug, Clone, Copy)]
pub enum ToolCallFormat {
    Hermes,
    Qwen3Coder,
    Gemma4,
    Mistral,
    MinimaxXml,
    BareJson,
}

impl std::str::FromStr for ToolCallFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hermes" => Ok(Self::Hermes),
            "qwen3_coder" => Ok(Self::Qwen3Coder),
            "gemma4" => Ok(Self::Gemma4),
            "mistral" => Ok(Self::Mistral),
            "minimax_xml" => Ok(Self::MinimaxXml),
            "bare_json" => Ok(Self::BareJson),
            other => Err(format!(
                "Unknown tool call parser '{other}'. Supported: hermes, qwen3_coder, gemma4, mistral, minimax_xml, bare_json",
            )),
        }
    }
}

impl ToolCallFormat {
    /// Create a boxed parser implementation for this format.
    pub fn into_parser(self) -> Box<dyn ToolCallParser> {
        match self {
            Self::Hermes => Box::new(HermesParser),
            Self::Qwen3Coder => Box::new(Qwen3CoderParser),
            Self::Gemma4 => Box::new(Gemma4Parser),
            Self::Mistral => Box::new(MistralNativeParser),
            Self::MinimaxXml => Box::new(MinimaxXmlParser),
            Self::BareJson => Box::new(BareJsonParser),
        }
    }

    /// F66/F69 (2026-04-29): does this parser have a registered XGrammar
    /// schema? Parsers without a grammar fall through to unconstrained
    /// decoding, which (per fix39 testing on MiniMax M2.7) can produce
    /// token-doubled output at the tool-call boundary.
    ///
    /// F69 routed this through the [`ToolCallParser::has_tool_grammar`]
    /// trait method so the per-parser answer is the parser's own — the
    /// previous duplicated `match self` on the enum variant could drift
    /// from the parser's actual `compile_tool_grammar` override.
    pub fn has_grammar(self) -> bool {
        self.into_parser().has_tool_grammar()
    }

    /// F66 (2026-04-29): canonical name used in CLI flags and logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hermes => "hermes",
            Self::Qwen3Coder => "qwen3_coder",
            Self::Gemma4 => "gemma4",
            Self::Mistral => "mistral",
            Self::MinimaxXml => "minimax_xml",
            Self::BareJson => "bare_json",
        }
    }
}

// ── MiniMax XML parser (MiniMax M2.7) ──

// ── Sub-modules (split from monolithic file) ──
mod bare_json;
mod gemma4;
mod helpers_a;
mod helpers_b;
mod hermes;
mod minimax_xml;
mod mistral;
mod parse_dispatch;
mod parse_single_a;
mod parse_single_b;
mod parse_tools_tag;
mod pipeline;
mod pipeline_helpers;
mod qwen3_coder;
mod streaming;
mod streaming_impl;
mod validation;

pub use bare_json::*;
pub use gemma4::*;
use helpers_a::*;
use helpers_b::*;
pub use hermes::*;
pub use minimax_xml::*;
pub use mistral::*;
pub use parse_dispatch::*;
use parse_single_a::*;
use parse_single_b::*;
use parse_tools_tag::*;
pub use pipeline::*;
use pipeline_helpers::*;
pub use qwen3_coder::*;
pub use streaming::*;
pub use validation::*;

#[cfg(test)]
mod tests;
