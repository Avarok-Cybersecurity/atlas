// SPDX-License-Identifier: AGPL-3.0-only

//! Reasoning parser — model-agnostic thinking/reasoning block detection.
//!
//! Each served model family delimits its chain-of-thought differently.
//! This module abstracts that behind the [`ReasoningParser`] trait so the
//! server can extract reasoning vs. final content for any model:
//!
//! | Format                  | Delimiters                  | Models |
//! |-------------------------|-----------------------------|--------|
//! | [`ReasoningFormat::Qwen`]       | `<think>` / `</think>`      | Qwen3.5/3.6/Next/VL |
//! | [`ReasoningFormat::DeepSeekR1`] | `<think>` / `</think>`      | Nemotron-H / Nano-3 / Super (`nano_v3`) |
//! | [`ReasoningFormat::MiniMax`]    | `<think>` / `</think>`      | MiniMax M2 / M2.7 |
//! | [`ReasoningFormat::Mistral`]    | `[THINK]` / `[/THINK]`      | Mistral Small 4 / Magistral |
//! | [`ReasoningFormat::Gemma4`]     | `<|channel>` / `<channel|>` | Gemma 4 (channel format) |
//!
//! The Qwen / DeepSeek-R1 / MiniMax families all use `<think>` tags with
//! the *same* extraction contract — the chat template injects the opening
//! tag into the prompt, so output begins inside the reasoning block — and
//! share one implementation, configured per-family with a distinct
//! identity. Mistral differs structurally: the model emits its own
//! `[THINK]`, so the parser does not assume an open block. Gemma 4 uses a
//! channel format entirely unlike `<think>` tags and has its own parser.
//!
//! Follows the same trait + enum + TOML-auto-detect pattern as
//! `ToolCallParser`.

mod parsers;
#[cfg(test)]
mod tests;

use std::str::FromStr;

use crate::tokenizer::ChatTokenizer;

/// Parses reasoning/thinking blocks from completed model output.
pub trait ReasoningParser: Send + Sync {
    /// Parser name for logging (e.g. `"qwen"`, `"deepseek_r1"`).
    fn name(&self) -> &str;

    /// Opening delimiter (e.g. `"<think>"`, `"[THINK]"`, `"<|channel>"`).
    fn start_tag(&self) -> &str;

    /// Closing delimiter (e.g. `"</think>"`, `"[/THINK]"`, `"<channel|>"`).
    fn end_tag(&self) -> &str;

    /// Resolve the end-of-thinking token ID from the tokenizer.
    /// Returns `None` if the end tag doesn't encode to a single token.
    fn end_token_id(&self, tokenizer: &ChatTokenizer) -> Option<u32> {
        match tokenizer.encode(self.end_tag()) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        }
    }

    /// Split completed generation text into `(reasoning, content)`.
    ///
    /// `enable_thinking` is the resolved per-request thinking state.
    /// When `false` the reasoning is discarded (`None`); the answer is
    /// always returned in `content`.
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String);
}

/// Supported reasoning-block formats. One variant per model family — see
/// the module docs for the delimiter/contract of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// `<think>...</think>` — Qwen3 family.
    Qwen,
    /// `<think>...</think>` — DeepSeek-R1 / NVIDIA Nemotron (`nano_v3`).
    DeepSeekR1,
    /// `<think>...</think>` — MiniMax M2 / M2.7.
    MiniMax,
    /// `[THINK]...[/THINK]` — Mistral / Magistral.
    Mistral,
    /// `<|channel>thought ... <channel|> ...` — Gemma 4 channel format.
    Gemma4,
}

impl FromStr for ReasoningFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "qwen" | "qwen3" => Ok(Self::Qwen),
            "deepseek_r1" | "deepseek" | "nemotron" | "nemotron_h" | "nano_v3" => {
                Ok(Self::DeepSeekR1)
            }
            "minimax" | "minimax_m2" => Ok(Self::MiniMax),
            "mistral" => Ok(Self::Mistral),
            "gemma4" | "gemma" => Ok(Self::Gemma4),
            other => Err(format!(
                "Unknown reasoning parser '{other}'. Supported: qwen, \
                 deepseek_r1, minimax, mistral, gemma4"
            )),
        }
    }
}

impl ReasoningFormat {
    /// Create a boxed parser for this format.
    pub fn into_parser(self) -> Box<dyn ReasoningParser> {
        parsers::build(self)
    }
}
