// SPDX-License-Identifier: AGPL-3.0-only

//! Concrete reasoning-parser implementations.
//!
//! `<think>`-style families (Qwen, DeepSeek-R1/Nemotron, MiniMax) share
//! [`extract_tag_thinking`] — the algorithm is identical, so SSOT is kept
//! by configuring one [`TagReasoningParser`] per family rather than
//! duplicating logic. Mistral reuses the same struct with `[THINK]` tags
//! and `prompt_opens_think = false`. Gemma 4's channel format is distinct
//! and has its own [`Gemma4ReasoningParser`].

use super::{ReasoningFormat, ReasoningParser};

/// Build the boxed parser for a [`ReasoningFormat`].
pub(super) fn build(fmt: ReasoningFormat) -> Box<dyn ReasoningParser> {
    match fmt {
        // `<think>` families. The chat template injects the opening
        // `<think>` into the prompt, so model output begins inside the
        // reasoning block (`prompt_opens_think = true`).
        ReasoningFormat::Qwen => Box::new(TagReasoningParser::THINK_PROMPT_OPENED.named("qwen")),
        ReasoningFormat::DeepSeekR1 => {
            Box::new(TagReasoningParser::THINK_PROMPT_OPENED.named("deepseek_r1"))
        }
        ReasoningFormat::MiniMax => {
            Box::new(TagReasoningParser::THINK_PROMPT_OPENED.named("minimax"))
        }
        // Mistral emits its own `[THINK]`; the parser must not assume an
        // already-open block.
        ReasoningFormat::Mistral => Box::new(TagReasoningParser {
            name: "mistral",
            start: "[THINK]",
            end: "[/THINK]",
            prompt_opens_think: false,
        }),
        ReasoningFormat::Gemma4 => Box::new(Gemma4ReasoningParser),
    }
}

// ── Tag-delimited parser (Qwen / DeepSeek-R1 / MiniMax / Mistral) ────────────

/// A `<tag>`-delimited reasoning parser. One implementation, configured
/// per model family — each gets a distinct [`ReasoningParser::name`] so
/// the families are decoupled even though the extraction logic is shared.
pub(super) struct TagReasoningParser {
    pub(super) name: &'static str,
    pub(super) start: &'static str,
    pub(super) end: &'static str,
    /// `true` when the chat template injects the opening tag into the
    /// prompt, so the model's output begins *inside* the reasoning block
    /// (Qwen / DeepSeek-R1 / MiniMax). `false` when the model emits its
    /// own opening tag (Mistral).
    pub(super) prompt_opens_think: bool,
}

impl TagReasoningParser {
    /// Base config for the `<think>`-tag, prompt-opened families.
    const THINK_PROMPT_OPENED: Self = Self {
        name: "qwen",
        start: "<think>",
        end: "</think>",
        prompt_opens_think: true,
    };

    /// This config with a different family name.
    const fn named(self, name: &'static str) -> Self {
        Self { name, ..self }
    }
}

impl ReasoningParser for TagReasoningParser {
    fn name(&self) -> &str {
        self.name
    }
    fn start_tag(&self) -> &str {
        self.start
    }
    fn end_tag(&self) -> &str {
        self.end
    }
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        extract_tag_thinking(
            text,
            self.start,
            self.end,
            self.prompt_opens_think,
            enable_thinking,
        )
    }
}

/// Split `text` into `(reasoning, content)` for a `<tag>`-delimited
/// format. Mirrors the vLLM `BaseThinkingReasoningParser` / DeepSeek-R1
/// contract:
///
///  * A leading `start` tag (if the model emitted one) is consumed; any
///    text before it is content.
///  * The reasoning block ends at the **first** `end` tag; everything
///    after is content.
///  * No `end` tag — if we are inside the block (`prompt_opens_think &&
///    enable_thinking`, or the model emitted its own `start`), the output
///    was cut off mid-think and is **all reasoning** (never leaked into
///    content as raw chain-of-thought). Otherwise the model emitted no
///    reasoning and the whole output is content.
pub(super) fn extract_tag_thinking(
    text: &str,
    start: &str,
    end: &str,
    prompt_opens_think: bool,
    enable_thinking: bool,
) -> (Option<String>, String) {
    // Locate the reasoning region. `pre` is any content before the
    // block, `body` runs from the start of reasoning, `in_block` records
    // whether `body` begins inside the reasoning block.
    let (pre, body, in_block): (&str, &str, bool) = if prompt_opens_think {
        // The chat template opened the block in the prompt, so output
        // begins inside it. A `start` tag matters only if the model
        // redundantly re-emitted it at the very start; a `start` later
        // in the text is leaked content, not a delimiter.
        match text.trim_start().strip_prefix(start) {
            Some(rest) => ("", rest, true),
            None => ("", text, enable_thinking),
        }
    } else {
        // The model emits its own `start` tag; text before it is content.
        match text.find(start) {
            Some(p) => (&text[..p], &text[p + start.len()..], true),
            None => ("", text, false),
        }
    };

    let (reasoning, content) = match body.find(end) {
        // Closed block: reasoning before `end`, content = pre + tail.
        Some(e) => {
            let mut c = String::with_capacity(pre.len() + body.len() - e);
            c.push_str(pre);
            c.push_str(&body[e + end.len()..]);
            (body[..e].to_string(), c)
        }
        // No `end` tag, inside the block → cut off mid-think: all
        // reasoning, `pre` is the only content.
        None if in_block => (body.to_string(), pre.to_string()),
        // No tags, not inside a block → no reasoning emitted.
        None => (String::new(), text.to_string()),
    };

    // Defensive: a model occasionally emits a stray `start..end` pair
    // after the real block — strip any such leftovers from content.
    let content = strip_tag_pairs(&content, start, end);

    let reasoning = reasoning.trim();
    let content = content.trim().to_string();
    if enable_thinking && !reasoning.is_empty() {
        (Some(reasoning.to_string()), content)
    } else {
        (None, content)
    }
}

/// Remove every balanced `start..end` pair from `s` (an unmatched
/// trailing `start` is left intact).
fn strip_tag_pairs(s: &str, start: &str, end: &str) -> String {
    let mut out = s.to_string();
    while let Some(a) = out.find(start) {
        match out[a + start.len()..].find(end) {
            Some(rel) => {
                let b = a + start.len() + rel + end.len();
                out.replace_range(a..b, "");
            }
            None => break,
        }
    }
    out
}

// ── Gemma 4 channel parser ──────────────────────────────────────────────────

/// Gemma 4 reasoning: the model emits `<|channel>thought\n … <channel|> …`
/// inside its turn — reasoning lives between the channel open/close
/// delimiters, the answer follows the close. Note the asymmetric
/// bracketing (`<|channel>` open, `<channel|>` close).
pub(super) struct Gemma4ReasoningParser;

impl Gemma4ReasoningParser {
    const OPEN: &'static str = "<|channel>";
    const CLOSE: &'static str = "<channel|>";
}

impl ReasoningParser for Gemma4ReasoningParser {
    fn name(&self) -> &str {
        "gemma4"
    }
    fn start_tag(&self) -> &str {
        Self::OPEN
    }
    fn end_tag(&self) -> &str {
        Self::CLOSE
    }
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        let Some((before, after_open)) = text.split_once(Self::OPEN) else {
            // No reasoning channel — the whole output is the answer.
            return (None, text.trim().to_string());
        };
        // `<|channel>` is followed by a literal `thought` role label.
        let after_label = after_open
            .trim_start()
            .strip_prefix("thought")
            .unwrap_or(after_open);
        match after_label.split_once(Self::CLOSE) {
            Some((reasoning, answer)) => {
                let reasoning = reasoning.trim();
                let content = format!("{}{}", before.trim(), answer.trim());
                let content = content.trim().to_string();
                if enable_thinking && !reasoning.is_empty() {
                    (Some(reasoning.to_string()), content)
                } else {
                    (None, content)
                }
            }
            // Open channel, no close — cut off mid-thought.
            None => {
                let reasoning = after_label.trim();
                if enable_thinking && !reasoning.is_empty() {
                    (Some(reasoning.to_string()), before.trim().to_string())
                } else {
                    (None, before.trim().to_string())
                }
            }
        }
    }
}
