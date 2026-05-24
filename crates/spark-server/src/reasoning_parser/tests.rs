// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the reasoning parsers.

use super::{ReasoningFormat, ReasoningParser};

/// Build the parser for a format.
fn parser(fmt: ReasoningFormat) -> Box<dyn ReasoningParser> {
    fmt.into_parser()
}

// ── `<think>`-tag families: closed-block (normal) case ──────────────────────

#[test]
fn qwen_closed_block_splits_on_end_tag() {
    let p = parser(ReasoningFormat::Qwen);
    let (reasoning, content) = p.extract_thinking(
        "I need to think about this\n</think>\nThe answer is 42.",
        true,
    );
    assert_eq!(reasoning.unwrap(), "I need to think about this");
    assert_eq!(content, "The answer is 42.");
}

#[test]
fn deepseek_r1_closed_block_splits_on_end_tag() {
    // Nemotron output: prompt opened `<think>`, so no opening tag.
    let p = parser(ReasoningFormat::DeepSeekR1);
    let (reasoning, content) =
        p.extract_thinking("counting steps</think>Your name is Zephyr.", true);
    assert_eq!(reasoning.unwrap(), "counting steps");
    assert_eq!(content, "Your name is Zephyr.");
}

#[test]
fn minimax_strips_model_emitted_opening_tag() {
    // M2.7 re-emits its own `<think>` even though the prompt opened one.
    let p = parser(ReasoningFormat::MiniMax);
    let (reasoning, content) =
        p.extract_thinking("<think>weighing options</think>final answer", true);
    assert_eq!(reasoning.unwrap(), "weighing options");
    assert_eq!(content, "final answer");
}

// ── THE BUG FIX: thinking-on, no tags (cut off mid-think) ───────────────────

#[test]
fn thinking_on_tagless_output_is_reasoning_not_content() {
    // Budget exhausted before `</think>`: the chat template opened the
    // think block in the prompt, so a tagless output is unfinished
    // reasoning — it must NOT leak into user-visible content.
    for fmt in [
        ReasoningFormat::Qwen,
        ReasoningFormat::DeepSeekR1,
        ReasoningFormat::MiniMax,
    ] {
        let p = parser(fmt);
        let (reasoning, content) =
            p.extract_thinking("The user typed 'My name is' and then... wait", true);
        assert_eq!(
            reasoning.as_deref(),
            Some("The user typed 'My name is' and then... wait"),
            "{} must route tagless thinking-on output to reasoning",
            p.name(),
        );
        assert_eq!(
            content,
            "",
            "{}: no raw chain-of-thought in content",
            p.name()
        );
    }
}

#[test]
fn thinking_off_tagless_output_is_content() {
    // Thinking disabled: the template emits a closed `<think></think>`,
    // so a tagless output is the plain answer — it belongs in content.
    let p = parser(ReasoningFormat::DeepSeekR1);
    let (reasoning, content) = p.extract_thinking("Your name is Zephyr.", false);
    assert!(reasoning.is_none());
    assert_eq!(content, "Your name is Zephyr.");
}

#[test]
fn thinking_disabled_discards_reasoning_keeps_answer() {
    // Model reasoned anyway with thinking off — still split on the tag,
    // keep only the answer.
    let p = parser(ReasoningFormat::Qwen);
    let (reasoning, content) = p.extract_thinking("reasoning\n</think>\ncontent", false);
    assert!(reasoning.is_none());
    assert_eq!(content, "content");
}

#[test]
fn truncated_reasoning_with_partial_tag_streamed_in() {
    // `<think>` re-emitted, no close, thinking on → all reasoning.
    let p = parser(ReasoningFormat::DeepSeekR1);
    let (reasoning, content) = p.extract_thinking("<think>still reasoning", true);
    assert_eq!(reasoning.unwrap(), "still reasoning");
    assert_eq!(content, "");
}

// ── Mistral: model emits its own `[THINK]` ──────────────────────────────────

#[test]
fn mistral_closed_block() {
    let p = parser(ReasoningFormat::Mistral);
    let (reasoning, content) = p.extract_thinking(
        "[THINK]Let me reason here[/THINK]Paris is the capital.",
        true,
    );
    assert_eq!(reasoning.unwrap(), "Let me reason here");
    assert_eq!(content, "Paris is the capital.");
}

#[test]
fn mistral_no_tags_is_all_content() {
    // Mistral emits its own `[THINK]`; with none emitted there is no
    // reasoning — even with thinking enabled, the output is content.
    let p = parser(ReasoningFormat::Mistral);
    let (reasoning, content) = p.extract_thinking("Paris is the capital of France.", true);
    assert!(reasoning.is_none());
    assert_eq!(content, "Paris is the capital of France.");
}

#[test]
fn mistral_cut_off_mid_think() {
    let p = parser(ReasoningFormat::Mistral);
    let (reasoning, content) = p.extract_thinking("[THINK]reasoning not finished", true);
    assert_eq!(reasoning.unwrap(), "reasoning not finished");
    assert_eq!(content, "");
}

// ── Leaked / stray blocks ───────────────────────────────────────────────────

#[test]
fn first_end_tag_bounds_reasoning_leftover_blocks_stripped() {
    // Reasoning ends at the FIRST `</think>`; a stray `<think>..</think>`
    // pair leaking into content is defensively removed.
    let p = parser(ReasoningFormat::Qwen);
    let (reasoning, content) =
        p.extract_thinking("first thought</think>middle<think>leaked</think>end", true);
    assert_eq!(reasoning.unwrap(), "first thought");
    assert_eq!(content, "middleend");
}

#[test]
fn content_before_model_emitted_start_tag_is_content() {
    // Text before a model-emitted opening tag is content, not reasoning.
    let p = parser(ReasoningFormat::Mistral);
    let (reasoning, content) =
        p.extract_thinking("prefix [THINK]the reasoning[/THINK] the answer", true);
    assert_eq!(reasoning.unwrap(), "the reasoning");
    assert_eq!(content, "prefix  the answer");
}

// ── Gemma 4 channel format ──────────────────────────────────────────────────

#[test]
fn gemma4_channel_split() {
    let p = parser(ReasoningFormat::Gemma4);
    let (reasoning, content) = p.extract_thinking(
        "<|channel>thought\nworking it out<channel|>The final answer.",
        true,
    );
    assert_eq!(reasoning.unwrap(), "working it out");
    assert_eq!(content, "The final answer.");
}

#[test]
fn gemma4_no_channel_is_all_content() {
    let p = parser(ReasoningFormat::Gemma4);
    let (reasoning, content) = p.extract_thinking("just a direct answer", true);
    assert!(reasoning.is_none());
    assert_eq!(content, "just a direct answer");
}

#[test]
fn gemma4_open_channel_no_close_is_reasoning() {
    let p = parser(ReasoningFormat::Gemma4);
    let (reasoning, content) = p.extract_thinking("<|channel>thought\ncut off here", true);
    assert_eq!(reasoning.unwrap(), "cut off here");
    assert_eq!(content, "");
}

// ── Identity / format parsing ───────────────────────────────────────────────

#[test]
fn parsers_have_distinct_identity() {
    // Nemotron is decoupled from qwen — its own name, not "qwen".
    assert_eq!(parser(ReasoningFormat::Qwen).name(), "qwen");
    assert_eq!(parser(ReasoningFormat::DeepSeekR1).name(), "deepseek_r1");
    assert_eq!(parser(ReasoningFormat::MiniMax).name(), "minimax");
    assert_eq!(parser(ReasoningFormat::Mistral).name(), "mistral");
    assert_eq!(parser(ReasoningFormat::Gemma4).name(), "gemma4");
}

#[test]
fn parser_tags() {
    assert_eq!(parser(ReasoningFormat::DeepSeekR1).start_tag(), "<think>");
    assert_eq!(parser(ReasoningFormat::DeepSeekR1).end_tag(), "</think>");
    assert_eq!(parser(ReasoningFormat::Mistral).start_tag(), "[THINK]");
    assert_eq!(parser(ReasoningFormat::Mistral).end_tag(), "[/THINK]");
    assert_eq!(parser(ReasoningFormat::Gemma4).end_tag(), "<channel|>");
}

#[test]
fn format_from_str() {
    use std::str::FromStr;
    assert_eq!(
        ReasoningFormat::from_str("qwen").unwrap(),
        ReasoningFormat::Qwen
    );
    assert_eq!(
        ReasoningFormat::from_str("deepseek_r1").unwrap(),
        ReasoningFormat::DeepSeekR1
    );
    // Nemotron aliases all resolve to the DeepSeek-R1 family.
    for alias in ["nemotron", "nemotron_h", "nano_v3"] {
        assert_eq!(
            ReasoningFormat::from_str(alias).unwrap(),
            ReasoningFormat::DeepSeekR1,
        );
    }
    assert_eq!(
        ReasoningFormat::from_str("minimax").unwrap(),
        ReasoningFormat::MiniMax
    );
    assert_eq!(
        ReasoningFormat::from_str("mistral").unwrap(),
        ReasoningFormat::Mistral
    );
    assert_eq!(
        ReasoningFormat::from_str("gemma4").unwrap(),
        ReasoningFormat::Gemma4
    );
    assert!(ReasoningFormat::from_str("nonsense").is_err());
}
