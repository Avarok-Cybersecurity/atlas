// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-native DeepSeek-V4 chat encoding.
//!
//! The official 0731 checkpoint ships `encoding_dsv4.py`, not a Hugging Face
//! chat template. Keeping the preprocessing in Rust avoids approximating its
//! tool-result merge, reasoning-history, and task-transition rules in Jinja.

use anyhow::{Result, bail};

mod preprocess;
mod render;

pub(super) const BOS: &str = "<｜begin▁of▁sentence｜>";
pub(super) const EOS: &str = "<｜end▁of▁sentence｜>";
pub(super) const USER: &str = "<｜User｜>";
pub(super) const ASSISTANT: &str = "<｜Assistant｜>";
pub(super) const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub(super) const THINK_START: &str = "<think>";
pub(super) const THINK_END: &str = "</think>";
pub(super) const DSML: &str = "｜DSML｜";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReasoningEffort {
    Low,
    High,
    Max,
}

impl ReasoningEffort {
    fn parse(value: Option<&str>, thinking: bool) -> Result<Self> {
        if !thinking {
            return Ok(Self::Low);
        }
        match value.unwrap_or("high") {
            "low" | "minimal" | "medium" | "none" => Ok(Self::Low),
            "high" => Ok(Self::High),
            "max" | "xhigh" => Ok(Self::Max),
            other => bail!("invalid DeepSeek-V4 reasoning effort '{other}'"),
        }
    }

    pub(super) fn prefix(self) -> &'static str {
        match self {
            Self::Low => "",
            Self::High => {
                "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
                 You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
                 Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
            }
            Self::Max => {
                "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
                 You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
                 Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
            }
        }
    }
}

pub(super) fn encode_messages(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<String> {
    if messages.is_empty() {
        bail!("DeepSeek-V4 requires at least one message");
    }

    let effort = ReasoningEffort::parse(reasoning_effort, enable_thinking)?;
    let mut messages = messages.to_vec();
    preprocess::attach_top_level_tools(&mut messages, tools)?;
    let mut messages = preprocess::merge_tool_messages(&messages)?;
    preprocess::sort_tool_results_by_call_order(&mut messages);

    let drop_thinking = !messages.iter().any(|m| {
        m.get("tools")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|v| !v.is_empty())
    });
    if enable_thinking && drop_thinking {
        messages = preprocess::drop_thinking_messages(&messages);
    }

    render::render_messages(&messages, enable_thinking, drop_thinking, effort)
}

pub(super) fn python_json(value: &serde_json::Value) -> Result<String> {
    render::python_json(value)
}
