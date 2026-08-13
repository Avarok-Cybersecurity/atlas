// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;
use super::sanitizer::*;

pub(crate) fn strip_thinking_tags(text: &str) -> String {
    let default_parser = crate::reasoning_parser::ReasoningFormat::Qwen.into_parser();
    extract_thinking(text, false, Some(&*default_parser)).1
}

/// Residual thinking-marker scrub for the assistant `content` channel.
///
/// The model's chat template opens `<think>` on the generation turn, and the
/// model closes it by emitting `</think>` (token id 19 for Laguna). It may emit
/// the close MORE THAN ONCE — observed live: `[19, …answer…, 19, …answer…]` for
/// "What is 17*23?", where the model declines to reason, closes immediately,
/// answers, then closes and answers again. The split only consumes the FIRST
/// close, so any later one lands in `content` verbatim.
///
/// The streaming path already removed these inline
/// (`chat_stream/handle_token.rs`); this is that logic hoisted so the blocking
/// path shares one implementation. Semantics are preserved exactly: drop every
/// occurrence and `trim_start` what follows, so `"…391.</think>17 × 23"` becomes
/// `"…391.17 × 23"` rather than leaving a ragged gap.
///
/// NOTE this only removes COMPLETE markers. Splitting reasoning from content is
/// the caller's job; this is defense-in-depth for markers that survive it.
pub(crate) fn scrub_think_markers(text: &str) -> String {
    const MARKERS: [&str; 5] = [
        "</think>",
        "</thinking>",
        "<thinking>",
        "</analysis>",
        "<analysis>",
    ];
    let mut out = text.to_string();
    for tag in MARKERS {
        while let Some(pos) = out.find(tag) {
            out = format!("{}{}", &out[..pos], out[pos + tag.len()..].trim_start());
        }
    }
    out
}

#[cfg(test)]
mod scrub_think_tests {
    use super::scrub_think_markers;

    #[test]
    fn removes_second_close_left_by_the_split() {
        // The exact live regression: model emits close, answers, closes, answers.
        assert_eq!(
            scrub_think_markers("17 × 23 = 391.</think>17 × 23 = 391."),
            "17 × 23 = 391.17 × 23 = 391."
        );
    }

    #[test]
    fn trims_whitespace_after_a_removed_marker() {
        assert_eq!(scrub_think_markers("</think>\n\n  hello"), "hello");
    }

    #[test]
    fn removes_every_occurrence_not_just_the_first() {
        assert_eq!(scrub_think_markers("a</think>b</think>c"), "abc");
    }

    #[test]
    fn leaves_ordinary_content_untouched() {
        let s = "Merge sort splits [38, 27] then merges. O(n log n).";
        assert_eq!(scrub_think_markers(s), s);
    }

    #[test]
    fn does_not_eat_partial_or_lookalike_markers() {
        // An incomplete marker is not a complete one; leave it for the caller.
        assert_eq!(
            scrub_think_markers("think about </thin"),
            "think about </thin"
        );
    }
}

/// Truncate assistant `content` at the first tool-call opener.
///
/// This model (and other tool-trained models) sometimes emit a
/// `<tool_call>…` block after a normal answer even when the request defined NO
/// tools — observed live: `…the story you keep telling yourself."<tool_call>catch_error({…})`.
/// When tools ARE active the tool parser extracts and strips these; when they
/// are NOT active nothing runs, so the raw markup lands in `content`.
///
/// `<tool_call>` (token id 25) and `<function=` are control markers, not prose,
/// so their presence in content is always spurious here. Everything from the
/// first one onward is the orphan tool block; cut it and trim. Apply ONLY when
/// no real tool call was produced (otherwise the tool parser already handled
/// the content).
pub(crate) fn strip_orphan_tool_markup(text: &str) -> String {
    const OPENERS: [&str; 2] = ["<tool_call>", "<function="];
    let cut = OPENERS
        .iter()
        .filter_map(|op| text.find(op))
        .min()
        .unwrap_or(text.len());
    text[..cut].trim_end().to_string()
}

#[cfg(test)]
mod orphan_tool_tests {
    use super::strip_orphan_tool_markup;

    #[test]
    fn cuts_the_live_leak() {
        let s = "You are the story you keep telling yourself.\
                 <tool_call>catch_error({'error': {'message': \"nope\"}})";
        assert_eq!(
            strip_orphan_tool_markup(s),
            "You are the story you keep telling yourself."
        );
    }

    #[test]
    fn cuts_at_function_opener() {
        assert_eq!(
            strip_orphan_tool_markup("Here you go.<function=foo>{}</function>"),
            "Here you go."
        );
    }

    #[test]
    fn leaves_clean_content_untouched() {
        let s = "The sky is blue due to Rayleigh scattering.";
        assert_eq!(strip_orphan_tool_markup(s), s);
    }

    #[test]
    fn does_not_trip_on_the_word_function() {
        // Only the control markers "<tool_call>" / "<function=" cut; prose is safe.
        let s = "You can call a function to do that.";
        assert_eq!(strip_orphan_tool_markup(s), s);
    }
}
