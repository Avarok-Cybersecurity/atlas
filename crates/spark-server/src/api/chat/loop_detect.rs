// SPDX-License-Identifier: AGPL-3.0-only
//
// Loop detection + spinning detection + task-pin re-anchor block,
// extracted from `chat::chat_completions_inner` (wave 4g).
//
// Inputs: the request (so we can scan history) and the in-progress
// `messages` vec (mutated to inject hints / anchors). Outputs are
// returned via `LoopDetectOut` so the orchestrator can wire them
// into the downstream sampling-bias logic.

use crate::openai::ChatCompletionRequest;

pub(super) struct LoopDetectOut {
    /// True when the verdict was Suppress OR spinning detection
    /// fired. Caller flips the `<tool_call>` token bias to avoid
    /// re-emitting tool calls for one turn.
    pub(super) suppress_tool_call: bool,
    /// Run-length of the most recent loop (or 0). Caller threads
    /// this into the exponential `<tool_call>` logit-bias decay.
    pub(super) tool_call_repeat_count: usize,
}

/// BW1/SPINFIX relaxation (Iter 52, 2026-06-02): the loop detector hard-masks
/// the `<tool_call>` token (Suppress verdict + spinning) after only ~3 similar
/// turns. During legitimate agentic coding the agent repeats commands
/// (`ls`/`cargo check`/`cargo run` while iterating), trips this at turns>=3,
/// and gets its next tool call BLOCKED — forcing a content/`<response>`
/// fast-fail. vLLM applies no such mask. With `ATLAS_LOOP_NO_SUPPRESS=1` the
/// verdict is still detected, logged, and metered, but the `<tool_call>`
/// hard-mask is NOT applied (the benign Hint path is unaffected). Default OFF
/// ⇒ byte-identical to today; additive, model-agnostic.
fn loop_suppress_disabled() -> bool {
    std::env::var("ATLAS_LOOP_NO_SUPPRESS").as_deref() == Ok("1")
}

pub(super) fn check_loops(
    req: &ChatCompletionRequest,
    tools_active: bool,
) -> LoopDetectOut {
    let mut suppress_tool_call = false;
    let mut tool_call_repeat_count: usize = 0;

    if !tools_active {
        return LoopDetectOut {
            suppress_tool_call,
            tool_call_repeat_count,
        };
    }

    let signatures: Vec<crate::loop_detector::Signature> = req
        .messages
        .iter()
        .rev()
        .filter(|m| m.role == "assistant")
        .map(|m| {
            let calls: Vec<(&str, &str)> = m
                .tool_calls
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|tc| (tc.function.name.as_str(), tc.function.arguments.as_str()))
                .collect();
            crate::loop_detector::Signature::build(&m.content.text, calls)
        })
        .take(8)
        .collect();
    let verdict = crate::loop_detector::detect(&signatures);

    // Spinning detection — independent signal: if the model has
    // produced ≥5 consecutive short, low-content responses,
    // something is structurally wrong even if no two are similar.
    let mut recent_short: usize = 0;
    for m in req.messages.iter().rev() {
        if m.role != "assistant" {
            continue;
        }
        let tool_args_len: usize = m.tool_calls.as_ref().map_or(0, |tcs| {
            tcs.iter().map(|tc| tc.function.arguments.len()).sum()
        });
        // A turn that issued ANY tool call is taking an action (progress) — it
        // is NOT spinning, even when the args are short. In an agentic coding
        // loop the verify cycle (`bash cargo build`, `bash cargo run`,
        // `bash curl`, `read`, small `edit`) is a run of legitimately
        // short-arg tool calls; counting those as "short" tripped the
        // recent_short>=5 spinning suppressor and hard-masked the NEXT
        // tool_call, killing the build→error→fix→rebuild loop after ~5 turns
        // (Atlas capped at ~4-5 turns vs vLLM's 12-17 on the same task).
        // Genuine repeated-tool-call loops are caught separately by
        // `loop_detector::detect` (the Suppress verdict above); spinning here
        // should only fire on consecutive short PURE-TEXT turns (no action).
        let made_tool_call = m
            .tool_calls
            .as_ref()
            .is_some_and(|tcs| !tcs.is_empty());
        let is_substantial = made_tool_call || m.content.text.len() >= 500 || tool_args_len >= 100;
        if is_substantial {
            break;
        }
        recent_short += 1;
        if recent_short >= 8 {
            break;
        }
    }
    let spinning = recent_short >= 5;

    match &verdict {
        crate::loop_detector::LoopState::Suppress {
            score,
            run_length,
            channel,
        } => {
            tracing::warn!(
                score = *score,
                run_length = *run_length,
                channel = channel.name(),
                "Loop detector → SUPPRESS: hard-mask <tool_call> for one turn"
            );
            suppress_tool_call = !loop_suppress_disabled();
            tool_call_repeat_count = *run_length;
            crate::metrics::LOOP_DETECTOR_VERDICTS
                .with_label_values(&["suppress", channel.name(), if spinning { "1" } else { "0" }])
                .inc();
        }
        crate::loop_detector::LoopState::Hint {
            score,
            run_length,
            channel,
        } => {
            tracing::info!(
                score = *score,
                run_length = *run_length,
                channel = channel.name(),
                "Loop detector → HINT: inject progress notice (no hard-mask)"
            );
            tool_call_repeat_count = *run_length;
            crate::metrics::LOOP_DETECTOR_VERDICTS
                .with_label_values(&["hint", channel.name(), if spinning { "1" } else { "0" }])
                .inc();
        }
        crate::loop_detector::LoopState::None => {
            crate::metrics::LOOP_DETECTOR_VERDICTS
                .with_label_values(&["none", "n/a", if spinning { "1" } else { "0" }])
                .inc();
        }
    }
    if spinning {
        tracing::warn!(
            recent_short,
            "Spinning detection fired — suppressing <tool_call>"
        );
        suppress_tool_call = !loop_suppress_disabled();
    }

    LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    }
}
