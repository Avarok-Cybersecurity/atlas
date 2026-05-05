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

use super::msg_entry::MsgEntry;

pub(super) struct LoopDetectOut {
    /// True when the verdict was Suppress OR spinning detection
    /// fired. Caller flips the `<tool_call>` token bias to avoid
    /// re-emitting tool calls for one turn.
    pub(super) suppress_tool_call: bool,
    /// Run-length of the most recent loop (or 0). Caller threads
    /// this into the exponential `<tool_call>` logit-bias decay.
    pub(super) tool_call_repeat_count: usize,
}

pub(super) fn check_loops(
    req: &ChatCompletionRequest,
    messages: &mut [MsgEntry],
    consecutive_tool_errors: u32,
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
        let is_substantial = m.content.text.len() >= 500 || tool_args_len >= 100;
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
            suppress_tool_call = true;
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
        suppress_tool_call = true;
    }

    // Single hint, used for both Suppress and Hint verdicts.
    let loop_active = !matches!(verdict, crate::loop_detector::LoopState::None);
    let inject_hint = loop_active || spinning;
    if inject_hint {
        let hint = "\n\n<IMPORTANT>\nYour recent turns have produced \
                    output very similar to earlier turns. Before \
                    continuing: (1) inspect the CURRENT state with \
                    read-only tools so you can see what is already \
                    done; (2) if the user's request is already \
                    satisfied, summarise and stop; (3) otherwise \
                    identify the SPECIFIC remaining gap and address \
                    only that — do not retry the same approach or \
                    regenerate work that already exists.\n</IMPORTANT>";
        if let Some(last) = messages.last_mut() {
            last.content.push_str(hint);
        }
    }

    // Goal re-anchor (P1.3 + #4 per-turn spread).
    if crate::task_pin::should_inject(loop_active || spinning, consecutive_tool_errors)
        && let Some(goal) = crate::task_pin::extract_original_goal(&req.messages, |m| {
            (m.role.as_str(), m.content.text.as_str())
        })
    {
        let n_failures = consecutive_tool_errors as usize + tool_call_repeat_count;
        let reminder = crate::task_pin::build_reminder(goal, n_failures.max(1));
        // Find indices of the last two tool/user messages.
        let mut anchor_idxs: Vec<usize> = Vec::with_capacity(2);
        for (i, m) in messages.iter().enumerate().rev() {
            if m.role == "tool" || m.role == "user" {
                anchor_idxs.push(i);
                if anchor_idxs.len() >= 2 {
                    break;
                }
            }
        }
        let anchored_count = anchor_idxs.len();
        for idx in anchor_idxs {
            messages[idx].content.push_str(&reminder);
        }
        // Fallback: still anchor on the last message.
        if anchored_count == 0
            && let Some(last) = messages.last_mut()
        {
            last.content.push_str(&reminder);
        }
        tracing::info!(
            n_failures,
            anchored_count = anchored_count.max(1),
            "task_pin: injected verbatim-goal reminder"
        );
        crate::metrics::TASK_PIN_INJECTIONS.inc();
    }

    LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    }
}
