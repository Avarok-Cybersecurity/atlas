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
    /// True ONLY when the content-similarity detector returns
    /// [`crate::loop_detector::LoopState::Suppress`] — i.e. strong,
    /// *measured* repetition across recent turns. The caller flips the
    /// `<tool_call>` token bias to break a genuine loop.
    ///
    /// The size-based "spinning" signal deliberately does NOT set this.
    /// A run of short *distinct* tool calls (reading files one at a
    /// time) is legitimate agentic exploration, not a loop; masking
    /// `<tool_call>` there strands the model — it narrates "let me read
    /// X" with the tool token biased to -inf, never progresses, never
    /// emits EOS, and is finally truncated mid-sentence by the
    /// inter-tool prose budget (the opencode "silent death").
    pub(super) suppress_tool_call: bool,
    /// Run-length of the most recent loop (or 0). Caller threads
    /// this into the exponential `<tool_call>` logit-bias decay.
    pub(super) tool_call_repeat_count: usize,
}

/// A turn is "short" (low-signal) when it carries little prose AND
/// little tool-call payload. These bounds are deliberately generous: a
/// single `read`/`grep`/`glob` call — the bread-and-butter of agentic
/// exploration — is short, so a run of them is normal progress, not a
/// loop. See [`leading_short_turn_run`].
const SHORT_TURN_CONTENT_MAX: usize = 500;
const SHORT_TURN_TOOL_ARGS_MAX: usize = 100;

/// `recent_short >= this` raises the (weak) size-based spinning signal.
const SPINNING_RUN_THRESHOLD: usize = 5;

/// How far back the spinning scan walks — it only needs to learn
/// whether the leading run reaches [`SPINNING_RUN_THRESHOLD`].
const SPINNING_SCAN_CAP: usize = 8;

/// Length of the leading run of consecutive "short" assistant turns,
/// newest-first, stopping at the first substantial turn. Pure and
/// allocation-free so it is unit-testable without an openai request.
///
/// Each item is `(content_len, tool_args_len)` for one assistant turn.
fn leading_short_turn_run<I>(assistant_turns_newest_first: I) -> usize
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut run = 0usize;
    for (content_len, tool_args_len) in assistant_turns_newest_first {
        let substantial =
            content_len >= SHORT_TURN_CONTENT_MAX || tool_args_len >= SHORT_TURN_TOOL_ARGS_MAX;
        if substantial {
            break;
        }
        run += 1;
        if run >= SPINNING_SCAN_CAP {
            break;
        }
    }
    run
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

    // Size-based "spinning" signal: a leading run of consecutive
    // short, low-content assistant turns (newest-first). WEAK by
    // design — see `leading_short_turn_run` and the suppression note
    // below. It feeds only the soft hint + goal re-anchor, never the
    // hard `<tool_call>` mask.
    let recent_short = leading_short_turn_run(
        req.messages
            .iter()
            .rev()
            .filter(|m| m.role == "assistant")
            .map(|m| {
                let tool_args_len: usize = m.tool_calls.as_ref().map_or(0, |tcs| {
                    tcs.iter().map(|tc| tc.function.arguments.len()).sum()
                });
                (m.content.text.len(), tool_args_len)
            }),
    );
    let spinning = recent_short >= SPINNING_RUN_THRESHOLD;

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
        // Spinning is a SIZE signal, not a repetition signal. A run of
        // short *distinct* tool calls is legitimate exploration
        // (reading files one at a time), so we must NOT hard-mask
        // `<tool_call>` here: doing so strands the model — it keeps
        // narrating "let me read X" with the tool token biased to -inf
        // (decode_logits_seq.rs), never progresses, never emits EOS,
        // and is finally truncated mid-sentence by the inter-tool prose
        // budget (NoSsmSnapshot hard stop) — the opencode "silent
        // death". Hard suppression stays gated on the content-similarity
        // verdict (`LoopState::Suppress`) above, which measures ACTUAL
        // repetition. Spinning only contributes the soft hint + goal
        // re-anchor below.
        tracing::warn!(
            recent_short,
            "Spinning detection fired (many short turns) — injecting hint + \
             goal re-anchor only; NOT masking <tool_call> (distinct short tool \
             calls are legitimate exploration)"
        );
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

#[cfg(test)]
mod spinning_tests {
    use super::{SPINNING_RUN_THRESHOLD, leading_short_turn_run};

    #[test]
    fn run_of_five_short_reads_reaches_spinning_threshold() {
        // The opencode failure shape (dump line 21): five short, distinct
        // read/glob turns, then a substantial `task` turn ends the run.
        // Items are (content_len, tool_args_len), newest-first.
        let turns: [(usize, usize); 6] = [(2, 98), (2, 89), (0, 79), (123, 45), (2, 47), (2, 1607)];
        let run = leading_short_turn_run(turns);
        assert_eq!(
            run, 5,
            "five leading short turns before the substantial task"
        );
        assert!(run >= SPINNING_RUN_THRESHOLD);
    }

    #[test]
    fn substantial_turn_stops_the_run() {
        // A long prose answer (>= content bound) ends the run...
        assert_eq!(
            leading_short_turn_run([(2usize, 98usize), (600, 0), (2, 1)]),
            1
        );
        // ...as does a large tool-call payload (>= args bound).
        assert_eq!(
            leading_short_turn_run([(2usize, 98usize), (0, 100), (2, 1)]),
            1
        );
    }

    #[test]
    fn empty_history_has_no_run() {
        let none: [(usize, usize); 0] = [];
        assert_eq!(leading_short_turn_run(none), 0);
    }
}
