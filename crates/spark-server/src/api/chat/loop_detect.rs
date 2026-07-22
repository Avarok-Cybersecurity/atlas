// SPDX-License-Identifier: AGPL-3.0-only
//
// Loop detection + spinning detection + task-pin re-anchor block,
// extracted from `chat::chat_completions_inner` (wave 4g).
//
// Inputs: the request (so we can scan history) and the in-progress
// `messages` vec (mutated to inject hints / anchors). Outputs are
// returned via `LoopDetectOut` so the orchestrator can wire them
// into the downstream sampling-bias logic.

use crate::ir::{Message, Role};

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

pub(super) fn check_loops(messages: &[Message], tools_active: bool) -> LoopDetectOut {
    let mut suppress_tool_call = false;
    let mut tool_call_repeat_count: usize = 0;

    if !tools_active {
        return LoopDetectOut {
            suppress_tool_call,
            tool_call_repeat_count,
        };
    }

    // IR tool-call arguments are structured JSON; the signature hasher
    // wants strings. Re-serialization is deterministic per turn, so
    // turn-to-turn similarity (the only thing the detector compares) is
    // unaffected by the wire's original whitespace.
    let signatures: Vec<crate::loop_detector::Signature> = messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::Assistant)
        .map(|m| {
            let text = m.text();
            let owned: Vec<(String, String)> = m
                .tool_calls
                .iter()
                .map(|tc| (tc.name.clone(), tc.arguments.to_string()))
                .collect();
            let calls: Vec<(&str, &str)> = owned
                .iter()
                .map(|(n, a)| (n.as_str(), a.as_str()))
                .collect();
            crate::loop_detector::Signature::build(&text, calls)
        })
        .take(8)
        .collect();
    let verdict = crate::loop_detector::detect(&signatures);

    // P1-5 (2026-07-09): pair each assistant turn with the tool
    // results that follow it, so the exact-match failing-call fast
    // path and the Suppress failing-gate can see error-shaped
    // outcomes — `Signature` only covers assistant content, never the
    // role=="tool" results. Forward scan, then reversed to
    // newest-first (same window as the signatures above).
    // (unit, saw_result, all_err, result_text)
    let mut outcomes_fwd: Vec<(Option<String>, bool, bool, String)> = Vec::new();
    for m in messages.iter() {
        match m.role {
            Role::Assistant => {
                // IR arguments are structured JSON; `to_string()` is
                // deterministic per turn, so the exact-repeat comparison
                // is unaffected (same reasoning as the signatures above).
                let unit = if m.tool_calls.is_empty() {
                    None
                } else {
                    let mut s = String::new();
                    for tc in &m.tool_calls {
                        if !s.is_empty() {
                            s.push('\u{1e}');
                        }
                        s.push_str(&tc.name);
                        s.push('\u{1f}');
                        s.push_str(&tc.arguments.to_string());
                    }
                    Some(s)
                };
                outcomes_fwd.push((unit, false, true, String::new()));
            }
            Role::Tool => {
                if let Some(last) = outcomes_fwd.last_mut() {
                    let text = m.text();
                    last.1 = true;
                    last.2 &= crate::hint_injector::looks_like_error(&text);
                    // P1-5b: capture the result text (bounded) so the
                    // Suppress gate can measure result-to-result progress.
                    let take = text.chars().take(2000);
                    last.3.extend(take);
                    last.3.push('\n');
                }
            }
            _ => {}
        }
    }
    let call_outcomes: Vec<crate::loop_detector::CallOutcome> = outcomes_fwd
        .into_iter()
        .rev()
        .take(8)
        .map(
            |(unit, saw_result, all_err, result_text)| crate::loop_detector::CallOutcome {
                call_unit: unit,
                failing: saw_result && all_err,
                result_unit: if saw_result { Some(result_text) } else { None },
            },
        )
        .collect();
    let exact_failing_run = crate::loop_detector::detect_exact_failing_repeat(&call_outcomes);

    // Spinning detection — independent signal: if the model has
    // produced ≥5 consecutive short, low-content responses,
    // something is structurally wrong even if no two are similar.
    let mut recent_short: usize = 0;
    for m in messages.iter().rev() {
        if m.role != Role::Assistant {
            continue;
        }
        let tool_args_len: usize = m
            .tool_calls
            .iter()
            .map(|tc| tc.arguments.to_string().len())
            .sum();
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
        let made_tool_call = !m.tool_calls.is_empty();
        let is_substantial = made_tool_call || m.text().len() >= 500 || tool_args_len >= 100;
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
            // P1-5 (2026-07-09): corrective, not masking — when the
            // repeated unit is a FAILING call (every following tool
            // result error-shaped), do NOT hard-mask <tool_call>: it
            // is the model's only escape action while it loops on a
            // failing call (45k collapse). Suppress stays unchanged
            // for repeated SUCCEEDING calls (BW1/SPINFIX true
            // infinite-loop case). The server-authored corrective
            // feedback for the failing call itself ships in P0-3.
            let failing_repeat =
                crate::loop_detector::recent_calls_all_failing(&call_outcomes, *run_length);
            // P1-5b (2026-07-09): progress gate. A productive
            // similar-call cycle (cargo check → fix → check) has
            // DIFFERENT results each round; masking <tool_call> there
            // cornered the model into an empty "..." EOS at 57k. Only
            // a loop whose results are ALSO near-identical (no new
            // information) keeps the hard-mask.
            let progressing = crate::loop_detector::recent_results_progressing(
                &call_outcomes,
                (*run_length).max(2),
            );
            if progressing && !failing_repeat {
                tracing::warn!(
                    score = *score,
                    run_length = *run_length,
                    channel = channel.name(),
                    "Loop detector → SUPPRESS on PROGRESSING cycle: results differ                      round-to-round; <tool_call> hard-mask SKIPPED (soft bias decay only)"
                );
            }
            if failing_repeat {
                tracing::warn!(
                    score = *score,
                    run_length = *run_length,
                    channel = channel.name(),
                    "Loop detector → SUPPRESS on FAILING repeated call: <tool_call> hard-mask \
                     SKIPPED (escape action stays available); soft bias decay only"
                );
            } else {
                tracing::warn!(
                    score = *score,
                    run_length = *run_length,
                    channel = channel.name(),
                    "Loop detector → SUPPRESS: hard-mask <tool_call> for one turn"
                );
            }
            suppress_tool_call = !failing_repeat && !progressing && !loop_suppress_disabled();
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
            // P1-5 (2026-07-09): log reworded — the old text claimed
            // "inject progress notice" but nothing is injected; the
            // only effect is the soft <tool_call> logit-bias decay
            // driven by tool_call_repeat_count (sampling_setup.rs).
            tracing::info!(
                score = *score,
                run_length = *run_length,
                channel = channel.name(),
                "Loop detector → HINT: soft <tool_call> bias decay via tool_call_repeat_count \
                 (no hard-mask, nothing injected)"
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
    // P1-5 (2026-07-09) exact-match fast path: byte-identical FAILING
    // calls whose short units sit below MIN_CHANNEL_TOKENS are
    // invisible to detect() (45k collapse: 3-token empty-call
    // signature, escalation ~4 turns late). Treat as a loop candidate
    // NOW, but do NOT hard-mask <tool_call> (the escape action) —
    // apply only the soft bias decay via tool_call_repeat_count.
    if let Some(run) = exact_failing_run
        && run > tool_call_repeat_count
    {
        tracing::warn!(
            run_length = run,
            "Loop detector → FAILING-REPEAT fast path: byte-identical failing tool calls; \
             soft <tool_call> bias decay only (no hard-mask — escape action stays available)"
        );
        tool_call_repeat_count = run;
        crate::metrics::LOOP_DETECTOR_VERDICTS
            .with_label_values(&["failing_repeat", "tools", if spinning { "1" } else { "0" }])
            .inc();
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
