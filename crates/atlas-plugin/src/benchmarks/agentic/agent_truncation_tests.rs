// SPDX-License-Identifier: AGPL-3.0-only

//! The truncated-turn half of the agent tests, split from `agent_tests.rs` for
//! size. Its own section banner in that file was already the seam.

use super::*;
use crate::http;

// ── truncated turns ────────────────────────────────────────────────

#[test]
fn a_turn_cut_off_at_the_token_cap_is_not_a_turn_that_finished() {
    // The failure this encodes: the model loops inside the turn that writes
    // src/main.rs, hits max_tokens, and returns no tool call. Read as "the
    // agent stopped calling tools", the loop exits and the run scores 0/6 —
    // recorded as a task failure when it was a truncation.
    let cut_off = http::ChatOutcome {
        text: "fn main() {".into(),
        finish_reason: Some("length".into()),
        ..Default::default()
    };
    assert!(was_cut_off(&cut_off), "length + no tool call is resumable");

    // A natural stop with no tool calls IS the agent finishing, and must still
    // end the run — otherwise every completed run would be prodded to continue.
    let done = http::ChatOutcome {
        text: "All six steps pass.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!was_cut_off(&done), "a natural stop ends the run");

    // Truncation WITH a tool call is not this case: the call is actionable, so
    // the loop proceeds normally and nothing needs resuming.
    let cut_off_with_call = http::ChatOutcome {
        tool_calls: vec![http::ToolCall {
            id: String::new(),
            name: "write".into(),
            arguments: "{}".into(),
        }],
        finish_reason: Some("length".into()),
        ..Default::default()
    };
    assert!(!was_cut_off(&cut_off_with_call));

    // A server that reports no finish_reason at all must not be guessed at.
    let unknown = http::ChatOutcome::default();
    assert!(!was_cut_off(&unknown));
}

#[test]
fn a_tool_call_left_in_the_message_body_is_not_a_turn_that_finished() {
    // Verbatim from gate run 7 at 66b20718, which is how this was found: after
    // five thinking-loop watchdog fires the model repeated a sentence, emitted
    // its next call as raw syntax inside the CONTENT, and stopped. The server's
    // parser rejected the malformed block, so `tool_calls` came back empty with
    // finish_reason `stop` — indistinguishable, to the old code, from the agent
    // deciding it was done. The curl never ran, the loop exited, the server was
    // never killed, and the run lost `tore_down`: 9/10 on a 10/10 gate.
    let degenerate = http::ChatOutcome {
        text: "Now let me test the other endpoints:\n\
               Now let me test the other endpoints:\n\
               <tool_call>\n<function=bash>\n<parameter=command>\n\
               timeout 15 curl -s http://localhost:3001/pong\n\
               </parameter>\n</function>\n</tool_call>oints:"
            .into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(
        tools::emitted_unparsed_call(&degenerate),
        "tool-call syntax in the body with nothing parsed must be re-asked"
    );
    // ★ and it is NOT the truncation case — that is why it needed its own arm.
    assert!(!was_cut_off(&degenerate));

    // Plain prose still ends the run. If this ever returns true the gate can
    // never terminate: every finished run would be prodded to continue forever.
    let done = http::ChatOutcome {
        text: "All six steps pass. The server is stopped.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!tools::emitted_unparsed_call(&done));

    // Prose that TALKS about tool calls without emitting the wire syntax is
    // ordinary text — the detector keys on the markers, not on the words.
    let talks_about_it = http::ChatOutcome {
        text: "I would normally call the bash function to curl the endpoint.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!tools::emitted_unparsed_call(&talks_about_it));

    // A turn whose call DID parse never reaches this branch, however much
    // syntax the accompanying prose quotes.
    let parsed = http::ChatOutcome {
        text: "running <tool_call> now".into(),
        tool_calls: vec![http::ToolCall {
            id: String::new(),
            name: "bash".into(),
            arguments: "{}".into(),
        }],
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!tools::emitted_unparsed_call(&parsed));
}

// ── preserve-thinking history (ATLAS_AGENTIC_PRESERVE_THINKING) ─────

/// Old reasoning is elided to a MARKER, never dropped: a turn with the field
/// removed renders an empty `<think></think>` wrapper, which is the
/// empty-think poisoning the flag exists to avoid. Recent turns keep their
/// full reasoning because that is what the model is working from.
#[test]
fn compaction_elides_old_reasoning_to_a_marker_and_keeps_the_recent() {
    let big = "r".repeat(20_000);
    let mut msgs = vec![json!({"role": "user", "content": "task"})];
    for i in 0..10 {
        msgs.push(json!({"role": "assistant", "content": Value::Null,
            "reasoning_content": big,
            "tool_calls": [{"id": format!("c{i}")}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": "ok"}));
    }
    compact(&mut msgs);

    let think: Vec<&str> = msgs
        .iter()
        .filter_map(|m| m["reasoning_content"].as_str())
        .collect();
    assert_eq!(think.len(), 10, "reasoning must never be removed outright");
    assert!(
        think[0].contains("elided"),
        "the oldest reasoning should be elided: {}",
        &think[0][..think[0].len().min(60)]
    );
    for kept in think.iter().rev().take(LIVE_REASONING) {
        assert_eq!(*kept, big, "the live window keeps full reasoning");
    }
    let total: usize = msgs
        .iter()
        .map(|m| {
            m["content"].as_str().map_or(64, str::len)
                + m["reasoning_content"].as_str().map_or(0, str::len)
        })
        .sum();
    assert!(total <= HISTORY_BUDGET, "{total}");
}
