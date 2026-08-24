// SPDX-License-Identifier: AGPL-3.0-only

//! Loader tests against the DOCUMENTED trajectory format (upstream README +
//! `AgenticInferenceDataset` semantics at 7935df4). No real dataset exists to
//! test against, so these prove the port of the documented behaviour — the
//! first calibration run is where reality gets a vote.

use serde_json::{Value, json};

use super::*;
use crate::benchmarks::mlperf_agentic::scoring::Domain;

fn write(name: &str, rows: &[Value]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "atlas-mlperf-ds-{name}-{}.jsonl",
        std::process::id()
    ));
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(&p, text).unwrap();
    p
}

/// A well-formed coding conversation exercising every message shape: system,
/// tool_calls with no content, tool_results expansion, plain assistant.
fn coding_rows(id: &str) -> Vec<Value> {
    vec![
        json!({"conversation_id": id, "turn": 1, "role": "user", "system": "be an agent",
               "content": "fix the bug", "tools": [{"type": "function"}]}),
        json!({"conversation_id": id, "turn": 2, "role": "assistant", "reasoning_content": "look first",
               "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": {"cmd": "grep -r bug src"}}}]}),
        json!({"conversation_id": id, "turn": 3, "role": "tool",
               "tool_results": [{"tool_call_id": "c1", "content": "src/x.py: bug"}], "delay_seconds": 1.5}),
        json!({"conversation_id": id, "turn": 4, "role": "assistant",
               "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": {"cmd": "python fix.py"}}}]}),
    ]
}

fn workflow_rows(id: &str) -> Vec<Value> {
    vec![
        json!({"conversation_id": id, "turn": 1, "role": "user", "system": "support bot", "content": "where is my order"}),
        json!({"conversation_id": id, "turn": 2, "role": "assistant", "content": "intent: I042", "intent_codes": ["I042"]}),
    ]
}

#[test]
fn prebuilt_messages_are_teacher_forced_snapshots() {
    let p = write("snap", &coding_rows("proj-1"));
    let convs = load(&p, &DrawSpec::all()).unwrap();
    assert_eq!(convs.len(), 1);
    let c = &convs[0];
    assert_eq!(c.domain, Domain::Coding);
    assert_eq!(c.client_turns.len(), 2, "turns 1 (user) and 3 (tool)");

    // Turn 1: system + the user row, with dataset-only fields removed.
    let t1 = &c.client_turns[0];
    assert_eq!(t1.turn, 1);
    assert_eq!(
        t1.messages,
        vec![
            json!({"role": "system", "content": "be an agent"}),
            json!({"role": "user", "content": "fix the bug"}),
        ]
    );

    // Turn 3: system + user + RECORDED assistant (content: null filled in) +
    // the tool_results expanded to a tool message.
    let t3 = &c.client_turns[1];
    assert_eq!(t3.turn, 3);
    assert_eq!(
        t3.messages,
        vec![
            json!({"role": "system", "content": "be an agent"}),
            json!({"role": "user", "content": "fix the bug"}),
            json!({
                "role": "assistant",
                "content": null,
                "reasoning_content": "look first",
                "tool_calls": [{
                    "id": "c1",
                    "function": {
                        "name": "bash",
                        "arguments": {"cmd": "grep -r bug src"},
                    },
                }],
            }),
            json!({
                "role": "tool",
                "tool_call_id": "c1",
                "content": "src/x.py: bug",
            }),
        ]
    );

    // Ground truth: the immediately-following assistant row, raw.
    let source = coding_rows("proj-1");
    assert_eq!(t1.ground_truth.as_ref(), Some(&source[1]));
    assert_eq!(t3.ground_truth.as_ref(), Some(&source[3]));

    // Tools come from the first user row.
    assert_eq!(c.tools, Some(json!([{"type": "function"}])));
}

#[test]
fn a_trailing_client_turn_has_no_ground_truth() {
    let mut rows = workflow_rows("sim_001");
    rows.push(
        json!({"conversation_id": "sim_001", "turn": 3, "role": "user", "content": "thanks"}),
    );
    let p = write("trailing", &rows);
    let convs = load(&p, &DrawSpec::all()).unwrap();
    assert_eq!(convs[0].domain, Domain::Workflow);
    let turns = &convs[0].client_turns;
    assert_eq!(turns.len(), 2);
    assert!(turns[0].ground_truth.is_some());
    assert!(turns[1].ground_truth.is_none());
}

#[test]
fn the_draw_takes_the_first_k_per_domain_in_file_order() {
    let mut rows = Vec::new();
    rows.extend(coding_rows("proj-a"));
    rows.extend(workflow_rows("sim_001"));
    rows.extend(coding_rows("proj-b"));
    rows.extend(coding_rows("proj-c"));
    rows.extend(workflow_rows("sim_002"));
    let p = write("draw", &rows);
    let convs = load(
        &p,
        &DrawSpec {
            coding: 2,
            workflow: 1,
        },
    )
    .unwrap();
    let ids: Vec<&str> = convs.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["proj-a", "sim_001", "proj-b"],
        "head(K), not a sample"
    );

    // 0 takes everything.
    assert_eq!(load(&p, &DrawSpec::all()).unwrap().len(), 5);
}

#[test]
fn the_fingerprint_tracks_draw_content_and_order() {
    let mut rows = coding_rows("proj-a");
    rows.extend(coding_rows("proj-b"));
    rows.extend(workflow_rows("sim_001"));
    let p = write("fp", &rows);
    let both = load(&p, &DrawSpec::all()).unwrap();
    let same = load(
        &p,
        &DrawSpec {
            coding: 2,
            workflow: 1,
        },
    )
    .unwrap();
    assert_eq!(
        draw_fingerprint(&both),
        draw_fingerprint(&same),
        "equal draws, equal identity"
    );

    let smaller = load(
        &p,
        &DrawSpec {
            coding: 1,
            workflow: 1,
        },
    )
    .unwrap();
    assert_ne!(
        draw_fingerprint(&both),
        draw_fingerprint(&smaller),
        "a different draw must never fingerprint the same"
    );

    let mut changed_rows = rows.clone();
    changed_rows[0]["content"] = json!("fix a different bug");
    let changed_path = write("fp-content", &changed_rows);
    let changed = load(&changed_path, &DrawSpec::all()).unwrap();
    assert_ne!(
        draw_fingerprint(&both),
        draw_fingerprint(&changed),
        "prompt content is part of the draw identity"
    );

    let mut changed_tools_rows = rows.clone();
    changed_tools_rows[0]["tools"] = json!([{"type": "different-function-schema"}]);
    let changed_tools_path = write("fp-tools", &changed_tools_rows);
    let changed_tools = load(&changed_tools_path, &DrawSpec::all()).unwrap();
    assert_ne!(
        draw_fingerprint(&both),
        draw_fingerprint(&changed_tools),
        "tool schemas are part of the draw identity"
    );

    let mut changed_truth_rows = rows.clone();
    changed_truth_rows[3]["tool_calls"][0]["function"]["arguments"]["cmd"] =
        json!("python different.py");
    let changed_truth_path = write("fp-ground-truth", &changed_truth_rows);
    let changed_truth = load(&changed_truth_path, &DrawSpec::all()).unwrap();
    assert_ne!(
        draw_fingerprint(&both),
        draw_fingerprint(&changed_truth),
        "terminal ground truth is part of the draw identity"
    );

    let mut reversed = both.clone();
    reversed.reverse();
    assert_ne!(
        draw_fingerprint(&both),
        draw_fingerprint(&reversed),
        "conversation order is part of the draw identity"
    );
    assert_eq!(draw_fingerprint(&both).len(), 64, "full sha256 hex");
}

#[test]
fn interleaved_conversations_are_refused() {
    let a = coding_rows("proj-a");
    let b = coding_rows("proj-b");
    let rows = vec![a[0].clone(), b[0].clone(), a[1].clone()];
    let p = write("interleave", &rows);
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("not consecutive"), "{err}");
}

#[test]
fn turn_numbers_must_be_exactly_one_to_n() {
    let mut rows = workflow_rows("sim_001");
    rows[1]["turn"] = json!(5);
    let p = write("turns", &rows);
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("turn numbers must be exactly"), "{err}");

    let mut reversed = workflow_rows("sim_002");
    reversed.reverse();
    let p = write("turn-order", &reversed);
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("file order"), "{err}");
}

#[test]
fn the_role_grammar_is_enforced() {
    // assistant first.
    let p = write(
        "grammar1",
        &[json!({"conversation_id": "c", "turn": 1, "role": "assistant", "content": "hi"})],
    );
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("after \"start\""), "{err}");

    // tool row without a tool-calling assistant before it.
    let p = write(
        "grammar2",
        &[
            json!({"conversation_id": "c", "turn": 1, "role": "user", "content": "hi"}),
            json!({"conversation_id": "c", "turn": 2, "role": "assistant", "content": "plain"}),
            json!({"conversation_id": "c", "turn": 3, "role": "tool",
                   "tool_results": [{"tool_call_id": "x", "content": "y"}]}),
        ],
    );
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("non-empty tool_calls"), "{err}");

    // tool row without results.
    let mut rows = coding_rows("proj-a");
    rows[2] = json!({"conversation_id": "proj-a", "turn": 3, "role": "tool"});
    let p = write("grammar3", &rows);
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("non-empty tool_results"), "{err}");

    // A client turn must be answered before another client turn begins.
    let p = write(
        "grammar4",
        &[
            json!({"conversation_id": "c", "turn": 1, "role": "user", "content": "one"}),
            json!({"conversation_id": "c", "turn": 2, "role": "user", "content": "two"}),
        ],
    );
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("after \"user\""), "{err}");

    // Undocumented roles cannot silently disappear from formatted history.
    let p = write(
        "grammar5",
        &[
            json!({"conversation_id": "c", "turn": 1, "role": "user", "content": "one"}),
            json!({"conversation_id": "c", "turn": 2, "role": "developer", "content": "two"}),
        ],
    );
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("role \"developer\""), "{err}");
}

#[test]
fn conversation_ids_must_be_printable_ascii_strings() {
    let p = write(
        "ids",
        &[json!({"conversation_id": "sim_00\u{e9}", "turn": 1, "role": "user", "content": "x"})],
    );
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("non-printable or non-ASCII"), "{err}");

    let p = write(
        "ids2",
        &[json!({"turn": 1, "role": "user", "content": "x"})],
    );
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("non-empty string"), "{err}");

    for (name, id) in [("empty", json!("")), ("control", json!("sim\n001"))] {
        let p = write(
            name,
            &[json!({"conversation_id": id, "turn": 1, "role": "user", "content": "x"})],
        );
        let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
        assert!(
            err.contains("non-empty string") || err.contains("non-printable or non-ASCII"),
            "{err}"
        );
    }
}

#[test]
fn a_malformed_line_names_its_line_number() {
    let p = write("bad", &coding_rows("proj-a"));
    let mut text = std::fs::read_to_string(&p).unwrap();
    text.push_str("\n{not json}\n");
    std::fs::write(&p, text).unwrap();
    let err = load(&p, &DrawSpec::all()).unwrap_err().to_string();
    assert!(err.contains("line 6"), "{err}");
}
