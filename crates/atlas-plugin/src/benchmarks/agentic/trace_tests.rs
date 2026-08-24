// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

use crate::http::ToolCall;

fn sandbox(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("atlas-trace-{}-{name}", std::process::id()))
        .join("run-07");
    let _ = std::fs::remove_dir_all(dir.parent().expect("has a parent"));
    std::fs::create_dir_all(&dir).expect("sandbox");
    dir
}

fn outcome() -> ChatOutcome {
    ChatOutcome {
        reasoning: "check the port".into(),
        text: "running the tests".into(),
        tool_calls: vec![ToolCall {
            id: "call_0".into(),
            name: "bash".into(),
            arguments: "{\"command\":\"cargo test\"}".into(),
        }],
        finish_reason: Some("tool_calls".into()),
        ..ChatOutcome::default()
    }
}

#[test]
fn the_trace_lands_beside_the_sandbox_never_inside_it() {
    // Inside, the agent's own glob/grep/read would show it the transcript of
    // the turn that just happened, and the scorer would walk it.
    let sb = sandbox("outside");
    let trace = Trace::start(&sb, "do the thing");
    trace.turn(0, &outcome());
    trace.result("bash", "test result: ok. 1 passed");

    let path = sb.with_file_name("run-07.trajectory.txt");
    assert!(path.is_file(), "no trace at {}", path.display());
    assert_eq!(
        std::fs::read_dir(&sb).expect("sandbox readable").count(),
        0,
        "the sandbox must stay exactly as the agent left it"
    );
}

#[test]
fn a_turn_records_everything_that_could_diverge() {
    let sb = sandbox("content");
    let trace = Trace::start(&sb, "do the thing");
    trace.turn(1, &outcome());
    trace.result("bash", "test result: ok. 1 passed");
    let text = std::fs::read_to_string(sb.with_file_name("run-07.trajectory.txt")).expect("trace");

    assert_eq!(
        text,
        "[prompt]\ndo the thing\n\
         \n── turn 2 ───────────────────────────────\n\
         [reasoning]\ncheck the port\n\
         [text]\nrunning the tests\n\
         [call call_0] bash {\"command\":\"cargo test\"}\n\
         [finish] tool_calls\n\
         [result bash]\ntest result: ok. 1 passed\n"
    );
}

#[test]
fn a_new_run_replaces_the_previous_trace() {
    // Two runs concatenated would diff against nothing.
    let sb = sandbox("truncate");
    let first = Trace::start(&sb, "prompt one");
    first.result("bash", "OLD_RUN_MARKER");
    let second = Trace::start(&sb, "prompt one");
    second.result("bash", "new");
    let text = std::fs::read_to_string(sb.with_file_name("run-07.trajectory.txt")).expect("trace");
    assert!(!text.contains("OLD_RUN_MARKER"), "{text}");
    assert!(text.contains("new"), "{text}");
}

#[test]
fn an_unwritable_trace_location_does_not_fail_the_run() {
    let missing = Path::new("/proc/atlas-does-not-exist/run-00");
    let trace = Trace::start(missing, "prompt");
    trace.turn(0, &outcome());
    trace.result("bash", "anything");
}
