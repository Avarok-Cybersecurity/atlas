// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::http;

/// A sandbox nobody else is using. There is no `tempfile` in this crate's
/// dependency set, and adding one to run a few tests is not worth the supply
/// chain — the crate is deliberately dependency-light.
pub fn sandbox(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-agent-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn cfg(sandbox: PathBuf) -> AgentConfig {
    AgentConfig {
        sandbox,
        max_turns: 1,
        command_timeout: Duration::from_secs(20),
        request_timeout: Duration::from_secs(1),
        max_tokens: 16,
        cargo_target_dir: None,
    }
}

// ── containment ────────────────────────────────────────────────────

#[test]
fn paths_cannot_escape_the_sandbox() {
    let sb = Path::new("/tmp/sandbox");
    assert_eq!(resolve(sb, "src/main.rs").unwrap(), sb.join("src/main.rs"));
    assert_eq!(resolve(sb, "./Cargo.toml").unwrap(), sb.join("Cargo.toml"));
    assert!(resolve(sb, "../../etc/passwd").is_err());
    assert!(resolve(sb, "/etc/passwd").is_err());
    assert!(resolve(sb, "src/../../../etc/shadow").is_err());
}

#[test]
fn an_absolute_path_inside_the_sandbox_is_accepted() {
    // opencode's file tools ask for absolute paths and its environment block
    // hands the model the working directory to build them from, so the
    // prompt-compliant call must not be rejected.
    let sb = Path::new("/tmp/sandbox");
    assert_eq!(
        resolve(sb, "/tmp/sandbox/src/main.rs").unwrap(),
        sb.join("src/main.rs")
    );
    assert!(resolve(sb, "/tmp/sandbox/../escape").is_err());
}

#[cfg(unix)]
#[test]
fn a_symlink_out_of_the_sandbox_is_not_a_way_out_of_it() {
    // Lexically `escape/passwd` never leaves the sandbox. `ln -s /etc escape`
    // is one bash call away, and `read`/`write`/`edit` follow symlinks, so the
    // lexical rule alone let the file tools read and overwrite anything this
    // user owns.
    let sb = sandbox("symlink-escape");
    std::os::unix::fs::symlink("/etc", sb.join("escape")).unwrap();
    assert!(resolve(&sb, "escape/passwd").is_err());
    assert!(resolve(&sb, "escape").is_err());
    // A symlink that stays inside is still usable, and an ordinary path that
    // does not exist yet — every `write` of a new file — must still resolve.
    std::fs::create_dir(sb.join("src")).unwrap();
    std::os::unix::fs::symlink(sb.join("src"), sb.join("inside")).unwrap();
    assert!(resolve(&sb, "inside/main.rs").is_ok());
    assert!(resolve(&sb, "src/deep/new.rs").is_ok());
}

// ── truncation ─────────────────────────────────────────────────────

#[test]
fn truncation_keeps_both_ends() {
    let text = format!("{}ERROR_AT_END", "a".repeat(20_000));
    let t = truncate(&text);
    assert!(t.ends_with("ERROR_AT_END"), "tail must survive");
    assert!(t.starts_with("aaa"));
    assert!(t.contains("elided"));
    assert!(t.chars().count() < 20_100);
}

#[test]
fn output_at_or_below_the_cap_is_untouched() {
    assert_eq!(truncate("hello"), "hello");
    let at_cap = "x".repeat(MAX_TOOL_OUTPUT);
    assert_eq!(truncate(&at_cap), at_cap);
}

#[test]
fn truncation_caps_at_the_harness_output_cap_and_never_splits_a_char() {
    let text = "€".repeat(20_000);
    let cut = (MAX_TOOL_OUTPUT - shell::TEST_ELISION_NOTE) / 2;
    assert!(
        !text.is_char_boundary(cut),
        "fixture head cut is a character boundary"
    );
    assert!(
        !text.is_char_boundary(text.len() - cut),
        "fixture tail cut is a character boundary"
    );
    let t = truncate(&text);
    assert!(t.len() < MAX_TOOL_OUTPUT + 200, "{}", t.len());
    assert!(t.contains("characters elided from the middle"));
}

// ── prompt and wire format ─────────────────────────────────────────

#[test]
fn assistant_message_substitutes_empty_arguments_with_an_object() {
    let outcome = http::ChatOutcome {
        tool_calls: vec![http::ToolCall {
            id: String::new(),
            name: "bash".into(),
            arguments: String::new(),
        }],
        ..Default::default()
    };
    let m = assistant_message(&outcome, 0);
    assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{}");
    assert_eq!(m["tool_calls"][0]["id"], "call_0_0");
    assert!(m["content"].is_null());
}

#[test]
fn tool_call_ids_are_positional_and_never_the_servers() {
    // Atlas mints ids from a per-process counter, so the same turn of the same
    // work carries a different id depending on what that server did earlier —
    // measured: five identical requests, five distinct id sets, identical text.
    // Echoing it wrote a value from outside the run into the model's context.
    let outcome = http::ChatOutcome {
        tool_calls: vec![
            http::ToolCall {
                id: "call_0000000000000004".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            },
            http::ToolCall {
                id: "call_0000000000000005".into(),
                name: "read".into(),
                arguments: "{}".into(),
            },
        ],
        ..Default::default()
    };
    let m = assistant_message(&outcome, 3);
    assert_eq!(m["tool_calls"][0]["id"], "call_3_0");
    assert_eq!(m["tool_calls"][1]["id"], "call_3_1");
    // The pairing is what the server validates: the id on the assistant message
    // and the one on the matching tool reply must be the same string.
    assert_eq!(m["tool_calls"][1]["id"].as_str().unwrap(), call_id(3, 1));
    // Two turns never collide, so an old reply cannot pair with a new call.
    assert_ne!(call_id(3, 1), call_id(4, 1));
}

#[test]
fn the_system_prompt_is_the_harness_agent_prompt_plus_the_environment() {
    let p = system_prompt(Path::new("/tmp/run-03"), "Qwen/Qwen3.6-35B-A3B-FP8");
    assert!(p.starts_with("You are a coding assistant running locally on Atlas Spark."));
    // The line that keeps a thinking model from walking the session past the
    // window — the failure the harness header blames for the slow runs.
    assert!(p.contains("Keep thinking short (under 50 words)"));
    assert!(p.contains("Working directory: /tmp/run-03"));
    assert!(p.contains("Qwen/Qwen3.6-35B-A3B-FP8"));
    // Every tool the prompt advertises must actually exist, or the model
    // spends turns calling something that answers "unknown tool".
    for name in ["bash", "read", "write", "edit", "glob", "grep"] {
        assert!(p.contains(&format!("**{name}**")), "{name}");
        assert!(
            tool_schema()
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t["function"]["name"] == name),
            "{name}"
        );
    }
}

#[test]
fn the_gate_request_pins_sampling_messages_and_tools() {
    // This asserted `TEMPERATURE == 0.3` — opencode's own setting — until the
    // gate's bar (an exact 10/10 on two counts) made a sampled instrument
    // useless: the same binary measured 10/10 then 8/10. The deviation from the
    // ported harness is deliberate and is documented at the constant.
    const { assert!(TEMPERATURE == 0.0) };
    let messages = [json!({"role": "user", "content": "hi"})];
    let body = request_body("Qwen/Qwen3.6-35B-A3B-FP8", &messages, &tool_schema(), 8192);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["seed"], SEED);
    assert_eq!(body["model"], "Qwen/Qwen3.6-35B-A3B-FP8");
    assert_eq!(body["stream"], true);
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["messages"], json!(messages));
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"], tool_schema());
}

// ── context compaction ─────────────────────────────────────────────

#[test]
fn compaction_elides_the_oldest_tool_results_and_keeps_the_pairing() {
    let big = "x".repeat(20_000);
    let mut msgs = vec![json!({"role": "system", "content": "s"})];
    for i in 0..10 {
        msgs.push(json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": format!("c{i}")}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": big}));
    }
    let before = msgs.len();
    compact(&mut msgs);

    assert_eq!(
        msgs.len(),
        before,
        "a dropped tool reply is a 400, not a saving"
    );
    let total: usize = msgs
        .iter()
        .map(|m| m["content"].as_str().map_or(64, str::len))
        .sum();
    assert!(total <= HISTORY_BUDGET, "{total}");
    assert!(msgs[2]["content"].as_str().unwrap().contains("elided"));
    // The most recent results are what the model is working from.
    let last = msgs.last().unwrap()["content"].as_str().unwrap();
    assert_eq!(last.len(), big.len(), "the live window must survive intact");

    // Force more elision than the live-window rule permits. The budget remains
    // exceeded on purpose; preserving the four results the model is actively
    // using takes precedence over shrinking farther.
    let live = "y".repeat(HISTORY_BUDGET);
    let mut pressured = vec![json!({"role": "system", "content": "s"})];
    for i in 0..5 {
        pressured.push(json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": format!("p{i}")}]}));
        pressured.push(json!({"role": "tool", "tool_call_id": format!("p{i}"), "content": live}));
    }
    compact(&mut pressured);
    assert!(pressured[2]["content"].as_str().unwrap().contains("elided"));
    for i in 1..5 {
        assert_eq!(
            pressured[2 + 2 * i]["content"].as_str().unwrap().len(),
            live.len(),
            "live tool result {i} was compacted"
        );
    }
}

#[test]
fn a_session_below_the_history_budget_is_left_alone() {
    let mut msgs = vec![json!({"role": "system", "content": "s"})];
    for i in 0..5 {
        msgs.push(json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": format!("c{i}")}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": format!("c{i}"),
            "content": format!("small-{i}")}));
    }
    let before = msgs.clone();
    compact(&mut msgs);
    assert_eq!(msgs, before);
}

// ── shell ──────────────────────────────────────────────────────────

#[tokio::test]
async fn stderr_and_a_non_zero_exit_are_both_reported() {
    let c = cfg(std::env::temp_dir());
    let out = run_shell(&c, "echo hi; echo bad >&2; exit 7", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(out.contains("hi"), "stdout is missing: {out}");
    assert!(out.contains("bad"), "stderr is missing: {out}");
    assert!(
        out.contains("exit status: 7"),
        "the exact exit status is missing: {out}"
    );
}

#[tokio::test]
async fn a_backgrounded_process_holding_the_pipe_does_not_stall_the_command() {
    // The prompt tells the model to redirect a detached server's output; when it
    // forgets, waiting for end-of-pipe charged the whole command timeout to a
    // command that had already finished, and returned none of its output.
    let c = cfg(std::env::temp_dir());
    let started = std::time::Instant::now();
    let out = run_shell(&c, "sleep 25 & echo started", Duration::from_secs(20))
        .await
        .unwrap();
    assert!(out.contains("started"), "{out}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_timed_out_command_still_returns_what_it_printed() {
    let c = cfg(std::env::temp_dir());
    let out = run_shell(
        &c,
        "echo early; echo late >&2; sleep 30",
        Duration::from_millis(400),
    )
    .await
    .unwrap();
    assert!(
        out.contains("early"),
        "stdout before the kill is lost: {out}"
    );
    assert!(
        out.contains("late"),
        "stderr before the kill is lost: {out}"
    );
}

#[tokio::test]
async fn shell_output_is_normalised_before_it_is_truncated() {
    // The wiring, not the rules (those are `norm_tests`): every byte the bash
    // tool returns has been through the normaliser, because that is the only
    // path by which run-to-run noise enters the conversation.
    let c = cfg(std::env::temp_dir());
    let out = run_shell(
        &c,
        "echo '   Compiling pingpong v0.1.0 (/tmp/x)'; \
         echo '    Finished `test` profile [unoptimized] target(s) in 1.23s'; \
         echo 'kill: (1417733) - No such process'",
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(!out.contains("Compiling"), "{out}");
    assert!(out.contains("target(s) in <elapsed>"), "{out}");
    assert!(out.contains("kill: (<pid>)"), "{out}");

    // The payload fits under the model-output cap only after the progress line
    // is removed. Truncating first would leave a permanent elision marker and
    // different retained bytes for equivalent cold and warm builds.
    let out = run_shell(
        &c,
        "printf '%08000d\\n' 0; printf '   Compiling %0300d\\n' 0",
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(!out.contains("Compiling"), "{out}");
    assert!(!out.contains("elided from the middle"), "{out}");
    assert_eq!(out.len(), 8001);
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_detached_survivor_in_the_sandbox_is_reaped() {
    // The prompt tells the model to use `setsid`, so the server it leaves
    // behind is not our child and `kill_on_drop` never sees it; it would hold
    // its port into the next iteration.
    let sb = sandbox("reap");
    let mut victim = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("exec sleep 45")
        .current_dir(&sb)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false)
        .spawn()
        .unwrap();
    let pid = victim.id().unwrap();
    reap(&sb).await;
    let seen = tokio::time::timeout(Duration::from_secs(5), victim.wait()).await;
    assert!(seen.is_ok(), "pid {pid} survived the reap");
    // Nothing outside the sandbox is a candidate — this process included.
    assert!(std::path::Path::new("/proc").exists());
}

#[tokio::test]
async fn output_past_the_pipe_buffer_does_not_deadlock_the_writer() {
    // Draining only after the process exits would block a command that writes
    // more than the 64 KiB pipe buffer; it would be reported as timed out.
    let c = cfg(std::env::temp_dir());
    let out = run_shell(
        &c,
        "head -c 8000000 /dev/zero | tr '\\0' 'a'",
        Duration::from_secs(2),
    )
    .await
    .unwrap();
    assert!(!out.contains("timed out"), "{}", &out[..80.min(out.len())]);
    assert!(!out.contains("[exit"), "the writer did not finish: {out}");
    assert!(out.contains("elided"), "the cap still applies");
}

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
        emitted_unparsed_call(&degenerate),
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
    assert!(!emitted_unparsed_call(&done));

    // Prose that TALKS about tool calls without emitting the wire syntax is
    // ordinary text — the detector keys on the markers, not on the words.
    let talks_about_it = http::ChatOutcome {
        text: "I would normally call the bash function to curl the endpoint.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!emitted_unparsed_call(&talks_about_it));

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
    assert!(!emitted_unparsed_call(&parsed));
}
