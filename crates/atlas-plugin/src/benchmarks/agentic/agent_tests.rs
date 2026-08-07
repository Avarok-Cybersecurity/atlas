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
fn short_output_is_untouched() {
    assert_eq!(truncate("hello"), "hello");
}

#[test]
fn truncation_caps_at_the_harness_output_cap_and_never_splits_a_char() {
    let t = truncate(&"é".repeat(20_000));
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
    let m = assistant_message(&outcome);
    assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{}");
    assert_eq!(m["tool_calls"][0]["id"], "call_0");
    assert!(m["content"].is_null());
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
fn sampling_matches_the_harness_opencode_config() {
    // ~/.config/opencode/opencode.json, providers atlas1/atlas2/atlas3,
    // options.temperature. The 10/10 tier was NOT greedy.
    const { assert!(TEMPERATURE == 0.3) };
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
}

#[test]
fn a_short_session_is_left_alone() {
    let mut msgs = vec![
        json!({"role": "system", "content": "s"}),
        json!({"role": "tool", "tool_call_id": "c0", "content": "small"}),
    ];
    compact(&mut msgs);
    assert_eq!(msgs[1]["content"], "small");
}

// ── shell ──────────────────────────────────────────────────────────

#[tokio::test]
async fn a_hanging_command_is_killed_at_the_timeout() {
    let mut c = cfg(std::env::temp_dir());
    c.command_timeout = Duration::from_millis(300);
    // `exec` so the kill lands on the sleep itself and this test leaks nothing.
    let out = run_shell(&c, "exec sleep 30", c.command_timeout)
        .await
        .unwrap();
    assert!(out.contains("timed out"), "{out}");
}

#[tokio::test]
async fn stderr_and_a_non_zero_exit_are_both_reported() {
    let c = cfg(std::env::temp_dir());
    let out = run_shell(&c, "echo hi; echo bad >&2; exit 7", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(
        out.contains("hi") && out.contains("bad") && out.contains("exit"),
        "{out}"
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
        "head -c 400000 /dev/zero | tr '\\0' 'a'",
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(!out.contains("timed out"), "{}", &out[..80.min(out.len())]);
    assert!(out.contains("elided"), "the cap still applies");
}
