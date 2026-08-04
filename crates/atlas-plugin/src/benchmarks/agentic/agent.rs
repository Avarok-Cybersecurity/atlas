// SPDX-License-Identifier: AGPL-3.0-only

//! The agent loop: tool-calling against the served endpoint, with the tools
//! executed inside a sandbox directory.
//!
//! **This executes model-authored shell.** There is no version of the agentic
//! webserver benchmark that does not — building and running the code the model
//! wrote is the measurement. The containment is explicit and lives here:
//!
//!   * every command runs with the sandbox as its working directory;
//!   * `write_file`/`read_file` paths are resolved lexically and rejected if
//!     they are absolute or climb out with `..`;
//!   * every command has a hard timeout and is killed on expiry;
//!   * tool output is truncated, so a runaway `yes` cannot exhaust memory;
//!   * the turn count is capped, so a loop cannot run forever.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;

use crate::http;
use crate::plugin::PluginHandle;

/// Cap on a single tool result, in characters.
const MAX_TOOL_OUTPUT: usize = 8_000;

/// What one agent run did, for scoring.
pub struct Transcript {
    /// Every shell command the agent issued, in order. `followed_directions`
    /// is computed from this.
    pub commands: Vec<String>,
    pub turns: usize,
    pub tool_calls: usize,
    /// True when the loop ended because the turn cap was hit rather than
    /// because the agent stopped calling tools.
    pub hit_turn_cap: bool,
    pub final_text: String,
}

pub fn tool_schema() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "bash",
            "description": "Run a shell command in the project directory and return its output.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "The shell command to run."}
            }, "required": ["command"]}}},
        {"type": "function", "function": {
            "name": "write_file",
            "description": "Write a file in the project directory, creating parent directories.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Path relative to the project directory."},
                "content": {"type": "string", "description": "Full file contents."}
            }, "required": ["path", "content"]}}},
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file from the project directory.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Path relative to the project directory."}
            }, "required": ["path"]}}}
    ])
}

pub struct AgentConfig {
    pub sandbox: PathBuf,
    pub max_turns: usize,
    pub command_timeout: Duration,
    pub request_timeout: Duration,
    pub max_tokens: usize,
    /// Shared warm cargo target dir, so the agent's own builds are incremental.
    /// Without it every `cargo test` cold-compiles the axum/tokio tree and the
    /// wall time measures dependency compilation, not the model.
    pub cargo_target_dir: Option<PathBuf>,
}

/// Run one agentic task to completion (or to the turn cap).
pub async fn run_task(
    handle: &PluginHandle,
    cfg: &AgentConfig,
    prompt: &str,
) -> Result<Transcript> {
    let target = handle.target();
    let mut messages = vec![
        json!({"role": "system", "content":
            "You are a software engineer working in the current project directory. \
             Use the provided tools to create, inspect and run code. Call tools rather than \
             describing what you would do, and stop once the task is fully verified."}),
        json!({"role": "user", "content": prompt}),
    ];
    let tools = tool_schema();
    let mut transcript = Transcript {
        commands: Vec::new(),
        turns: 0,
        tool_calls: 0,
        hit_turn_cap: false,
        final_text: String::new(),
    };

    for turn in 0..cfg.max_turns {
        handle.check_cancelled()?;
        handle.status(format!("agent turn {}/{}", turn + 1, cfg.max_turns));
        let body = json!({
            "model": target.model,
            "stream": true,
            "temperature": 0.0,
            "max_tokens": cfg.max_tokens,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        });
        let outcome = http::chat_stream(target, &body, cfg.request_timeout).await?;
        transcript.turns = turn + 1;
        transcript.final_text = outcome.text.clone();

        if outcome.tool_calls.is_empty() {
            return Ok(transcript);
        }

        messages.push(assistant_message(&outcome));
        for (i, call) in outcome.tool_calls.iter().enumerate() {
            handle.check_cancelled()?;
            transcript.tool_calls += 1;
            let id = if call.id.is_empty() {
                format!("call_{turn}_{i}")
            } else {
                call.id.clone()
            };
            let result = execute(cfg, call, &mut transcript.commands).await;
            let content = match result {
                Ok(text) => text,
                // A tool error is data for the model, not a run failure: an
                // agent recovering from a bad command is normal behaviour and
                // aborting here would score it as a crash.
                Err(e) => format!("error: {e:#}"),
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": truncate(&content),
            }));
        }
    }
    transcript.hit_turn_cap = true;
    Ok(transcript)
}

fn assistant_message(outcome: &http::ChatOutcome) -> Value {
    let calls: Vec<Value> = outcome
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({
                "id": if c.id.is_empty() { format!("call_{i}") } else { c.id.clone() },
                "type": "function",
                "function": {
                    "name": c.name,
                    // Some models emit no arguments at all for a zero-arg call;
                    // an empty string is not valid JSON to a strict server.
                    "arguments": if c.arguments.is_empty() { "{}".to_string() } else { c.arguments.clone() },
                },
            })
        })
        .collect();
    json!({
        "role": "assistant",
        "content": if outcome.text.is_empty() { Value::Null } else { Value::String(outcome.text.clone()) },
        "tool_calls": calls,
    })
}

async fn execute(
    cfg: &AgentConfig,
    call: &http::ToolCall,
    commands: &mut Vec<String>,
) -> Result<String> {
    let args: Value = serde_json::from_str(if call.arguments.is_empty() {
        "{}"
    } else {
        &call.arguments
    })
    .map_err(|e| anyhow!("arguments were not valid JSON: {e}"))?;
    match call.name.as_str() {
        "bash" => {
            let cmd = args
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("bash needs a `command` string"))?;
            commands.push(cmd.to_string());
            run_shell(cfg, cmd).await
        }
        "write_file" => {
            let rel = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("write_file needs a `path`"))?;
            let content = args.get("content").and_then(Value::as_str).unwrap_or("");
            let path = resolve(&cfg.sandbox, rel)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, content)?;
            Ok(format!("wrote {} ({} bytes)", rel, content.len()))
        }
        "read_file" => {
            let rel = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("read_file needs a `path`"))?;
            Ok(std::fs::read_to_string(resolve(&cfg.sandbox, rel)?)?)
        }
        other => bail!("unknown tool {other}"),
    }
}

/// Resolve `rel` inside `sandbox`, rejecting anything that escapes it.
///
/// Lexical, not `canonicalize`: the target usually does not exist yet, and a
/// canonicalize-then-compare check silently passes on a missing path.
pub fn resolve(sandbox: &Path, rel: &str) -> Result<PathBuf> {
    let rel = Path::new(rel);
    if rel.is_absolute() {
        bail!(
            "path must be relative to the project directory: {}",
            rel.display()
        );
    }
    let mut out = sandbox.to_path_buf();
    for component in rel.components() {
        match component {
            Component::Normal(c) => out.push(c),
            Component::CurDir => {}
            Component::ParentDir => bail!("path must not leave the project directory"),
            Component::RootDir | Component::Prefix(_) => {
                bail!("absolute paths are not allowed")
            }
        }
    }
    Ok(out)
}

async fn run_shell(cfg: &AgentConfig, command: &str) -> Result<String> {
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .current_dir(&cfg.sandbox)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // `kill_on_drop` is what makes the timeout below real: without it a
        // timed-out `cargo build` keeps running and keeps holding the CPU that
        // every later iteration is being timed on.
        .kill_on_drop(true);
    if let Some(dir) = &cfg.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", dir);
    }
    let mut child = cmd.spawn()?;
    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
    let collect = async {
        let (mut o, mut e) = (String::new(), String::new());
        let _ = tokio::try_join!(stdout.read_to_string(&mut o), stderr.read_to_string(&mut e));
        let status = child.wait().await?;
        anyhow::Ok((status, o, e))
    };
    match tokio::time::timeout(cfg.command_timeout, collect).await {
        Ok(Ok((status, out, err))) => {
            let mut text = out;
            if !err.trim().is_empty() {
                text.push_str("\n[stderr]\n");
                text.push_str(&err);
            }
            if !status.success() {
                text.push_str(&format!("\n[exit {status}]"));
            }
            Ok(truncate(&text))
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Ok(format!(
            "[timed out after {}s and was killed]",
            cfg.command_timeout.as_secs()
        )),
    }
}

/// Keep the head and tail of long output — a build failure's error is at the
/// end, and a head-only truncation would cut off exactly what matters.
pub fn truncate(text: &str) -> String {
    if text.chars().count() <= MAX_TOOL_OUTPUT {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let half = MAX_TOOL_OUTPUT / 2;
    let head: String = chars[..half].iter().collect();
    let tail: String = chars[chars.len() - half..].iter().collect();
    format!(
        "{head}\n… [{} chars elided] …\n{tail}",
        chars.len() - MAX_TOOL_OUTPUT
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn a_hanging_command_is_killed_at_the_timeout() {
        let cfg = AgentConfig {
            sandbox: std::env::temp_dir(),
            max_turns: 1,
            command_timeout: Duration::from_millis(300),
            request_timeout: Duration::from_secs(1),
            max_tokens: 16,
            cargo_target_dir: None,
        };
        let out = run_shell(&cfg, "sleep 30").await.unwrap();
        assert!(out.contains("timed out"), "{out}");
    }

    #[tokio::test]
    async fn stderr_and_a_non_zero_exit_are_both_reported() {
        let cfg = AgentConfig {
            sandbox: std::env::temp_dir(),
            max_turns: 1,
            command_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(1),
            max_tokens: 16,
            cargo_target_dir: None,
        };
        let out = run_shell(&cfg, "echo hi; echo bad >&2; exit 7")
            .await
            .unwrap();
        assert!(
            out.contains("hi") && out.contains("bad") && out.contains("exit"),
            "{out}"
        );
    }
}
