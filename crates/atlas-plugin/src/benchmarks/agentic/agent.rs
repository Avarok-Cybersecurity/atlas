// SPDX-License-Identifier: AGPL-3.0-only

//! The agent loop: tool-calling against the served endpoint, with the tools
//! executed inside a sandbox directory.
//!
//! **A port of one client, not a generic agent.** The recorded Gate A history
//! was measured by driving `opencode` 1.18.14 from
//! `bench/fp8_dgx2_drift/harness/run_tier.sh`, so "faithful" means reproducing
//! the scaffolding opencode put in front of the model: the six tools the
//! harness's own agent enables (see [`tools`]), that agent's system prompt plus
//! opencode's environment block, its sampling, and its output caps. Each is
//! cited at the constant or function that carries it.
//!
//! **This executes model-authored shell.** There is no version of the agentic
//! webserver benchmark that does not — building and running the code the model
//! wrote is the measurement. The containment is explicit and lives here: every
//! command runs in the sandbox, under a hard timeout, and is killed on expiry;
//! file-tool paths are rejected if they leave the sandbox; tool output is
//! capped so a runaway `yes` cannot exhaust memory; turns are capped so a loop
//! cannot run forever.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;

#[path = "norm.rs"]
pub mod norm;
#[path = "agent_tools.rs"]
pub mod tools;
#[path = "trace.rs"]
pub mod trace;
pub use tools::{glob_match, tool_schema};

/// Bytes of one tool result.
///
/// opencode's own bash cap is 30000 characters, tail-only. This is deliberately
/// tighter, and middle-elided, because `run_tier.sh` explains what a big result
/// costs on this window: past it "the model leaks repeated `<tool_call>` XML as
/// plain text and runs a turn to the max_tokens cap". One `cargo build` error
/// dump gets there in a single turn. 8192 matches the *model* output cap the
/// harness pins beside it (`ATLAS_OPENCODE_OUTPUT_CAP` → `limit.output`, which
/// `mod.rs` mirrors as `max_tokens`), so one tool result can never cost more
/// context than one whole reply.
const MAX_TOOL_OUTPUT: usize = 8192;

/// Conversation characters kept before old tool results are elided. opencode
/// never lets a session exceed the window (`SessionPrompt.run` checks
/// `isOverflow` every step, then compacts); `mod.rs`'s Gate A recipe serves
/// `--max-seq-len 32768`, less one 8192-token reply ≈ 24k tokens.
const HISTORY_BUDGET: usize = 96_000;

/// Recent tool results compaction never touches — the model is mid-edit here.
const LIVE_TOOL_RESULTS: usize = 4;

/// **The one place this gate deliberately departs from the harness it ports.**
///
/// `~/.config/opencode/opencode.json` sets `options.temperature: 0.3` on every
/// `atlas*` model, and that is right for a research harness: it samples the
/// model's behaviour distribution, and 10 runs at 0.3 say something about the
/// spread. A PR gate has the opposite job. Its bar is an exact 10-of-10, so a
/// sampled instrument cannot separate a regression from a draw — the same
/// binary measured 10/10 then 8/10 on `webserver_ok` and 9/10 then 5/10 on
/// `followed_directions`, and re-running until green is not a gate.
///
/// At 0 the sampler is argmax (`adaptive_sampler::should_use_greedy` short-
/// circuits on `base_temperature == 0.0`), and Atlas is bitwise-deterministic
/// at batch 1 — which is what this benchmark runs, one agent at a time. Greedy
/// decoding is a necessary condition for a repeatable trajectory, not a
/// sufficient one: see [`norm`] for the other half.
const TEMPERATURE: f64 = 0.0;

/// Pinned beside the temperature. At 0 the sampler never draws, so the seed is
/// unused today; it is sent so that a serve path which ever *does* sample
/// samples the same way twice rather than silently reintroducing the spread
/// this gate just removed.
const SEED: u64 = 0;

/// Grace for the output pumps once the process is gone. Only a grace: a pipe
/// inherited by a detached child never reaches EOF at all.
const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// The harness agent's prompt, verbatim from the body of
/// `~/.config/opencode/agents/atlas.md` — the agent `run_tier.sh` selects with
/// `default_agent: atlas`. `LLMRequestPrep.prepare` uses an agent's own prompt
/// *instead of* the built-in provider prompt, so this is the whole of it.
///
/// The last paragraph is the load-bearing one for a thinking model on a 32k
/// window: without "keep thinking short", reasoning alone walks the session into
/// the degeneration zone the harness header describes.
const AGENT_PROMPT: &str = "\
You are a coding assistant running locally on Atlas Spark. No data leaves this machine.

You have access to tools for interacting with the filesystem and running commands:
- **bash**: Execute shell commands (ls, cat, grep, find, git, etc.)
- **read**: Read file contents
- **write**: Create or overwrite files
- **edit**: Edit existing files (find and replace)
- **glob**: Find files matching a pattern
- **grep**: Search file contents with regex

When asked to list files, check directories, or run commands, use the **bash** tool.
When asked to read a file, use the **read** tool.

IMPORTANT: Think briefly, then act. Do NOT describe tool calls in your thinking — just make \
them directly. Keep thinking short (under 50 words). Never put tool calls inside thinking tags. \
Use the write tool (not edit) when creating new files.";

/// What one agent run did, for scoring.
#[derive(Default)]
pub struct Transcript {
    /// Every shell command the agent issued, in order. `followed_directions`
    /// is computed from this.
    pub commands: Vec<String>,
    pub turns: usize,
    pub tool_calls: usize,
    /// True when the loop ended at the turn cap rather than because the agent
    /// stopped calling tools.
    pub hit_turn_cap: bool,
    pub final_text: String,
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

/// opencode's environment block, appended to the agent prompt inside one system
/// message (`LLMRequestPrep.prepare`). Naming the working directory is what
/// makes the absolute paths its file tools ask for constructible.
///
/// `Today's date` is omitted deliberately: a prompt that changes at midnight is
/// not a fixed benchmark, and `run_tier.sh` holds the task prompt constant for
/// that very reason ("a bit-identical token sequence for every run").
fn system_prompt(sandbox: &Path, model: &str) -> String {
    let dir = sandbox.display();
    format!(
        "{AGENT_PROMPT}\nYou are powered by the model named {model}. The exact model ID is \
         {model}\nHere is some useful information about the environment you are running in:\n\
         <env>\n  Working directory: {dir}\n  Workspace root folder: {dir}\n  \
         Is directory a git repo: no\n  Platform: linux\n</env>"
    )
}

/// Run one agentic task to completion (or to the turn cap).
pub async fn run_task(
    handle: &crate::plugin::PluginHandle,
    cfg: &AgentConfig,
    prompt: &str,
) -> Result<Transcript> {
    let mut transcript = Transcript::default();
    let outcome = agent_loop(handle, cfg, prompt, &mut transcript).await;
    // Reap on every path, including a transport error: a leaked server holds
    // its port into the next iteration, and the scorer has not run yet.
    reap(&cfg.sandbox).await;
    outcome.map(|()| transcript)
}

/// Kill anything still running out of the sandbox.
///
/// `run_tier.sh:329` reaps the same way and says why: on the timeout SIGTERM a
/// backgrounded server "reparents to init (PPID=1) and KEEPS HOLDING ITS PORT".
/// `kill_on_drop` cannot reach it — the prompt tells the model to use `setsid`,
/// so the process is deliberately not our child any more. Victims are
/// identified by working directory alone, exactly as the harness does, so
/// nothing outside this run's sandbox is ever touched. Without `/proc` (i.e.
/// not Linux) this is a no-op.
async fn reap(sandbox: &Path) {
    let real = std::fs::canonicalize(sandbox).unwrap_or_else(|_| sandbox.to_path_buf());
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return;
    };
    let me = std::process::id().to_string();
    let victims: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) && *n != me)
        .filter(|pid| {
            std::fs::read_link(format!("/proc/{pid}/cwd")).is_ok_and(|c| c.starts_with(&real))
        })
        .collect();
    if !victims.is_empty() {
        let _ = tokio::process::Command::new("kill")
            .arg("-9")
            .args(&victims)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

async fn agent_loop(
    handle: &crate::plugin::PluginHandle,
    cfg: &AgentConfig,
    prompt: &str,
    transcript: &mut Transcript,
) -> Result<()> {
    let target = handle.target();
    let mut messages = vec![
        json!({"role": "system", "content": system_prompt(&cfg.sandbox, &target.model)}),
        json!({"role": "user", "content": prompt}),
    ];
    let tools = tool_schema();
    let trace = trace::Trace::start(&cfg.sandbox, prompt);

    for turn in 0..cfg.max_turns {
        handle.check_cancelled()?;
        handle.status(format!("agent turn {}/{}", turn + 1, cfg.max_turns));
        compact(&mut messages);
        let body = request_body(&target.model, &messages, &tools, cfg.max_tokens);
        let outcome = crate::http::chat_stream(target, &body, cfg.request_timeout).await?;
        transcript.turns = turn + 1;
        transcript.final_text = outcome.text.clone();
        trace.turn(turn, &outcome);

        if outcome.tool_calls.is_empty() {
            return Ok(());
        }

        messages.push(assistant_message(&outcome, turn));
        for (i, call) in outcome.tool_calls.iter().enumerate() {
            handle.check_cancelled()?;
            transcript.tool_calls += 1;
            // A tool error is data for the model, not a run failure: an agent
            // recovering from a bad command is normal behaviour, and aborting
            // here would score it as a crash.
            let content = match tools::execute(cfg, call, &mut transcript.commands).await {
                Ok(text) => text,
                Err(e) => format!("error: {e:#}"),
            };
            let content = truncate(&content);
            trace.result(&call.name, &content);
            messages.push(json!({"role": "tool", "content": content,
                "tool_call_id": call_id(turn, i)}));
        }
    }
    transcript.hit_turn_cap = true;
    Ok(())
}

/// One chat request. Split out so the gate's sampling pins are asserted by a
/// test rather than trusted: a silent drift back to sampled decoding would not
/// fail anything, it would just make the gate flaky again.
fn request_body(model: &str, messages: &[Value], tools: &Value, max_tokens: usize) -> Value {
    json!({
        "model": model, "stream": true, "temperature": TEMPERATURE, "seed": SEED,
        "max_tokens": max_tokens, "messages": messages,
        "tools": tools, "tool_choice": "auto",
    })
}

/// The `tool_call_id` this conversation carries — **ours, never the server's.**
///
/// Atlas mints ids from a per-process counter (`call_0000000000000004`), so the
/// same turn of the same work is labelled differently depending on how many
/// tool calls that server has answered since it started. Echoing it wrote a
/// value from outside the run into the model's context, where it changes the
/// next turn's tokens: measured here, five identical requests came back with
/// five distinct id sets and identical text. An id only has to pair one
/// assistant `tool_calls` entry with its `role: "tool"` reply inside this
/// request, so a positional one is both legal and reproducible.
///
/// Turn *and* index, because the two sites must agree — a `tool_call_id` that
/// pairs with nothing on the assistant message is a 400. They previously
/// numbered from different bases (`i` against `turn * 100 + i`) and only
/// matched because both echoed the server's id; a model that emits no ids hit
/// the mismatch.
fn call_id(turn: usize, nth: usize) -> String {
    format!("call_{turn}_{nth}")
}

/// Elide the oldest tool results once the session outgrows the window — the
/// port of opencode's auto-compaction (`isOverflow` → `compaction`).
///
/// It rewrites tool *contents* and never removes a message: an assistant
/// `tool_calls` block whose matching `role: "tool"` reply went missing is a 400
/// from the server, which would end the run rather than shorten it.
fn compact(messages: &mut [Value]) {
    let size = |m: &Value| m["content"].as_str().map_or(64, str::len);
    let mut total: usize = messages.iter().map(size).sum();
    let tools: Vec<usize> = (0..messages.len())
        .filter(|i| messages[*i]["role"] == "tool")
        .collect();
    for &i in tools
        .iter()
        .take(tools.len().saturating_sub(LIVE_TOOL_RESULTS))
    {
        if total <= HISTORY_BUDGET {
            return;
        }
        let was = size(&messages[i]);
        let marker = format!("[{was} characters elided to stay inside the context window]");
        total = total - was + marker.len();
        messages[i]["content"] = Value::String(marker);
    }
}

fn assistant_message(outcome: &crate::http::ChatOutcome, turn: usize) -> Value {
    let calls: Vec<Value> = outcome
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({"id": call_id(turn, i), "type": "function", "function": {"name": c.name,
                // Some models emit no arguments at all for a zero-arg call; an
                // empty string is not valid JSON to a strict server.
                "arguments": if c.arguments.is_empty() { "{}" } else { &c.arguments }}})
        })
        .collect();
    let text = &outcome.text;
    json!({"role": "assistant", "tool_calls": calls,
        "content": if text.is_empty() { Value::Null } else { Value::String(text.clone()) }})
}

/// Resolve `path` inside `sandbox`, rejecting anything that escapes it.
///
/// Lexical, not `canonicalize`: the target usually does not exist yet, and a
/// canonicalize-then-compare check silently passes on a missing path.
///
/// An absolute path is accepted **only** when it is already inside the sandbox.
/// opencode's file tools ask for absolute paths and its environment block hands
/// the model the working directory to build them from, so rejecting every
/// absolute path — as this did — failed the prompt-compliant call.
pub fn resolve(sandbox: &Path, path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    let path = match path.strip_prefix(sandbox) {
        Ok(inside) => inside,
        Err(_) if path.is_absolute() => bail!(
            "path must be inside the project directory {}: {}",
            sandbox.display(),
            path.display()
        ),
        Err(_) => path,
    };
    let mut out = sandbox.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(c) => out.push(c),
            Component::CurDir => {}
            Component::ParentDir => bail!("path must not leave the project directory"),
            Component::RootDir | Component::Prefix(_) => bail!("absolute paths are not allowed"),
        }
    }
    Ok(out)
}

pub(crate) async fn run_shell(cfg: &AgentConfig, command: &str, limit: Duration) -> Result<String> {
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
        .kill_on_drop(true)
        // `run_tier.sh:296` sets this on the opencode process. It is the signal
        // the cargo shim (`/workspace/.cargo-shim/cargo`) reads to force-detach
        // `cargo run` "regardless of how the model writes the command". Set on
        // THIS child only, never process-wide: the shim reserves the detach for
        // the agent, and the scorer must keep `cargo run` in the foreground.
        .env("ATLAS_AGENT_SHELL", "1");
    if let Some(dir) = &cfg.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", dir);
    }
    let mut child = cmd.spawn()?;
    let (out, err) = (Arc::default(), Arc::default());
    // Drain concurrently with the wait: output past the pipe buffer blocks the
    // writer until someone reads it.
    let pumps = (
        tokio::spawn(pump(child.stdout.take(), Arc::clone(&out))),
        tokio::spawn(pump(child.stderr.take(), Arc::clone(&err))),
    );
    // Wait on the PROCESS, not on end-of-pipe. `setsid cargo run &` inherits
    // this tool's stdout, so the pipe never reaches EOF even though `sh` has
    // exited — reading to EOF first (as this did) charged the whole timeout to a
    // command that finished instantly, then reported none of its output. The
    // prompt's `> /tmp/server.log 2>&1` avoids it; a model that forgets should
    // lose one command, not the run.
    let status = match tokio::time::timeout(limit, child.wait()).await {
        Ok(s) => Some(s?),
        Err(_) => {
            let _ = child.kill().await;
            None
        }
    };
    let _ = tokio::time::timeout(DRAIN_GRACE, async {
        let _ = pumps.0.await;
        let _ = pumps.1.await;
    })
    .await;
    let mut text = String::from_utf8_lossy(&out.lock()).into_owned();
    let stderr = String::from_utf8_lossy(&err.lock()).into_owned();
    if !stderr.trim().is_empty() {
        // A failing command's stderr is the most valuable signal the model gets,
        // so it survives every path — including the timeout.
        text.push_str("\n[stderr]\n");
        text.push_str(&stderr);
    }
    match status {
        Some(s) if !s.success() => text.push_str(&format!("\n[exit {s}]")),
        Some(_) => {}
        None => text.push_str(&format!(
            "\n[timed out after {}s and was killed; the output above is what it had produced. \
             If this was a server, start it detached with its output redirected to a file.]",
            limit.as_secs()
        )),
    }
    // Normalise BEFORE truncating. The other order looks equivalent and is not:
    // `truncate` reports how many characters it elided and cuts at a byte
    // offset, so a duration that is one digit longer on one run shifts the cut
    // and changes text the model reads on both sides of it. See [`norm`] for
    // what is rewritten and, more importantly, what is not.
    Ok(truncate(&norm::normalize(&text)))
}

/// Bytes, not `String`: a UTF-8 sequence split across two reads would be
/// mangled if each chunk were decoded on its own.
async fn pump<R: AsyncReadExt + Unpin>(reader: Option<R>, sink: Arc<Mutex<Vec<u8>>>) {
    let Some(mut reader) = reader else { return };
    let mut buf = [0u8; 8192];
    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            return;
        }
        sink.lock().extend_from_slice(&buf[..n]);
    }
}

/// Keep the head and tail of long output — a build failure's first error is at
/// the top and its summary is at the bottom, and either alone is a worse signal.
/// A silent truncation would be worse still, so the elision is stated.
pub fn truncate(text: &str) -> String {
    if text.len() <= MAX_TOOL_OUTPUT {
        return text.to_string();
    }
    let (mut cut, mut from) = (MAX_TOOL_OUTPUT / 2, text.len() - MAX_TOOL_OUTPUT / 2);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    while !text.is_char_boundary(from) {
        from += 1;
    }
    let (head, tail) = (&text[..cut], &text[from..]);
    let elided = text.len() - head.len() - tail.len();
    format!("{head}\n… [{elided} characters elided from the middle] …\n{tail}")
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
