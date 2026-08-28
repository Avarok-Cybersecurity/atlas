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

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};

#[path = "agent_history.rs"]
mod history;
#[path = "norm.rs"]
pub mod norm;
#[path = "agent_path.rs"]
mod path_guard;
#[path = "agent_shell.rs"]
pub mod shell;
#[path = "agent_tools.rs"]
pub mod tools;
#[path = "trace.rs"]
pub mod trace;
use history::{assistant_message, call_id, compact, preserve_thinking};
pub use path_guard::resolve;
pub(crate) use shell::run_shell;
pub use shell::truncate;
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
pub(super) const MAX_TOOL_OUTPUT: usize = 8192;

/// Conversation characters kept before old tool results are elided. opencode
/// never lets a session exceed the window (`SessionPrompt.run` checks
/// `isOverflow` every step, then compacts); `mod.rs`'s Gate A recipe serves
/// `--max-seq-len 32768`, less one 8192-token reply ≈ 24k tokens.
const HISTORY_BUDGET: usize = 96_000;

/// Recent tool results compaction never touches — the model is mid-edit here.
const LIVE_TOOL_RESULTS: usize = 4;

/// Assistant turns that keep their full reasoning when preserve-thinking is
/// on. Older ones are elided first (see [`compact`]).
const LIVE_REASONING: usize = 4;

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
pub(super) const DRAIN_GRACE: Duration = Duration::from_secs(2);

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
    /// Turns cut off at `max_tokens` and resumed rather than mistaken for the
    /// agent finishing. Counted because it is the signature of greedy
    /// repetition degeneration, and a run that needed several of these is worth
    /// looking at even when it ends up passing.
    pub truncated_turns: usize,
    /// Turns that carried tool-call syntax in the CONTENT while the server
    /// parsed none, and were re-asked rather than mistaken for the agent
    /// finishing. Counted for the same reason as `truncated_turns`: it is a
    /// degeneration signature, and a run that needed one is worth looking at
    /// even when it passes.
    pub unparsed_call_turns: usize,
    /// Decoded tokens across every turn, server-reported where the server
    /// sends `usage`. The honest denominator for a speed claim — turns vary in
    /// how much they generate, tokens do not. Recorded rather than gated: no
    /// bound can be set before it has been measured on a variant.
    pub completion_tokens: usize,
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
        transcript.completion_tokens += outcome.completion_tokens;
        trace.turn(turn, &outcome);

        if outcome.tool_calls.is_empty() {
            // A turn that hit the token cap did not FINISH — it was CUT OFF, and
            // those are not the same event. The agent stops calling tools when
            // it considers the task done; a truncated reply says nothing about
            // whether it was done, only that it ran out of room mid-sentence.
            // Treating the two alike is what made one stuck turn cost an entire
            // run: the model would loop inside the turn that writes
            // `src/main.rs`, hit `max_tokens`, return no tool call, and the loop
            // would exit as if it had chosen to — scoring 0/6 steps on a run
            // that had not actually failed the task, only failed to fit.
            //
            // So say so and let it continue. The partial text goes back in as
            // the assistant turn it was, followed by the fact of the truncation,
            // which is information the model cannot otherwise have: from its
            // side the reply simply ended. This is what a real agent client does
            // with a `length` stop, and it is correct independently of any score
            // — a harness that silently reinterprets truncation as completion is
            // measuring something other than the agent.
            if was_cut_off(&outcome) {
                transcript.truncated_turns += 1;
                messages.push(json!({"role": "assistant", "content": outcome.text}));
                messages.push(json!({"role": "user", "content":
                    "Your previous message was cut off at the output limit before you \
                     finished. Do not repeat it. Continue from where it stopped, and make \
                     the tool call you intended."}));
                continue;
            }
            // The same mistake wearing a different stop reason. A turn can
            // degenerate into repetition, emit its tool call as raw syntax
            // inside the CONTENT, and stop naturally — the server's parser
            // rejects the malformed block, so `tool_calls` is empty and
            // `finish_reason` is `stop`. Nothing about that says the agent
            // chose to finish; it says the reply came apart. Observed for real
            // on gate run 7 at `66b20718`: after five thinking-loop watchdog
            // fires the model emitted
            // `<tool_call><function=bash>…curl …/pong…</function></tool_call>`
            // wrapped in repeated prose, the call never executed, the loop
            // exited, and the run lost `tore_down` — 9/10 on a gate that wants
            // 10/10, from one unparsed call.
            //
            // Re-ask instead. This cannot mask a real failure: if the model
            // meant to stop it simply stops again next turn, with no syntax in
            // the text, and the run ends one turn later than it would have.
            if tools::emitted_unparsed_call(&outcome) {
                transcript.unparsed_call_turns += 1;
                messages.push(json!({"role": "assistant", "content": outcome.text}));
                messages.push(json!({"role": "user", "content":
                    "Your previous message contained tool-call syntax in the message body, so \
                     no tool actually ran. Re-issue exactly that one call as a real tool call, \
                     with nothing else in the message. If you are finished, say so in plain \
                     text with no tool-call syntax."}));
                continue;
            }
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
/// Did this turn run out of room, rather than run out of things to do?
///
/// The distinction is the whole point: no tool calls AND a natural stop means
/// the agent is finished, while no tool calls AND `length` means it never got
/// to say what it wanted. Only the second is resumable, and only the first
/// should end the run.
fn was_cut_off(outcome: &crate::http::ChatOutcome) -> bool {
    outcome.tool_calls.is_empty() && outcome.finish_reason.as_deref() == Some("length")
}

/// Did this turn try to call a tool and fail to be understood as one?
///
/// True when the server parsed no tool calls but the text still carries the
/// opening syntax of one. Both markers are *opening* tags on purpose: the
/// failure mode is a block the parser could not close, so requiring a
/// well-formed pair would miss precisely the case this exists to catch.
///
fn request_body(model: &str, messages: &[Value], tools: &Value, max_tokens: usize) -> Value {
    let mut body = json!({
        "model": model, "stream": true,
        "max_tokens": max_tokens, "messages": messages,
        "tools": tools, "tool_choice": "auto",
    });
    // ATLAS_AGENTIC_SAMPLING=model-card: omit the greedy pins so the server's
    // card presets own sampling (A/B arm; the GATE default stays pinned-greedy
    // per the rationale above TEMPERATURE).
    if std::env::var("ATLAS_AGENTIC_SAMPLING").as_deref() != Ok("model-card") {
        body["temperature"] = json!(TEMPERATURE);
        body["seed"] = json!(SEED);
    }
    // Only `preserve_thinking` goes in: the other kwargs (notably
    // `reasoning_effort`) are left absent ON PURPOSE so the serve-level
    // default keeps owning them.
    if preserve_thinking() {
        body["chat_template_kwargs"] = json!({"preserve_thinking": true});
    }
    body
}

/// Resolve `path` inside `sandbox`, rejecting anything that escapes it.
///
#[cfg(test)]
#[path = "agent_loop_tests.rs"]
mod loop_tests;
#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
