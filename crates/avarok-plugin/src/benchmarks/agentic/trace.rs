// SPDX-License-Identifier: AGPL-3.0-only

//! The per-iteration trajectory record — the gate's determinism evidence.
//!
//! Greedy decoding plus normalised tool output ([`super::norm`]) are supposed to
//! make two consecutive tiers the same measurement. "Supposed to" is not a
//! claim anyone should accept on a gate, and the outcome columns
//! (`webserver_ok`, `steps`, `turns`) are far too coarse to check it: two runs
//! can agree on all three and have diverged on turn 2. So every run writes down
//! exactly what it saw and what it said, in a plain-text shape that `diff`
//! answers in one line — identical trajectories, or the first turn that split.
//!
//! It is written **beside** the sandbox (`…/run-07.trajectory.txt`, never
//! inside `…/run-07/`) for a reason that is easy to get wrong: the agent's
//! `glob`, `grep` and `read` tools walk the sandbox, and the scorer's
//! `has_tests` reads every `.rs` under it. A trace file inside would be visible
//! to the model it is recording, and would feed the previous turn's transcript
//! back into the next one.
//!
//! Failures are logged, never propagated: a full disk must not fail a run that
//! is otherwise measuring correctly.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::http::ChatOutcome;

pub struct Trace {
    path: Option<PathBuf>,
}

impl Trace {
    /// Truncate (not append) — the trace describes THIS run of this index, and
    /// a file carrying two runs concatenated is not diffable against anything.
    pub fn start(sandbox: &Path, prompt: &str) -> Self {
        let path = path_for(sandbox);
        let trace = Self { path: Some(path) };
        match std::fs::write(trace.path.as_ref().expect("just set"), header(prompt)) {
            Ok(()) => trace,
            Err(e) => {
                tracing::warn!("agentic: trajectory trace disabled: {e}");
                Self { path: None }
            }
        }
    }

    /// One model turn: its reasoning, its reply, and the calls it asked for.
    ///
    /// Reasoning is recorded even though it never re-enters the conversation
    /// (`ChatOutcome` keeps it out of `text` deliberately): when a trajectory
    /// does split, it splits in the thinking first, and a diff that starts at
    /// the tool call has already lost the cause.
    pub fn turn(&self, index: usize, outcome: &ChatOutcome) {
        let mut s = format!("\n── turn {} ───────────────────────────────\n", index + 1);
        section(&mut s, "reasoning", &outcome.reasoning);
        section(&mut s, "text", &outcome.text);
        for call in &outcome.tool_calls {
            s.push_str(&format!(
                "[call {}] {} {}\n",
                call.id, call.name, call.arguments
            ));
        }
        if let Some(reason) = &outcome.finish_reason {
            s.push_str(&format!("[finish] {reason}\n"));
        }
        self.append(&s);
    }

    /// One tool result, exactly as the model received it — normalised and
    /// truncated. Recording the raw bytes instead would make every trace differ
    /// on durations the model never saw.
    pub fn result(&self, tool: &str, content: &str) {
        let mut s = format!("[result {tool}]\n");
        s.push_str(content.trim_end());
        s.push('\n');
        self.append(&s);
    }

    fn append(&self, text: &str) {
        let Some(path) = &self.path else { return };
        let written = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(text.as_bytes()));
        if let Err(e) = written {
            tracing::warn!("agentic: could not extend {}: {e}", path.display());
        }
    }
}

/// `…/sandbox/run-07` → `…/sandbox/run-07.trajectory.txt`.
fn path_for(sandbox: &Path) -> PathBuf {
    let mut name = sandbox.file_name().unwrap_or_default().to_os_string();
    name.push(".trajectory.txt");
    sandbox.with_file_name(name)
}

fn header(prompt: &str) -> String {
    format!("[prompt]\n{}\n", prompt.trim_end())
}

fn section(out: &mut String, name: &str, body: &str) {
    if body.trim().is_empty() {
        return;
    }
    out.push_str(&format!("[{name}]\n{}\n", body.trim_end()));
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
