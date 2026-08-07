// SPDX-License-Identifier: AGPL-3.0-only

//! Running the shell the model authored, and bounding what that can cost.
//!
//! Split out of [`super`] because it is a different concern from the agent
//! loop: the loop decides WHAT to run, this decides what running it is allowed
//! to do. Every containment rule the benchmark relies on lives here — the
//! sandbox-relative path resolution, the hard timeout and kill, the concurrent
//! pipe drain that keeps a backgrounded child from wedging the run, and the
//! output cap that stops a runaway command from eating the context window.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use parking_lot::Mutex;
use tokio::io::AsyncReadExt;

use super::{AgentConfig, DRAIN_GRACE, MAX_TOOL_OUTPUT, norm};

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
