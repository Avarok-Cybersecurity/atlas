// SPDX-License-Identifier: AGPL-3.0-only

//! Running the shell the model authored, and bounding what that can cost.
//!
//! Split out of [`super`] because it is a different concern from the agent
//! loop: the loop decides WHAT to run, this decides what running it is allowed
//! to do. Every containment rule the benchmark relies on lives here — the hard
//! timeout, the kill that takes the whole process GROUP so a fork cannot outlive
//! it, the concurrent pipe drain that keeps a backgrounded child from wedging
//! the run, the bounded capture that keeps a runaway writer from exhausting the
//! box, and the output cap that stops a big result from eating the context
//! window.

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use parking_lot::Mutex;
use tokio::io::AsyncReadExt;

use super::{AgentConfig, DRAIN_GRACE, MAX_TOOL_OUTPUT, norm};

/// Bytes of one stream held at each end while the command is still running.
///
/// [`truncate`] caps what the *model* sees, but it only runs once the command
/// is over, so on its own it caps nothing: `yes` writes into the pipe as fast as
/// the drain empties it, and the drain used to append every byte to a `Vec` that
/// only stopped growing when the command did. At a 180 s default command timeout
/// that is tens of gigabytes on a box whose 121 GB is shared with the served
/// model — the run does not fail, the machine does. This is the real cap; 16×
/// the tool-output cap is far more than the 8 KiB that survives truncation, and
/// leaves room for [`norm`] to see whole lines at both ends.
const CAPTURE_END: usize = 16 * MAX_TOOL_OUTPUT;

/// Room reserved for the elision note [`truncate`] writes, so its own result is
/// never longer than [`MAX_TOOL_OUTPUT`]. The agent loop truncates tool results
/// a second time (results from the file tools have not been through here at
/// all); without the reserve that second pass found an over-cap string and cut
/// it again, nesting one elision note inside another and dropping the text
/// around the first one.
const ELISION_NOTE: usize = 96;

#[cfg(test)]
pub(super) const TEST_ELISION_NOTE: usize = ELISION_NOTE;

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
    // Its own process group, so the timeout below can kill everything the
    // command started rather than only the shell that started it. A `setsid`
    // server is deliberately outside it — that is what `setsid` is for, and
    // reaping those is `agent::reap`'s job.
    #[cfg(unix)]
    cmd.process_group(0);
    if let Some(dir) = &cfg.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", dir);
    }
    let mut child = cmd.spawn()?;
    let pid = child.id();
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
            // Kill the GROUP, not just `sh`. A command that forked and waited
            // (`cargo build`, `sleep 300`, anything the model wrote without
            // `setsid`) left the fork running when only the shell was killed: it
            // kept the CPU that the rest of the tier is timed on, and it kept it
            // until the end of the whole iteration, because `agent::reap` does
            // not run until then. Σwall is a gate criterion, so this is a
            // measurement bug as much as a leak.
            if let Some(pid) = pid {
                kill_group(pid).await;
            }
            let _ = child.kill().await;
            None
        }
    };
    let aborts = (pumps.0.abort_handle(), pumps.1.abort_handle());
    let _ = tokio::time::timeout(DRAIN_GRACE, async {
        let _ = pumps.0.await;
        let _ = pumps.1.await;
    })
    .await;
    // The grace expiring means the pipe is still held open by something we did
    // not kill, and it will never reach EOF. Left alone the pump outlives the
    // command for the life of the process, holding the read end open and filling
    // a buffer nobody will ever look at again — one leaked task and one leaked
    // fd per detached server, for the whole tier.
    aborts.0.abort();
    aborts.1.abort();
    let mut text = out.lock().text();
    let stderr = err.lock().text();
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
async fn pump<R: AsyncReadExt + Unpin>(reader: Option<R>, sink: Arc<Mutex<Capture>>) {
    let Some(mut reader) = reader else { return };
    let mut buf = [0u8; 8192];
    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            return;
        }
        sink.lock().push(&buf[..n]);
    }
}

/// A bounded window over one stream: the first [`CAPTURE_END`] bytes and the
/// last [`CAPTURE_END`], with everything between them counted and thrown away.
///
/// Both ends, for the same reason [`truncate`] keeps both: a build failure's
/// first error is at the top and its summary is at the bottom. Reading is never
/// stopped to enforce the bound — a writer blocked on a full pipe would be
/// reported as a timeout, turning a chatty command that worked into a failure.
#[derive(Default)]
pub(crate) struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    dropped: usize,
}

impl Capture {
    fn push(&mut self, mut bytes: &[u8]) {
        if self.head.len() < CAPTURE_END {
            let n = (CAPTURE_END - self.head.len()).min(bytes.len());
            self.head.extend_from_slice(&bytes[..n]);
            bytes = &bytes[n..];
        }
        if bytes.len() >= CAPTURE_END {
            self.dropped += self.tail.len() + bytes.len() - CAPTURE_END;
            self.tail.clear();
            self.tail.extend(&bytes[bytes.len() - CAPTURE_END..]);
            return;
        }
        let overflow = (self.tail.len() + bytes.len()).saturating_sub(CAPTURE_END);
        self.dropped += overflow;
        drop(self.tail.drain(..overflow));
        self.tail.extend(bytes);
    }

    /// Decoded once, over the concatenation, so a multi-byte character split
    /// across the head/tail seam is not mangled into two replacement characters.
    fn text(&self) -> String {
        let mut bytes = self.head.clone();
        if self.dropped > 0 {
            bytes.extend_from_slice(
                format!("\n… [{} bytes dropped from the middle] …\n", self.dropped).as_bytes(),
            );
        }
        bytes.extend(self.tail.iter().copied());
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[cfg(test)]
    fn held(&self) -> usize {
        self.head.len() + self.tail.len()
    }
}

/// SIGKILL the whole group led by `pid` (which is the group, because the child
/// was spawned with `process_group(0)`).
///
/// **The `--` is load-bearing.** util-linux's `kill` accepts `kill -9 -<pgid>`,
/// exits **0**, and kills nothing — the negative pid is eaten as an option. The
/// end-of-options marker is what makes it reach the group, and the difference
/// between the two spellings is invisible in the exit status, so it is asserted
/// by a test rather than read off a man page.
#[cfg(unix)]
async fn kill_group(pid: u32) {
    let _ = tokio::process::Command::new("kill")
        .arg("-9")
        .arg("--")
        .arg(format!("-{pid}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
}

#[cfg(not(unix))]
async fn kill_group(_pid: u32) {}

/// Keep the head and tail of long output — a build failure's first error is at
/// the top and its summary is at the bottom, and either alone is a worse signal.
/// A silent truncation would be worse still, so the elision is stated.
///
/// Idempotent, because the caller cannot always know whether a result has been
/// through here: the result is at most `MAX_TOOL_OUTPUT` bytes, so truncating
/// it again returns it unchanged.
pub fn truncate(text: &str) -> String {
    if text.len() <= MAX_TOOL_OUTPUT {
        return text.to_string();
    }
    let keep = MAX_TOOL_OUTPUT - ELISION_NOTE;
    let (mut cut, mut from) = (keep / 2, text.len() - keep / 2);
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
#[path = "agent_shell_tests.rs"]
mod tests;
