// SPDX-License-Identifier: AGPL-3.0-only

//! Scoring for the agentic webserver task, ported from the harness's
//! `score_run.py` + `followed_directions.py`.
//!
//! Two orthogonal axes, and keeping them apart is the point:
//!
//! * `webserver_ok` — **outcome**. The scorer builds and runs the code the
//!   agent left behind and asks `/ping` for a `pong`. It is true even if the
//!   agent never built or verified anything itself.
//! * `followed_directions` — **process**. Did the agent do the six things the
//!   prompt told it to? This is what separates a real agentic run from one that
//!   wrote a correct `main.rs`, stopped, and let the scorer carry it.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, BufReader};

/// The six prompt-mandated process steps. `followed_directions` is their AND.
pub const REQUIRED_STEPS: [&str; 6] = [
    "wrote_project",
    "wrote_tests",
    "ran_tests",
    "ran_server",
    "curled",
    "tore_down",
];

#[derive(Clone, Debug, Default)]
pub struct WebserverResult {
    pub webserver_ok: bool,
    pub build_ok: bool,
    pub error: String,
    pub port_used: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Directions {
    pub steps: Vec<(&'static str, bool)>,
}

impl Directions {
    /// True only when every mandated step is evidenced.
    pub fn overall(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|(_, ok)| *ok)
    }
    pub fn met(&self) -> usize {
        self.steps.iter().filter(|(_, ok)| *ok).count()
    }
}

/// An ephemeral port that is free right now.
///
/// A fresh OS-assigned port per iteration is what makes this self-isolating: a
/// zombie server from an earlier run can neither collide with ours nor answer
/// our `curl` — the bug class that invalidated every earlier `webserver_ok`
/// number when the port was hardcoded.
pub fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Build the project, run it on `port`, and check `/ping` answers `pong`.
pub async fn webserver_test(
    sandbox: &Path,
    cargo_target_dir: Option<&Path>,
    build_timeout: Duration,
    serve_timeout: Duration,
) -> WebserverResult {
    let mut out = WebserverResult::default();
    if !sandbox.join("Cargo.toml").is_file() {
        out.error = "no Cargo.toml was written".into();
        return out;
    }
    let port = match free_port() {
        Ok(p) => p,
        Err(e) => {
            out.error = format!("could not reserve a port: {e}");
            return out;
        }
    };
    out.port_used = port;

    let mut build = tokio::process::Command::new("cargo");
    build
        .args(["build", "--release"])
        .current_dir(sandbox)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(dir) = cargo_target_dir {
        build.env("CARGO_TARGET_DIR", dir);
    }
    match tokio::time::timeout(build_timeout, build.output()).await {
        Ok(Ok(o)) if o.status.success() => out.build_ok = true,
        Ok(Ok(o)) => {
            let err = String::from_utf8_lossy(&o.stderr);
            out.error =
                super::super::one_line(err.lines().rev().take(6).collect::<Vec<_>>().join(" "));
            return out;
        }
        Ok(Err(e)) => {
            out.error = format!("cargo build could not start: {e}");
            return out;
        }
        Err(_) => {
            out.error = format!("cargo build exceeded {}s", build_timeout.as_secs());
            return out;
        }
    }

    let mut serve = tokio::process::Command::new("cargo");
    serve
        .args(["run", "--release"])
        .current_dir(sandbox)
        // The prompt told the model to read this variable. If it hardcoded a
        // port instead, the server binds elsewhere, the probe fails, and
        // `webserver_ok = false` is the correct answer — not a harness bug.
        .env("ATLAS_HARNESS_PORT", port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    if let Some(dir) = cargo_target_dir {
        serve.env("CARGO_TARGET_DIR", dir);
    }
    let mut child = match serve.spawn() {
        Ok(c) => c,
        Err(e) => {
            out.error = format!("cargo run could not start: {e}");
            return out;
        }
    };
    // `child` is killed on drop, so every early return below also tears the
    // server down — a leaked server would hold the CPU the next iteration is
    // timed on, and its wall time would land on the wrong run.
    let deadline = tokio::time::Instant::now() + serve_timeout;
    while tokio::time::Instant::now() < deadline {
        if let Some(body) = ping(port).await
            && body.to_lowercase().contains("pong")
        {
            out.webserver_ok = true;
            let _ = child.kill().await;
            return out;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let _ = child.kill().await;
    out.error = format!(
        "/ping did not answer 'pong' within {}s",
        serve_timeout.as_secs()
    );
    out
}

/// One `GET /ping`. `None` means "not up yet", which is the normal state while
/// the server is starting.
async fn ping(port: u16) -> Option<String> {
    use tokio::io::AsyncWriteExt;
    let mut sock = tokio::time::timeout(
        Duration::from_millis(500),
        tokio::net::TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    .ok()?
    .ok()?;
    let req = format!("GET /ping HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    sock.write_all(req.as_bytes()).await.ok()?;
    let mut body = String::new();
    let mut lines = BufReader::new(sock).lines();
    let read = async {
        while let Ok(Some(line)) = lines.next_line().await {
            body.push_str(&line);
            body.push('\n');
        }
    };
    tokio::time::timeout(Duration::from_secs(2), read)
        .await
        .ok()?;
    Some(body)
}

/// Did the agent perform the six steps the prompt mandated?
///
/// Evidence is the shell commands it issued plus the tree it left behind — the
/// same two sources `followed_directions.py` uses.
pub fn followed_directions(commands: &[String], sandbox: &Path) -> Directions {
    let joined = commands.join("\n").to_lowercase();
    let wrote_project = sandbox.join("Cargo.toml").is_file() && has_source(sandbox);
    let wrote_tests = has_tests(sandbox);
    let ran_tests = contains_cargo(&joined, &["test", "nextest"]);
    let ran_server = contains_cargo(&joined, &["run"])
        || joined.contains("target/release/")
        || joined.contains("target/debug/");
    let curled = ["curl ", "wget ", "httpie", "httpx", "nc -z"]
        .iter()
        .any(|k| joined.contains(k));
    let tore_down = ["kill", "fuser -k", "pkill"]
        .iter()
        .any(|k| joined.contains(k));
    Directions {
        steps: REQUIRED_STEPS
            .iter()
            .zip([
                wrote_project,
                wrote_tests,
                ran_tests,
                ran_server,
                curled,
                tore_down,
            ])
            .map(|(name, ok)| (*name, ok))
            .collect(),
    }
}

/// `cargo <sub>` for any of `subs`, tolerating extra whitespace.
fn contains_cargo(haystack: &str, subs: &[&str]) -> bool {
    haystack.split("cargo").skip(1).any(|rest| {
        let head = rest.trim_start();
        subs.iter().any(|s| {
            head.strip_prefix(s).is_some_and(|after| {
                after.is_empty() || after.starts_with(|c: char| !c.is_alphanumeric())
            })
        })
    })
}

fn has_source(sandbox: &Path) -> bool {
    walk(sandbox).any(|p| p.extension().is_some_and(|e| e == "rs"))
}

/// A `tests/` directory, or `#[test]` / `#[cfg(test)]` anywhere in the source.
fn has_tests(sandbox: &Path) -> bool {
    if sandbox.join("tests").is_dir() {
        return true;
    }
    walk(sandbox)
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .any(|s| s.contains("#[test]") || s.contains("#[cfg(test)]"))
}

/// Shallow recursive walk that skips build output and VCS metadata — a
/// `target/` tree contains thousands of files and none of them are evidence.
fn walk(root: &Path) -> impl Iterator<Item = std::path::PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            if matches!(name.to_str(), Some("target") | Some(".git")) {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.into_iter()
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
