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
    /// The steps that were NOT evidenced, in declaration order.
    ///
    /// ★ Without this a failure is undiagnosable. The run record stored only
    /// the COUNT (`"5/6"`) while `steps` carried the names all along, and the
    /// per-iteration trajectory is truncated by the next run of the same index
    /// — so the 2026-08-09 investigation into an intermittent 9/10 had to be
    /// reconstructed from a leftover `/tmp/agent_server.log` four hours later.
    /// A gate that cannot say WHY it failed is a gate nobody can fix.
    pub fn missing(&self) -> Vec<&'static str> {
        self.steps
            .iter()
            .filter(|(_, ok)| !*ok)
            .map(|(name, _)| *name)
            .collect()
    }
    pub fn met(&self) -> usize {
        self.steps.iter().filter(|(_, ok)| *ok).count()
    }
}

/// Return an OS-assigned ephemeral port for a caller that will spawn immediately.
///
/// Gate self-start uses this after all fallible resolution work and directly
/// before spawning the model server. The scorer has a longer Cargo-build gap,
/// so it uses the private reservation path below and retains the listener
/// through that gap.
pub fn free_port() -> Result<u16> {
    let listener = reserve_port()?;
    Ok(listener.local_addr()?.port())
}

/// Hold an OS-assigned ephemeral port until the scored project is ready to run.
///
/// A fresh OS-assigned port per iteration is what makes this self-isolating: a
/// zombie server from an earlier run can neither collide with ours nor answer
/// our `curl` — the bug class that invalidated every earlier `webserver_ok`
/// number when the port was hardcoded. Keeping the listener through the Cargo
/// build closes the minutes-long window in which another process could take a
/// port that was free only before compilation started.
fn reserve_port() -> Result<std::net::TcpListener> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?)
}

/// Build the project, run it on `port`, and check `/ping` answers `pong`.
pub async fn webserver_test(
    sandbox: &Path,
    cargo_target_dir: Option<&Path>,
    build_timeout: Duration,
    serve_timeout: Duration,
) -> WebserverResult {
    let mut out = WebserverResult::default();
    // `score_run.py:306` refuses the same two before spending a build on it:
    // "no Cargo.toml or src/ — skipping webserver test". A manifest with no
    // `src/` cannot produce a binary, and a 600 s cold build is a lot to spend
    // on finding that out.
    if !sandbox.join("Cargo.toml").is_file() {
        out.error = "no Cargo.toml was written".into();
        return out;
    }
    if !sandbox.join("src").is_dir() {
        out.error = "no src/ was written — skipping webserver test".into();
        return out;
    }
    let port_reservation = match reserve_port() {
        Ok(reservation) => reservation,
        Err(e) => {
            out.error = format!("could not reserve a port: {e}");
            return out;
        }
    };
    let port = match port_reservation.local_addr() {
        Ok(address) => address.port(),
        Err(e) => {
            out.error = format!("could not inspect the reserved port: {e}");
            return out;
        }
    };
    out.port_used = port;

    let mut build = tokio::process::Command::new("cargo");
    build
        .args(["build", "--release"])
        .current_dir(sandbox)
        // The reference harness exports this for the whole project lifecycle,
        // and an agent may compile it with `env!` as well as read it at runtime.
        // The listener remains held while the build receives the number.
        .env("ATLAS_HARNESS_PORT", port.to_string())
        // The harness's cargo shim detaches `cargo run` only when
        // ATLAS_AGENT_SHELL is set, and its own comment reserves that for the
        // agent: "The SCORER (no ATLAS_AGENT_SHELL) + build/test pass through."
        // Inheriting it here would detach the server we are about to supervise.
        .env_remove("ATLAS_AGENT_SHELL")
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

    // Capture the server's stderr to a file, exactly as `score_run.py:348-351`
    // does and for its stated reason: so a bind panic "is recorded as a
    // distinct, diagnosable failure instead of being silently swallowed and
    // mislabeled as a generic 'didn't respond' timeout". `/ping did not answer`
    // was this benchmark's most common failure string while the cause was
    // invisible.
    let err_log =
        std::env::temp_dir().join(format!("atlas-ws-stderr-{}-{port}.log", std::process::id()));
    // `create_new`, because the name of this file is predictable and the thing
    // whose code we are about to run may still have processes in the sandbox.
    // `create` would follow a symlink planted at that path and truncate whatever
    // it points at; failing closed costs one diagnosis, not a file.
    let sink = match std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&err_log)
    {
        Ok(f) => Stdio::from(f),
        Err(_) => Stdio::null(),
    };

    let mut serve = tokio::process::Command::new("cargo");
    serve
        .args(["run", "--release"])
        .current_dir(sandbox)
        // The prompt told the model to read this variable. If it hardcoded a
        // port instead, the server binds elsewhere, the probe fails, and
        // `webserver_ok = false` is the correct answer — not a harness bug.
        .env("ATLAS_HARNESS_PORT", port.to_string())
        // `score_run.py:355`: keeps a `tracing_subscriber` server from burying
        // the panic we are capturing under info-level request logs.
        .env("RUST_LOG", "warn")
        .env_remove("ATLAS_AGENT_SHELL")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(sink)
        .kill_on_drop(true);
    if let Some(dir) = cargo_target_dir {
        serve.env("CARGO_TARGET_DIR", dir);
    }
    // Release only when the child is ready to bind. A small spawn race remains,
    // but the potentially minutes-long build no longer leaves the port open.
    drop(port_reservation);
    let mut child = match serve.spawn() {
        Ok(c) => c,
        Err(e) => {
            out.error = format!("cargo run could not start: {e}");
            let _ = std::fs::remove_file(&err_log);
            return out;
        }
    };
    // `child` is killed on drop, so every early return below also tears the
    // server down — a leaked server would hold the CPU the next iteration is
    // timed on, and its wall time would land on the wrong run.
    let deadline = tokio::time::Instant::now() + serve_timeout;
    let mut exited = None;
    while tokio::time::Instant::now() < deadline {
        if let Some(body) = ping(port).await
            && body.to_lowercase().contains("pong")
        {
            out.webserver_ok = true;
            let _ = child.kill().await;
            let _ = std::fs::remove_file(&err_log);
            return out;
        }
        // A process that has already exited will never answer, so waiting out
        // the rest of the budget only turns one diagnosable failure (no binary
        // target, bind panic, instant-exit main) into a timeout that reads as a
        // model failure.
        if let Ok(Some(status)) = child.try_wait() {
            exited = Some(status);
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let _ = child.kill().await;
    let detail = server_stderr(&err_log);
    let _ = std::fs::remove_file(&err_log);
    out.error = match exited {
        Some(status) => format!("server exited ({status}) before answering /ping{detail}"),
        None => format!(
            "/ping did not answer 'pong' within {}s{detail}",
            serve_timeout.as_secs()
        ),
    };
    out
}

/// The tail of the server's stderr, plus the one diagnosis `score_run.py`
/// singles out (`score_run.py:418-421`): a port already in use is a harness
/// collision, not a model failure, and must never look like one.
fn server_stderr(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    if text.trim().is_empty() {
        return String::new();
    }
    let mut note = String::new();
    if text.contains("Address already in use") || text.contains("EADDRINUSE") {
        note.push_str(" | server bind failed (port in use)");
    }
    // 800 chars, matching the harness's window: enough for a panic and its
    // location, short enough to stay one line in the results table.
    let mut tail: Vec<char> = text.chars().rev().take(800).collect();
    tail.reverse();
    let tail: String = tail.into_iter().collect();
    format!("{note} | stderr: {}", super::super::one_line(tail))
}

/// One `GET /ping`. `None` means "not up yet", which is the normal state while
/// the server is starting.
async fn ping(port: u16) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut sock = tokio::time::timeout(
        Duration::from_millis(500),
        tokio::net::TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    .ok()?
    .ok()?;
    let req = format!("GET /ping HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    sock.write_all(req.as_bytes()).await.ok()?;
    // Raw bytes, not lines: a `pong` body with no trailing newline is only
    // flushed by a line reader at EOF, so a server that ignores
    // `Connection: close` never surfaced its answer at all and the run was
    // scored "/ping did not answer 'pong'". `curl -sS -m 2` (score_run.py:383)
    // prints whatever arrived before its own timer expired; so must we.
    let mut buf = Vec::new();
    let read = async {
        let mut chunk = [0u8; 4096];
        while let Ok(n) = sock.read(&mut chunk).await {
            if n == 0 || buf.len() > 64 * 1024 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), read).await;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Did the agent perform the six steps the prompt mandated?
///
/// Evidence is the shell commands it issued plus the tree it left behind — the
/// same two sources `followed_directions.py` uses.
pub fn followed_directions(commands: &[String], sandbox: &Path) -> Directions {
    let joined = commands.join("\n").to_lowercase();
    // `followed_directions.py:124-136`, step for step. Its detectors are all
    // word-anchored (`\bcurl\b`, `\bp?kill\b`) so that "killed" in a log line or
    // a path containing "cargo" cannot stand in as evidence of the step.
    let wrote_project = regular(&sandbox.join("Cargo.toml")) && has_main(sandbox);
    let wrote_tests = has_tests(sandbox);
    let ran_tests = contains_cargo(&joined, &["test", "nextest"]);
    let ran_server = contains_cargo(&joined, &["run"]) || ran_binary(&joined);
    let curled = ["curl", "wget", "httpie", "httpx"]
        .iter()
        .any(|k| word(&joined, k))
        || word_then(&joined, "nc", "-z");
    let tore_down =
        word(&joined, "kill") || word(&joined, "pkill") || word_then(&joined, "fuser", "-k");
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

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Byte offsets just past every occurrence of `needle` that starts on a word
/// boundary — the `\b` the harness's detectors are all anchored with.
fn after_word_start<'a>(hay: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    hay.match_indices(needle)
        .filter(|(i, _)| !hay[..*i].chars().next_back().is_some_and(is_word))
        .map(|(i, m)| i + m.len())
}

/// `\bneedle\b`. Bare `contains("kill")` also fired on "killed" and "skill",
/// crediting `tore_down` to an agent that never killed anything.
fn word(hay: &str, needle: &str) -> bool {
    after_word_start(hay, needle).any(|end| !hay[end..].starts_with(is_word))
}

/// `\bfirst\s+second\b` — the shape of `_RE_KILL`'s `fuser -k` and `_RE_CURL`'s
/// `nc -z`.
fn word_then(hay: &str, first: &str, second: &str) -> bool {
    after_word_start(hay, first).any(|end| {
        let rest = &hay[end..];
        let head = rest.trim_start();
        head.len() < rest.len()
            && head
                .strip_prefix(second)
                .is_some_and(|after| !after.starts_with(is_word))
    })
}

/// `\btarget/(?:debug|release)/\S` — `_RE_RUN`'s other half: the agent ran the
/// built binary directly instead of through `cargo run`.
fn ran_binary(hay: &str) -> bool {
    ["target/debug/", "target/release/"].iter().any(|p| {
        after_word_start(hay, p).any(|end| hay[end..].starts_with(|c: char| !c.is_whitespace()))
    })
}

/// `\bcargo\s+<sub>\b` for any of `subs` — `_RE_TEST` / `_RE_RUN`.
fn contains_cargo(haystack: &str, subs: &[&str]) -> bool {
    after_word_start(haystack, "cargo").any(|end| {
        let rest = &haystack[end..];
        let head = rest.trim_start();
        head.len() < rest.len()
            && subs.iter().any(|s| {
                head.strip_prefix(s)
                    .is_some_and(|after| !after.starts_with(is_word))
            })
    })
}

/// `followed_directions.py:125` counts the project written only when a `main.rs`
/// exists — a lone `Cargo.toml` plus a stray `build.rs` is not an Axum project.
fn has_main(sandbox: &Path) -> bool {
    regular(&sandbox.join("src/main.rs"))
        || walk(sandbox).any(|p| p.file_name().is_some_and(|n| n == "main.rs"))
}

/// A `tests/` directory *containing Rust*, or a test attribute anywhere in the
/// source (`followed_directions.py:88-99`). An empty `tests/` is not evidence —
/// `cargo init` creates directories the agent never filled in.
fn has_tests(sandbox: &Path) -> bool {
    let tests = sandbox.join("tests");
    let real_dir = std::fs::symlink_metadata(&tests).is_ok_and(|m| m.is_dir());
    if real_dir && walk(&tests).any(|p| p.extension().is_some_and(|e| e == "rs")) {
        return true;
    }
    walk(sandbox)
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .filter_map(|p| std::fs::read_to_string(p).ok())
        // `#[tokio::test]` is how an async Axum test is spelled and the built-in
        // rubric did not look for it.
        .any(|s| {
            s.contains("#[test]") || s.contains("#[cfg(test)]") || s.contains("#[tokio::test]")
        })
}

/// Shallow recursive walk that skips build output and VCS metadata — a
/// `target/` tree contains thousands of files and none of them are evidence.
///
/// Symlinks are neither followed nor collected, for two reasons that both come
/// back to this walking a tree written by the thing it is scoring. `ln -s . a`
/// (three of them, one bash call) makes the walk explode combinatorially up to
/// the kernel's `ELOOP` depth, and nothing here has a timeout — the agent has
/// already finished by the time the scorer runs, so that hangs the whole
/// benchmark, not one iteration. And a link is not evidence: `ln -s
/// ~/atlas/tests/foo.rs tests/foo.rs` would otherwise credit `wrote_tests` to an
/// agent that wrote no tests.
fn walk(root: &Path) -> impl Iterator<Item = std::path::PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if matches!(name.to_str(), Some("target") | Some(".git")) {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            match (kind.is_dir(), kind.is_file()) {
                (true, _) => stack.push(entry.path()),
                (_, true) => files.push(entry.path()),
                _ => {}
            }
        }
    }
    files.into_iter()
}

/// A file the agent actually wrote, rather than a link to one it did not.
fn regular(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file())
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;

/// How many iterations evidenced each directive, keyed `step:<name>`.
///
/// ★ Lives here rather than in the driving loop because it is SCORING. It also
/// keeps `mod.rs` under the repo's 500-line ceiling, which the diagnosability
/// work pushed it past.
///
/// The record needs this because `followed_directions` is all-or-nothing per
/// iteration: a 9/10 says one iteration failed and nothing about WHICH
/// directive, and the trajectory that would have said is truncated by the next
/// run of the same index.
pub fn per_step_tallies(all: &[&Directions]) -> Vec<(String, f64)> {
    let Some(first) = all.first() else {
        return Vec::new();
    };
    first
        .steps
        .iter()
        .map(|(name, _)| {
            let met = all
                .iter()
                .filter(|d| d.steps.iter().any(|(n, ok)| n == name && *ok))
                .count();
            (format!("step:{name}"), met as f64)
        })
        .collect()
}
