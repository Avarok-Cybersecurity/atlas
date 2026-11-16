// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Code extraction and a limited child process for HumanEval and MBPP, written from scratch.
//! Responsibility: pull the last fenced code block out of a reply, build the test program, and run it in a sandboxed
//! `python3` child (the graded answers are Python programs, so the grader still spawns an interpreter with no
//! third-party imports). The sandbox keeps the old limits: address space, CPU time, file size, no core files, a temp
//! directory, a minimal environment, a wall-clock timeout, and the whole process group killed on timeout.
//! `unshare -rn` (new user and network namespaces: no network) is used when the host allows it.

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant};

use regex::Regex;
use serde_json::Value;

use super::Grade;
use crate::route_gate::error::{GResult, value_err};

static FENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```").expect("static regex")
});

/// The last fenced code block (python-tagged blocks win over untagged ones); `None` when there is no fence.
pub fn extract_code(text: &str) -> Option<String> {
    let blocks: Vec<(String, String)> = FENCE
        .captures_iter(text)
        .map(|c| (c[1].to_lowercase(), c[2].to_string()))
        .collect();
    if blocks.is_empty() {
        return None;
    }
    let tagged: Vec<&(String, String)> = blocks
        .iter()
        .filter(|(t, _)| matches!(t.as_str(), "python" | "py" | "python3"))
        .collect();
    let pick = if tagged.is_empty() {
        blocks.last()
    } else {
        tagged.last().copied()
    };
    pick.map(|(_, body)| body.clone())
}

static UNSHARE_OK: OnceLock<bool> = OnceLock::new();

fn python_exe() -> String {
    std::env::var("ROUTE_GATE_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// `"off"` (no code runs), `"subprocess"` (a limited child process, network NOT blocked) or `"unshare"` (the same
/// inside new user and network namespaces: no network). `"auto"` picks `"unshare"` when this host allows it, else
/// `"off"`.
pub fn resolve_sandbox(mode: &str) -> GResult<&'static str> {
    match mode {
        "off" => return Ok("off"),
        "subprocess" => return Ok("subprocess"),
        "auto" | "unshare" => {}
        other => return value_err(format!("unknown code sandbox mode {other:?}")),
    }
    let ok = *UNSHARE_OK.get_or_init(|| {
        let spawned = Command::new("unshare")
            .args(["-rn", "true"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = spawned else {
            return false;
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return status.success(),
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
            }
        }
    });
    if ok {
        return Ok("unshare");
    }
    if mode == "unshare" {
        return value_err(
            "unshare -rn is not available on this host; use --code-exec subprocess or off",
        );
    }
    Ok("off")
}

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A temp directory removed on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> std::io::Result<Self> {
        let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("route_gate_exec_{}_{n}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The last 64 KB of a stream.
fn read_tail(mut r: impl Read) -> Vec<u8> {
    const KEEP: usize = 64 * 1024;
    let mut tail: Vec<u8> = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                tail.extend_from_slice(&buf[..n]);
                if tail.len() > KEEP {
                    tail.drain(..tail.len() - KEEP);
                }
            }
        }
    }
    tail
}

/// Runs `source` in a fresh interpreter. Returns `(Some(passed), detail)`; `None` means code execution is off.
pub fn run_python(
    source: &str,
    timeout: f64,
    mode: &str,
    mem_mb: u64,
) -> GResult<(Option<bool>, String)> {
    let mode = resolve_sandbox(mode)?;
    if mode == "off" {
        return Ok((None, "code_exec_off".to_string()));
    }
    let limit = mem_mb * 1024 * 1024;
    let cpu = timeout as u64 + 5;
    let dir = ScratchDir::new()?;
    let script = dir.path().join("main.py");
    std::fs::write(&script, source)?;
    let mut cmd = if mode == "unshare" {
        let mut c = Command::new("unshare");
        c.arg("-rn").arg(python_exe());
        c
    } else {
        Command::new(python_exe())
    };
    cmd.arg("-I").arg(&script);
    cmd.current_dir(dir.path())
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", dir.path())
        .env("PYTHONHASHSEED", "0")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .process_group(0);
    // SAFETY: the closure runs between fork and exec and only calls `setrlimit`, which is async-signal-safe.
    unsafe {
        cmd.pre_exec(move || {
            for (res, v) in [
                (libc::RLIMIT_AS, limit),
                (libc::RLIMIT_CPU, cpu),
                (libc::RLIMIT_FSIZE, 16 * 1024 * 1024),
                (libc::RLIMIT_CORE, 0),
            ] {
                let lim = libc::rlimit {
                    rlim_cur: v as libc::rlim_t,
                    rlim_max: v as libc::rlim_t,
                };
                if libc::setrlimit(res, &lim) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn()?;
    let stderr = child
        .stderr
        .take()
        .map(|s| std::thread::spawn(move || read_tail(s)));
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let Some(status) = status else {
        // SAFETY: `kill` with a negative pid signals the child's process group, which `process_group(0)` created.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.kill();
        let _ = child.wait();
        if let Some(h) = stderr {
            let _ = h.join();
        }
        return Ok((Some(false), "timeout".to_string()));
    };
    let err = stderr.and_then(|h| h.join().ok()).unwrap_or_default();
    if status.success() {
        return Ok((Some(true), "ok".to_string()));
    }
    let text = String::from_utf8_lossy(&err);
    let last = text.trim().lines().next_back().map(str::to_string);
    let tail = last.unwrap_or_else(|| format!("exit {}", status.code().unwrap_or(-1)));
    Ok((
        Some(false),
        format!("error: {}", tail.chars().take(200).collect::<String>()),
    ))
}

fn meta_str<'a>(item: &'a Value, key: &str) -> &'a str {
    item["meta"][key].as_str().unwrap_or_default()
}

/// HumanEval: prompt + the model's code + the dataset's `check(entry_point)`. The prompt stays in front so helper
/// functions it defines exist; a complete `def entry_point` in the reply overrides the stub. A body-only reply
/// continues the stub.
pub fn humaneval_program(item: &Value, reply: &str) -> Option<String> {
    let code = extract_code(reply)?;
    let entry = meta_str(item, "entry_point");
    let prompt = meta_str(item, "prompt");
    let def = Regex::new(&format!(r"def\s+{}\s*\(", regex::escape(entry))).ok()?;
    let body = if def.is_match(&code) {
        code
    } else {
        code.lines()
            .map(|ln| {
                if !ln.trim().is_empty() && !ln.starts_with([' ', '\t']) {
                    format!("    {ln}")
                } else {
                    ln.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    Some(format!(
        "{prompt}\n{body}\n\n{}\n\ncheck({entry})\n",
        meta_str(item, "test")
    ))
}

/// MBPP: optional setup code, the model's code, then the dataset's assert lines.
pub fn mbpp_program(item: &Value, reply: &str) -> Option<String> {
    let code = extract_code(reply)?;
    let setup = meta_str(item, "test_setup_code");
    let tests: Vec<&str> = item["meta"]["test_list"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    Some(format!("{setup}\n{code}\n\n{}\n", tests.join("\n")))
}

/// Grades a code reply by running its program; `passed` is `None` when code execution is off.
pub fn grade_code(item: &Value, reply: &str, mode: &str, timeout: f64) -> GResult<Grade> {
    let program = if item["source"].as_str() == Some("humaneval") {
        humaneval_program(item, reply)
    } else {
        mbpp_program(item, reply)
    };
    let Some(program) = program else {
        return Ok(Grade::no_answer());
    };
    let (passed, detail) = run_python(&program, timeout, mode, 2048)?;
    Ok(Grade {
        passed,
        extracted: None,
        reason: detail,
    })
}
