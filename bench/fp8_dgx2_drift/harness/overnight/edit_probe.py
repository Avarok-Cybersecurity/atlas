#!/usr/bin/env python3
"""Edit-flow probe — stresses the Read+Edit+Bash agentic path (NOT Write).

Seeds an EXISTING cargo project whose src/lib.rs has 3 trivially-easy `todo!()`
stubs pinned by tests, and instructs the agent to EDIT the stubs (not rewrite
the file). Coding is trivial on purpose so a failure points at the agentic
mechanics / Atlas tool path (Edit old_string/new_string arg handling, multi-turn
read→edit→verify, tool-arg corruption) rather than model coding competence.

Runs BOTH clients against the live Atlas at :8888, then `cargo test`. Writes
/workspace/editprobe_<client>.json with completeness + the cargo tail.

Usage: edit_probe.py [opencode|claude-code|both]
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

SHARED_TARGET = "/tmp/ovn-cargo-target"
AGENT_TIMEOUT = int(os.environ.get("EDIT_AGENT_TIMEOUT", "480"))
CHECK_TIMEOUT = 600
CLAUDE_BIN = "/workspace/.local/bin/claude"

LIB_RS = '''//! Tiny math utility crate. Three functions are unimplemented (marked
//! `todo!()`). Implement each so the tests pass. The signatures and tests
//! are the spec — do not change them.

/// Return a + b.
pub fn add(a: i64, b: i64) -> i64 {
    todo!("implement add")
}

/// Return a * b.
pub fn multiply(a: i64, b: i64) -> i64 {
    todo!("implement multiply")
}

/// Return true iff n is even.
pub fn is_even(n: i64) -> bool {
    todo!("implement is_even")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-4, 4), 0);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(3, 4), 12);
        assert_eq!(multiply(-2, 5), -10);
    }

    #[test]
    fn test_is_even() {
        assert!(is_even(0));
        assert!(is_even(8));
        assert!(!is_even(7));
    }
}
'''

CARGO_TOML = '''[package]
name = "mathkit"
version = "0.1.0"
edition = "2021"

[dependencies]
'''

PROMPT = (
    "The Rust project in the current directory has three unimplemented functions "
    "in src/lib.rs, each marked with `todo!()`: add, multiply, and is_even. "
    "Implement all three so that `cargo test` passes. The function signatures and "
    "the existing #[cfg(test)] tests are the specification — do not change the "
    "signatures or the tests. EDIT the existing src/lib.rs in place to replace the "
    "three todo!() bodies; do not rewrite the whole file from scratch. After "
    "editing, run `cargo test` to confirm all tests pass."
)


def run(cmd, cwd=None, timeout=None, stdin=None, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, input=stdin,
                           capture_output=True, text=True,
                           env={**os.environ, **(env or {})})
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or "") + "\n[timeout]"
    except Exception as e:  # noqa: BLE001
        return 125, "", f"[launch error: {e}]"


def seed(target: pathlib.Path):
    subprocess.run(["rm", "-rf", str(target)])
    (target / "src").mkdir(parents=True, exist_ok=True)
    (target / "Cargo.toml").write_text(CARGO_TOML)
    (target / "src" / "lib.rs").write_text(LIB_RS)
    subprocess.run(["chmod", "-R", "777", str(target)])


def kill_opencode():
    # Reap leaked opencode (Bun) instances before+after every run — a leaked
    # instance hangs the next run. `-x opencode` (exact name) so we don't kill
    # this script via a cmdline self-match. pkill exit 1 (no match) is fine.
    subprocess.run(["pkill", "-9", "-x", "opencode"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_opencode(target):
    kill_opencode()
    try:
        return run(["timeout", "-k", "10", str(AGENT_TIMEOUT), "opencode", "run",
                    "--dangerously-skip-permissions", "--dir", str(target),
                    "--format", "json", PROMPT],
                   timeout=AGENT_TIMEOUT + 20,
                   env={"CARGO_TARGET_DIR": SHARED_TARGET})
    finally:
        kill_opencode()


def run_claude(target):
    return run(["sudo", "-n", "-u", "claude",
                "timeout", "-k", "10", str(AGENT_TIMEOUT),
                "env", "ANTHROPIC_BASE_URL=http://localhost:8888",
                "ANTHROPIC_AUTH_TOKEN=dummy", f"CARGO_TARGET_DIR={SHARED_TARGET}",
                CLAUDE_BIN, "-p", "--output-format", "json",
                "--permission-mode", "bypassPermissions"],
               cwd=str(target), stdin=PROMPT, timeout=AGENT_TIMEOUT + 30)


def check(target):
    rc, out, err = run(["bash", "-lc", "cargo test 2>&1"], cwd=str(target),
                       timeout=CHECK_TIMEOUT, env={"CARGO_TARGET_DIR": SHARED_TARGET})
    return rc == 0, (out[-1800:] + "\n" + err[-800:]).strip()


def still_has_todo(target):
    # Only count todo!() in CODE, not the doc-comment that mentions it.
    try:
        code = [ln for ln in (target / "src" / "lib.rs").read_text().splitlines()
                if not ln.lstrip().startswith("//")]
        return any("todo!(" in ln for ln in code)
    except Exception:  # noqa: BLE001
        return None


def one(client: str):
    target = pathlib.Path(f"/tmp/editprobe-{client}")
    seed(target)
    t0 = time.time()
    print(f"[{client}] running… @ {time.strftime('%H:%M:%S')}", flush=True)
    (run_opencode if client == "opencode" else run_claude)(target)
    ok, tail = check(target)
    rec = {"client": client, "complete": ok, "wall_s": round(time.time() - t0, 1),
           "still_has_todo": still_has_todo(target),
           "lib_lines": len((target / "src" / "lib.rs").read_text().splitlines())
           if (target / "src" / "lib.rs").exists() else 0,
           "target": str(target), "check_tail": "" if ok else tail}
    print(f"[{client}] complete={ok} still_todo={rec['still_has_todo']} "
          f"lib_lines={rec['lib_lines']} wall={rec['wall_s']}s", flush=True)
    pathlib.Path(f"/workspace/editprobe_{client}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    clients = ["opencode", "claude-code"] if which == "both" else [which]
    for c in clients:
        one(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
