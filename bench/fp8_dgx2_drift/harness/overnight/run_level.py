#!/usr/bin/env python3
"""Overnight graduated-completion executor — runs ONE difficulty level against
BOTH agent clients (opencode + Claude Code), checks task completion via the
level's completion_check (cargo build + test), commits on success, and writes a
bug sentinel on failure.

Usage:  run_level.py <level-int>
Exit:   0 = level passed (both clients completed); 1 = failed; 2 = setup error.

Reads the ladder from prompts/graduated.jsonl. Each level:
  {level, name, prompt, completion_check, is_server}

Client invocation (both run in a fresh /tmp target dir, ATLAS at :8888):
  opencode    : opencode run --dangerously-skip-permissions --dir T --format json <prompt>
  claude-code : (cwd=T) sudo -u claude env ANTHROPIC_BASE_URL=http://localhost:8888
                ANTHROPIC_AUTH_TOKEN=dummy timeout -k10 T claude -p
                --permission-mode bypassPermissions     (executing mode = real usage;
                plan mode can't self-approve non-interactively)
Completion = run the level's completion_check (bash) in the target; exit 0 == done.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

HARNESS = pathlib.Path(__file__).resolve().parent.parent
LADDER = HARNESS / "prompts" / "graduated.jsonl"
RESULTS = HARNESS / "overnight" / "results.jsonl"
BUG_SENTINEL = pathlib.Path("/workspace/overnight_BUG.json")
ATLAS_PORT = os.environ.get("ATLAS_HARNESS_PORT", "3001")
# claude-code (opus + alwaysThinking/high-effort) systematically hit the old
# 360s wall mid-write on multi-file server tasks (L2 360.6s barely done, L3/L4
# truncated → unclosed delimiter) while opencode finishes in ~190s. 360s tested
# speed-under-deadline, not completion. 600s gives coherent work room to finish;
# opencode is unaffected (well under budget).
AGENT_TIMEOUT = int(os.environ.get("OVN_AGENT_TIMEOUT", "600"))
# Cold axum/tokio debug builds take minutes; 300s spuriously failed L3-claude
# mid-dep-compile. 600s is ample. A SHARED cargo target dir makes deps compile
# ONCE across all levels/clients so completion checks (and agent builds) are
# fast + never timeout on a cold build.
CHECK_TIMEOUT = int(os.environ.get("OVN_CHECK_TIMEOUT", "600"))
SHARED_TARGET = os.environ.get("OVN_CARGO_TARGET", "/tmp/ovn-cargo-target")
CLAUDE_BIN = "/workspace/.local/bin/claude"


def load_level(level: int) -> dict:
    for ln in LADDER.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        if int(d.get("level", -1)) == level:
            return d
    raise SystemExit(f"level {level} not found in {LADDER}")


def run(cmd, cwd=None, timeout=None, stdin=None, env=None):
    """Run a command, capture output, never raise. Returns (rc, out, err)."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, timeout=timeout, input=stdin,
            capture_output=True, text=True,
            env={**os.environ, **(env or {})},
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or "") + "\n[timeout]"
    except Exception as e:  # noqa: BLE001
        return 125, "", f"[launch error: {e}]"


def kill_opencode() -> None:
    """Kill every lingering opencode process. opencode (a Bun binary) leaks
    multi-threaded instances when a `run` is SIGTERM'd or hangs; a leaked
    instance makes the NEXT run hang (shared-state contention under the box's
    tight unified-memory budget — the 35B model holds ~114/121 GB). So we reap
    before AND after every run. Use `-x opencode` (EXACT process name): `-f
    opencode` would match this script's own cmdline and kill the caller.
    `pkill` exit 1 (no match) is fine; never raise."""
    subprocess.run(["pkill", "-9", "-x", "opencode"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_opencode(prompt: str, target: pathlib.Path) -> None:
    kill_opencode()                      # clear leaks from any prior run
    try:
        run(
            # timeout -k 10: escalate to SIGKILL 10s after SIGTERM so a wedged
            # opencode is actually killed on expiry, not left hanging.
            ["timeout", "-k", "10", str(AGENT_TIMEOUT), "opencode", "run",
             "--dangerously-skip-permissions", "--dir", str(target), "--format", "json", prompt],
            timeout=AGENT_TIMEOUT + 20,
            env={"ATLAS_HARNESS_PORT": ATLAS_PORT, "CARGO_TARGET_DIR": SHARED_TARGET},
        )
    finally:
        kill_opencode()                  # reap leaked instances on done/timeout


def run_claude(prompt: str, target: pathlib.Path) -> None:
    # Executing mode (bypassPermissions) = the real-usage path. timeout INSIDE
    # sudo so the claude grandchild is actually killed on expiry.
    run(
        ["sudo", "-n", "-u", "claude",
         "timeout", "-k", "10", str(AGENT_TIMEOUT),
         "env", "ANTHROPIC_BASE_URL=http://localhost:8888", "ANTHROPIC_AUTH_TOKEN=dummy",
         f"ATLAS_HARNESS_PORT={ATLAS_PORT}", f"CARGO_TARGET_DIR={SHARED_TARGET}",
         CLAUDE_BIN, "-p", "--output-format", "json",
         "--permission-mode", "bypassPermissions"],
        cwd=str(target), stdin=prompt, timeout=AGENT_TIMEOUT + 30,
    )


def completion_check(check: str, target: pathlib.Path) -> tuple[bool, str]:
    rc, out, err = run(["bash", "-lc", check], cwd=str(target), timeout=CHECK_TIMEOUT,
                       env={"ATLAS_HARNESS_PORT": ATLAS_PORT, "CARGO_TARGET_DIR": SHARED_TARGET})
    tail = (out[-1500:] + "\n" + err[-1500:]).strip()
    return rc == 0, tail


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_level.py <level>", file=sys.stderr)
        return 2
    level = int(sys.argv[1])
    lv = load_level(level)
    name, prompt, check = lv["name"], lv["prompt"], lv["completion_check"]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== overnight L{level} {name} @ {time.strftime('%H:%M:%S')} ===", flush=True)
    clients = {"opencode": run_opencode, "claude-code": run_claude}
    per_client = {}
    for cname, runner in clients.items():
        target = pathlib.Path(f"/tmp/overnight-L{level}-{cname}")
        subprocess.run(["rm", "-rf", str(target)])
        target.mkdir(parents=True, exist_ok=True)
        # claude as user `claude` must be able to write here
        subprocess.run(["chmod", "777", str(target)])
        t0 = time.time()
        print(f"  [{cname}] running…", flush=True)
        runner(prompt, target)
        ok, tail = completion_check(check, target)
        files = sum(1 for _ in target.rglob("*") if _.is_file() and ".git" not in _.parts)
        per_client[cname] = {"complete": ok, "files": files,
                             "wall_s": round(time.time() - t0, 1),
                             "target": str(target),
                             "check_tail": tail if not ok else ""}
        print(f"  [{cname}] complete={ok} files={files} wall={per_client[cname]['wall_s']}s", flush=True)

    level_pass = all(c["complete"] for c in per_client.values())
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": level, "name": name,
           "pass": level_pass, "clients": per_client, "prompt": prompt}
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    if level_pass:
        BUG_SENTINEL.unlink(missing_ok=True)
        print(f"=== L{level} {name}: PASS (both clients) ===", flush=True)
    else:
        failing = [c for c, v in per_client.items() if not v["complete"]]
        BUG_SENTINEL.write_text(json.dumps({
            "level": level, "name": name, "failing_clients": failing,
            "prompt": prompt, "completion_check": check,
            "clients": per_client,
            "deployed_image": _deployed_image(),
        }, indent=2))
        print(f"=== L{level} {name}: FAIL (clients: {failing}) → bug sentinel written ===", flush=True)
    return 0 if level_pass else 1


def _deployed_image() -> str:
    rc, out, _ = run(["sudo", "docker", "inspect", "atlas-camp", "--format", "{{.Config.Image}}"])
    return out.strip() if rc == 0 else "?"


if __name__ == "__main__":
    raise SystemExit(main())
