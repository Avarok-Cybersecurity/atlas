#!/usr/bin/env python3
"""followed_directions — process-fidelity metric for the opencode webserver probe.

This is the THIRD axis, orthogonal to the existing two:
  - cargo.cargo_toml_valid   : artifact validity (does the TOML parse?)
  - webserver.webserver_ok   : OUTCOME — does the left-behind code, when the
                               SCORER builds+runs it, answer /ping with pong?
  - followed_directions      : PROCESS — did the AGENT ITSELF do the things the
                               prompt instructed (build, add tests, run tests,
                               run the server, curl it, tear it down)?

`webserver_ok` can be True even when the agent under-followed the prompt — e.g.
it ran `cargo init`, wrote a correct main.rs, and stopped, never building or
self-verifying (observed: lmfp8_postfix run 8 — 3 files, 59s). The scorer's own
build+curl then carried it. `followed_directions` distinguishes that lazy
early-stop from a run that genuinely performed the full agentic workflow.

The harness PROMPT (run_tier.sh) instructs, verbatim:
  "create a pure rust Axum project ... ping/pong endpoint. The server MUST bind
   to the port from the ATLAS_HARNESS_PORT env var ... Add tests, run them and
   prove all tests pass, then run the server and use curl to prove it works.
   Finally, tear down the server."

Each instruction becomes a checked sub-step. Evidence sources:
  - bash commands the agent ran (opencode tool_use events: state.input.command)
  - the filesystem the agent left behind (tests/ dir, #[test] in source,
    ATLAS_HARNESS_PORT reference)

Pure-stdlib, no I/O beyond reading the target dir. Importable
(`compute_followed_directions`) and runnable standalone for backfill.
"""
from __future__ import annotations

import pathlib
import re
from typing import Any

# The six prompt-mandated process steps. `followed_directions` (overall) is the
# AND of all of them — the agent did the complete workflow it was told to.
_REQUIRED_STEPS = (
    "wrote_project",  # Cargo.toml + src/main.rs authored
    "wrote_tests",    # "Add tests"        — tests/ dir or #[test]/#[cfg(test)]
    "ran_tests",      # "run them"         — cargo test
    "ran_server",     # "run the server"   — cargo run / executes the binary
    "curled",         # "use curl to prove it works"
    "tore_down",      # "tear down the server"
)

# Bash-command detectors. Anchored on word boundaries so "cargo test" matches
# but "cargo test" inside an unrelated path/string is unlikely to false-positive.
_RE_BUILD = re.compile(r"\bcargo\s+(?:build|check|run|test)\b")
_RE_TEST = re.compile(r"\bcargo\s+(?:test|nextest)\b")
_RE_RUN = re.compile(r"\bcargo\s+run\b|\./target/(?:debug|release)/|\btarget/(?:debug|release)/\S")
_RE_CURL = re.compile(r"\bcurl\b|\bwget\b|\bhttp(?:ie|x)\b|\bnc\s+-z\b")
_RE_KILL = re.compile(r"\bp?kill\b|\bkill\s+(?:%|-9|-TERM|-SIGTERM|\$|\d)|\bfuser\s+-k\b")


def _bash_commands(events: list[dict[str, Any]]) -> list[str]:
    """Extract every bash command string the agent issued."""
    cmds: list[str] = []
    for e in events:
        if e.get("type") != "tool_use":
            continue
        part = e.get("part", {}) or {}
        if part.get("tool") != "bash":
            continue
        state = part.get("state", {}) or {}
        inp = state.get("input", {}) or {}
        cmd = inp.get("command")
        if isinstance(cmd, str) and cmd:
            cmds.append(cmd)
    return cmds


def _source_files(target: pathlib.Path) -> list[pathlib.Path]:
    """Agent-authored .rs files (skip build artifacts under target/)."""
    if not target.exists():
        return []
    out: list[pathlib.Path] = []
    for p in target.rglob("*.rs"):
        parts = p.parts
        if "target" in parts or ".git" in parts:
            continue
        out.append(p)
    return out


def _has_tests(target: pathlib.Path) -> bool:
    """tests/ integration dir, or a unit-test attribute in any source file."""
    if (target / "tests").is_dir() and any((target / "tests").rglob("*.rs")):
        return True
    for p in _source_files(target):
        try:
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        if "#[test]" in txt or "#[cfg(test)]" in txt or "#[tokio::test]" in txt:
            return True
    return False


def _reads_port_env(target: pathlib.Path) -> bool:
    for p in _source_files(target):
        try:
            if "ATLAS_HARNESS_PORT" in p.read_text(errors="replace"):
                return True
        except Exception:
            continue
    return False


def compute_followed_directions(
    events: list[dict[str, Any]], target: pathlib.Path
) -> dict[str, Any]:
    """Return the followed_directions verdict + per-step evidence.

    Never raises on malformed input — returns best-effort flags. The overall
    `followed_directions` bool is the AND of the six required process steps.
    """
    target = pathlib.Path(target)
    cmds = _bash_commands(events)
    blob = "\n".join(cmds)

    has_cargo_toml = (target / "Cargo.toml").exists()
    has_main = (target / "src" / "main.rs").exists() or any(
        p.name == "main.rs" for p in _source_files(target)
    )

    steps = {
        "wrote_project": bool(has_cargo_toml and has_main),
        "wrote_tests": _has_tests(target),
        "ran_tests": bool(_RE_TEST.search(blob)),
        "ran_server": bool(_RE_RUN.search(blob)),
        "curled": bool(_RE_CURL.search(blob)),
        "tore_down": bool(_RE_KILL.search(blob)),
    }
    completed = sum(1 for s in _REQUIRED_STEPS if steps[s])
    return {
        "followed_directions": all(steps[s] for s in _REQUIRED_STEPS),
        "steps_completed": completed,
        "steps_total": len(_REQUIRED_STEPS),
        # informational sub-flags (not part of the strict overall):
        "ran_build": bool(_RE_BUILD.search(blob)),
        "reads_port_env": _reads_port_env(target),
        "bash_command_count": len(cmds),
        # per-step booleans for diagnosis:
        **steps,
    }


# ── standalone / backfill entrypoint ───────────────────────────────────
def _load_events(path: pathlib.Path) -> list[dict[str, Any]]:
    import json

    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Compute followed_directions for one probe.")
    ap.add_argument("--target", required=True, type=pathlib.Path,
                    help="the project dir the agent wrote (e.g. /tmp/harness-<tier>-r<N>)")
    ap.add_argument("--opencode-json", required=True, type=pathlib.Path,
                    help="the saved opencode JSONL event log for that run")
    args = ap.parse_args()
    events = _load_events(args.opencode_json)
    print(json.dumps(compute_followed_directions(events, args.target), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
