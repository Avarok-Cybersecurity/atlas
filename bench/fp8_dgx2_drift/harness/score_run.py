#!/usr/bin/env python3
"""Score a single opencode probe run.

Extracts structured drift metrics from:
  - The opencode JSONL stdout (tool calls, content, timing)
  - The Atlas server's stderr/stdout (logged via docker logs)
  - The actual filesystem state of the target directory

Usage:
    python3 score_run.py --tier TIER --run N --target TARGET_DIR \
        --opencode-json /tmp/oc-<tier>-r<N>.json \
        --opencode-stderr /tmp/oc-<tier>-r<N>.err \
        --atlas-log-window /tmp/atlas-log-<tier>-r<N>.txt \
        --probe-start-ts <epoch_ms> \
        --probe-end-ts <epoch_ms> \
        --out /path/to/run_<tier>_<N>.json

Output: structured JSON containing one record. Atomic write — caller can
collect across many runs without locking.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

# followed_directions (process-fidelity metric) — additive + fully guarded so a
# missing/broken module can never disturb the existing scoring path.
try:
    _hd = str(pathlib.Path(__file__).resolve().parent)
    if _hd not in sys.path:
        sys.path.insert(0, _hd)
    from followed_directions import compute_followed_directions as _compute_followed
except Exception:  # pragma: no cover — defensive: never break scoring
    _compute_followed = None


def load_events(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def find_files_written(target: pathlib.Path) -> list[str]:
    if not target.exists():
        return []
    files = []
    for p in target.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            files.append(str(p.relative_to(target)))
    return sorted(files)


def cargo_check(target: pathlib.Path) -> dict[str, Any]:
    """Syntactic validation of Cargo.toml: parses as TOML, has a
    `[package]` section with required `name` + `version`. We do NOT
    run `cargo metadata` because it pulls deps (slow + network) and
    requires src/main.rs which may not exist if the model wrote
    Cargo.toml but not main.rs (a legitimate intermediate state).

    The drift modes we want to flag — newline-collapsed sections,
    embedded line numbers, reasoning text in content — are all
    detectable by `tomllib.loads` failing.
    """
    out: dict[str, Any] = {
        "cargo_toml_present": False,
        "cargo_toml_valid": False,
        "cargo_toml_error": "",
        "cargo_toml_has_package_name": False,
    }
    cargo_path = target / "Cargo.toml"
    if not cargo_path.exists():
        return out
    out["cargo_toml_present"] = True
    try:
        import tomllib  # Python 3.11+
        text = cargo_path.read_bytes()
        try:
            data = tomllib.loads(text.decode("utf-8", errors="replace"))
            out["cargo_toml_valid"] = True
            pkg = data.get("package", {})
            if isinstance(pkg, dict) and pkg.get("name") and pkg.get("version"):
                out["cargo_toml_has_package_name"] = True
        except tomllib.TOMLDecodeError as e:
            out["cargo_toml_error"] = str(e)[:500]
    except Exception as e:
        out["cargo_toml_error"] = f"check error: {e}"
    return out


def count_drift_events(events: list[dict[str, Any]], target: pathlib.Path) -> dict[str, int]:
    """Per-write-tool-call drift signatures.

    All counts are over the WRITE tool invocations only. Drift modes
    catalogued in research3_drift_catalog.md:
      - #1: one-byte mutation in path
      - #2: phantom directory
      - #5: XML-attribute leak in args
      - #7: `lean://` / `lean ` prefix
      - #9: empty parameter
      - #11: whitespace/newline collapse (content)
      - bash-as-content: model wrote a bash command into file content
    """
    counts = {
        "write_calls": 0,
        "write_empty_path": 0,
        "write_path_drift_from_target": 0,
        "write_path_has_literal_space": 0,
        "write_content_starts_with_lean": 0,
        "write_content_is_bash_command": 0,
        "write_content_xml_attr_leak": 0,
        "write_content_newlines_collapsed_toml": 0,
    }
    target_prefix = str(target.resolve())
    for e in events:
        if e.get("type") != "tool_use":
            continue
        p = e.get("part", {})
        if p.get("tool") != "write":
            continue
        counts["write_calls"] += 1
        st = p.get("state", {})
        if not isinstance(st, dict):
            continue
        ip = st.get("input", {}) or {}
        if not isinstance(ip, dict):
            continue
        path = (ip.get("filePath") or ip.get("path") or "").strip()
        content = ip.get("content") or ""

        if not path:
            counts["write_empty_path"] += 1
        else:
            # opencode runs with --dir <target>, so the model may emit
            # relative paths (Cargo.toml, src/main.rs). Resolve those
            # against the target directory, not score_run.py's cwd.
            try:
                p_obj = pathlib.Path(path)
                if not p_obj.is_absolute():
                    p_obj = pathlib.Path(target) / p_obj
                resolved = str(p_obj.resolve())
            except Exception:
                resolved = path
            if not resolved.startswith(target_prefix):
                counts["write_path_drift_from_target"] += 1
            if " " in path:
                counts["write_path_has_literal_space"] += 1

        if content[:5].lower().startswith("lean ") or content[:5].lower() == "lean":
            counts["write_content_starts_with_lean"] += 1
        # Heuristic bash-as-content: content looks like a shell command
        # (no TOML/Rust structure, starts with a known shell verb).
        bash_starters = (
            "cargo ", "ls ", "rm ", "mkdir", "cd ", "echo ", "cat ",
            "grep ", "find ", "python3 ", "sh ",
        )
        cstrip = content.lstrip()
        if any(cstrip.startswith(s) for s in bash_starters):
            counts["write_content_is_bash_command"] += 1
        # XML attribute leak: drift mode #5
        if 'filePath="' in content or 'content="' in content[:200]:
            counts["write_content_xml_attr_leak"] += 1
        # TOML newline collapse: if Cargo.toml and section header `[package]`
        # appears on a line with key=value following (no newline between).
        if path.endswith("Cargo.toml") or "Cargo.toml" in (path or ""):
            if re.search(r"\[\w+\][^\n]+=", content):
                counts["write_content_newlines_collapsed_toml"] += 1
    return counts


def count_tool_calls(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts + per-turn breakdown.

    `tool_calls_per_turn[i]` = count of tool_use events that happened
    during the i-th assistant message in the opencode session. Detects
    "model produced N tool calls in turn 0 then collapsed at turn 1".
    """
    by_tool: dict[str, int] = {}
    per_turn: dict[str, int] = {}      # messageID → count
    turn_order: list[str] = []         # first-seen order
    for e in events:
        if e.get("type") == "tool_use":
            tool = (e.get("part", {}) or {}).get("tool", "?")
            by_tool[tool] = by_tool.get(tool, 0) + 1
            mid = (e.get("part", {}) or {}).get("messageID")
            if mid:
                if mid not in per_turn:
                    per_turn[mid] = 0
                    turn_order.append(mid)
                per_turn[mid] += 1
    per_turn_counts = [per_turn[m] for m in turn_order]
    return {
        "total": sum(by_tool.values()),
        "by_tool": by_tool,
        "tool_calls_per_turn": per_turn_counts,
        "turns_observed": len(turn_order),
    }


def _free_port() -> int:
    """Acquire an OS-assigned ephemeral port that is free *right now*.

    Bind 127.0.0.1:0, read the assigned port, close. The scorer then runs
    the agent's server on this port and curls it. Using a fresh per-run port
    (instead of a fixed 3001) is what makes the webserver test self-isolating:
    a leaked/zombie server from a prior run can never occupy it, so the scorer
    can never (a) collide and EADDRINUSE-panic its own server while (b) a stale
    holder answers curl — the false-positive/false-negative bug that
    contaminated every prior webserver_ok number. SO_REUSEADDR is NOT set, so
    a port still actively held is not handed out.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _warm_target_dir() -> str:
    """SSOT for the shared, pre-warmed cargo target directory.

    Mirrors warm_cargo_cache.sh's ATLAS_WARM_TARGET_DIR (same env var, same
    explicit default). When the warm cache has been populated, exporting
    CARGO_TARGET_DIR to this path makes each per-project `cargo build` reuse
    the already-compiled dependency rlibs (libc/proc-macro2/hyper/tokio/axum/…)
    instead of cold-compiling the whole tree — turning a ~150-300s cold build
    under CPU contention (which blows the timeout on VALID code) into a few-
    second incremental build of just the project's own crate.
    """
    return os.environ.get(
        "ATLAS_WARM_TARGET_DIR",
        os.path.join(os.path.expanduser("~"), ".cargo", "atlas-warm-target"),
    )


def _build_timeout_s() -> int:
    """SSOT for the scorer's `cargo build` timeout.

    Default raised to 600s so a cold build under heavy CPU contention (the
    realistic worst case when the warm cache is cold or a generation pins an
    off-version dep) completes rather than being mislabeled as build_ok=false.
    With the warm target dir the typical build is a few seconds, so the high
    ceiling only ever bites pathological cold cases. Override via
    ATLAS_WS_BUILD_TIMEOUT for de-confounding experiments.
    """
    try:
        return int(os.environ.get("ATLAS_WS_BUILD_TIMEOUT", "600"))
    except ValueError:
        return 600


def webserver_test(target: pathlib.Path, port: int, timeout_s: int = 15) -> dict[str, Any]:
    """Build + run the just-written Axum project and verify /ping → 'pong'.

    Steps:
      1. Acquire a fresh ephemeral port (self-isolating — see `_free_port`).
         The passed `port` is advisory only; the OS-assigned port is
         authoritative and recorded in `port_used`.
      2. Run `cargo build --release` in the target dir (must succeed). Builds
         reuse the shared warm target dir (CARGO_TARGET_DIR) so dependency
         compilation is incremental — see `_warm_target_dir`.
      3. Spawn `cargo run --release` as background process with
         ATLAS_HARNESS_PORT={port} env. The prompt instructed the model
         to read this env var; if the model misread the instruction the
         server binds elsewhere and /ping curl fails — that's a valid
         "webserver_ok=false" signal.
      4. Poll `http://127.0.0.1:{port}/ping` for up to `timeout_s`s.
      5. Assert response body contains "pong".
      6. Kill the server process group (and any cargo subprocesses).

    Returns the test verdict + diagnostic strings. The build step is
    the expensive part (~30-60s); skip via webserver_ok=false if
    Cargo.toml didn't validate.
    """
    import os
    import signal
    import tempfile
    import time
    out: dict[str, Any] = {
        "webserver_ok": False,
        "build_ok": False,
        "build_error": "",
        "bind_ok": False,
        "ping_ok": False,
        "ping_response": "",
        "port_used": 0,
        "server_stderr": "",
    }
    if not (target / "Cargo.toml").exists() or not (target / "src").exists():
        out["build_error"] = "no Cargo.toml or src/ — skipping webserver test"
        return out

    # Phase 0: self-isolating ephemeral port (overrides the advisory `port`).
    port = _free_port()
    out["port_used"] = port

    # Shared warm target dir → incremental dependency builds. Routing
    # CARGO_TARGET_DIR at the warm dir means build AND run share the same
    # already-compiled rlibs, so phase 1 (build) is fast and phase 2 (run)
    # does not re-link from scratch.
    warm_target = _warm_target_dir()
    build_timeout = _build_timeout_s()
    build_env = {
        **os.environ,
        "ATLAS_HARNESS_PORT": str(port),
        "CARGO_TARGET_DIR": warm_target,
    }

    # Phase 1: build
    try:
        build_proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(target),
            capture_output=True,
            timeout=build_timeout,
            env=build_env,
        )
    except subprocess.TimeoutExpired:
        out["build_error"] = f"cargo build timed out (>{build_timeout}s)"
        return out
    except Exception as e:
        out["build_error"] = f"cargo build launch error: {e}"
        return out
    if build_proc.returncode != 0:
        # Truncate stderr to first ~600 chars for the report
        err = (build_proc.stderr or b"").decode("utf-8", errors="replace")
        out["build_error"] = err[:600]
        return out
    out["build_ok"] = True

    # Phase 2: spawn server in background. Capture stderr to a temp file so a
    # bind panic (e.g. EADDRINUSE / "Address already in use") is recorded as a
    # distinct, diagnosable failure instead of being silently swallowed and
    # mislabeled as a generic "didn't respond" timeout.
    env = {
        **os.environ,
        "ATLAS_HARNESS_PORT": str(port),
        "RUST_LOG": "warn",
        "CARGO_TARGET_DIR": warm_target,
    }
    server_err = tempfile.NamedTemporaryFile(
        mode="w+", prefix="atlas-ws-stderr-", suffix=".log", delete=False
    )
    try:
        server = subprocess.Popen(
            ["cargo", "run", "--release"],
            cwd=str(target),
            stdout=subprocess.DEVNULL,
            stderr=server_err,
            env=env,
            preexec_fn=os.setsid,  # so we can kill the whole process group
        )
    except Exception as e:
        out["build_error"] = f"cargo run launch error: {e}"
        server_err.close()
        return out

    try:
        # Phase 3: poll /ping
        url = f"http://127.0.0.1:{port}/ping"
        deadline = time.time() + timeout_s
        last_err = ""
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                r = subprocess.run(
                    ["curl", "-sS", "-m", "2", url],
                    capture_output=True,
                    timeout=4,
                )
                if r.returncode == 0:
                    out["bind_ok"] = True
                    body = (r.stdout or b"").decode("utf-8", errors="replace").strip()
                    out["ping_response"] = body[:200]
                    if "pong" in body.lower():
                        out["ping_ok"] = True
                        out["webserver_ok"] = True
                    break
                else:
                    last_err = (r.stderr or b"").decode("utf-8", errors="replace")[:200]
            except Exception as e:
                last_err = str(e)[:200]
        if not out["bind_ok"]:
            out["ping_response"] = f"timeout: {last_err}"
    finally:
        # Phase 4: tear down — kill the whole process group
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except Exception:
            pass
        # Collect the server's stderr (bind panics etc.) for diagnosis.
        try:
            server_err.flush()
            server_err.seek(0)
            err_txt = server_err.read()
            out["server_stderr"] = err_txt[-800:]
            if not out["bind_ok"] and (
                "Address already in use" in err_txt or "EADDRINUSE" in err_txt
            ):
                out["build_error"] = (out.get("build_error") or "") + " | server bind failed (port in use)"
        except Exception:
            pass
        finally:
            try:
                server_err.close()
                os.unlink(server_err.name)
            except Exception:
                pass
    return out


def atlas_log_metrics(log_text: str) -> dict[str, Any]:
    """Extract atlas-side counters from the captured docker log window.

    `runaway_length_capped_turns` + `max_turn_output_tokens` make the dominant
    webserver_ok wall-time driver VISIBLE per run: a deep-context FP8
    degeneration where the model leaks repeated tool-call XML as plain text and
    runs the final turn to the max_tokens cap (~8200 tokens @ ~31 tok/s ≈ 260s).
    Surfacing it here means the slow-run cost is attributable to the MODEL floor
    in the JSON record itself, not buried in the raw docker log — no masking.
    """
    # `... Done: <N> tokens (length) ...` is the scheduler's per-turn finish
    # line; finish_reason "(length)" == hit the max_tokens cap (the runaway).
    length_capped = log_text.count("(length)")
    # Largest single-turn token count = the runaway turn. Parse the
    # authoritative `Done: <N> tokens (<reason>)` finish line — `output_tokens=`
    # is logged only intermittently and `docker logs --since` can clip it, so
    # `Done:` is the reliable source for the per-turn maximum.
    max_out = 0
    for m in re.finditer(r"Done:\s+(\d+)\s+tokens", log_text):
        max_out = max(max_out, int(m.group(1)))
    return {
        "ws1_mask_active_fires": log_text.count("ws1/am1 mask active"),
        "b1_drift_gauge_fires": log_text.count("B1 drift gauge"),
        "tier_5c_retries": log_text.count("Tier 5c (stream): retry produced") + log_text.count("Tier 5c: retry produced"),
        "a2_fuzzy_repair_fires": log_text.count("A2 fuzzy_repair: rescued"),
        "doom_loop_trips": log_text.count("Bug-2 name-run cap tripped"),
        "tool_call_lines": log_text.count("Tool call: "),
        "runaway_length_capped_turns": length_capped,
        "max_turn_output_tokens": max_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--target", required=True, type=pathlib.Path)
    ap.add_argument("--opencode-json", required=True, type=pathlib.Path)
    ap.add_argument("--opencode-stderr", required=True, type=pathlib.Path)
    ap.add_argument("--atlas-log-window", required=True, type=pathlib.Path)
    ap.add_argument("--probe-start-ts", required=True, type=float)
    ap.add_argument("--probe-end-ts", required=True, type=float)
    ap.add_argument("--webserver-port", type=int, default=3001,
                    help="port the Axum server should bind to; harness curls /ping here")
    ap.add_argument("--skip-webserver", action="store_true",
                    help="skip the build + run + /ping webserver check")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    events = load_events(args.opencode_json)
    files = find_files_written(args.target)
    cargo = cargo_check(args.target)
    drift = count_drift_events(events, args.target)
    tools = count_tool_calls(events)
    # Process-fidelity: did the AGENT itself build/test/run/curl/teardown as the
    # prompt instructed? Orthogonal to webserver_ok (outcome). Fully guarded.
    try:
        followed = (
            _compute_followed(events, args.target)
            if _compute_followed is not None
            else {"followed_directions": None, "error": "module-unavailable"}
        )
    except Exception as _e:  # pragma: no cover — never break scoring
        followed = {"followed_directions": None, "error": f"compute-error: {_e}"}
    atlas_log_text = (
        args.atlas_log_window.read_text(errors="replace") if args.atlas_log_window.exists() else ""
    )
    atlas = atlas_log_metrics(atlas_log_text)

    # Webserver test: only if cargo_toml_valid (no point building if TOML
    # doesn't parse) and not skipped.
    if cargo.get("cargo_toml_valid") and not args.skip_webserver:
        ws = webserver_test(args.target, args.webserver_port)
    else:
        ws = {
            "webserver_ok": False,
            "build_ok": False,
            "build_error": "" if args.skip_webserver else "cargo_toml not valid; webserver test skipped",
            "bind_ok": False,
            "ping_ok": False,
            "ping_response": "",
        }

    record = {
        "tier": args.tier,
        "run": args.run,
        "wall_time_s": args.probe_end_ts - args.probe_start_ts,
        "opencode": {
            "events_total": len(events),
            "stderr_bytes": args.opencode_stderr.stat().st_size if args.opencode_stderr.exists() else 0,
        },
        "filesystem": {
            "files_written": files,
            "files_count": len(files),
        },
        "cargo": cargo,
        "webserver": ws,
        "followed_directions": followed,
        "tool_calls": tools,
        "drift": drift,
        "atlas": atlas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
