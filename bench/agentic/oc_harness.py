#!/usr/bin/env python3
"""
Agentic test harness driving OpenCode (`opencode run`) against a locally-served Atlas
model (OpenAI-compatible endpoint on :8888). Mirrors pool_harness.py's tasks +
verification + scoring, OpenCode variant of omp_harness.py — IDENTICAL tasks/verification/scoring so the
two harnesses are directly comparable. Vanilla stdlib only.

Tasks:
  - 'unittest'  : Python task, verified by running `python3 -m unittest`.
  - 'webserver' : Rust Axum ping/pong — build -> run on an ephemeral port ->
                  curl /ping == "pong" -> teardown, plus a followed-directions
                  score from the agent's shell commands. Uses a WARM
                  CARGO_TARGET_DIR so axum/tokio/hyper compile once.

Prereqs:
  - omp on PATH (~/.bun/bin) with provider 'openai-custom' -> :8888/v1 and the
    model id present in ~/.omp/agent/models.yml (laguna-s-2.1 / laguna-xs-2.1).
  - Atlas server up on :8888.
Usage:
  python3 omp_harness.py                         # all tasks, laguna-s-2.1
  python3 omp_harness.py --model laguna-xs-2.1
  python3 omp_harness.py --only webserver --keep
  python3 omp_harness.py --timeout 480 --json out.json
"""
import argparse, json, os, re, shutil, signal, socket, subprocess, tempfile, time
from pathlib import Path

OC = shutil.which("opencode") or "/home/ms/.nvm/versions/node/v24.15.0/bin/opencode"
WARM_TARGET = os.environ.get("OC_HARNESS_WARM_TARGET", "/home/ms/pool-harness-runs/warm-target")


def _sccache_env():
    sc = shutil.which("sccache") or "/home/ms/.cargo/bin/sccache"
    if not Path(sc).exists():
        return {}
    return {"RUSTC_WRAPPER": sc, "CARGO_INCREMENTAL": "0"}


SCCACHE_ENV = _sccache_env()

WEBSERVER_PROMPT = (
    "Please create a pure Rust Axum project here in the current working directory. "
    "Just have a ping/pong endpoint (GET /ping returns the body 'pong'). The server "
    "MUST bind to the port from the ATLAS_HARNESS_PORT env var (default 3001) — use "
    "`let port: u16 = std::env::var(\"ATLAS_HARNESS_PORT\").unwrap_or_else(|_| \"3001\".to_string()).parse().unwrap();` "
    "then bind to 0.0.0.0:port. Add tests, run them and prove all tests pass, then run "
    "the server and use curl to prove it works. Whenever you run the server or any "
    "long-lived process in the background, start it detached with output redirected to a "
    "file (e.g. `setsid cargo run > /tmp/server.log 2>&1 &`) so your shell never blocks, "
    "and wrap any command that might hang (curl checks, kills) in a short `timeout 15`. "
    "Finally, tear down the server by killing whatever is listening on its port rather "
    "than guessing the process name, wrapped in a short timeout, e.g. "
    "`timeout 5 fuser -k ${ATLAS_HARNESS_PORT:-3001}/tcp 2>/dev/null || true`."
)

# name, kind, prompt, verify_spec
TASKS = [
    ("calc", "unittest",
     "Create a Python file 'calc.py' with functions add, subtract, multiply, and divide "
     "(divide raises ZeroDivisionError on a zero divisor). Then create 'test_calc.py' using "
     "Python's built-in unittest module (NOT pytest) covering normal and edge cases (divide "
     "by zero, negatives, floats). Run `python3 -m unittest test_calc -v` and ensure all pass.",
     ["python3", "-m", "unittest", "test_calc", "-v"]),
    ("sorter", "unittest",
     "Create 'sortlib.py' with bubble_sort(arr), merge_sort(arr), and binary_search(arr, target) "
     "returning the index or -1. Then create 'test_sort.py' using unittest with tests for: empty "
     "list, single element, already sorted, reverse sorted, duplicates, and binary_search hit/miss. "
     "Run `python3 -m unittest test_sort -v` and ensure all tests pass.",
     ["python3", "-m", "unittest", "test_sort", "-v"]),
    ("webserver", "webserver", WEBSERVER_PROMPT, None),
]


# ── omp driver ───────────────────────────────────────────────────────────────
def run_omp(prompt, workdir, timeout, model, thinking=None):
    """Drive `opencode run`. Model is provider-qualified (atlas/<id>)."""
    m = model if "/" in model else f"atlas/{model}"
    cmd = [OC, "run", "--auto", "--format", "json", "--dir", str(workdir), "-m", m, prompt]
    env = {**os.environ,
           "ATLAS_HARNESS_PORT": "3001",
           "CARGO_TARGET_DIR": WARM_TARGET,
           **SCCACHE_ENV}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 124, (e.stdout or ""), (e.stderr or "")
    wall = time.time() - t0
    events = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            events.append(json.loads(ln))
        except json.JSONDecodeError:
            # Non-JSON line in the event stream (banner text, or a partial line
            # from a truncated run). Skip it; the surrounding events are usable.
            pass
    return rc, events, wall, err


def shell_cmds(events):
    """Agent's executed bash commands (opencode `bash` tool)."""
    out = []
    for e in events:
        if e.get("type") != "tool_use":
            continue
        part = e.get("part") or {}
        if part.get("tool") != "bash":
            continue
        c = ((part.get("state") or {}).get("input") or {}).get("command")
        if c:
            out.append(c)
    return out


def count_tools(events):
    return sum(1 for e in events if e.get("type") == "tool_use")


def had_tool_error(events):
    n = 0
    for e in events:
        if e.get("type") != "tool_use":
            continue
        st = ((e.get("part") or {}).get("state") or {}).get("status")
        if st and st != "completed":
            n += 1
    return n


# ── unittest verify ──────────────────────────────────────────────────────────
def verify_unittest(workdir, cmd):
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=60)
        return p.returncode == 0, (p.stderr or p.stdout)[-300:]
    except Exception as e:
        return False, str(e)[:200]


# ── webserver verify ─────────────────────────────────────────────────────────
def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def verify_webserver(workdir, build_timeout=420, run_timeout=20):
    wd = Path(workdir)
    out = {"cargo_valid": False, "webserver_ok": False, "note": ""}
    if not (wd / "Cargo.toml").exists():
        out["note"] = "no Cargo.toml"
        return out
    port = _free_port()
    env = {**os.environ, "ATLAS_HARNESS_PORT": str(port),
           "CARGO_TARGET_DIR": WARM_TARGET, **SCCACHE_ENV}
    try:
        b = subprocess.run(["cargo", "build", "--release"], cwd=workdir,
                           capture_output=True, timeout=build_timeout, env=env)
    except subprocess.TimeoutExpired:
        out["note"] = f"cargo build >{build_timeout}s"
        return out
    if b.returncode != 0:
        out["note"] = (b.stderr or b"").decode(errors="replace")[:300]
        return out
    out["cargo_valid"] = True
    srv = subprocess.Popen(["cargo", "run", "--release"], cwd=workdir,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env={**env, "RUST_LOG": "warn"}, preexec_fn=os.setsid)
    try:
        url = f"http://127.0.0.1:{port}/ping"
        deadline = time.time() + run_timeout
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                r = subprocess.run(["curl", "-sS", "-m", "2", url], capture_output=True, timeout=4)
                if r.returncode == 0:
                    body = (r.stdout or b"").decode(errors="replace").strip()
                    out["note"] = f"/ping -> {body[:40]!r}"
                    if "pong" in body.lower():
                        out["webserver_ok"] = True
                    break
            except (subprocess.SubprocessError, OSError):
                # Server not up yet: curl exits non-zero or times out until the
                # port is listening. Keep polling until the deadline.
                pass
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
            try:
                srv.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Process group already reaped (server exited on its own) or not
            # ours to signal. Teardown is best-effort; nothing left to kill.
            pass
    return out


# ── followed_directions ──────────────────────────────────────────────────────
_RE_TEST = re.compile(r"\bcargo\s+test\b")
_RE_RUN = re.compile(r"\bcargo\s+run\b")
_RE_CURL = re.compile(r"\bcurl\b")
_RE_KILL = re.compile(r"\bp?kill\b|\bkill\s+(?:%|-9|-TERM|-SIGTERM|\$|\d)|\bfuser\s+-k\b")


def followed_directions(events, workdir):
    wd = Path(workdir)
    blob = "\n".join(shell_cmds(events))
    srcs = list(wd.rglob("*.rs"))
    src_text = "\n".join(p.read_text(errors="replace") for p in srcs) if srcs else ""
    steps = {
        "wrote_project": (wd / "Cargo.toml").exists() and any(p.name == "main.rs" for p in srcs),
        "wrote_tests": "#[test]" in src_text or "#[tokio::test]" in src_text,
        "ran_tests": bool(_RE_TEST.search(blob)),
        "ran_server": bool(_RE_RUN.search(blob)),
        "curled": bool(_RE_CURL.search(blob)),
        "tore_down": bool(_RE_KILL.search(blob)),
        "reads_port_env": "ATLAS_HARNESS_PORT" in src_text,
    }
    req = ["wrote_project", "wrote_tests", "ran_tests", "ran_server", "curled", "tore_down"]
    return steps, all(steps[s] for s in req), req


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laguna-s-2.1")
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--only", default=None, help="run one task by name")
    ap.add_argument("--thinking", default=None, help="unused for opencode (kept for arg parity)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--base", default="/home/ms/oc-harness-runs")
    ap.add_argument("--json", default=None, help="write full results JSON here")
    args = ap.parse_args()
    Path(args.base).mkdir(parents=True, exist_ok=True)
    Path(WARM_TARGET).mkdir(parents=True, exist_ok=True)

    tasks = [t for t in TASKS if not args.only or t[0] == args.only]
    rows = []
    for name, kind, prompt, vspec in tasks:
        wd = tempfile.mkdtemp(prefix=f"{name}-", dir=args.base)
        print(f"\n=== {name} ({kind}) [{args.model}] ===", flush=True)
        rc, events, wall, err = run_omp(prompt, wd, args.timeout, args.model, args.thinking)
        ntools = count_tools(events)
        nerr = had_tool_error(events)
        rec = {"name": name, "kind": kind, "model": args.model, "rc": rc,
               "tools": ntools, "tool_errors": nerr, "wall": round(wall, 1)}
        if rc == 124:
            print(f"  TIMEOUT after {wall:.0f}s (tools={ntools})", flush=True)
        if kind == "unittest":
            ok, log = verify_unittest(wd, vspec)
            rec.update({"tests": ok, "log": log})
            print(f"  oc_rc={rc} tools={ntools} tool_err={nerr} "
                  f"tests={'PASS' if ok else 'FAIL'} {wall:.0f}s", flush=True)
            if not ok:
                print(f"    {log.strip()[-200:]}")
        else:
            ws = verify_webserver(wd)
            steps, followed, req = followed_directions(events, wd)
            nf = sum(1 for s in req if steps[s])
            rec.update({"cargo_valid": ws["cargo_valid"], "webserver_ok": ws["webserver_ok"],
                        "note": ws["note"], "steps": steps, "followed": followed,
                        "nf": nf, "nreq": len(req)})
            print(f"  oc_rc={rc} tools={ntools} tool_err={nerr} cargo_valid={ws['cargo_valid']} "
                  f"webserver_ok={ws['webserver_ok']} followed={nf}/{len(req)} {wall:.0f}s", flush=True)
            print(f"    {ws['note']}")
            for s in req + ["reads_port_env"]:
                print(f"    {'✓' if steps[s] else '✗'} {s}")
        rows.append(rec)
        if not args.keep:
            shutil.rmtree(wd, ignore_errors=True)

    print("\n=== SUMMARY ===")
    npass = 0
    for r in rows:
        if r["kind"] == "unittest":
            ok = r.get("tests")
            npass += bool(ok)
            print(f"  {r['name']:10} tests={'PASS' if ok else 'FAIL'} rc={r['rc']} "
                  f"tools={r['tools']} err={r['tool_errors']} {r['wall']:.0f}s")
        else:
            ok = r.get("webserver_ok")
            npass += bool(ok)
            print(f"  {r['name']:10} cargo_valid={r.get('cargo_valid')} webserver_ok={ok} "
                  f"followed={r.get('nf')}/{r.get('nreq')} rc={r['rc']} tools={r['tools']} {r['wall']:.0f}s")
    print(f"  ── {npass}/{len(rows)} tasks passed ──")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
