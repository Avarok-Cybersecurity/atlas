#!/usr/bin/env python3
"""Concurrent agentic harness — measures how Atlas holds up with C agents at once.

The sequential harness (oc_harness.py) can't answer concurrency questions, and its
Rust/axum task is unusable in parallel: cargo serializes on the shared target dir
and a single build already costs 80-370s, so wall-clock measures cargo contention
rather than the server. Python is near-free to verify (`python3 -m unittest`), and
C# turns out to be too: 8 parallel `dotnet new` + `dotnet run` complete in ~3s
total on this box, so it gives compiled-language coverage without the pathology.

Each concurrent slot gets a DISTINCT task from the pool, so the agents do not all
send byte-identical prompts — otherwise the prefix cache would dedupe the whole
batch and the run would measure cache hits instead of concurrent work.

Reports per-level: pass rate, per-task latency (median/p95), batch makespan,
speedup vs C=1, plus server-side KV health and decode throughput scraped from the
container log for exactly the batch's time window.

Usage:
  python3 conc_harness.py --levels 1,4,8
  python3 conc_harness.py --levels 4 --repeats 3 --include-csharp
  python3 conc_harness.py --levels 8 --python-only
"""
import argparse, json, os, re, shutil, statistics as st, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OC = os.environ.get("OPENCODE_BIN") or shutil.which("opencode") or "opencode"
CONTAINER = os.environ.get("ATLAS_CONTAINER", "laguna-s")
PROVIDER = os.environ.get("OPENCODE_PROVIDER", "atlas")

# ── Task pool ────────────────────────────────────────────────────────────────
# (name, kind, prompt, verify_argv). Kept deliberately similar in size/shape so
# per-task latencies are comparable across slots; distinct enough that no two
# agents send the same prompt.
PY = "python3"
PY_TASKS = [
    ("calc", "unittest",
     "Create a Python file 'calc.py' with functions add, subtract, multiply, and divide "
     "(divide raises ZeroDivisionError on a zero divisor). Then create 'test_calc.py' using "
     "Python's built-in unittest module (NOT pytest) covering normal and edge cases (divide "
     "by zero, negatives, floats). Run `python3 -m unittest test_calc -v` and ensure all pass.",
     [PY, "-m", "unittest", "test_calc", "-v"]),
    ("sorter", "unittest",
     "Create 'sortlib.py' with bubble_sort(arr), merge_sort(arr), and binary_search(arr, target) "
     "returning the index or -1. Then create 'test_sort.py' using unittest with tests for: empty "
     "list, single element, already sorted, reverse sorted, duplicates, and binary_search hit/miss. "
     "Run `python3 -m unittest test_sort -v` and ensure all tests pass.",
     [PY, "-m", "unittest", "test_sort", "-v"]),
    ("textstats", "unittest",
     "Create 'textstats.py' with word_count(text), char_frequency(text) returning a dict, and "
     "longest_word(text) (ties -> first). Then create 'test_textstats.py' using Python's built-in "
     "unittest module (NOT pytest) covering empty string, punctuation, mixed case, and ties. "
     "Run `python3 -m unittest test_textstats -v` and ensure all tests pass.",
     [PY, "-m", "unittest", "test_textstats", "-v"]),
    ("matrix", "unittest",
     "Create 'matrix.py' with transpose(m), multiply(a, b) (raise ValueError on shape mismatch), "
     "and identity(n). Then create 'test_matrix.py' using Python's built-in unittest module (NOT "
     "pytest) covering non-square matrices, the shape-mismatch error, and identity edge cases. "
     "Run `python3 -m unittest test_matrix -v` and ensure all tests pass.",
     [PY, "-m", "unittest", "test_matrix", "-v"]),
    ("lru", "unittest",
     "Create 'lru.py' with an LRUCache class supporting get(key) (-1 when missing), put(key, value), "
     "and a fixed capacity that evicts the least-recently-used entry. Then create 'test_lru.py' using "
     "Python's built-in unittest module (NOT pytest) covering eviction order, updating an existing "
     "key, and capacity 1. Run `python3 -m unittest test_lru -v` and ensure all tests pass.",
     [PY, "-m", "unittest", "test_lru", "-v"]),
    ("roman", "unittest",
     "Create 'roman.py' with to_roman(n) for 1..3999 and from_roman(s), each raising ValueError on "
     "invalid input. Then create 'test_roman.py' using Python's built-in unittest module (NOT pytest) "
     "covering subtractive forms (4, 9, 40, 90, 400, 900), round-tripping, and invalid inputs. "
     "Run `python3 -m unittest test_roman -v` and ensure all tests pass.",
     [PY, "-m", "unittest", "test_roman", "-v"]),
    ("intervals", "unittest",
     "Create 'intervals.py' with merge(intervals) merging overlapping [start, end] pairs and "
     "insert(intervals, new) inserting into a sorted non-overlapping list. Then create "
     "'test_intervals.py' using Python's built-in unittest module (NOT pytest) covering empty input, "
     "touching intervals, and full containment. Run `python3 -m unittest test_intervals -v` and "
     "ensure all tests pass.",
     [PY, "-m", "unittest", "test_intervals", "-v"]),
    ("stack", "unittest",
     "Create 'stackcalc.py' with evaluate_rpn(tokens) evaluating reverse-Polish notation with "
     "+ - * / (integer division truncating toward zero), raising ValueError on malformed input. "
     "Then create 'test_stackcalc.py' using Python's built-in unittest module (NOT pytest) covering "
     "negative results, division truncation, and malformed input. "
     "Run `python3 -m unittest test_stackcalc -v` and ensure all tests pass.",
     [PY, "-m", "unittest", "test_stackcalc", "-v"]),
]

CS_TASKS = [
    # NOTE: a third, harder task (`cs_orders`: 3-file project, LINQ, System.Text.Json,
    # async) lived here and was REMOVED — it drove the model into the
    # `bash({"command":true})` retry loop in 6 of 6 appearances (C=1/4/8 across two
    # sweeps), so it measured that degeneration rather than concurrency. The loop is
    # NOT specific to it: `matrix`, `roman` and `textstats` were also loop-killed at
    # C=8, i.e. the behaviour correlates with concurrency, not with one prompt.
    ("cs_calc", "dotnet",
     "In the current directory run `dotnet new console -o app` to scaffold a C# console project. "
     "Edit app/Program.cs so it defines a static class Calc with methods Add, Subtract, Multiply and "
     "Divide(int a, int b) (Divide throws DivideByZeroException when b is 0), and a Main that "
     "exercises every method including the divide-by-zero case using try/catch. Main must print "
     "exactly ALL TESTS PASSED as its final line when every check succeeds. "
     "Then run `cd app && dotnet run` and make sure it prints ALL TESTS PASSED.",
     None),
    ("cs_strings", "dotnet",
     "In the current directory run `dotnet new console -o app` to scaffold a C# console project. "
     "Edit app/Program.cs so it defines a static class TextUtil with Reverse(string), "
     "IsPalindrome(string) (ignoring case and non-letters) and WordCount(string), and a Main that "
     "exercises each with normal and edge cases (empty string, punctuation, mixed case). Main must "
     "print exactly ALL TESTS PASSED as its final line when every check succeeds. "
     "Then run `cd app && dotnet run` and make sure it prints ALL TESTS PASSED.",
     None),
]


def _tool_signature(ev):
    """Stable identity of a tool call, for loop detection."""
    if ev.get("type") not in ("tool", "tool_use", "tool_call"):
        return None
    name = ev.get("name") or ev.get("tool") or ""
    args = ev.get("input") or ev.get("arguments") or ev.get("args") or ""
    if not isinstance(args, str):
        try:
            args = json.dumps(args, sort_keys=True)
        except Exception:
            args = str(args)
    return f"{name}({args})"[:400]


def run_agent(prompt, workdir, timeout, model, max_repeat=0):
    """Drive one opencode session, streaming so we can cut off a stuck agent.

    Two independent stops:
      * `timeout`   — hard wall per task.
      * `max_repeat`— identical tool call repeated this many times in a row.

    The second exists because an agent can burn the ENTIRE wall budget in a
    tight retry loop and still look 'busy'. Observed live: the model emitted
    `bash({"command":true})` (a literal boolean where a command string belongs),
    opencode ran it, errored, and retried — 91 identical calls over 3.5 minutes,
    each a 30-token generation against an 8.4K-token prompt. That is pure waste
    that also distorts every concurrency number in the run.
    """
    # Provider-qualify unless already qualified. A bare `"/" in model` test is
    # WRONG: HF-style ids contain a slash themselves (Hcompany/Holo-3.1-...),
    # so that heuristic passed them through unprefixed and opencode exited in
    # ~1s with every task failing before the agent ever started.
    m = model if model.startswith(f"{PROVIDER}/") else f"{PROVIDER}/{model}"
    cmd = [OC, "run", "--auto", "--format", "json", "--dir", str(workdir), "-m", m, prompt]
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1",
               ATLAS_HARNESS_PORT="3001")
    t0 = time.time()
    events, last_sig, repeats, killed = [], None, 0, ""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, cwd=str(workdir), env=env)
    # Hard wall-clock watchdog. The in-loop `time.time() - t0 > timeout` check
    # below is NOT sufficient on its own: `for line in p.stdout` blocks in
    # readline, so that check only runs when the child actually emits something.
    # An agent that hangs SILENTLY (observed: opencode alive 15+ min, zero
    # requests reaching the server, no child processes) never yields a line, so
    # the loop parks forever and stalls the whole sweep. A timer thread kills
    # the child on schedule no matter what it is or isn't printing.
    timed_out = threading.Event()

    def _fire():
        timed_out.set()
        try:
            p.kill()
        except Exception:
            pass

    watchdog = threading.Timer(timeout, _fire)
    watchdog.daemon = True
    watchdog.start()
    try:
        for line in p.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                events.append(ev)
                sig = _tool_signature(ev)
                if sig:
                    repeats = repeats + 1 if sig == last_sig else 1
                    last_sig = sig
                    if max_repeat and repeats >= max_repeat:
                        killed = f"loop: {sig[:90]} x{repeats}"
                        p.kill()
                        break
            if time.time() - t0 > timeout:
                killed = "timeout"
                p.kill()
                break
    finally:
        watchdog.cancel()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
    if timed_out.is_set() and not killed:
        killed = "timeout"
    wall = time.time() - t0
    if killed:
        return (124 if killed == "timeout" else 125), events, wall, killed
    return p.returncode, events, wall, ""


def count_tools(events):
    return sum(1 for e in events if e.get("type") in ("tool", "tool_use", "tool_call"))


def verify(kind, workdir, argv):
    """Return (passed, note)."""
    try:
        if kind == "unittest":
            r = subprocess.run(argv, cwd=str(workdir), capture_output=True, text=True, timeout=120)
            return r.returncode == 0, (r.stderr or r.stdout or "")[-200:]
        if kind == "dotnet":
            app = Path(workdir) / "app"
            if not app.is_dir():
                return False, "no app/ project scaffolded"
            r = subprocess.run(["dotnet", "run"], cwd=str(app), capture_output=True,
                               text=True, timeout=300,
                               env=dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1",
                                        DOTNET_NOLOGO="1"))
            ok = r.returncode == 0 and "ALL TESTS PASSED" in (r.stdout or "")
            return ok, (r.stdout or r.stderr or "")[-200:]
    except subprocess.TimeoutExpired:
        return False, "verify timeout"
    except Exception as e:
        return False, f"verify error: {e}"
    return False, "unknown kind"


def kv_counters():
    out = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True)
    log = (out.stdout or "") + (out.stderr or "")
    return {
        "exhaustions": log.count("no free blocks"),
        "decref": log.count("dec_ref on block"),
        "preempts": log.count("preempting slot"),
        "evict_unowned": log.count("returned a block it holds no reference on"),
    }


TOKS = re.compile(r"Done: (\d+) tokens .*?([\d.]+) tok/s")


def require_container():
    """Fail loudly when CONTAINER names something that isn't running.

    Throughput is scraped from the server's own log, so a wrong container name
    yields zero matches — which renders as a clean `0.0 tok/s` table rather than
    an error. That looks exactly like a real measurement and silently wasted a
    full sweep. Check once, up front, and say which name to set.
    """
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                         capture_output=True, text=True)
    names = (out.stdout or "").split()
    if CONTAINER not in names:
        sys.exit(f"ATLAS_CONTAINER={CONTAINER!r} is not a running container "
                 f"(running: {', '.join(names) or 'none'}). Throughput is read from "
                 f"its log, so the sweep would report 0.0 tok/s. Set ATLAS_CONTAINER.")


def decode_stats(since_s):
    """Per-stream decode rates the server reported during the batch window."""
    out = subprocess.run(["docker", "logs", "--since", f"{int(since_s)+2}s", CONTAINER],
                         capture_output=True, text=True)
    log = (out.stdout or "") + (out.stderr or "")
    rates, toks = [], 0
    for m in TOKS.finditer(log):
        toks += int(m.group(1))
        rates.append(float(m.group(2)))
    return toks, rates


def run_level(conc, tasks, model, timeout, keep, max_repeat=0):
    """Fire `conc` agents at once, one distinct task each."""
    picked = [tasks[i % len(tasks)] for i in range(conc)]
    workdirs = [Path(tempfile.mkdtemp(prefix=f"conc-{n}-")) for n, _, _, _ in picked]

    def one(i):
        name, kind, prompt, argv = picked[i]
        wd = workdirs[i]
        t0 = time.time()
        rc, events, wall, err = run_agent(prompt, wd, timeout, model, max_repeat)
        ok, note = verify(kind, wd, argv)
        # Keep BOTH reasons: `err` carries why the agent was cut short (loop /
        # timeout) and `note` carries what verification saw. Reporting only the
        # latter hid the loop diagnosis behind an unmodified template's output.
        why = "loop-killed" if rc == 125 else ("timeout" if rc == 124 else "")
        detail = "; ".join(x for x in (err, note) if x)
        return {"task": name, "kind": kind, "ok": ok, "wall": round(wall, 1),
                "tools": count_tools(events), "rc": rc, "start": t0, "why": why,
                "note": detail.replace("\n", " ")[:160]}

    t_batch = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, range(conc)))
    makespan = time.time() - t_batch
    toks, rates = decode_stats(makespan)
    if not keep:
        for wd in workdirs:
            shutil.rmtree(wd, ignore_errors=True)
    return res, makespan, toks, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,8", help="comma-separated concurrency levels")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="laguna-s-2.1")
    ap.add_argument("--timeout", type=int, default=480,
                    help="hard wall per task (s); a looping agent otherwise burns all of it")
    ap.add_argument("--max-repeat", type=int, default=6,
                    help="kill a task after N identical tool calls in a row (0 = off)")
    ap.add_argument("--include-csharp", action="store_true", help="mix C# tasks into the pool")
    ap.add_argument("--csharp-only", action="store_true")
    ap.add_argument("--python-only", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    require_container()

    if args.csharp_only:
        pool = CS_TASKS
    elif args.include_csharp and not args.python_only:
        # INTERLEAVE, don't concatenate. `run_level` takes the first C entries,
        # so appending C# after 8 Python tasks meant a C=8 run was all Python and
        # the C# tasks never executed below C=9. Round-robin so every level gets
        # a language mix, hardest C# task first.
        pool, i, j = [], 0, 0
        while i < len(PY_TASKS) or j < len(CS_TASKS):
            if j < len(CS_TASKS):
                pool.append(CS_TASKS[j]); j += 1
            for _ in range(3):              # ~3 Python per C# keeps the mix realistic
                if i < len(PY_TASKS):
                    pool.append(PY_TASKS[i]); i += 1
    else:
        pool = PY_TASKS

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    base0 = kv_counters()
    all_rows, baseline = [], None

    for c in levels:
        if c > len(pool):
            print(f"  note: C={c} exceeds the {len(pool)}-task pool; tasks will repeat", flush=True)
        for rep in range(1, args.repeats + 1):
            res, makespan, toks, rates = run_level(c, pool, args.model, args.timeout,
                                                   args.keep, args.max_repeat)
            walls = sorted(r["wall"] for r in res)
            npass = sum(1 for r in res if r["ok"])
            p95 = walls[min(len(walls) - 1, int(round(0.95 * (len(walls) - 1))))]
            row = {"conc": c, "rep": rep, "pass": npass, "n": len(res),
                   "median_wall": st.median(walls), "p95_wall": p95,
                   "makespan": round(makespan, 1), "tokens": toks,
                   "median_tok_s": round(st.median(rates), 1) if rates else 0.0,
                   "agg_tok_s": round(toks / makespan, 1) if makespan else 0.0,
                   "tasks": res}
            all_rows.append(row)
            if c == 1 and baseline is None:
                baseline = row["median_wall"]
            print(f"  C={c:<2} rep{rep}: {npass}/{len(res)} pass  makespan={makespan:6.1f}s  "
                  f"median={row['median_wall']:6.1f}s  p95={p95:6.1f}s  "
                  f"decode median={row['median_tok_s']:5.1f} tok/s  agg={row['agg_tok_s']:6.1f} tok/s",
                  flush=True)
            nloop = sum(1 for r in res if r.get("why") == "loop-killed")
            if nloop:
                print(f"      ({nloop} task(s) cut short by the repeat-loop detector)", flush=True)
            for r in res:
                if not r["ok"]:
                    tag = f"[{r['why']}] " if r.get("why") else ""
                    print(f"      FAIL {r['task']}: {tag}rc={r['rc']} {r['note']}", flush=True)

    print(f"\n{'=' * 78}\n  CONCURRENCY SUMMARY\n{'=' * 78}")
    print(f"{'C':>3} {'pass':>8} {'median s':>10} {'p95 s':>8} {'makespan':>10} "
          f"{'tok/s med':>10} {'tok/s agg':>10} {'vs C=1':>8}")
    for c in levels:
        rows = [r for r in all_rows if r["conc"] == c]
        if not rows:
            continue
        med = st.median([r["median_wall"] for r in rows])
        print(f"{c:>3} {sum(r['pass'] for r in rows):>4}/{sum(r['n'] for r in rows):<3} "
              f"{med:>10.1f} {st.median([r['p95_wall'] for r in rows]):>8.1f} "
              f"{st.median([r['makespan'] for r in rows]):>10.1f} "
              f"{st.median([r['median_tok_s'] for r in rows]):>10.1f} "
              f"{st.median([r['agg_tok_s'] for r in rows]):>10.1f} "
              f"{(med / baseline if baseline else float('nan')):>7.2f}x")

    c1 = kv_counters()
    print("\n  KV over the whole run: " + "  ".join(
        f"{k}={c1[k] - base0[k]}" for k in sorted(c1)))
    print("  (per-task latency SHOULD rise with C — the question is whether aggregate")
    print("   throughput scales and whether pass rate and KV health hold.)")

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
