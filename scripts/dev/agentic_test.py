#!/usr/bin/env python3
"""
Atlas Agentic Multi-Turn Session Tester

Simulates a Claude Code / OpenCode style agentic session:
- Multi-turn tool calling (write_file, read_file, run_command)
- Actually executes tools and feeds results back
- Builds a real project and verifies it compiles/runs
- Tests coherence across 10+ turns

Usage: python3 agentic_test.py [--anthropic] [--iterations N]
"""
import json, urllib.request, subprocess, os, sys, time, shutil, re, argparse
from pathlib import Path

PORT = 8888
BASE_URL = f"http://localhost:{PORT}"
WORK_DIR = Path("/tmp/atlas-agentic-test")
MAX_TURNS = 15

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file. Creates parent directories if needed.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to project root"},
            "content": {"type": "string", "description": "File content to write"}
        }, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read content of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to project root"}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the project directory. Returns stdout+stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to execute"}
        }, "required": ["command"]}
    }},
]

def api_call(messages, max_tokens=8000):
    body = json.dumps({
        "model": "test",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def execute_tool(name, args):
    """Execute a tool call and return the result string."""
    if name == "write_file":
        path = WORK_DIR / args["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return f"File written: {args['path']} ({len(args['content'])} bytes)"
    elif name == "read_file":
        path = WORK_DIR / args["path"]
        if path.exists():
            return path.read_text()[:4000]
        return f"Error: file not found: {args['path']}"
    elif name == "run_command":
        try:
            r = subprocess.run(
                args["command"], shell=True, capture_output=True, text=True,
                timeout=30, cwd=str(WORK_DIR))
            output = r.stdout + r.stderr
            return output[:4000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s)"
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {name}"

def run_session(task, language, verify_cmd):
    """Run a full agentic session and return (success, turns, summary)."""
    # Clean workspace
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    messages = [
        {"role": "system", "content": (
            "You are a coding assistant. Use tools to write code files. "
            "RULES: 1) Write ONE file per tool call. 2) After writing all files, run tests. "
            "3) If tests fail, read the error carefully, fix the file, re-run. "
            "4) Keep going until all tests pass. Do NOT give up."
        )},
        {"role": "user", "content": task},
    ]

    turns = 0
    tool_calls_made = 0
    errors_fixed = 0
    last_verify_output = ""

    for turn in range(MAX_TURNS):
        turns += 1
        try:
            r = api_call(messages)
        except Exception as e:
            print(f"  Turn {turns}: API ERROR: {e}")
            return False, turns, f"API error: {e}"

        choice = r["choices"][0]
        msg = choice["message"]
        finish = choice["finish_reason"]
        comp_tok = r["usage"].get("completion_tokens", 0)
        tps = r["usage"].get("response_token/s", 0)

        # Check for content
        content = msg.get("content") or ""
        if content:
            print(f"  Turn {turns}: [{comp_tok} tok, {tps:.0f} tok/s] {content[:100]}...")

        # Check for tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            # Add assistant message with tool calls to history
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except:
                    args = {}
                tool_calls_made += 1
                result = execute_tool(fn, args)
                print(f"  Turn {turns}: tool={fn}({', '.join(f'{k}={repr(v)[:30]}' for k,v in args.items())}) → {result[:80]}")

                # Add tool result to history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

                # Track verification
                if fn == "run_command" and verify_cmd and args.get("command", "").strip().startswith(verify_cmd.split()[0]):
                    last_verify_output = result
                    if "error" in result.lower() or "Error" in result:
                        errors_fixed += 1
        elif finish == "stop":
            # Model stopped without tool calls — done or stuck
            if content:
                messages.append({"role": "assistant", "content": content})
                # Check if model is asking to verify
                if any(w in content.lower() for w in ["verify", "test", "run", "check"]):
                    messages.append({"role": "user", "content": f"Please run: {verify_cmd}"})
                    continue
            break
        elif finish == "length":
            print(f"  Turn {turns}: HIT MAX TOKENS ({comp_tok})")
            break

    # Final verification
    print(f"  Running verification: {verify_cmd}")
    verify_result = execute_tool("run_command", {"command": verify_cmd})
    print(f"  Verify result: {verify_result[:200]}")

    # Check for test success patterns
    has_ok = "ok" in verify_result.lower() or "OK" in verify_result or "passed" in verify_result.lower()
    has_fail = "FAIL" in verify_result or "error" in verify_result.lower() or "Error" in verify_result or "Traceback" in verify_result
    success = has_ok and not has_fail
    return success, turns, f"tools={tool_calls_made}, errors_fixed={errors_fixed}, verify={'PASS' if success else 'FAIL'}: {verify_result[:80]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hours", type=float, default=9.0)
    args = parser.parse_args()

    end_time = time.time() + args.hours * 3600

    # Test cases: (task_description, language, verify_command)
    test_cases = [
        (
            "Create a Python file 'calculator/calc.py' with functions: add, subtract, multiply, divide (with ZeroDivisionError). "
            "Then create 'calculator/test_calc.py' using Python's built-in unittest module (NOT pytest). "
            "Include tests for normal operations AND edge cases (divide by zero, negative numbers, floats). "
            "Run: python3 -m unittest calculator/test_calc.py -v",
            "python_calc",
            "python3 -m unittest calculator.test_calc -v"
        ),
        (
            "Create a Python file 'sorter/sort_lib.py' with: "
            "1) bubble_sort(arr) 2) merge_sort(arr) 3) binary_search(arr, target) returning index or -1 "
            "Then create 'sorter/test_sort.py' using unittest with tests for: empty list, single element, "
            "already sorted, reverse sorted, duplicates, and binary_search hit/miss. "
            "Run: python3 -m unittest sorter/test_sort.py -v",
            "python_sort",
            "python3 -m unittest sorter.test_sort -v"
        ),
        (
            "Create a Python file 'converter/convert.py' with: "
            "1) celsius_to_fahrenheit(c) 2) fahrenheit_to_celsius(f) 3) km_to_miles(km) "
            "4) miles_to_km(m) 5) kg_to_lbs(kg) 6) lbs_to_kg(lbs) "
            "Then create 'converter/test_convert.py' using unittest testing all functions "
            "with known values (0C=32F, 100C=212F, 1km=0.621371mi, 1kg=2.20462lbs). "
            "Run: python3 -m unittest converter/test_convert.py -v",
            "python_convert",
            "python3 -m unittest converter.test_convert -v"
        ),
    ]

    iteration = 0
    results = []

    print(f"{'='*60}")
    print(f"Atlas Agentic Multi-Turn Session Tester")
    print(f"Duration: {args.hours}h, Test cases: {len(test_cases)}")
    print(f"{'='*60}\n")

    while time.time() < end_time:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} (elapsed: {(time.time() - (end_time - args.hours*3600))/60:.0f}m)")
        print(f"{'='*60}")

        for i, (task, lang, verify) in enumerate(test_cases):
            if time.time() >= end_time:
                break
            print(f"\n--- Test {i+1}/{len(test_cases)}: {lang} project ---")
            t0 = time.time()
            success, turns, summary = run_session(task, lang, verify)
            elapsed = time.time() - t0
            status = "PASS" if success else "FAIL"
            result = {"iteration": iteration, "test": lang, "status": status, "turns": turns, "summary": summary, "elapsed": f"{elapsed:.1f}s"}
            results.append(result)
            print(f"  → {status} in {turns} turns ({elapsed:.1f}s): {summary}")

            if not success:
                print(f"  ⚠ FAILED — investigating...")
                # List what was created
                if WORK_DIR.exists():
                    files = list(WORK_DIR.rglob("*"))
                    print(f"  Files created: {[str(f.relative_to(WORK_DIR)) for f in files if f.is_file()][:10]}")

        # Print summary table
        print(f"\n{'='*60}")
        print(f"{'Iter':>4} {'Test':>12} {'Status':>6} {'Turns':>5} {'Time':>8} Summary")
        print(f"{'-'*60}")
        for r in results:
            print(f"{r['iteration']:>4} {r['test']:>12} {r['status']:>6} {r['turns']:>5} {r['elapsed']:>8} {r['summary']}")
        print(f"{'='*60}")

        passes = sum(1 for r in results if r['status'] == 'PASS')
        total = len(results)
        print(f"Score: {passes}/{total} ({100*passes/max(total,1):.0f}%)")

        if passes == total and iteration >= args.iterations:
            print(f"\n✓ ALL TESTS PASSING after {iteration} iterations. Done!")
            break

    # Save results
    with open("/workspace/atlas/agentic_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /workspace/atlas/agentic_test_results.json")

if __name__ == "__main__":
    main()
