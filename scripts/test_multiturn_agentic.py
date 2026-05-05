#!/usr/bin/env python3
"""Multi-turn agentic coherence test for Atlas.
Simulates a real coding agent session: write code → save → execute → modify → verify.
Each turn carries the FULL conversation history (system prompt + all prior turns).
Usage: python3 test_multiturn_agentic.py [host:port]
"""
import json, subprocess, re, sys, os, tempfile

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost:8888"
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

r = json.loads(subprocess.run(
    ["curl", "-s", f"http://{HOST}/v1/models"], capture_output=True, text=True
).stdout)
MODEL = r["data"][0]["id"]

# Load system prompts
with open(os.path.join(FIXTURES, "opencode_system_prompt.txt")) as f:
    opencode_sp = f.read()
with open(os.path.join(FIXTURES, "claude_code_system_prompt.txt")) as f:
    claude_sp = f.read()

def chat(messages, max_tokens=500):
    """Send a chat completion request and return the assistant's content."""
    payload = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0})
    with open("/tmp/atlas_mt_payload.json", "w") as f:
        f.write(payload)
    result = subprocess.run(
        ["curl", "-s", "--max-time", "180", f"http://{HOST}/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", "@/tmp/atlas_mt_payload.json"],
        capture_output=True, text=True)
    r = json.loads(result.stdout)
    if "error" in r:
        return None, f"ERROR: {r['error'].get('message', r['error'])}"
    content = r["choices"][0]["message"]["content"]
    tokens = r.get("usage", {}).get("prompt_tokens", "?")
    return content, tokens

def extract_code(text):
    """Extract Python code from a response."""
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    return m.group(1) if m else text.strip()

def run_session(name, system_prompt):
    """Run a 5-turn agentic session with full history carryover."""
    print(f"\n{'='*60}")
    print(f"  Multi-Turn Session: {name}")
    print(f"  Model: {MODEL}")
    print(f"{'='*60}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    passed = 0
    failed = 0
    tmpdir = tempfile.mkdtemp(prefix="atlas_test_")

    # ── Turn 1: Write fibonacci ──
    messages.append({"role": "user", "content":
        "Write a Python function called fibonacci(n) that returns the nth Fibonacci number. "
        "It must be 0-indexed: fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1, fibonacci(3)=2, "
        "fibonacci(4)=3, fibonacci(5)=5. Use an iterative approach with variables a and b. "
        "Return ONLY the Python code, no explanation."})
    
    resp, tokens = chat(messages)
    if resp is None:
        print(f"  Turn 1 (write code): FAIL — {tokens}")
        return 0, 5
    messages.append({"role": "assistant", "content": resp})
    
    code = extract_code(resp)
    try:
        ns = {}; exec(code, ns)
        fn = ns.get("fibonacci") or ns.get("fib")
        if fn is None:
            for v in ns.values():
                if callable(v): fn = v; break
        result = [fn(i) for i in range(10)]
        if result == [0,1,1,2,3,5,8,13,21,34]:
            print(f"  Turn 1 (write code, {tokens} prompt tok): PASS")
            passed += 1
        else:
            print(f"  Turn 1 (write code, {tokens} prompt tok): FAIL — {result}")
            failed += 1
            return passed, passed + (5 - passed)
    except Exception as e:
        print(f"  Turn 1 (write code, {tokens} prompt tok): FAIL — {e}")
        print(f"    Output: {repr(resp[:200])}")
        failed += 1
        return passed, passed + (5 - passed)

    # ── Turn 2: Save to file ──
    fib_path = os.path.join(tmpdir, "fibonacci.py")
    messages.append({"role": "user", "content":
        f"Now add a main block that prints fibonacci(10), fibonacci(20), and fibonacci(30). "
        f"Show the complete file content."})
    
    resp, tokens = chat(messages)
    if resp is None:
        print(f"  Turn 2 (add main): FAIL — {tokens}"); failed += 1
        return passed, passed + failed + 3
    messages.append({"role": "assistant", "content": resp})
    
    code = extract_code(resp)
    with open(fib_path, "w") as f:
        f.write(code)
    
    # Execute the file
    run = subprocess.run(["python3", fib_path], capture_output=True, text=True, timeout=10)
    if run.returncode == 0 and "55" in run.stdout:  # fibonacci(10) = 55
        print(f"  Turn 2 (add main + exec, {tokens} prompt tok): PASS — output: {run.stdout.strip()}")
        passed += 1
    else:
        print(f"  Turn 2 (add main + exec, {tokens} prompt tok): FAIL")
        print(f"    returncode={run.returncode}, stdout={run.stdout[:100]}, stderr={run.stderr[:100]}")
        failed += 1

    # ── Turn 3: Modify to add memoization ──
    messages.append({"role": "user", "content":
        "Modify the fibonacci function to use memoization with a dictionary cache. "
        "Keep the main block. Show the complete file."})
    
    resp, tokens = chat(messages)
    if resp is None:
        print(f"  Turn 3 (memoize): FAIL — {tokens}"); failed += 1
        return passed, passed + failed + 2
    messages.append({"role": "assistant", "content": resp})
    
    code = extract_code(resp)
    with open(fib_path, "w") as f:
        f.write(code)
    
    run = subprocess.run(["python3", fib_path], capture_output=True, text=True, timeout=10)
    if run.returncode == 0 and "55" in run.stdout and "memo" in code.lower() or "cache" in code.lower() or "{}" in code:
        print(f"  Turn 3 (memoize + exec, {tokens} prompt tok): PASS — output: {run.stdout.strip()}")
        passed += 1
    else:
        print(f"  Turn 3 (memoize + exec, {tokens} prompt tok): FAIL")
        print(f"    returncode={run.returncode}, stdout={run.stdout[:100]}")
        failed += 1

    # ── Turn 4: Add error handling ──
    messages.append({"role": "user", "content":
        "Add input validation: raise ValueError for negative n. Add a test that catches the error. "
        "Show the complete file."})
    
    resp, tokens = chat(messages)
    if resp is None:
        print(f"  Turn 4 (validation): FAIL — {tokens}"); failed += 1
        return passed, passed + failed + 1
    messages.append({"role": "assistant", "content": resp})
    
    code = extract_code(resp)
    with open(fib_path, "w") as f:
        f.write(code)
    
    run = subprocess.run(["python3", fib_path], capture_output=True, text=True, timeout=10)
    if run.returncode == 0 and "ValueError" not in run.stderr:
        print(f"  Turn 4 (validation + exec, {tokens} prompt tok): PASS — output: {run.stdout.strip()[:80]}")
        passed += 1
    else:
        print(f"  Turn 4 (validation + exec, {tokens} prompt tok): FAIL")
        print(f"    returncode={run.returncode}, stderr={run.stderr[:100]}")
        failed += 1

    # ── Turn 5: Context check — does it remember everything? ──
    messages.append({"role": "user", "content":
        "What were the three modifications you made to the original fibonacci function? "
        "List them briefly."})
    
    resp, tokens = chat(messages, max_tokens=200)
    if resp is None:
        print(f"  Turn 5 (memory): FAIL — {tokens}"); failed += 1
        return passed, passed + failed
    
    resp_lower = resp.lower()
    remembered = sum([
        "main" in resp_lower or "print" in resp_lower,
        "memo" in resp_lower or "cache" in resp_lower,
        "valid" in resp_lower or "error" in resp_lower or "negative" in resp_lower,
    ])
    if remembered >= 2:
        print(f"  Turn 5 (memory check, {tokens} prompt tok): PASS — remembered {remembered}/3 modifications")
        passed += 1
    else:
        print(f"  Turn 5 (memory check, {tokens} prompt tok): FAIL — only remembered {remembered}/3")
        print(f"    Response: {resp[:200]}")
        failed += 1

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return passed, failed

# Run sessions
total_pass = total_fail = 0

# Session 1: OpenCode system prompt (~7K tokens)
p, f = run_session("OpenCode (~7K system prompt)", opencode_sp)
total_pass += p; total_fail += f

# Session 2: Claude Code system prompt (~30K tokens)
p, f = run_session("Claude Code (~30K system prompt)", claude_sp)
total_pass += p; total_fail += f

print(f"\n{'='*60}")
print(f"  TOTAL: {total_pass} passed, {total_fail} failed")
print(f"{'='*60}")
sys.exit(0 if total_fail == 0 else 1)
