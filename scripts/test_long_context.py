#!/usr/bin/env python3
"""Long-context coherence test for Atlas.
Tests fibonacci generation with realistic agentic system prompts.
Usage: python3 test_long_context.py [host:port]
"""
import json, subprocess, re, sys, os

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost:8888"
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

r = json.loads(subprocess.run(
    ["curl", "-s", f"http://{HOST}/v1/models"], capture_output=True, text=True
).stdout)
MODEL = r["data"][0]["id"]

print("=" * 55)
print(f"  Long-Context Coherence Test")
print(f"  Model: {MODEL}")
print(f"  Host:  {HOST}")
print("=" * 55)

with open(os.path.join(FIXTURES, "opencode_system_prompt.txt")) as f:
    opencode_sp = f.read()
with open(os.path.join(FIXTURES, "claude_code_system_prompt.txt")) as f:
    claude_sp = f.read()

passed = failed = 0

def test_fib(name, system_prompt):
    global passed, failed
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content":
        "Write a Python function called fibonacci(n) that returns the nth Fibonacci number. "
        "It must be 0-indexed: fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1, fibonacci(3)=2, "
        "fibonacci(4)=3, fibonacci(5)=5. Use an iterative approach with variables a and b. "
        "Return ONLY the Python code, no explanation."})
    payload = json.dumps({"model": MODEL, "messages": messages, "max_tokens": 300, "temperature": 0})
    with open("/tmp/atlas_test_payload.json", "w") as f:
        f.write(payload)
    result = subprocess.run(
        ["curl", "-s", "--max-time", "180", f"http://{HOST}/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", "@/tmp/atlas_test_payload.json"],
        capture_output=True, text=True)
    try:
        r = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  x {name}: ERROR (no response)"); failed += 1; return
    if "error" in r:
        print(f"  x {name}: ERROR: {r['error'].get('message', r['error'])}"); failed += 1; return
    c = r["choices"][0]["message"]["content"]
    tokens = r.get("usage", {}).get("prompt_tokens", "?")
    m = re.search(r'```python\s*\n(.*?)```', c, re.DOTALL)
    code = m.group(1) if m else c.strip()
    try:
        ns = {}; exec(code, ns)
        fn = ns.get("fibonacci") or ns.get("fib")
        if fn is None:
            for v in ns.values():
                if callable(v): fn = v; break
        result = [fn(i) for i in range(10)]
        if result == [0,1,1,2,3,5,8,13,21,34]:
            print(f"  PASS: {name} ({tokens} prompt tokens)"); passed += 1
        else:
            print(f"  FAIL: {name} ({tokens} tokens): {result}"); failed += 1
    except Exception as e:
        print(f"  FAIL: {name} ({tokens} tokens): {e}"); print(f"    Output: {repr(c[:200])}"); failed += 1

test_fib("Baseline", None)
test_fib("OpenCode (~7K tok)", opencode_sp)
test_fib("Claude Code (~30K tok)", claude_sp)
print(f"\n{'='*55}\n  Results: {passed} passed, {failed} failed\n{'='*55}")
sys.exit(0 if failed == 0 else 1)
