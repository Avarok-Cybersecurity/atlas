#!/usr/bin/env python3
"""Benchmark a single model: throughput, TTFT, coherence, multi-turn, code recall."""
import json, sys, time, requests

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost:8888"
URL = f"http://{HOST}/v1"

def get_model():
    r = requests.get(f"{URL}/models", timeout=10)
    return r.json()["data"][0]["id"]

def chat(messages, max_tokens=50, temperature=0.0):
    r = requests.post(f"{URL}/chat/completions", json={
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }, timeout=300)
    return r.json()

MODEL = get_model()
results = {"model": MODEL, "tests": {}}
print(f"=== Benchmarking: {MODEL} ===")

# 1. Throughput + TTFT (150 tokens)
print("  [1/5] Throughput...", end=" ", flush=True)
t0 = time.time()
r = chat([{"role": "user", "content": "Write a Python fibonacci function with detailed docstring."}], max_tokens=150, temperature=0.7)
elapsed = time.time() - t0
u = r.get("usage", {})
gen = u.get("completion_tokens", 0)
ttft = r["usage"].get("time_to_first_token_ms", 0)
tps = r["usage"].get("response_token/s", gen / max(elapsed, 0.01))
results["tests"]["throughput"] = {"tok_s": round(tps, 1), "ttft_ms": round(ttft, 1), "gen_tokens": gen}
print(f"{tps:.1f} tok/s, TTFT={ttft:.0f}ms")

# 2. Fibonacci coherence
print("  [2/5] Fibonacci...", end=" ", flush=True)
r = chat([
    {"role": "user", "content": "Write a Python function that returns the first N fibonacci numbers as a list. Just the function, no explanation."}
], max_tokens=200, temperature=0.0)
content = r["choices"][0]["message"]["content"]
fib_pass = "def " in content and ("fib" in content.lower() or "fibonacci" in content.lower())
results["tests"]["fibonacci"] = "PASS" if fib_pass else "FAIL"
print("PASS" if fib_pass else "FAIL")

# 3. Multi-turn (5 turns)
print("  [3/5] Multi-turn...", end=" ", flush=True)
msgs = [{"role": "system", "content": "You are a helpful coding assistant."}]
mt_pass = 0
for i in range(5):
    prompts = [
        "Write a Python function to check if a number is prime.",
        "Now add type hints and a docstring to it.",
        "Add error handling for negative numbers.",
        "Write 3 test cases for the function.",
        "What modifications did we make to the prime function across our conversation?",
    ]
    msgs.append({"role": "user", "content": prompts[i]})
    r = chat(msgs, max_tokens=300, temperature=0.7)
    content = r["choices"][0]["message"]["content"]
    msgs.append({"role": "assistant", "content": content})
    if len(content) > 20:
        mt_pass += 1
results["tests"]["multi_turn"] = f"{mt_pass}/5"
print(f"{mt_pass}/5")

# 4. Code recall at ~8K context
print("  [4/5] Code recall (8K)...", end=" ", flush=True)
msgs = [{"role": "system", "content": "You are helpful. SECRET CODE: ALPHA-BRAVO-7. Remember this."}]
for i in range(5):
    msgs.append({"role": "user", "content": f"Explain topic {i+1}: distributed systems, compilers, databases, networking, or OS design."})
    r = chat(msgs, max_tokens=1500, temperature=0.7)
    msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
msgs.append({"role": "user", "content": "What is the secret code? ONLY the code."})
r = chat(msgs, max_tokens=30, temperature=0.0)
content = r["choices"][0]["message"]["content"]
ctx = r["usage"]["prompt_tokens"]
recall_pass = "ALPHA-BRAVO-7" in content
results["tests"]["code_recall_8k"] = {"result": "PASS" if recall_pass else "FAIL", "ctx_tokens": ctx}
print(f"{'PASS' if recall_pass else 'FAIL'} at {ctx} tokens")

# 5. Extended context (16K+ if possible)
print("  [5/5] Extended context...", end=" ", flush=True)
msgs = [{"role": "system", "content": "CRITICAL: Code is DELTA-99. Remember."}]
for i in range(10):
    msgs.append({"role": "user", "content": f"Write about topic {i+1} in great detail."})
    r = chat(msgs, max_tokens=1500, temperature=0.7)
    d = r
    if "error" in d:
        break
    content = d["choices"][0]["message"]["content"]
    gen = d["usage"]["completion_tokens"]
    if gen < 10:
        break
    msgs.append({"role": "assistant", "content": content})
msgs.append({"role": "user", "content": "Code?"})
r = chat(msgs, max_tokens=20, temperature=0.0)
content = r["choices"][0]["message"]["content"]
ctx = r["usage"]["prompt_tokens"]
ext_pass = "DELTA-99" in content
results["tests"]["extended_ctx"] = {"result": "PASS" if ext_pass else "FAIL", "ctx_tokens": ctx}
print(f"{'PASS' if ext_pass else 'FAIL'} at {ctx} tokens")

# Summary
print(f"\n  Results: {json.dumps(results['tests'], indent=2)}")
json.dump(results, open(f"/tmp/bench_{MODEL.replace('/', '_')}.json", "w"), indent=2)
print(f"  Saved to /tmp/bench_{MODEL.replace('/', '_')}.json")
