#!/usr/bin/env python3
"""Aggregate decode throughput at C=N, counting tokens robustly.

WHY NOT THE OLDER DRIVER. It summed `usage.completion_tokens`, and at
max_tokens=400 with reasoning_effort low a request can spend its whole budget
inside <think> and come back with EMPTY content -- some builds then report
completion_tokens=0 while 400 tokens of decode work actually happened. That
silently dropped a quarter of the tokens out of one aggregate (34.39 reported
where the same run did 1600 tokens in 34.90s = 45.8 tok/s). Here the token
count falls back to the generation budget when usage under-reports, and every
empty completion is named in the output rather than quietly changing the
number.

Aggregate is total tokens / wall-clock of the whole concurrent phase, which is
the figure of merit for a serving box: how many tokens the machine produced per
second while N requests were in flight.

REPS runs the concurrent phase several times so the reported number has a
spread attached, not a single sample.
"""
import os, json, time, threading, urllib.request, sys

PORT = int(os.environ.get("ATLAS_BENCH_PORT", "8892"))
MODEL = "qwen4exp-nvfp4"
C = int(sys.argv[1]) if len(sys.argv) > 1 else 4
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAXTOK = 400

PROMPTS = [
    "Write a Python LRU cache class with capacity 512 using a dict plus a doubly linked list. Explain the eviction order.",
    "Implement a min-heap in Python with push, pop, and heapify. Explain the sift-down invariant.",
    "Write a Rust function that merges overlapping intervals in place. Explain the sort key and the merge condition.",
    "Explain how a copy-on-write B-tree handles a concurrent reader during a node split, step by step.",
    "Write a Go worker pool that bounds in-flight work with a semaphore channel. Explain the shutdown ordering.",
    "Implement binary search over a rotated sorted array in C. Explain why the pivot test picks the sorted half.",
    "Describe how a write-ahead log recovers a database after a crash mid-transaction, step by step.",
    "Write a SQL query that finds the median order value per customer, and explain the window function choice.",
]


def one(prompt, out, idx):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": MAXTOK, "temperature": 0.0,
                         "chat_template_kwargs": {"reasoning_effort": "low"}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
    except Exception as e:
        out[idx] = ("ERROR", 0, 0.0, str(e)[:100])
        return
    w = time.perf_counter() - t0
    u = d.get("usage") or {}
    ct = u.get("completion_tokens") or 0
    body = d["choices"][0]["message"].get("content") or ""
    fin = d["choices"][0].get("finish_reason")
    # A length-truncated request did MAXTOK tokens of work whatever usage says.
    if ct == 0 and fin == "length":
        ct = MAXTOK
        st = "empty-content(length)"
    elif not body:
        st = "empty-content"
    else:
        st = "ok"
    out[idx] = (st, ct, w, "")


def run(n):
    out = [None] * n
    ths = [threading.Thread(target=one, args=(PROMPTS[i % len(PROMPTS)], out, i)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    tot = sum(r[1] for r in out)
    bad = [f"req{i}:{r[0]}" for i, r in enumerate(out) if r[0] != "ok"]
    return tot / wall, tot, wall, bad


print(f"--- C=1 reference ---")
a1, t1, w1, b1 = run(1)
print(f"  {a1:.2f} tok/s ({t1} tok / {w1:.2f}s) {' '.join(b1)}")

vals = []
for rep in range(REPS):
    a, t, w, bad = run(C)
    vals.append(a)
    print(f"--- C={C} rep{rep}: {a:.2f} tok/s ({t} tok / {w:.2f}s) {' '.join(bad)}")

vals.sort()
print(f"\nC={C} AGGREGATE median {vals[len(vals)//2]:.2f} tok/s   min {vals[0]:.2f}  max {vals[-1]:.2f}")
print(f"C={C} vs C=1 scaling: {vals[len(vals)//2]/a1:.2f}x")
