# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Decode throughput at C=1/2/4, measured over the ALL-SLOTS-BUSY window.

WHY THE WINDOW, AND NOT FIRST-TOKEN-TO-LAST-TOKEN
-------------------------------------------------
The obvious measurement -- time the whole wave and divide by the tokens it
produced -- is wrong here, and the reason is worth knowing before trusting any
concurrency number on this model.

It would be sound if every request retired at the same moment, which is what
capping all of them at the same `max_tokens` is supposed to guarantee. But
temp-0 output on this engine is NOT concurrency-invariant: the same prompt
yields 1024 tokens alone and 726 beside one other request (deterministic across
repetitions, and NOT streaming truncation -- streamed and non-streamed counts
agree exactly at C=1). Batch width selects a different per-M GEMM kernel, the
reduction order changes, and a near-tie token flips; bit-losslessness across
batch widths is unreachable on this model. So slots retire at different times
no matter what cap is set, and the tail of the wave runs at less than width C.
A number computed over the whole wave therefore reports the retirement pattern
as if it were a property of concurrency.

This measures the steady state instead. Every token's arrival time is recorded;
the window is [max(first token), min(last token)] across the wave -- the
interval during which ALL C slots were provably streaming. Tokens are counted
only inside it. At C=1 this degenerates to the ordinary decode window.

`occupancy` reports the window as a fraction of the shortest request's own
span. A small occupancy means the slots barely overlapped and the arm should
not be quoted.
"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("CVEC_BASE", "http://127.0.0.1:8899")
MODEL = os.environ.get("CVEC_MODEL", "qwen3.8-flash-next-nvfp4-tp2-ep2")
MAX_TOKENS = int(os.environ.get("CVEC_MAX_TOKENS", "1024"))
REPS = int(os.environ.get("CVEC_REPS", "3"))
# The vector to compare against no steering. Must be registered on the server.
ARM_VECTOR = os.environ.get("CVEC_ARM", "vcd_m15")

PROMPTS = [
    "Implement a thread-safe LRU cache in Python with an explanation of every "
    "design decision, the concurrency invariants, and a full test suite.",
    "Implement a red-black tree in Rust with insertion, deletion and lookup. "
    "Explain the rebalancing cases in detail and include unit tests.",
    "Write a complete TCP echo server and client in Go using goroutines. "
    "Explain the concurrency model, error handling, and graceful shutdown.",
    "Implement a bounded multi-producer multi-consumer queue in C++17. "
    "Explain the memory ordering choices and include a stress test.",
]


def one_request(prompt, cvec, out, idx):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "none"},
        "control_vector": cvec,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    stamps = []
    usage_tok = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if d.get("usage"):
                    usage_tok = d["usage"].get("completion_tokens")
                for ch in d.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        stamps.append(time.perf_counter())
    except Exception as e:  # noqa: BLE001
        out[idx] = {"error": repr(e)}
        return
    out[idx] = {"t0": t0, "stamps": stamps, "usage_tok": usage_tok}


def wave(conc, cvec):
    out = [None] * conc
    threads = [
        threading.Thread(target=one_request, args=(PROMPTS[i % len(PROMPTS)], cvec, out, i))
        for i in range(conc)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if any(r is None or "error" in r for r in out):
        bad = next(r for r in out if r is None or "error" in r)
        return {"bad": bad.get("error", "no result") if bad else "no result"}
    if any(len(r["stamps"]) < 2 for r in out):
        return {"bad": "a request produced fewer than 2 tokens"}

    # The interval during which every slot was provably streaming.
    w_start = max(r["stamps"][0] for r in out)
    w_end = min(r["stamps"][-1] for r in out)
    if w_end <= w_start:
        return {"bad": "slots did not overlap at all"}

    counted = sum(sum(1 for s in r["stamps"] if w_start < s <= w_end) for r in out)
    agg = counted / (w_end - w_start)
    per_req = agg / conc

    shortest_span = min(r["stamps"][-1] - r["stamps"][0] for r in out)
    occupancy = (w_end - w_start) / shortest_span if shortest_span > 0 else 0.0

    return {
        "agg": agg,
        "per_req": per_req,
        "occupancy": occupancy,
        "ttft_median": statistics.median(r["stamps"][0] - r["t0"] for r in out),
        "toks": [r["usage_tok"] or len(r["stamps"]) for r in out],
        "counted": counted,
    }


def main():
    print(f"model      : {MODEL}")
    print(f"window     : all-slots-busy  [max(first tok), min(last tok)]")
    print(f"max_tokens : {MAX_TOKENS}   reps: {REPS}   temp 0, reasoning_effort=none\n")

    results = {}
    hdr = (
        f"{'arm':18} {'C':>2} {'rep':>3} {'agg tok/s':>10} {'per-req':>8} "
        f"{'occ':>5} {'TTFT':>6}  tokens"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, cvec in (("cvec OFF (null)", None), (f"cvec {ARM_VECTOR}", ARM_VECTOR)):
        for conc in (1, 2, 4):
            aggs, pers, occs = [], [], []
            for rep in range(REPS):
                r = wave(conc, cvec)
                if "bad" in r:
                    print(f"{label:18} {conc:>2} {rep:>3}  BAD: {r['bad']}")
                    continue
                print(
                    f"{label:18} {conc:>2} {rep:>3} {r['agg']:>10.2f} {r['per_req']:>8.2f} "
                    f"{r['occupancy']:>5.2f} {r['ttft_median']:>6.2f}  {r['toks']}"
                )
                sys.stdout.flush()
                aggs.append(r["agg"])
                pers.append(r["per_req"])
                occs.append(r["occupancy"])
            if aggs:
                results[(label, conc)] = (
                    statistics.median(aggs),
                    statistics.median(pers),
                    statistics.median(occs),
                )

    print("\n=== medians (steady state) ===")
    print(
        f"{'arm':18} {'C':>2} {'agg tok/s':>10} {'per-req':>8} {'occ':>5} {'scaling':>8}"
    )
    for label in ("cvec OFF (null)", f"cvec {ARM_VECTOR}"):
        base = results.get((label, 1), (None,))[0]
        for conc in (1, 2, 4):
            if (label, conc) not in results:
                continue
            agg, per, occ = results[(label, conc)]
            scale = f"{agg / base:.2f}x" if base else "-"
            flag = "  <- LOW OVERLAP, do not quote" if occ < 0.30 else ""
            print(
                f"{label:18} {conc:>2} {agg:>10.2f} {per:>8.2f} {occ:>5.2f} {scale:>8}{flag}"
            )

    print("\nocc = all-slots-busy window as a fraction of the shortest request's span.")


if __name__ == "__main__":
    main()
