"""Warm-turn TPOT / decode-speed probe for the K=3 vs K=4 (num-drafts 2 vs 3) A/B.

Speculative decoding is trajectory-dependent, so this reports emitted token
counts alongside timing -- a tok/s delta is only meaningful if both configs
emitted comparable output (see reference_spec_decode_tokps_is_trajectory_dependent).
Turn 0 is a cold warmup and is excluded from the aggregate.
"""
import json, sys, time, urllib.request

URL = "http://localhost:8081/v1/chat/completions"
MODEL = "nvidia/Qwen3.6-27B-NVFP4"

TURNS = [
    "Write a Python function to merge two sorted lists. Code only.",
    "Now make it merge K sorted lists using a heap. Full code.",
    "Add type hints and a docstring with complexity analysis. Full code.",
    "Write 5 pytest tests for it, including empty and single-list cases.",
    "Refactor to a class KWayMerger with an iterator interface. Full code.",
    "Explain the time and space complexity of each method in a bulleted list.",
]

def turn(msgs, max_tok=400):
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": max_tok,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None; n = 0; parts = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                c = json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content")
            except Exception:
                continue
            if c:
                if tfirst is None:
                    tfirst = time.time() - t0
                n += 1; parts.append(c)
    wall = time.time() - t0
    dec = max(wall - (tfirst or 0.0), 1e-9)
    return dict(ttft_ms=(tfirst or wall) * 1000, chunks=n, decode_s=dec,
                tps=(n - 1) / dec if n > 1 else 0.0, text="".join(parts))

msgs = []
tot_tok = 0.0; tot_dec = 0.0
print("%5s %10s %8s %10s %9s" % ("turn", "ttft_ms", "chunks", "decode_s", "tok/s"))
for i, u in enumerate(TURNS):
    msgs.append({"role": "user", "content": u})
    r = turn(msgs)
    msgs.append({"role": "assistant", "content": r["text"]})
    print("%5d %10.1f %8d %10.2f %9.2f" % (i, r["ttft_ms"], r["chunks"], r["decode_s"], r["tps"]))
    if i > 0:
        tot_tok += r["chunks"] - 1; tot_dec += r["decode_s"]
print("\nWARM AGGREGATE: %.0f chunks / %.2f s = %.2f tok/s   (TPOT %.2f ms)" % (
    tot_tok, tot_dec, tot_tok / tot_dec, 1000.0 * tot_dec / tot_tok))
