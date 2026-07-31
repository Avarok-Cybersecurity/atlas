import json, sys, time, urllib.request
URL = "http://localhost:8081/v1/chat/completions"
MODEL = "nvidia/Qwen3.6-27B-NVFP4"
# 12 growing turns to drive cumulative context into the multi-k range where the
# old recompute-all tail produced the 18-77s spikes.
TURNS = [
    "Write a Python class LRUCache with get and put, O(1). Explain briefly then code.",
    "Add type hints and a docstring with a usage example. Full class.",
    "Add a capacity property and a __len__. Keep it backward compatible. Full class.",
    "Add a peek(key) that doesn't update recency, and a clear(). Full class.",
    "Write 6 pytest tests: eviction order, update existing, peek, clear, capacity=1, overflow.",
    "Now add thread-safety with a lock around get/put/peek/clear. Full class.",
    "Explain the time complexity of each method and why the lock doesn't change it.",
    "Refactor to expose an OrderedDict-free doubly-linked-list implementation. Full code.",
    "Add a to_dict() that dumps entries most-recent-first, and from_dict(). Full class.",
    "Write 4 more tests for to_dict/from_dict round-trip and ordering.",
    "Summarize the final design's invariants in a bulleted list.",
    "Now write a short module docstring and __all__ for this file.",
]
def stream_turn(msgs, max_tok=400):
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": max_tok,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None; ntok = 0; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except Exception:
                continue
            c = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if c:
                if tfirst is None:
                    tfirst = time.time() - t0
                ntok += 1; text.append(c)
    wall = time.time() - t0
    return dict(ttft_ms=round((tfirst or wall) * 1000, 1), out_tok=ntok, text="".join(text))
msgs = []; ttfts = []
for i, u in enumerate(TURNS):
    msgs.append({"role": "user", "content": u})
    r = stream_turn(msgs)
    msgs.append({"role": "assistant", "content": r["text"]})
    ttfts.append(r["ttft_ms"])
    approx_ctx = sum(len(m["content"]) for m in msgs) // 4
    print("turn%2d: ttft=%7.1fms  out=%3dtok  ~ctx=%dtok" % (i, r["ttft_ms"], r["out_tok"], approx_ctx), flush=True)
warm = sorted(ttfts[1:])
n = len(warm)
p50 = warm[n // 2]; p90 = warm[min(n - 1, int(n * 0.9))]; mx = warm[-1]
print("WARM TTFT  p50=%.0fms  p90=%.0fms  max=%.0fms  (n=%d warm turns)" % (p50, p90, mx, n))
