"""Concurrency QUALITY at the model's own sampling settings.

At temperature 0.7 / top_k 4 an exact-match comparison is meaningless — two
runs of the same prompt legitimately differ. What must NOT change with
concurrency is whether the answer is CORRECT.

Sends no sampler parameters at all, so the server applies the model card's
own preset (repetition_penalty 1.06 / temperature 0.7 / top_p 0.95 / top_k 4).
"""
import json
import sys
import threading
import os
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else '8888'
C = int(sys.argv[2]) if len(sys.argv) > 2 else 4
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
MODEL = os.environ.get('PROBE_MODEL', 'longcat-full')
URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'

CASES = [
    ("What is the capital city of France? Answer in one sentence.", ["paris"]),
    ("What is 17 multiplied by 23? Give the number.", ["391"]),
    ("Which planet in our solar system is the largest? One sentence.", ["jupiter"]),
    ("Who wrote the play Hamlet? One sentence.", ["shakespeare"]),
]


def ask(case, out, idx):
    q, _ = case
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 250}          # NO sampler params: use the model card's
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        out[idx] = d["choices"][0]["message"]["content"]
    except Exception as e:                       # noqa: BLE001
        out[idx] = f"<ERROR {e}>"


def score(text, keys):
    t = (text or "").lower()
    return any(k in t for k in keys)


def run(concurrent):
    jobs = [(CASES[i % len(CASES)], i) for i in range(len(CASES) * REPEATS)]
    out = [None] * len(jobs)
    if concurrent:
        for base in range(0, len(jobs), C):
            batch = jobs[base:base + C]
            ts = [threading.Thread(target=ask, args=(c, out, i)) for c, i in batch]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
    else:
        for c, i in jobs:
            ask(c, out, i)
    ok = sum(score(out[i], c[1]) for c, i in jobs)
    return ok, len(jobs), out, jobs


for label, conc in (("serial  (C=1)", False), (f"batched (C={C})", True)):
    ok, n, out, jobs = run(conc)
    print(f'{label}: {ok}/{n} correct')
    for c, i in jobs:
        if not score(out[i], c[1]):
            print(f'    MISS {c[0][:44]!r} -> {(out[i] or "")[:110]!r}')
    print()
