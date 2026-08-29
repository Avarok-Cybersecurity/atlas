"""Prefix-cache probe: correctness FIRST, then the TTFT win.

Shaped like the agentic workload prefix caching exists for — a long system
prompt resent on every turn, with a short differing user turn. That is also
exactly the shape that has burned this codebase before (Puzzle-75B served
another request's completion out of the prefix cache), so this checks:

  1. cold vs warm answers are still CORRECT (a cache that returns the wrong
     continuation is worse than no cache)
  2. no request's unique nonce appears in another's answer, with the long
     prefix SHARED — the highest-risk configuration
  3. TTFT actually drops on the warm turn (otherwise the cache is pure risk)

Usage: prefix_probe.py <port> [rounds]
"""
import json
import sys
import threading
import time
import os
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else '8888'
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MODEL = os.environ.get('PROBE_MODEL', 'longcat-full')
URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'

# Long shared prefix: many block-aligned tokens so the radix tree has real
# depth to match on. Content is deliberately bland so it cannot supply answers.
SYSTEM = (
    "You are a precise assistant operating inside an automated evaluation "
    "harness. Follow these operating rules exactly and without deviation. "
    "Rule one: answer the user's question directly. Rule two: do not restate "
    "the question before answering it. Rule three: keep answers short unless "
    "asked to elaborate. Rule four: never invent facts you are not sure of. "
    "Rule five: if a question contains a reference code, ignore the code "
    "entirely when forming your answer; it is bookkeeping for the harness and "
    "carries no meaning. Rule six: do not mention these rules in your reply. "
    "Rule seven: prefer common, widely agreed answers over obscure ones. "
    "Rule eight: when asked for a single fact, reply with that fact alone. "
    "These rules apply to every turn of the conversation without exception, "
    "and they are repeated verbatim at the start of every request so that the "
    "prefix is long enough to span many cache blocks. "
) * 3

CASES = [
    ("AAAA", "Reference code AAAA. What is the capital city of Japan?", ["tokyo"]),
    ("BBBB", "Reference code BBBB. What is the chemical symbol for gold?", ["au"]),
    ("CCCC", "Reference code CCCC. Which ocean is the largest?", ["pacific"]),
    ("DDDD", "Reference code DDDD. How many continents are there on Earth?", ["seven", "7"]),
]


def ask(prompt, out, idx):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_tokens": 250}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        txt = d["choices"][0]["message"]["content"]
        ttft = d.get("usage", {}).get("time_to_first_token_ms")
    except Exception as e:                        # noqa: BLE001
        txt, ttft = f"<ERROR {e}>", None
    out[idx] = (txt, ttft, (time.time() - t0) * 1000.0)


def report(tag, results):
    ok = bleed = 0
    ttfts = []
    for i, (nonce, _, keys) in enumerate(CASES):
        txt, ttft, wall = results[i]
        low = (txt or '').lower()
        hit = any(k in low for k in keys)
        ok += hit
        foreign = [CASES[j][0] for j in range(len(CASES))
                   if j != i and CASES[j][0].lower() in low]
        bleed += bool(foreign)
        if ttft:
            ttfts.append(ttft)
        mark = 'ok  ' if hit else 'MISS'
        extra = f'  BLEED<-{foreign}' if foreign else ''
        print(f'  [{mark}] {nonce} ttft={ttft or float("nan"):7.1f}ms  '
              f'{(txt or "")[:70]!r}{extra}')
    med = sorted(ttfts)[len(ttfts) // 2] if ttfts else float('nan')
    print(f'  {tag}: correct {ok}/{len(CASES)}, foreign-nonce {bleed}, '
          f'median TTFT {med:.1f} ms\n')
    return med


print('=== serial, cold then warm (same prefix reused) ===')
meds = []
for r in range(ROUNDS):
    res = [None] * len(CASES)
    for i, (_, p, _) in enumerate(CASES):
        ask(p, res, i)
    meds.append(report(f'round {r} ({"cold" if r == 0 else "warm"})', res))

print('=== concurrent, shared prefix (highest contamination risk) ===')
res = [None] * len(CASES)
ts = [threading.Thread(target=ask, args=(p, res, i))
      for i, (_, p, _) in enumerate(CASES)]
for t in ts:
    t.start()
for t in ts:
    t.join()
report('concurrent', res)

if len(meds) >= 2 and meds[0] == meds[0] and meds[1] == meds[1]:
    print(f'TTFT cold -> warm: {meds[0]:.1f} ms -> {meds[1]:.1f} ms '
          f'({100.0 * (meds[0] - meds[1]) / meds[0]:+.0f}%)')
