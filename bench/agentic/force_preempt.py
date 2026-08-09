#!/usr/bin/env python3
"""Force KV exhaustion and verify preempt-and-retry fires.

Fires N concurrent long generations at a deliberately starved KV pool.
Pre-fix decode behavior: ONE exhaustion errors ALL N requests.
Post-fix: the largest victim is preempted, the rest complete.
"""
import os, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PROMPT_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 900
URL = os.environ.get("ATLAS_URL", "http://localhost:8888/v1/chat/completions")
MODEL = os.environ.get("ATLAS_MODEL", "laguna-s-2.1")
BASE = ("The paged KV cache stores per-layer key and value tensors in fixed-size blocks. "
        "Prefix caching reuses blocks across turns via a radix tree keyed on token spans. ")


def one(i):
    body = {"model": MODEL,
            # unique prefix per stream so prefix caching can't dedupe them
            "messages": [{"role": "user", "content": f"[stream{i}] " + BASE * PROMPT_REPS +
                          " Now write an extremely detailed essay about all of this."}],
            "max_tokens": MAXTOK, "temperature": 0.7, "ignore_eos": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        u = d.get("usage", {})
        return ("OK", i, u.get("completion_tokens"), time.time() - t0, "")
    except Exception as e:
        return ("ERR", i, None, time.time() - t0, str(e)[:70])


if __name__ == "__main__":
    print(f"firing {N} concurrent streams (prompt~{PROMPT_REPS*20} tok, max_tokens={MAXTOK})", flush=True)
    with ThreadPoolExecutor(max_workers=N) as ex:
        res = list(ex.map(one, range(N)))
    ok = [r for r in res if r[0] == "OK"]
    err = [r for r in res if r[0] == "ERR"]
    for st, i, ct, el, msg in sorted(res, key=lambda r: r[1]):
        print(f"  stream{i}: {st:3} gen={ct} {el:.0f}s {msg}")
    print(f"\n  RESULT: {len(ok)}/{N} completed, {len(err)}/{N} errored")
    print("  Pre-fix decode behavior would be: ALL streams error together on one exhaustion.")
