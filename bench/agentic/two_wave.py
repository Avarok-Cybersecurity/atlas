#!/usr/bin/env python3
"""Two-wave load: force the classic swap-out path, then decode exhaustion.

Wave 1 fills the pool and starts decoding. Wave 2 arrives while wave 1 is
still active, which is the only condition under which the scheduler's
swap-out valve engages (`new_reqs` non-empty AND `active` non-empty).
The failing run that produced 10 `dec_ref`-at-0 errors had exactly this
shape, so this is the shape that has to be re-run to clear it.
"""
import os, json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

W1 = int(sys.argv[1]) if len(sys.argv) > 1 else 6
W2 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 45
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 180
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 1200
URL = os.environ.get("ATLAS_URL", "http://localhost:8888/v1/chat/completions")
MODEL = os.environ.get("ATLAS_MODEL", "laguna-s-2.1")
BASE = ("The paged KV cache stores per-layer key and value tensors in fixed-size blocks. "
        "Prefix caching reuses blocks across turns via a radix tree keyed on token spans. ")
lock = threading.Lock()


def one(tag):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": f"[{tag}] " + BASE * REPS +
                          " Now write an extremely detailed essay about all of this."}],
            "max_tokens": MAXTOK, "temperature": 0.7, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        r = ("OK", tag, d.get("usage", {}).get("completion_tokens"), time.time() - t0, "")
    except Exception as e:
        r = ("ERR", tag, None, time.time() - t0, str(e)[:60])
    with lock:
        print(f"  {r[1]}: {r[0]:3} gen={r[2]} {r[3]:.0f}s {r[4]}", flush=True)
    return r


if __name__ == "__main__":
    print(f"wave1={W1} streams, +{GAP}s -> wave2={W2} streams "
          f"(prompt~{REPS*20} tok, max_tokens={MAXTOK})", flush=True)
    res = []
    with ThreadPoolExecutor(max_workers=W1 + W2) as ex:
        futs = [ex.submit(one, f"w1s{i}") for i in range(W1)]
        time.sleep(GAP)
        print(f"--- wave 2 firing at t={GAP}s ---", flush=True)
        futs += [ex.submit(one, f"w2s{i}") for i in range(W2)]
        res = [f.result() for f in futs]
    ok = sum(1 for r in res if r[0] == "OK")
    print(f"\n  RESULT: {ok}/{len(res)} completed, {len(res)-ok} errored")
