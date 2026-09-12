#!/usr/bin/env python3
"""Known-answer probes at C=1 and C=4, distinct answers per sequence.

WHY NOT A BYTE-IDENTITY TEST. Strict solo-vs-concurrent identity is confounded
on this model: the solo pass populates the prefix cache that the concurrent
pass then restores from, and qwen4exp has a documented warm-restore defect.
Measured on THIS build, the known-good per-sequence arm scores 2/4 on that test
and the batched arm 1/4 -- the test fails for both, so it cannot separate them.

These probes are the standard the batched verify was validated against before:
each question has a DISTINCT, short, unambiguous answer, so a cross-sequence
mix-up shows up as one sequence answering another's question rather than as a
token-level diff. Run cold (no solo pass first) so no warm restore is involved.
"""
import os, json, threading, urllib.request, sys

PORT = int(os.environ.get("ATLAS_BENCH_PORT", "8892"))
MODEL = os.environ.get("ATLAS_BENCH_MODEL", "qwen4exp-nvfp4")

PROBES = [
    ("capital", "What is the capital of France? Reply with only the city name.", "paris"),
    ("arith",   "What is 47 * 23? Reply with only the number.", "1081"),
    ("primes",  "List the first 5 prime numbers, comma separated, nothing else.", "11"),
    ("color",   "What colour do you get mixing red and yellow? One word.", "orange"),
]


def ask(prompt, out, idx):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 512, "temperature": 0.0,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        out[idx] = (d["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        out[idx] = f"__ERROR__ {e}"


def score(out, label):
    ok = 0
    for i, (tag, _, want) in enumerate(PROBES):
        txt = out[i] or ""
        hit = want in txt.lower()
        ok += hit
        # Cross-check: did this sequence answer a DIFFERENT probe?
        cross = [PROBES[j][0] for j in range(len(PROBES))
                 if j != i and PROBES[j][2] in txt.lower()]
        note = f"  << also contains {cross}" if cross else ""
        print(f"  {label} {tag}: {'PASS' if hit else 'FAIL'} want={want!r} got={txt[:70]!r}{note}")
    print(f"  {label}: {ok}/{len(PROBES)}")
    return ok


solo = [None] * len(PROBES)
for i, (_, p, _) in enumerate(PROBES):
    ask(p, solo, i)
s_ok = score(solo, "solo")

conc = [None] * len(PROBES)
ths = [threading.Thread(target=ask, args=(PROBES[i][1], conc, i)) for i in range(len(PROBES))]
for t in ths:
    t.start()
for t in ths:
    t.join()
c_ok = score(conc, "conc")

print(f"\nVERDICT solo={s_ok}/{len(PROBES)} concurrent={c_ok}/{len(PROBES)}")
sys.exit(0 if s_ok == len(PROBES) and c_ok == len(PROBES) else 1)
