#!/usr/bin/env python3
"""Warm-replay TTFT probe — the instrument for ATLAS_GDN_REGRESIDENT.

The register-resident GDN kernel only replaces WY4 on the WARM MARCONI REPLAY
path (`ctx.gdn_exact_replay`), i.e. the suffix re-run after an SSM snapshot is
restored on a prefix-cache hit. A cold prefill never reaches it (cold takes the
baked FLA chunked path), so a cold benchmark cannot see this lever at all.

To hit it we reproduce how the MLCommons agentic harness actually drives the
endpoint: a flat string prompt that is an EXACT prefix-extension of the previous
turn (verified against the golden run's events.jsonl — 987/987 warm turns are
exact prefix extensions). Turn N is sent first to populate the cache and leave a
tail checkpoint; turn N+1 = turn N + delta then hits the cache and replays only
`delta` tokens through the 48 GDN layers. That replay is what we are timing.

Replay cost scales with DELTA, not with prompt depth, so we sweep delta. The
golden run's real distribution is p50 210 / mean 331 / p90 698 new tokens, so the
sweep brackets it rather than testing one arbitrary point.

Emitted text is recorded for every call so the two legs can be checked for token
equality — the kernel claims cos 1.0 / max|dH|~1e-8 vs WY4, so a differing
completion is a red flag, not an expected lossy tradeoff.

Usage: warm_replay_probe.py <port> <tag> <out.json> [--reps 7]
"""
import argparse
import json
import statistics
import time
import urllib.request

MODEL = "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf"

# Code-shaped filler: the target workload is SWE-bench agentic traces, and
# tokenizer behaviour (chars/token) differs enough on prose to shift the
# delta->token mapping this probe reports.
CHUNK = (
    "def _resolve_readonly_fields(self, obj, name):\n"
    "    # walk the admin class MRO so inherited descriptors are visible\n"
    "    for klass in type(obj).__mro__:\n"
    "        if name in vars(klass):\n"
    "            return vars(klass)[name]\n"
    "    raise FieldDoesNotExist(f'{name} not on {type(obj).__name__}')\n\n"
)


def call(port, prompt, max_tokens=24):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 42, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    out = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            t = (d.get("choices") or [{}])[0].get("text")
            if t:
                if ttft is None:
                    ttft = (time.time() - t0) * 1000.0
                out.append(t)
    return (ttft or 0.0), "".join(out)


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port"); ap.add_argument("tag"); ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()

    # ~38k-char base = the golden run's p50 prompt depth.
    BASE = CHUNK * 130
    # Deltas bracketing the real distribution (p50 210 / mean 331 / p90 698 tok).
    DELTAS = {"d_128tok": 1, "d_400tok": 3, "d_900tok": 7, "d_2000tok": 15}

    results = {}
    for name, nchunk in DELTAS.items():
        delta = CHUNK * nchunk
        ttfts, texts = [], []
        for i in range(a.reps):
            # Re-seat the base as a cache entry, then measure the extension.
            #
            # The rep marker must come BEFORE the delta, not after. With it
            # appended, `BASE + delta` is identical across reps and is itself
            # cached from rep 0 onward, so reps 1..n-1 only ever prefill the few
            # marker tokens -- the measurement collapses to a ~5-token replay and
            # goes flat in delta (observed: 241 ms at 288 chars vs 245 ms at 4320,
            # i.e. +4 ms for 15x the delta). Putting the marker first forces the
            # divergence point up front so the whole delta is genuinely new and
            # the replayed suffix really is delta-sized.
            call(a.port, BASE, max_tokens=4)
            t, txt = call(a.port, BASE + f"\n# rep {i}\n" + delta)
            ttfts.append(t); texts.append(txt)
        results[name] = {
            "delta_chars": len(delta), "n": len(ttfts),
            "p50": statistics.median(ttfts), "min": min(ttfts),
            "p90": pct(ttfts, 90), "vals": ttfts, "texts": texts,
        }
        print(f"[{a.tag}] {name:10s} delta={len(delta):6d}ch  "
              f"p50={statistics.median(ttfts):8.1f}  min={min(ttfts):8.1f} ms", flush=True)

    with open(a.out, "w") as fh:
        json.dump({"tag": a.tag, "base_chars": len(BASE), "cells": results},
                  fh, indent=2)


if __name__ == "__main__":
    main()
