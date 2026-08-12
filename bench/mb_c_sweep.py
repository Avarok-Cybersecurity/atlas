#!/usr/bin/env python3
"""mb_c_sweep.py — Milestone-B concurrency sweep for Nemotron-H Lightning.

Fires C concurrent 400-token story completions and reports, per rung:

  per-stream tok/s   server-reported decode throughput of each stream
  sum-of-stream      sum of the per-stream decode tok/s (the figure the
                     milestone-A notes quote at C=8)
  aggregate          total completion tokens / wall time of the whole batch

Distinct prompts per stream so the prefix cache cannot serve one stream's
decode from another's blocks. Unbuffered by construction (print(flush=True)).

Usage:
  python3 -u bench/mb_c_sweep.py --port 8889 --reps 2 --conc 1,2,4,8
"""

import argparse
import json
import statistics
import sys
import threading
import time
from urllib.request import Request, urlopen

TOPICS = [
    "a lighthouse keeper who collects lost radio signals",
    "a cartographer mapping a city that rearranges itself nightly",
    "a beekeeper whose hives predict the weather",
    "a night-shift train dispatcher and the one train nobody scheduled",
    "a luthier who builds instruments from shipwreck timber",
    "a glacier researcher who finds a message frozen in the ice",
    "a retired chess champion teaching a village to play",
    "a diver who repairs undersea cables in the dark",
    "an archivist restoring the last reel of a lost film",
    "a baker whose sourdough starter is older than the town",
    "a park ranger tracking a wolf that avoids every camera",
    "a radio astronomer listening to a star that stutters",
    "a clockmaker asked to slow one single clock",
    "a ferry pilot on a river that changes course each spring",
    "a seed-bank curator during a long drought",
    "a translator working on a language with no word for 'goodbye'",
]


def one_stream(url, model, idx, max_tokens, out, lock):
    prompt = (
        f"Write a {max_tokens}-word short story about {TOPICS[idx % len(TOPICS)]}. "
        "Use vivid concrete detail and a clear beginning, middle and end."
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    n_tok = 0
    usage = {}
    try:
        with urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        n_tok += 1
    except Exception as exc:  # noqa: BLE001 — a dead stream must not kill the sweep
        with lock:
            out.append({"idx": idx, "error": repr(exc)})
        return
    t_end = time.perf_counter()
    comp = usage.get("completion_tokens") or n_tok
    decode_s = (t_end - t_first) if t_first else (t_end - t0)
    with lock:
        out.append(
            {
                "idx": idx,
                "ttft_ms": (t_first - t0) * 1000 if t_first else None,
                "completion_tokens": comp,
                "decode_s": decode_s,
                "tps": (comp - 1) / decode_s if decode_s > 0 and comp > 1 else 0.0,
                "e2e_s": t_end - t0,
            }
        )


def run_rung(url, model, conc, max_tokens):
    out, lock, threads = [], threading.Lock(), []
    t0 = time.perf_counter()
    for i in range(conc):
        th = threading.Thread(
            target=one_stream, args=(url, model, i, max_tokens, out, lock)
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    good = [r for r in out if "error" not in r]
    bad = [r for r in out if "error" in r]
    total_tok = sum(r["completion_tokens"] for r in good)
    return {
        "conc": conc,
        "wall_s": wall,
        "errors": bad,
        "per_stream_tps": sorted(round(r["tps"], 2) for r in good),
        "sum_of_stream_tps": round(sum(r["tps"] for r in good), 2),
        "aggregate_tps": round(total_tok / wall, 2) if wall > 0 else 0.0,
        "median_ttft_ms": round(
            statistics.median([r["ttft_ms"] for r in good if r["ttft_ms"]]), 1
        )
        if good
        else None,
        "total_tokens": total_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    url = args.url or f"http://localhost:{args.port}"
    rungs = [int(x) for x in args.conc.split(",")]

    for _ in range(args.warmup):
        print("warmup...", flush=True)
        run_rung(url, args.model, 1, 64)

    results = []
    for c in rungs:
        reps = []
        for rep in range(args.reps):
            r = run_rung(url, args.model, c, args.max_tokens)
            r["rep"] = rep
            reps.append(r)
            print(
                f"[{args.label}] C={c:<3} rep{rep}  "
                f"sum-of-stream={r['sum_of_stream_tps']:<8} "
                f"aggregate={r['aggregate_tps']:<8} "
                f"per-stream={r['per_stream_tps']} "
                f"ttft_ms={r['median_ttft_ms']} "
                f"errors={len(r['errors'])}",
                flush=True,
            )
            if r["errors"]:
                print(f"    ERRORS: {r['errors'][:2]}", flush=True)
        best = max(reps, key=lambda x: x["sum_of_stream_tps"])
        results.append({"conc": c, "reps": reps, "best": best})
        print(
            f"[{args.label}] C={c:<3} BEST sum-of-stream={best['sum_of_stream_tps']} "
            f"aggregate={best['aggregate_tps']}",
            flush=True,
        )

    print("\n=== SUMMARY " + args.label + " ===", flush=True)
    print(f"{'C':>4} {'sum-of-stream':>15} {'aggregate':>12} {'ttft_ms':>9}", flush=True)
    for r in results:
        b = r["best"]
        print(
            f"{r['conc']:>4} {b['sum_of_stream_tps']:>15} {b['aggregate_tps']:>12} "
            f"{b['median_ttft_ms']:>9}",
            flush=True,
        )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"label": args.label, "results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
