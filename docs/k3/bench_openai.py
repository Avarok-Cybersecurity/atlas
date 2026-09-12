#!/usr/bin/env python3
"""K3-LAB S0 bake-off: hit two OpenAI-compatible endpoints, write JSONL.

Does not write certified .benchmarks/*/BASELINE.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def chat(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[float, float, str]:
    """Return (ttft_ms, total_ms, text). TTFT is full-response latency here (non-stream)."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            # S0: match Atlas --disable-thinking. Without this, vLLM Qwen3.8 emits a think preamble.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    total_ms = (time.perf_counter() - t0) * 1000.0
    data = json.loads(raw)
    text = data["choices"][0]["message"].get("content") or ""
    return total_ms, total_ms, text


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="http://spark1:8888/v1")
    ap.add_argument("--vllm", default="http://spark2:8000/v1")
    ap.add_argument("--atlas-model", default="Qwen/Qwen3.8-27B-NVFP4")
    ap.add_argument("--vllm-model", default="/model")
    ap.add_argument("--isl", type=int, default=512)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="docs/k3/logs/bakeoff.jsonl")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    prompt = ("Say the word ping, then count from 1 to 8.\n" + ("lorem ipsum " * (args.isl // 3)))[: args.isl * 4]

    rows = []
    for engine, url, model, hardware in (
        ("atlas", args.atlas, args.atlas_model, "spark1-gb10"),
        ("vllm", args.vllm, args.vllm_model, "spark2-gb10"),
    ):
        lat: list[float] = []
        texts: list[str] = []
        for i in range(args.warmup + args.n):
            try:
                ttft, total, text = chat(url, model, prompt, args.osl, args.timeout)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                print(f"FAIL {engine} i={i}: {e}")
                raise
            if i >= args.warmup:
                lat.append(total)
                texts.append(text[:80])
            print(f"{engine} i={i} total_ms={total:.1f} chars={len(text)}")
        tok_s = (args.osl / (statistics.mean(lat) / 1000.0)) if lat else float("nan")
        row = {
            "engine": engine,
            "hardware": hardware,
            "model": model,
            "isl": args.isl,
            "osl": args.osl,
            "concurrency": 1,
            "ttft_p50_ms": pct(lat, 50),
            "ttft_p99_ms": pct(lat, 99),
            "itl_p50_ms": float("nan"),
            "tok_s_per_user": tok_s,
            "tok_s_system": tok_s,
            "gpu_mem_gb": None,
            "notes": args.notes
            + " non-stream so ttft==e2e; ITL not measured; S0 harness smoke not a certified baseline",
            "sample": texts[:2],
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
