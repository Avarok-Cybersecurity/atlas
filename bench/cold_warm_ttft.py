#!/usr/bin/env python3
"""Cold + warm TTFT bench for A/B comparison of --enable-prefix-caching.

For each prompt length, send the identical prompt twice:
  1st request: cold (populates prefix cache if enabled)
  2nd request: warm (hits cache if enabled)

Warm vs cold delta is the signal.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

INPUTS = [256, 1024, 4096]  # target prompt-token lengths

LOREM = (
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled "
    "with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with "
    "nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort. "
)


def _post(url, body, timeout=300.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_prompt(target_tokens, salt):
    # Salt so different prompt-length trials do not collide in the cache.
    target_chars = target_tokens * 4
    reps = (target_chars // len(LOREM)) + 1
    body = (LOREM * reps)[:target_chars]
    return f"[trial {salt}] " + body + "\n\nQuestion: summarize in one sentence.\nAnswer:"


def measure_ttft(url, model, prompt):
    t0 = time.perf_counter()
    r = _post(
        f"{url}/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    usage = r.get("usage", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "ttft_ms": round(elapsed_ms, 1),
        "cached_prompt_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="bench")
    args = ap.parse_args()

    # Warm-up — ensure server is live and CUDA graphs captured.
    print(f"[{args.tag}] warm-up...", flush=True)
    try:
        _post(f"{args.url}/v1/completions",
              {"model": args.model, "prompt": "Hello", "max_tokens": 8})
    except Exception as e:
        print(f"warm-up failed: {e}", file=sys.stderr)
        return 2

    results = {"tag": args.tag, "model": args.model, "trials": []}
    for pl in INPUTS:
        prompt = build_prompt(pl, salt=f"{args.tag}-{pl}")
        print(f"[{args.tag}] prompt_len={pl} cold...", flush=True)
        cold = measure_ttft(args.url, args.model, prompt)
        print(f"[{args.tag}]   cold ttft={cold['ttft_ms']}ms (tokens={cold['prompt_tokens']}, cached={cold['cached_prompt_tokens']})", flush=True)
        print(f"[{args.tag}] prompt_len={pl} warm...", flush=True)
        warm = measure_ttft(args.url, args.model, prompt)
        print(f"[{args.tag}]   warm ttft={warm['ttft_ms']}ms (tokens={warm['prompt_tokens']}, cached={warm['cached_prompt_tokens']})", flush=True)
        results["trials"].append({
            "prompt_len_target": pl,
            "prompt_tokens": cold["prompt_tokens"],
            "cold_ttft_ms": cold["ttft_ms"],
            "warm_ttft_ms": warm["ttft_ms"],
            "warm_cached_tokens": warm["cached_prompt_tokens"],
        })

    out_path = os.path.join(args.out_dir, f"cold_warm_ttft_{args.tag}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
