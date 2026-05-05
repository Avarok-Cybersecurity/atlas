#!/usr/bin/env python3
"""TTFT + decode-TPS benchmark for Qwen3.5/3.6-35B-A3B-FP8 on DGX Spark.

Used by the overnight /loop TTFT optimization — establishes a baseline,
then re-measures after each Atlas change. Fails the run if decode TPS
regresses more than DECODE_GUARD_PCT from the committed baseline.

Usage:
    python3 bench/qwen36_ttft.py --url http://localhost:8888 \
        --model Qwen/Qwen3.5-35B-A3B-FP8 --tag alpha-2.43-baseline

Output:
    bench/qwen36_ttft_<tag>.json
"""
import argparse
import json
import os
import sys
import time
import urllib.request

INPUTS = [256, 1024, 4096]  # prompt token target lengths for TTFT
DECODE_OUTPUT_TOKENS = [128, 512]  # decode TPS probe lengths
DECODE_GUARD_PCT = 0.03

LOREM = (
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled "
    "with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with "
    "nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort. "
)


def _post(url: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _count_tokens(url: str, model: str, prompt: str) -> int:
    # Best-effort: not all Atlas builds expose /v1/tokenize. Fall back to
    # a rough char/4 estimate.
    try:
        r = _post(f"{url}/v1/tokenize", {"model": model, "prompt": prompt})
        return int(r.get("count", len(prompt) // 4))
    except Exception:
        return len(prompt) // 4


def build_prompt(target_tokens: int) -> str:
    # Atlas's tokenizer is BPE-ish; 4 chars/token is a decent first pass.
    target_chars = target_tokens * 4
    reps = (target_chars // len(LOREM)) + 1
    prompt = (LOREM * reps)[:target_chars]
    return prompt + "\n\nQuestion: summarize the passage in one sentence.\nAnswer:"


def measure_ttft(url: str, model: str, prompt_len: int) -> dict:
    prompt = build_prompt(prompt_len)
    t0 = time.perf_counter()
    r = _post(
        f"{url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=300.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    usage = r.get("usage", {})
    return {
        "prompt_len_target": prompt_len,
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "ttft_ms": round(elapsed_ms, 1),
    }


def measure_decode_tps(url: str, model: str, out_tokens: int) -> dict:
    prompt = "Write a long, continuous paragraph about the history of computing. Start:\n"
    t0 = time.perf_counter()
    r = _post(
        f"{url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": out_tokens,
            "temperature": 0.0,
        },
        timeout=600.0,
    )
    elapsed = time.perf_counter() - t0
    usage = r.get("usage", {})
    produced = int(usage.get("completion_tokens", 0))
    tps = (produced / elapsed) if elapsed > 0 and produced > 0 else 0.0
    return {
        "out_tokens_target": out_tokens,
        "out_tokens_actual": produced,
        "elapsed_s": round(elapsed, 3),
        "tps": round(tps, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="bench")
    ap.add_argument("--baseline", default=None,
                    help="path to prior run's JSON; enforces decode guard")
    args = ap.parse_args()

    # Warm-up: ensure the model is loaded and CUDA graphs are captured.
    print("warm-up...", flush=True)
    try:
        _post(f"{args.url}/v1/completions",
              {"model": args.model, "prompt": "Hello", "max_tokens": 8},
              timeout=300.0)
    except Exception as e:
        print(f"warm-up failed: {e}", file=sys.stderr)
        return 2

    results = {"tag": args.tag, "model": args.model, "ttft": [], "decode": []}

    for pl in INPUTS:
        print(f"ttft: prompt_len={pl}", flush=True)
        results["ttft"].append(measure_ttft(args.url, args.model, pl))

    for ot in DECODE_OUTPUT_TOKENS:
        print(f"decode: out_tokens={ot}", flush=True)
        results["decode"].append(measure_decode_tps(args.url, args.model, ot))

    out_path = os.path.join(args.out_dir, f"qwen36_ttft_{args.tag}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    # Decode guard.
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            prior = json.load(f)
        prior_tps = {d["out_tokens_target"]: d["tps"] for d in prior.get("decode", [])}
        regressed = []
        for cur in results["decode"]:
            base = prior_tps.get(cur["out_tokens_target"])
            if base and base > 0 and cur["tps"] < base * (1.0 - DECODE_GUARD_PCT):
                regressed.append((cur["out_tokens_target"], base, cur["tps"]))
        if regressed:
            print("DECODE REGRESSION:", file=sys.stderr)
            for ot, base, now in regressed:
                pct = (1 - now / base) * 100
                print(f"  out_tokens={ot}: {base:.2f} → {now:.2f} ({pct:.1f}% slower)",
                      file=sys.stderr)
            return 3

    # Print a one-line summary for easy scraping.
    ttft_summary = ", ".join(f"{r['prompt_tokens']}→{r['ttft_ms']}ms"
                             for r in results["ttft"])
    decode_summary = ", ".join(f"{r['out_tokens_actual']}@{r['tps']}tps"
                               for r in results["decode"])
    print(f"TTFT: {ttft_summary}")
    print(f"DECODE: {decode_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
