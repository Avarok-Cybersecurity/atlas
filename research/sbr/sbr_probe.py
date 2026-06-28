#!/usr/bin/env python3
"""Minimal serial Marconi warm-hit probe — does the warm-hit SSM path fire at all?

Serial (no concurrency), ample slots, no distractors. Sends:
  1. P            (cold prefill)
  2. P            (exact repeat -> exact leaf-snapshot hit, should be fast)
  3. P + s200     (intermediate hit + ~200-tok replay)
  4. P + s2000    (deeper resume + ~2000-tok replay)
Reports TTFT for each; pair with server log greps for 'Marconi intermediate hit'.
"""
import argparse, json, time, urllib.request

WORDS = ("the model integrates state across tokens while the recurrent layer "
         "accumulates a decaying hidden matrix and attention reads the cache").split()


def filler(n_tokens):
    n = int(n_tokens / 0.75)
    return " ".join(WORDS[i % len(WORDS)] for i in range(n))


def ttft(base_url, model, messages, max_tokens=16):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    d = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if d.get("choices", [{}])[0].get("delta", {}).get("content"):
                    first = time.perf_counter() - t0
                    break
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8888/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    ap.add_argument("--base-tokens", type=int, default=2000)
    a = ap.parse_args()
    sysdoc = "Reference:\n" + filler(a.base_tokens)
    P = [{"role": "system", "content": sysdoc}, {"role": "user", "content": "Q0: summarize."}]

    print(f"1. cold P        TTFT={ttft(a.base_url, a.model, P):.3f}s", flush=True)
    print(f"2. repeat P      TTFT={ttft(a.base_url, a.model, P):.3f}s", flush=True)
    P2 = P + [{"role": "assistant", "content": filler(200)}, {"role": "user", "content": "Q1?"}]
    print(f"3. P+~200 replay TTFT={ttft(a.base_url, a.model, P2):.3f}s", flush=True)
    P3 = P + [{"role": "assistant", "content": filler(2000)}, {"role": "user", "content": "Q2?"}]
    print(f"4. P+~2000 reply TTFT={ttft(a.base_url, a.model, P3):.3f}s", flush=True)


if __name__ == "__main__":
    main()
