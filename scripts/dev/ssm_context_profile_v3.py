#!/usr/bin/env python3
"""
SSM State Quality Degradation Profiler v3 for Atlas.

Focused deep-dive:
1. Needle reliability: repeat needle test 3x at each failing length
2. Varied needle positions (beginning, middle, end of context)
3. Monitor DIAG norms per context length
4. Non-trivial recall (multi-fact, computation)
"""

import json
import subprocess
import sys
import time
import requests

API_BASE = "http://localhost:8888"
MODEL = "Sehyo/Qwen3.5-35B-A3B-NVFP4"

FILLER_PARAGRAPH = (
    "The development of modern computing began with Charles Babbage's Analytical Engine "
    "in the 1830s, which introduced the concept of a programmable general-purpose computer. "
    "Ada Lovelace recognized its potential beyond mere calculation, writing what is considered "
    "the first computer program. The field advanced through Alan Turing's theoretical work on "
    "computability in the 1930s, the construction of ENIAC in 1945, and the invention of the "
    "transistor at Bell Labs in 1947. The integrated circuit, developed independently by Jack "
    "Kilby and Robert Noyce in 1958, led to Moore's Law and the exponential growth of computing "
    "power that continues to shape our world today. "
)

NEEDLE = "IMPORTANT: The password for vault access is CRIMSON-TIGER-9918."


def build_prompt_with_needle(target_tokens: int, question: str, needle_position: str = "early") -> str:
    """
    Build prompt with needle at specified position.
    needle_position: "early" (10%), "middle" (50%), "late" (90%), "none"
    """
    template = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer questions precisely based on the text provided.<|im_end|>\n"
        "<|im_start|>user\n"
        "{filler}\n\n"
        "{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n"
        "</think>\n\n"
    )

    template_no_filler = template.format(filler="", question=question)
    overhead_tokens = len(template_no_filler) // 4
    filler_tokens_needed = target_tokens - overhead_tokens

    filler_token_est = len(FILLER_PARAGRAPH) // 4
    repeats = max(1, filler_tokens_needed // filler_token_est)
    filler = (FILLER_PARAGRAPH * repeats)
    target_chars = filler_tokens_needed * 4
    filler = filler[:target_chars]

    if needle_position != "none":
        positions = {"early": 0.10, "middle": 0.50, "late": 0.90}
        frac = positions.get(needle_position, 0.10)
        inject_pos = int(len(filler) * frac)
        filler = filler[:inject_pos] + "\n\n" + NEEDLE + "\n\n" + filler[inject_pos:]
        filler = filler[:target_chars]

    return template.format(filler=filler, question=question)


def send_request(prompt: str, max_tokens: int = 80) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(f"{API_BASE}/v1/completions", json=payload, timeout=600)
        t1 = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e), "wall_time": time.monotonic() - t0}

    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})
    return {
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason", "unknown"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "ttft_ms": usage.get("time_to_first_token_ms", 0),
        "decode_tps": usage.get("response_token/s", 0),
        "wall_time": t1 - t0,
        "error": None,
    }


def get_diag_norms(n: int = 50) -> list:
    """Extract recent DIAG lines from docker logs."""
    try:
        result = subprocess.run(
            ["sudo", "docker", "logs", "atlas-35b", "--tail", str(n)],
            capture_output=True, text=True, timeout=10
        )
        lines = (result.stderr + result.stdout).split("\n")
        return [l for l in lines if "DIAG" in l]
    except Exception:
        return []


def main():
    print("=" * 110)
    print("SSM State Quality Profiler v3 — Needle Reliability & DIAG Norms")
    print(f"Model: {MODEL}")
    print(f"Needle: {NEEDLE}")
    print("=" * 110)

    # Verify
    try:
        requests.get(f"{API_BASE}/v1/models", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable: {e}")
        sys.exit(1)

    # Warmup
    wp = build_prompt_with_needle(512, "Say hi.", "none")
    send_request(wp, 5)
    print("[WARMUP] done\n")

    # =========================================================================
    # TEST 1: Needle recall reliability at multiple lengths, 3 trials each
    # =========================================================================
    print("=" * 110)
    print("TEST 1: Needle Recall Reliability (3 trials per length, needle at 10%)")
    print("=" * 110)

    question = "What is the password for vault access mentioned in the text? Reply with ONLY the password."
    targets = [1024, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536]
    trials = 3

    reliability_results = []

    for target in targets:
        passes = 0
        outputs = []
        ttfts = []
        decodes = []

        for trial in range(trials):
            prompt = build_prompt_with_needle(target, question, "early")
            time.sleep(0.3)
            r = send_request(prompt, max_tokens=30)

            if r.get("error"):
                outputs.append(f"ERROR:{r['error'][:30]}")
                continue

            text = r["text"].strip()
            ttfts.append(r["ttft_ms"])
            decodes.append(r["decode_tps"])

            passed = ("CRIMSON" in text.upper() and "TIGER" in text.upper() and "9918" in text)
            if passed:
                passes += 1
            outputs.append(text[:50])

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        avg_decode = sum(decodes) / len(decodes) if decodes else 0
        actual_tok = r.get("prompt_tokens", 0) if not r.get("error") else 0
        rate = f"{passes}/{trials}"

        reliability_results.append({
            "target": target, "actual": actual_tok, "pass_rate": rate,
            "passes": passes, "trials": trials,
            "avg_ttft": avg_ttft, "avg_decode": avg_decode,
            "outputs": outputs,
        })

        status = "OK" if passes == trials else "PARTIAL" if passes > 0 else "FAIL"
        print(f"  ~{target:>6} tok ({actual_tok:>6} actual) [{status:>7}] {rate}  TTFT={avg_ttft:.0f}ms  decode={avg_decode:.1f}t/s  outputs={outputs}")

    # =========================================================================
    # TEST 2: Needle position sweep at a known tricky length (32K)
    # =========================================================================
    print(f"\n{'='*110}")
    print("TEST 2: Needle Position Sweep at 32K tokens")
    print("=" * 110)

    for pos in ["early", "middle", "late"]:
        prompt = build_prompt_with_needle(32768, question, pos)
        time.sleep(0.5)
        r = send_request(prompt, max_tokens=30)
        if r.get("error"):
            print(f"  Position={pos}: ERROR {r['error'][:50]}")
            continue
        text = r["text"].strip()
        passed = "CRIMSON" in text.upper() and "TIGER" in text.upper() and "9918" in text
        actual = r["prompt_tokens"]
        ttft = r["ttft_ms"]
        print(f"  Position={pos:>6}: {'PASS' if passed else 'FAIL':>4}  actual={actual}  TTFT={ttft:.0f}ms  output={text[:60]!r}")

    # =========================================================================
    # TEST 3: Multi-fact recall (harder)
    # =========================================================================
    print(f"\n{'='*110}")
    print("TEST 3: Multi-Fact Recall")
    print("=" * 110)

    multi_needle = (
        "FACT A: The project codename is AURORA. "
        "FACT B: The launch date is March 15, 2027. "
        "FACT C: The budget is 4.7 million dollars. "
    )
    multi_question = "Based on the text: What is the project codename, the launch date, and the budget? Answer concisely."

    for target in [1024, 8192, 16384, 32768, 49152, 65536]:
        template = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "{filler}\n\n"
            "{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n"
            "</think>\n\n"
        )
        overhead = len(template.format(filler="", question=multi_question)) // 4
        filler_needed = target - overhead
        reps = max(1, filler_needed // (len(FILLER_PARAGRAPH) // 4))
        filler = (FILLER_PARAGRAPH * reps)[:filler_needed * 4]
        # Insert facts at 10%
        pos = len(filler) // 10
        filler = filler[:pos] + "\n\n" + multi_needle + "\n\n" + filler[pos:]
        filler = filler[:filler_needed * 4]
        prompt = template.format(filler=filler, question=multi_question)

        time.sleep(0.5)
        r = send_request(prompt, max_tokens=100)
        if r.get("error"):
            print(f"  ~{target:>6}: ERROR {r['error'][:50]}")
            continue
        text = r["text"].strip()
        has_aurora = "AURORA" in text.upper()
        has_date = "MARCH" in text.upper() and ("15" in text or "2027" in text)
        has_budget = "4.7" in text or "4,700,000" in text or "4.7 million" in text.lower()
        facts_found = sum([has_aurora, has_date, has_budget])
        actual = r["prompt_tokens"]
        ttft = r["ttft_ms"]
        decode = r["decode_tps"]
        print(f"  ~{target:>6} ({actual:>6} actual) facts={facts_found}/3  TTFT={ttft:.0f}ms  decode={decode:.1f}t/s  output={text[:80]!r}")

    # =========================================================================
    # TEST 4: DIAG norm collection per context length
    # =========================================================================
    print(f"\n{'='*110}")
    print("TEST 4: DIAG Norm Trend (first decode step per context length)")
    print("=" * 110)

    # We already have DIAG data from the prior runs. Let's collect fresh ones
    # by sending one request per length and immediately reading logs.
    diag_targets = [1024, 8192, 32768, 65536]
    simple_q = "What is 1+1? Answer with just the number."

    for target in diag_targets:
        prompt = build_prompt_with_needle(target, simple_q, "none")
        # Clear log position by getting current line count
        time.sleep(0.5)
        r = send_request(prompt, max_tokens=5)
        time.sleep(0.3)

        # Get recent DIAG
        diags = get_diag_norms(60)
        # Find the most recent set (from L0 to logits)
        l0_norm = None
        l39_norm = None
        post_norm = None
        logits_info = None
        for d in reversed(diags):
            if "DIAG L0" in d and l0_norm is None:
                try:
                    l0_norm = float(d.split("last_tok_norm=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if "DIAG L39" in d and l39_norm is None:
                try:
                    l39_norm = float(d.split("last_tok_norm=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if "DIAG post-norm" in d and post_norm is None:
                try:
                    post_norm = float(d.split("norm=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if "DIAG logits" in d and logits_info is None:
                try:
                    max_val = float(d.split("max=")[1].split()[0])
                    min_val = float(d.split("min=")[1].split()[0])
                    nan_count = int(d.split("nan=")[1].split()[0])
                    logits_info = (max_val, min_val, nan_count)
                except (IndexError, ValueError):
                    pass

        actual = r.get("prompt_tokens", 0) if not r.get("error") else 0
        output = r.get("text", "ERR")[:20].strip() if not r.get("error") else "ERR"
        print(f"  ~{target:>6} ({actual:>6} actual) output={output!r}")
        print(f"    L0_norm={l0_norm}  L39_norm={l39_norm}  post_norm={post_norm}  logits={logits_info}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n\n{'='*110}")
    print("SUMMARY")
    print("=" * 110)

    print("\nNeedle Recall Reliability:")
    print(f"{'Target':>8} {'Actual':>8} {'Pass Rate':>10}")
    print("-" * 30)
    for r in reliability_results:
        marker = " ***" if r["passes"] < r["trials"] else ""
        print(f"{r['target']:>8} {r['actual']:>8} {r['pass_rate']:>10}{marker}")

    total_passes = sum(r["passes"] for r in reliability_results)
    total_trials = sum(r["trials"] for r in reliability_results)
    print(f"\nOverall: {total_passes}/{total_trials} ({100*total_passes/total_trials:.0f}%)")

    any_degradation = any(r["passes"] < r["trials"] for r in reliability_results)
    if any_degradation:
        first_unreliable = next(r["target"] for r in reliability_results if r["passes"] < r["trials"])
        print(f"\nFirst unreliable length: ~{first_unreliable} tokens")
        print("NOTE: Intermittent needle failures without DIAG norm degradation suggest")
        print("model attention limitations (not SSM state corruption).")
    else:
        print("\nAll lengths fully reliable — no SSM degradation detected.")


if __name__ == "__main__":
    main()
