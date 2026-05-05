#!/usr/bin/env python3
"""
SSM State Quality Degradation Profiler v2 for Atlas.

Extended version with:
- Harder questions (needle-in-haystack, multi-step reasoning)
- Higher context lengths (up to 64K)
- Quality scoring beyond simple correctness
"""

import json
import subprocess
import sys
import time
import requests

API_BASE = "http://localhost:8888"
MODEL = "Sehyo/Qwen3.5-35B-A3B-NVFP4"

# Paragraph to repeat for padding context
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

# Target token counts to test
TARGET_TOKENS = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536]

# The needle: a specific fact buried early in the context
NEEDLE_FACT = (
    "IMPORTANT FACT: The secret code for project Zephyr is BLUE-FALCON-7742. "
    "Remember this code, you will be asked about it later. "
)

# Test 1: Simple math (baseline)
TEST_MATH = {
    "name": "math_2+2",
    "question": "What is 2+2? Answer with just the number.",
    "check": lambda t: "4" in t.strip() and len(t.strip()) < 50,
    "needle": False,
}

# Test 2: Needle in haystack (recall specific fact)
TEST_NEEDLE = {
    "name": "needle_recall",
    "question": "What is the secret code for project Zephyr? Answer with just the code.",
    "check": lambda t: "BLUE-FALCON-7742" in t.upper().replace(" ", "").replace("-", "").replace("–", "")
                        or ("BLUE" in t.upper() and "FALCON" in t.upper() and "7742" in t),
    "needle": True,
}

# Test 3: Count/reasoning about the filler
TEST_REASON = {
    "name": "reasoning",
    "question": "Name three people mentioned in the text above. Answer with just the names, comma-separated.",
    "check": lambda t: sum(1 for name in ["Babbage", "Lovelace", "Turing", "Kilby", "Noyce"]
                          if name.lower() in t.lower()) >= 2,
    "needle": False,
}


def build_prompt(target_tokens: int, question: str, inject_needle: bool) -> str:
    """Build a prompt with approximately target_tokens of filler text."""
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

    template_no_filler = template.format(filler="", question=question)
    overhead_tokens = len(template_no_filler) // 4
    filler_tokens_needed = target_tokens - overhead_tokens

    filler_token_est = len(FILLER_PARAGRAPH) // 4
    repeats = max(1, filler_tokens_needed // filler_token_est)

    filler = (FILLER_PARAGRAPH * repeats)
    target_chars = filler_tokens_needed * 4
    filler = filler[:target_chars]

    # Inject needle near the beginning (after ~10% of filler)
    if inject_needle:
        inject_pos = len(filler) // 10
        filler = filler[:inject_pos] + "\n\n" + NEEDLE_FACT + "\n\n" + filler[inject_pos:]
        # Trim back to target
        filler = filler[:target_chars]

    return template.format(filler=filler, question=question)


def send_request(prompt: str, max_tokens: int = 50) -> dict:
    """Send a completion request and return parsed response with timing."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }

    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{API_BASE}/v1/completions",
            json=payload,
            timeout=600,
        )
        t1 = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"error": "TIMEOUT (600s)", "wall_time": 600.0}
    except requests.exceptions.ConnectionError as e:
        return {"error": f"CONNECTION_ERROR: {e}", "wall_time": time.monotonic() - t0}
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


def assess_output(text: str, check_fn) -> str:
    """Assess output quality."""
    cleaned = text.strip()
    if not cleaned:
        return "EMPTY"
    # Check for garbled output
    if cleaned.count("�") > 2:
        return "GARBLED"
    if len(set(cleaned)) < 3 and len(cleaned) > 10:
        return "GARBLED"
    # Extreme repetition check
    if len(cleaned) > 30:
        words = cleaned.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.1:
                return "GARBLED"
    if check_fn(cleaned):
        return "PASS"
    return "FAIL"


def get_docker_logs_tail(n: int = 5) -> str:
    """Get last N lines from atlas-35b docker logs."""
    try:
        result = subprocess.run(
            ["sudo", "docker", "logs", "atlas-35b", "--tail", str(n)],
            capture_output=True, text=True, timeout=10
        )
        return result.stderr + result.stdout
    except Exception as e:
        return f"(log fetch failed: {e})"


def run_test_suite(test: dict, targets: list) -> list:
    """Run a single test across all target context lengths."""
    results = []
    print(f"\n{'='*100}")
    print(f"TEST: {test['name']} — {test['question'][:60]}")
    print(f"{'='*100}")

    for target in targets:
        print(f"\n  [{test['name']}] ~{target} tokens ...", end=" ", flush=True)

        prompt = build_prompt(target, test["question"], test["needle"])
        time.sleep(0.5)

        result = send_request(prompt, max_tokens=80)

        if result.get("error"):
            print(f"ERROR: {result['error'][:60]}")
            results.append({
                "test": test["name"],
                "target": target,
                "actual": 0,
                "ttft_ms": 0,
                "prefill_tps": 0,
                "decode_tps": 0,
                "finish": "ERROR",
                "quality": "ERROR",
                "output": result["error"][:60],
                "wall_s": result["wall_time"],
            })
            # Check logs
            logs = get_docker_logs_tail(3)
            err_lines = [l for l in logs.split("\n") if "ERROR" in l or "panic" in l.lower() or "OOM" in l]
            if err_lines:
                for el in err_lines[:2]:
                    print(f"    LOG: {el.strip()[:100]}")
            continue

        actual = result["prompt_tokens"]
        ttft = result["ttft_ms"]
        prefill_rate = actual / (ttft / 1000.0) if ttft > 0 else 0
        decode = result["decode_tps"]
        finish = result["finish_reason"]
        quality = assess_output(result["text"], test["check"])
        output_preview = result["text"].strip()[:60].replace("\n", "\\n")

        status_char = "+" if quality == "PASS" else "!" if quality == "FAIL" else "X"
        print(f"[{status_char}] {actual} tok, TTFT={ttft:.0f}ms, decode={decode:.1f}t/s, {quality}: {output_preview!r}")

        results.append({
            "test": test["name"],
            "target": target,
            "actual": actual,
            "ttft_ms": ttft,
            "prefill_tps": prefill_rate,
            "decode_tps": decode,
            "finish": finish,
            "quality": quality,
            "output": output_preview,
            "wall_s": result["wall_time"],
            "completion_tokens": result["completion_tokens"],
        })

    return results


def main():
    print("=" * 100)
    print("SSM State Quality Degradation Profiler v2")
    print(f"Model: {MODEL}")
    print(f"Tests: math, needle-in-haystack, reasoning")
    print(f"Context lengths: {TARGET_TOKENS}")
    print(f"Max seq len: 65536 (server config)")
    print("=" * 100)

    # Verify server
    try:
        r = requests.get(f"{API_BASE}/v1/models", timeout=5)
        r.raise_for_status()
        print("[OK] Server responding")
    except Exception as e:
        print(f"[FAIL] Server not reachable: {e}")
        sys.exit(1)

    # Warmup
    print("[WARMUP] ...")
    warmup_prompt = build_prompt(512, "Say hello.", False)
    send_request(warmup_prompt, max_tokens=5)
    print("[WARMUP] done\n")

    all_results = []

    # Run all three test suites
    for test in [TEST_MATH, TEST_NEEDLE, TEST_REASON]:
        results = run_test_suite(test, TARGET_TOKENS)
        all_results.extend(results)

    # Print combined summary table
    print("\n\n")
    print("=" * 140)
    print("COMBINED RESULTS TABLE")
    print("=" * 140)

    # Group by test
    for test in [TEST_MATH, TEST_NEEDLE, TEST_REASON]:
        test_results = [r for r in all_results if r["test"] == test["name"]]
        print(f"\n--- {test['name']} ---")
        print(f"{'Target':>8} {'Actual':>8} {'TTFT(ms)':>10} {'Prefill':>10} {'Decode':>10} {'Finish':>8} {'Quality':>8}  Output")
        print("-" * 130)
        for r in test_results:
            pf = f"{r['prefill_tps']:.0f}t/s" if r['prefill_tps'] else "N/A"
            dc = f"{r['decode_tps']:.1f}t/s" if r['decode_tps'] else "N/A"
            print(
                f"{r['target']:>8} {r['actual']:>8} {r['ttft_ms']:>10.1f} {pf:>10} {dc:>10} "
                f"{r['finish']:>8} {r['quality']:>8}  {r['output'][:50]}"
            )

    # Performance trend table
    print(f"\n\n{'='*100}")
    print("PERFORMANCE TREND (math_2+2 — simplest test)")
    print(f"{'='*100}")
    math_results = [r for r in all_results if r["test"] == "math_2+2"]
    if math_results:
        print(f"{'Tokens':>8} {'TTFT(ms)':>10} {'ms/tok':>8} {'Decode':>10} {'Quality':>8}")
        print("-" * 60)
        for r in math_results:
            ms_per_tok = r["ttft_ms"] / r["actual"] if r["actual"] > 0 else 0
            dc = f"{r['decode_tps']:.1f}" if r['decode_tps'] else "N/A"
            print(f"{r['actual']:>8} {r['ttft_ms']:>10.1f} {ms_per_tok:>8.3f} {dc:>10} {r['quality']:>8}")

    # Degradation analysis
    print(f"\n\n{'='*100}")
    print("DEGRADATION ANALYSIS")
    print(f"{'='*100}")

    for test in [TEST_MATH, TEST_NEEDLE, TEST_REASON]:
        test_results = [r for r in all_results if r["test"] == test["name"]]
        last_pass = None
        first_fail = None
        for r in test_results:
            if r["quality"] == "PASS":
                last_pass = r["target"]
            elif r["quality"] in ("FAIL", "GARBLED", "EMPTY", "ERROR") and first_fail is None:
                first_fail = r["target"]

        if first_fail:
            print(f"  {test['name']}: DEGRADATION at ~{first_fail} tokens (last pass: ~{last_pass or 0})")
        else:
            max_t = test_results[-1]["target"] if test_results else 0
            fails = [r["target"] for r in test_results if r["quality"] == "FAIL"]
            if fails:
                print(f"  {test['name']}: No garbling, but WRONG answers at {fails}")
            else:
                print(f"  {test['name']}: All PASS up to ~{max_t} tokens")


if __name__ == "__main__":
    main()
