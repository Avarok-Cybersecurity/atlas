#!/usr/bin/env python3
"""
SSM State Quality Degradation Profiler for Atlas.

Sends increasing context lengths to find the exact threshold where SSM state
quality breaks down (garbled output, failure to stop at EOS, etc).

Uses /v1/completions with raw prompts (no chat template overhead).
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
TARGET_TOKENS = [1024, 2048, 4096, 8192, 16384, 24576, 32768]

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{filler}\n\n"
    "After reading the above text, what is 2+2? Answer with just the number.<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n"
    "</think>\n\n"
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 chars per token for English text."""
    return len(text) // 4


def build_prompt(target_tokens: int) -> str:
    """Build a prompt with approximately target_tokens of filler text."""
    # Template overhead (system, user wrapper, think tags, question)
    template_no_filler = PROMPT_TEMPLATE.format(filler="")
    overhead_tokens = estimate_tokens(template_no_filler)
    filler_tokens_needed = target_tokens - overhead_tokens

    filler_token_est = estimate_tokens(FILLER_PARAGRAPH)
    repeats = max(1, filler_tokens_needed // filler_token_est)

    filler = (FILLER_PARAGRAPH * repeats)
    # Trim to approximate target
    target_chars = filler_tokens_needed * 4
    filler = filler[:target_chars]

    return PROMPT_TEMPLATE.format(filler=filler)


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


def check_coherence(text: str) -> str:
    """Check if the output is coherent and answers the math question."""
    cleaned = text.strip()
    if not cleaned:
        return "EMPTY"
    if "4" in cleaned and len(cleaned) < 200:
        return "CORRECT"
    # Check for garbled output indicators
    garbled_indicators = 0
    if cleaned.count("�") > 0:
        garbled_indicators += 1
    if len(set(cleaned)) < 3 and len(cleaned) > 10:
        garbled_indicators += 1  # repetitive single chars
    # Check for extreme repetition
    if len(cleaned) > 20:
        chunks = [cleaned[i:i+5] for i in range(0, len(cleaned)-5, 5)]
        if len(chunks) > 3 and len(set(chunks)) <= 2:
            garbled_indicators += 1
    if garbled_indicators > 0:
        return "GARBLED"
    if len(cleaned) > 100:
        return "VERBOSE"
    return "WRONG"


def get_docker_logs_tail(n: int = 10) -> str:
    """Get last N lines from atlas-35b docker logs."""
    try:
        result = subprocess.run(
            ["sudo", "docker", "logs", "atlas-35b", "--tail", str(n)],
            capture_output=True, text=True, timeout=10
        )
        return result.stderr + result.stdout
    except Exception as e:
        return f"(log fetch failed: {e})"


def main():
    print("=" * 100)
    print("SSM State Quality Degradation Profiler")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {API_BASE}/v1/completions")
    print(f"Test lengths: {TARGET_TOKENS}")
    print("=" * 100)
    print()

    # Verify server is up
    try:
        r = requests.get(f"{API_BASE}/v1/models", timeout=5)
        r.raise_for_status()
        print("[OK] Server is responding")
    except Exception as e:
        print(f"[FAIL] Server not reachable: {e}")
        sys.exit(1)

    # Warmup request
    print("[WARMUP] Sending short warmup request...")
    warmup_prompt = PROMPT_TEMPLATE.format(filler="This is a short warmup.")
    warmup = send_request(warmup_prompt, max_tokens=10)
    if warmup.get("error"):
        print(f"  Warmup failed: {warmup['error']}")
    else:
        print(f"  Warmup OK: {warmup['text']!r} ({warmup['prompt_tokens']} prompt tokens)")
    print()

    results = []

    for target in TARGET_TOKENS:
        print(f"--- Testing ~{target} tokens ---")
        prompt = build_prompt(target)

        # Clear any prior request state by waiting briefly
        time.sleep(1.0)

        result = send_request(prompt, max_tokens=50)

        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            results.append({
                "target_tokens": target,
                "actual_prompt_tokens": 0,
                "ttft_ms": 0,
                "decode_tps": 0,
                "finish_reason": "ERROR",
                "coherence": "ERROR",
                "output": result["error"][:80],
                "wall_s": result["wall_time"],
            })
            # Check logs for errors
            logs = get_docker_logs_tail(5)
            print(f"  Docker logs tail:\n{logs}")
            continue

        coherence = check_coherence(result["text"])
        output_preview = result["text"].strip()[:80].replace("\n", "\\n")

        actual_tokens = result["prompt_tokens"]
        ttft = result["ttft_ms"]
        decode_speed = result["decode_tps"]
        finish = result["finish_reason"]
        wall = result["wall_time"]

        prefill_rate = actual_tokens / (ttft / 1000.0) if ttft > 0 else 0

        print(f"  Prompt tokens: {actual_tokens}")
        print(f"  TTFT: {ttft:.1f}ms ({prefill_rate:.0f} tok/s prefill)")
        print(f"  Decode: {decode_speed:.1f} tok/s")
        print(f"  Finish: {finish}")
        print(f"  Coherence: {coherence}")
        print(f"  Output: {output_preview!r}")
        print(f"  Wall time: {wall:.2f}s")

        results.append({
            "target_tokens": target,
            "actual_prompt_tokens": actual_tokens,
            "ttft_ms": ttft,
            "prefill_tps": prefill_rate,
            "decode_tps": decode_speed,
            "finish_reason": finish,
            "coherence": coherence,
            "output": output_preview,
            "wall_s": wall,
            "completion_tokens": result["completion_tokens"],
        })

        # Check docker logs for DIAG output
        logs = get_docker_logs_tail(3)
        diag_lines = [l for l in logs.split("\n") if "DIAG" in l or "norm" in l.lower() or "ERROR" in l or "panic" in l.lower()]
        if diag_lines:
            print(f"  DIAG from logs:")
            for dl in diag_lines[:3]:
                print(f"    {dl.strip()}")

        print()

    # Print summary table
    print()
    print("=" * 130)
    print(f"{'Target':>8} {'Actual':>8} {'TTFT(ms)':>10} {'Prefill':>10} {'Decode':>10} {'Finish':>8} {'Coherence':>10} {'Wall(s)':>8}  Output")
    print("-" * 130)
    for r in results:
        prefill_str = f"{r.get('prefill_tps', 0):.0f} t/s" if r.get('prefill_tps') else "N/A"
        decode_str = f"{r.get('decode_tps', 0):.1f} t/s" if r.get('decode_tps') else "N/A"
        print(
            f"{r['target_tokens']:>8} "
            f"{r.get('actual_prompt_tokens', 0):>8} "
            f"{r.get('ttft_ms', 0):>10.1f} "
            f"{prefill_str:>10} "
            f"{decode_str:>10} "
            f"{r.get('finish_reason', '?'):>8} "
            f"{r.get('coherence', '?'):>10} "
            f"{r.get('wall_s', 0):>8.2f}  "
            f"{r.get('output', '')[:60]}"
        )
    print("=" * 130)

    # Determine degradation point
    print("\n--- Analysis ---")
    last_good = None
    first_bad = None
    for r in results:
        coh = r.get("coherence", "")
        if coh in ("CORRECT", "VERBOSE"):
            last_good = r["target_tokens"]
        elif coh in ("GARBLED", "EMPTY", "ERROR") and first_bad is None:
            first_bad = r["target_tokens"]

    if first_bad:
        print(f"DEGRADATION DETECTED at ~{first_bad} tokens")
        if last_good:
            print(f"Last good output at ~{last_good} tokens")
        print(f"Quality cliff between {last_good or 0} and {first_bad} tokens")
    else:
        max_tested = results[-1]["target_tokens"] if results else 0
        print(f"No degradation detected up to ~{max_tested} tokens")
        if any(r.get("coherence") == "WRONG" for r in results):
            wrong_at = [r["target_tokens"] for r in results if r.get("coherence") == "WRONG"]
            print(f"WRONG (but coherent) answers at: {wrong_at}")


if __name__ == "__main__":
    main()
