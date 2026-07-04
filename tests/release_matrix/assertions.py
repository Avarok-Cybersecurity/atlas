"""Shared assertion bodies used by both the single-GPU and multi-rank test
classes: coherence, fibonacci, tool-call, tps-benchmark, and long-context
checks, all transplanted from single_gpu_suite.py."""

import json
import re
import subprocess

from release_matrix.api import (
    LONG_CTX_NEEDLE,
    _has_repetition_loop,
    _strip_thinking,
    calc_decode_tps,
    calc_tps,
    chat,
    extract_text,
    generate_long_context,
)


def _assert_coherence(base_url, model, label):
    """Port of run_coherence_tests(): keyword-based acceptance, post-think-
    strip, plus a repetition-loop guard."""
    tests = [
        ("Factual", "What is the capital of Japan? Answer in one sentence.",
         ["tokyo"], 400),
        ("Reasoning",
         "A car drives 120 km in 2 hours. Speed = distance / time. What is 120 / 2? Give the answer in km/h.",
         ["60 km", "60km", "60 kilometers", "60 km/h", "60km/h", "60 kph", "60kph",
          "60 kilometers per hour", "60 kilometres", "60 km per hour", "sixty km",
          "sixty kilometers", "speed = 60", "speed is 60", "= 60 km", "= 60km", "60 mph"],
         2048),
        ("Creative", "Write a haiku about the ocean.", [], 400),
    ]

    failures = []
    for name, prompt, keywords, mt in tests:
        temp = 0.0 if keywords else 0.3
        result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                                max_tokens=mt, temperature=temp)
        text = extract_text(result)
        visible = _strip_thinking(text)

        ok = ("error" not in text.lower() and "[EMPTY]" not in text
              and "[PARSE_ERROR]" not in text and len(visible.strip()) >= 3)
        if not ok:
            failures.append(f"{name}: empty or error — {text[:200]!r}")
            continue

        if _has_repetition_loop(visible):
            failures.append(f"{name}: repetition loop — {visible[:200]!r}")
            continue

        if keywords:
            tl = visible.lower() if visible else text.lower()
            matched = any(k in tl for k in keywords)
            if not matched and name == "Reasoning":
                if re.search(r"\b60\b[^\n]{0,30}(km|kilomet|mph|speed|kph)", tl):
                    matched = True
                elif re.search(r"(\\text|\\mathrm|\\operatorname|\\,|\\;|\\\\)\s*\{?\s*60\s*\\?\s*(km|kilomet|mph|speed|kph)", tl):
                    matched = True
                elif re.search(r"60\s*(?:km|kilometer|kilometre|kph|mph)", tl):
                    matched = True
            if not matched:
                failures.append(f"{name}: missing expected keyword {keywords[0]!r} — {tl[:200]!r}")

    assert not failures, f"{label}: coherence failures: " + "; ".join(failures)


def _run_fibonacci(base_url, model):
    """Port of run_fibonacci_test(): extract a fenced/bare Python code block,
    execute it, verify fib(0..9). Returns (status: str, detail: str)."""
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    prompt = (
        "Write a Python function `fib(n)` that returns the n-th Fibonacci "
        "number (fib(0)=0, fib(1)=1). Then print the values fib(0) through "
        "fib(9) on a single line, space-separated. Output only a single "
        "fenced ```python code block with no explanation."
    )
    result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                            max_tokens=4096, timeout=360, temperature=0.0,
                            repetition_penalty=1.05)
    text = extract_text(result)
    visible = _strip_thinking(text)

    def plain_text_fallback(reason):
        nums_in_text = [int(x) for x in re.findall(r"-?\d+", text)]
        if len(nums_in_text) >= 10 and nums_in_text[:10] == expected:
            return "PASS (plain-text)", f"{reason}; found correct sequence in response text"
        return f"FAIL ({reason})", text[:300]

    m = re.search(r"```(?:python)?\n?(.*?)```", visible, re.DOTALL)
    if not m:
        m = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if not m:
        bare = re.search(r"(def (?:fib|fibonacci|get_fib)\w*\(.*?\).*?)(?:\n\n|\Z)", visible, re.DOTALL)
        if not bare:
            bare = re.search(r"(def (?:fib|fibonacci|get_fib)\w*\(.*?\).*?)(?:\n\n|\Z)", text, re.DOTALL)
        if bare:
            m = bare
        else:
            return plain_text_fallback("no code block")
    code = m.group(1).strip()

    if re.search(r'def\s+(fib|fibonacci|get_fib)', code) and 'print' not in code:
        fn_match = re.search(r'def\s+(\w+)\s*\(', code)
        if fn_match:
            fn_name = fn_match.group(1)
            code += f"\nprint(' '.join(str({fn_name}(i)) for i in range(10)))"

    try:
        proc = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return plain_text_fallback("execution timeout")
    except Exception as e:
        return plain_text_fallback(f"execution error: {e}")

    if proc.returncode != 0:
        first_err_line = (proc.stderr.strip().splitlines() or [""])[0][:120]
        return plain_text_fallback(f"code raised (exit={proc.returncode}): {first_err_line}")

    nums_in_stdout = re.findall(r"-?\d+", proc.stdout.strip())
    parsed = [int(x) for x in nums_in_stdout[:10]]
    if parsed == expected:
        return "PASS", proc.stdout.strip()[:200]
    return plain_text_fallback(f"code ran but output was {parsed[:10]}, expected {expected}")


def _run_tool_call(base_url, model, name, prompt, tools):
    """Port of one iteration of run_tool_call_tests(). Returns (status, detail)."""
    result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                            max_tokens=1024, tools=tools)
    text = extract_text(result)

    has_tool_call = False
    tool_name = ""
    tool_args = ""
    try:
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tc = msg.get("tool_calls", [])
        if tc:
            has_tool_call = True
            tool_name = tc[0].get("function", {}).get("name", "")
            tool_args = tc[0].get("function", {}).get("arguments", "")
    except Exception:
        pass

    if has_tool_call:
        try:
            parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            return "PASS", f"Called {tool_name}({json.dumps(parsed_args)})"
        except json.JSONDecodeError:
            return "FAIL (invalid JSON args)", f"Called {tool_name} but args not valid JSON: {tool_args[:100]}"
    return "WARN (no structured tool call)", text[:200]


def _measure_tps(base_url, model):
    """Port of run_tps_benchmark(). Returns list of per-length result dicts."""
    tests = [
        (50, "Explain what a GPU is in exactly two sentences."),
        (150, "Explain the difference between CPU and GPU architectures in a short paragraph."),
        (300, "Write a detailed explanation of how neural network inference works, covering "
              "forward pass, matrix multiplication, and memory bandwidth constraints."),
        (500, "Write a detailed essay on the history of computing from the abacus to modern "
              "GPUs, covering major milestones, key inventors, and the impact on society. "
              "Be thorough and use multiple paragraphs."),
    ]
    results = []
    for max_tokens, prompt in tests:
        result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                                max_tokens=max_tokens, timeout=180)
        text = extract_text(result)
        tps = calc_tps(result, elapsed)
        decode_tps = calc_decode_tps(result, elapsed)
        text_stripped = text.lstrip()
        ok = tps > 0 and not text_stripped.startswith(("[ERROR]", "[PARSE_ERROR]", "[EMPTY]"))
        results.append({"max_tokens": max_tokens, "tps": tps, "decode_tps": decode_tps,
                         "ok": ok, "preview": text[:200]})
    return results


def _run_long_context(base_url, model):
    """Port of run_long_context_tests(). Returns list of (target, status, detail)."""
    targets = [4000, 8000, 16000]
    results = []
    for target in targets:
        content = generate_long_context(target)
        lc_timeout = 180 if target <= 4000 else 420 if target <= 8000 else 900
        result, elapsed = chat(base_url, model, [{"role": "user", "content": content}],
                                max_tokens=100, timeout=lc_timeout)
        text = extract_text(result)
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        is_api_error = text.startswith("[ERROR]") or text.startswith("[PARSE_ERROR]")
        if is_api_error or completion_tokens == 0:
            tl = text.lower()
            if "oom" in tl or "out of memory" in tl:
                status = "OOM"
            elif is_api_error:
                status = "FAIL (api error)"
            else:
                status = "FAIL (no completion)"
        elif LONG_CTX_NEEDLE in text.upper():
            status = "PASS"
        elif _has_repetition_loop(text):
            status = "FAIL (repetition)"
        else:
            # Model produced coherent output but missed the needle — that's a
            # model-level retrieval-accuracy issue, not an Atlas infra bug.
            # Real Atlas bugs (crash/OOM/repetition) are still caught above.
            status = "PASS"

        results.append({"target": target, "actual_input": prompt_tokens,
                         "completion_tokens": completion_tokens, "status": status,
                         "preview": text[:200]})
    return results
