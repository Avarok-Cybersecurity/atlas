"""HTTP-API helpers transplanted from single_gpu_suite.py: chat completion
wrapper, text extraction, tps calculation, repetition-loop detection, and the
needle-in-haystack long-context / tool-call fixtures."""

import json
import re
import time
import urllib.error
import urllib.request


def api_call(base_url, endpoint, payload, timeout=120):
    url = f"{base_url}/{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {body[:500]}"}
    except Exception as e:
        return {"error": str(e)}


def chat(base_url, model, messages, max_tokens=256, tools=None, timeout=120,
         temperature=0.3, repetition_penalty=None, extra_body=None):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if extra_body:
        payload.update(extra_body)
    t0 = time.time()
    result = api_call(base_url, "chat/completions", payload, timeout=timeout)
    elapsed = time.time() - t0
    return result, elapsed


def _has_repetition_loop(text: str, min_repeats: int = 4) -> bool:
    """Detect pathological repetition loops ('a a a a a a…'). Checks n-gram
    sizes 1-12 words, flags if ANY size repeats `min_repeats` times in a row."""
    if not text:
        return False
    toks = text.split()
    for ngram in range(1, 13):
        if len(toks) < ngram * min_repeats:
            continue
        for i in range(len(toks) - ngram * min_repeats + 1):
            window = toks[i: i + ngram]
            ok = True
            for r in range(1, min_repeats):
                start = i + r * ngram
                if toks[start: start + ngram] != window:
                    ok = False
                    break
            if ok:
                return True
    return False


def extract_text(result):
    """Falls back to `reasoning_content` when `content` is empty (reasoning-
    first models that exhaust their thinking budget without a closing
    `</think>`)."""
    if "error" in result:
        return f"[ERROR] {result['error']}"
    try:
        choice = result["choices"][0]
        msg = choice["message"]
        if msg.get("content"):
            return msg["content"]
        if msg.get("tool_calls"):
            return f"[TOOL_CALL] {json.dumps(msg['tool_calls'], indent=2)}"
        if msg.get("reasoning_content"):
            return f"[THINK-ONLY]\n{msg['reasoning_content']}"
        return "[EMPTY]"
    except (KeyError, IndexError) as e:
        return f"[PARSE_ERROR] {e} — {json.dumps(result)[:300]}"


def calc_tps(result, elapsed):
    """Wall-clock tokens/sec (includes prefill/TTFT)."""
    try:
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0 and elapsed > 0:
            return completion_tokens / elapsed
    except Exception:
        pass
    return 0.0


def calc_decode_tps(result, elapsed):
    """Decode-only tokens/sec (subtracts TTFT). Returns 0.0 if the server
    did not report time_to_first_token_ms."""
    try:
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        ttft_ms = usage.get("time_to_first_token_ms") or usage.get("ttft_ms")
        if completion_tokens <= 1 or ttft_ms is None:
            return 0.0
        decode_seconds = (elapsed * 1000.0 - ttft_ms) / 1000.0
        if decode_seconds <= 0:
            return 0.0
        return (completion_tokens - 1) / decode_seconds
    except Exception:
        return 0.0


LONG_CTX_NEEDLE = "PURPLE-DOLPHIN-42"


def generate_long_context(target_tokens):
    """Needle-in-haystack long-context prompt with diverse filler (not a
    repeated sentence, which trains models to emit degenerate continuations)."""
    filler = (
        "Alice walked quietly through the tall forest. The pines above "
        "her swayed with the afternoon wind, and somewhere in the distance "
        "a woodpecker tapped a rhythm against a hollow trunk. A narrow "
        "stream wound between mossy rocks, reflecting broken fragments of "
        "the sky. Squirrels chased one another up the bark while small "
        "blue flowers nodded along the path. She paused to breathe the "
        "cool green air before moving on, her footsteps soft on the needles. "
    )
    repeats = max(2, target_tokens // 100)
    middle = repeats // 2
    paragraphs = []
    for i in range(repeats):
        if i == middle:
            paragraphs.append(
                f"[Important fact — remember this] The secret code is {LONG_CTX_NEEDLE}. " + filler
            )
        else:
            paragraphs.append(filler)
    content = "\n\n".join(paragraphs)
    content += (
        f"\n\nQuestion: What is the secret code mentioned in the text above? "
        f"Answer in one short sentence that contains the exact code."
    )
    return content


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
}


def _strip_thinking(text: str) -> str:
    for pat in (r"<think>.*?</think>", r"\[THINK\].*?\[/THINK\]"):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    for pat in (r"<think>.*$", r"\[THINK\].*$"):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# Models whose Atlas tool-call parser is a known gap (lower-case substring
# match on model ID). Currently empty — nothing is unsupported. Kept as a
# named constant (not inlined) so a future gap can be recorded here without
# touching test bodies, matching single_gpu_suite.py's `_tool_calls_supported`.
TOOL_CALL_UNSUPPORTED_SUBSTRINGS: tuple = ()


def _tool_calls_supported(model: str) -> bool:
    m = model.lower()
    return not any(s in m for s in TOOL_CALL_UNSUPPORTED_SUBSTRINGS)
