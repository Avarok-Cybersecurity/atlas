#!/usr/bin/env python3
"""Coherency gate for one leg of the Hopper A/B.

Four claims, each a PRD gate. A leg that fails any of them has no comparable
numbers, however fast it was -- an engine that is nondeterministic at temp 0,
or that cannot emit a parseable tool call, or that leaks its scratchpad into
the reply, or that cannot answer a question whose answer is known, is not
serving the same workload as the engine it is being compared against.

  determinism   the same prompt twice at temperature 0 must be byte-identical,
                and the reply must not be degenerate or truncated
  toolcall      finish_reason == "tool_calls" and json.loads(arguments) succeeds
  think_leak    no <think>/</think> in content when thinking is off
  known_answer  three questions with checkable answers, answered correctly

Byte-equality alone is not coherence. Measured on a GB10 the determinism check
certified this reply (Nemotron 3 Nano FP8 through the nvfp4 bundle), because
both greedy runs produced it identically:

  101, 103, 107, 107, 109, 109, 113, 109, 107, 109, 109, 109, 109, ... 107, 1

Identical garbage is identical. `fixtures/degenerate_primes.txt` is that reply,
and the selftest now requires the gate to refuse it. The gates are meant to be
independent oracles: the tool-call check did catch that leg (finish_reason was
"length"), but a model that loops on plain text has to fail the check that is
looking at plain text.

The degeneration signals reuse `_has_degeneration` from
`scripts/test_coherence.py` (defined at line 813 there) rather than re-deriving
the signal list. That function is the repo's existing answer to "does this
reply look wrong", it already covers the two think tags plus raw <tool_call>
and script mixing, and two copies of a heuristic drift apart. What it does not
cover is a reply that repeats itself, so `_repetition_loop` below adds that,
here rather than there, because `scripts/test_coherence.py` is out of this
change's scope and the thresholds want the fixture beside them.

The known-answer probes are `CASES` from `bench/agentic/coherence_check.py`,
imported by path and judged the way that script judges them. Both imports are
hard -- a missing source is a broken gate, not a reason to fall back to a
private copy that says something else.

Usage:
  coherency_gate.py --url URL --model MODEL [--out FILE] [--timeout 300]
  coherency_gate.py --selftest

Exits non-zero if any check fails, so a driver can `set -e` around it.
"""

import argparse
import collections
import importlib.util
import http.client
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# ── the leak heuristic, from the suite that already owns it ──


def _load_degeneration_check():
    """`scripts/test_coherence.py::_has_degeneration`, imported by path.

    Safe to import: that module's only top-level statements are definitions and
    constants; everything with an effect is under `if __name__ == "__main__"`.
    """
    src = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "test_coherence.py"
    spec = importlib.util.spec_from_file_location("atlas_test_coherence", src)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the degeneration check from {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._has_degeneration


HAS_DEGENERATION = _load_degeneration_check()


def _load_known_answer_probes():
    """`bench/agentic/coherence_check.py`'s CASES and SYSTEM, imported by path.

    Safe to import for the same reason the degeneration check is: that module's
    top level is imports, three `os.environ.get` reads and the CASES list;
    everything that talks to a server is under `if __name__ == "__main__"`.
    Checked, not assumed -- an import-time request would make this gate open a
    socket at parse time.

    The system prompt travels with the cases on purpose. That module explains
    why it sends one (sending none leaves whatever the chat template bakes in,
    which is a different probe on every model), and a gate that dropped it
    would be asking a different question than the suite it borrowed from.
    """
    src = pathlib.Path(__file__).resolve().parents[1] / "agentic" / "coherence_check.py"
    spec = importlib.util.spec_from_file_location("atlas_coherence_check", src)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the known-answer probes from {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES, module.SYSTEM


KNOWN_ANSWER_CASES, KNOWN_ANSWER_SYSTEM = _load_known_answer_probes()

# ── the repetition-loop detector ──
#
# Thresholds, and the oracle each one is set against. The oracle is not taste:
# `fixtures/degenerate_primes.txt` -- the real GB10 reply -- must trip every
# one of the three, and the good replies in the selftest plus the three
# known-answer replies must trip none. Numbers, on that fixture:
#
#   top token share      0.73  (109, 38 of 52 tokens)   threshold > 0.50
#   distinct 3-grams     0.24  (12 of 50)               threshold < 0.35
#   consecutive repeats  20    ("109" twenty times)     threshold >= 8
#
# The margins are wide in the direction that matters. Prose does not reach a
# 0.50 single-token share (English's most frequent word, "the", sits near 0.07
# in running text), and it does not fall to 0.24 distinct 3-grams.
LOOP_MIN_TOKENS = 40
LOOP_TOP_TOKEN_SHARE = 0.50
LOOP_MIN_TRIGRAM_DIVERSITY = 0.35
LOOP_MAX_CONSECUTIVE_REPEATS = 8

# coherence_check.py's garbling floor, kept at its value: a reply is clean when
# MORE than this fraction of its characters are printable or whitespace.
PRINTABLE_RATIO_FLOOR = 0.98


def _repetition_loop(text):
    """Signals that a reply has fallen into a repetition loop. [] means prose.

    Three signals over one tokenisation -- split on whitespace OR commas,
    because the reply that motivated this check was a comma-separated list and
    a whitespace-only split would have counted "109," and "109" as different
    tokens:

      token_repeat      one token is more than LOOP_TOP_TOKEN_SHARE of the reply
      trigram_collapse  distinct 3-grams are under LOOP_MIN_TRIGRAM_DIVERSITY
                        of all 3-grams -- catches a loop over a repeating
                        PHRASE, which no single-token count can see
      segment_repeat    one line or comma segment repeats consecutively at
                        least LOOP_MAX_CONSECUTIVE_REPEATS times

    The first two are frequency statistics and say nothing about short text --
    "Tokyo" is a 100% single-token share and has no 3-grams at all -- so both
    are gated behind LOOP_MIN_TOKENS. The third needs no such gate: it already
    requires LOOP_MAX_CONSECUTIVE_REPEATS identical segments in a row, which no
    reply reaches by accident.

    TODO: an n-gram loop whose period is longer than the reply is invisible
    here (a 200-token answer that says the same thing twice in different
    words). Semantic repetition needs a model, not a counter.
    """
    tokens = [t for t in re.split(r"[\s,]+", text) if t]
    signals = []
    if len(tokens) >= LOOP_MIN_TOKENS:
        top, count = collections.Counter(tokens).most_common(1)[0]
        share = count / len(tokens)
        if share > LOOP_TOP_TOKEN_SHARE:
            signals.append(f"token_repeat: {top!r} is {share:.0%} of {len(tokens)} tokens")
        grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        if grams:
            diversity = len(set(grams)) / len(grams)
            if diversity < LOOP_MIN_TRIGRAM_DIVERSITY:
                signals.append(
                    f"trigram_collapse: {len(set(grams))} distinct of {len(grams)} 3-grams ({diversity:.2f})")
    segments = [s.strip() for s in re.split(r"[\n,]", text) if s.strip()]
    longest, run = 1, 1
    for previous, current in zip(segments, segments[1:]):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    if longest >= LOOP_MAX_CONSECUTIVE_REPEATS:
        signals.append(f"segment_repeat: a segment repeats {longest} times in a row")
    return signals


def degeneration_signals(text):
    """Everything wrong with one reply's text, as a list of reasons.

    The repo's existing leak/script heuristic plus the loop detector, applied
    to EVERY reply this gate reads. Splitting them by check -- leaks only on
    the think probe, loops only on the determinism probe -- would leave each
    check blind to the failure the other one is named after.
    """
    degenerate, detail = HAS_DEGENERATION(text)
    signals = [detail] if degenerate else []
    return signals + _repetition_loop(text)


# ── the fixed tool schema ──
#
# One tool, required arguments of two different types, because a tool call
# whose arguments are `{}` parses as JSON and proves nothing. Shaped after the
# fixtures in scripts/fixtures/ rather than copied: those are whole agent
# toolsets (~20 tools, thousands of tokens) aimed at a different question --
# whether a real agent's prompt survives -- and a gate that has to run before
# every leg wants the smallest input that can still fail.
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                    "days": {"type": "integer", "description": "Forecast horizon in days."},
                },
                "required": ["city", "days"],
            },
        },
    }
]

DETERMINISM_PROMPT = (
    "List exactly five prime numbers greater than one hundred, comma separated, "
    "with no other words."
)
TOOLCALL_PROMPT = "What is the weather in Reykjavik over the next three days? Use the tool."
THINK_PROMPT = "In one short sentence, say what a benchmark harness is for."


def post(url, payload, timeout, exchanges=None):
    request_json = json.dumps(payload)
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=request_json.encode(),
        headers={"Content-Type": "application/json"},
    )
    exchange = {"request_json": request_json, "response_status": None,
                "response_body": "", "response_complete": False}
    raw = b""
    try:
        error = None
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # HTTPError also owns a readable body; retain it before re-raising
            # so the existing gate failure remains inspectable.
            resp, error = exc, exc
        with resp:
            exchange["response_status"] = resp.status
            try:
                raw = resp.read()
            except http.client.IncompleteRead as exc:
                raw = exc.partial
                raise
            exchange["response_complete"] = True
        if error is not None:
            raise error
        return json.loads(raw)
    finally:
        exchange["response_body"] = raw.decode("utf-8", errors="replace")
        if exchanges is not None:
            exchanges.append(exchange)


def body(model, prompt, **extra):
    """The campaign's pinned sampling, on every request this gate makes.

    Identical to `bench/ladder38/harness_w55_conc_ladder.py` and
    `workloads.json`: pinning penalties explicitly stops Atlas's non_thinking
    preset injecting presence_penalty=1.5 where vLLM defaults to 0, and
    `chat_template_kwargs.enable_thinking=false` is the only key that disables
    thinking on vLLM ({"thinking": false} is silently ignored). A gate that
    checked a different configuration than the ladder measures would be
    certifying a server nobody benchmarked.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 42,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    return payload


def choice_of(response):
    """Validate the HTTP/JSON boundary before inspecting a completion."""
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("response must be a completion object without an error")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response must contain a nonempty choices array of objects")
    choice = choices[0]
    if not isinstance(choice.get("message"), dict):
        raise ValueError("completion message must be an object")
    return choice


def completion_of(response):
    """The reply text and the finish_reason that ended it."""
    choice = choice_of(response)
    content = choice["message"].get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("completion content must be a string or null")
    return content or "", choice.get("finish_reason")


def content_of(response):
    return completion_of(response)[0]


def check_determinism(url, model, timeout, exchanges=None):
    """Two identical requests at temp 0 must return identical, coherent bytes.

    Not "similar": at temperature 0 the sampler is argmax, so any difference is
    a difference in the compute -- a batching-dependent reduction order, a
    leaked cache entry, an uninitialised buffer. A/B numbers measured on a
    server that cannot reproduce itself describe nothing repeatable.

    Reproducing itself is necessary and not sufficient, which is the whole
    reason this check grew two more conditions:

      * the reply must not be degenerate. A decode loop is perfectly
        reproducible -- see the module docstring and
        `fixtures/degenerate_primes.txt` -- so byte-equality certifies it.
      * the reply must not have hit the token cap. DETERMINISM_PROMPT asks for
        exactly five numbers; five numbers cannot need 256 tokens, so
        finish_reason "length" here means the model never stopped, and a reply
        the sampler cut off mid-loop is not an answer. That inference is about
        THIS prompt being bounded, not a general rule about "length".
    """
    first, finish = completion_of(post(url, body(model, DETERMINISM_PROMPT), timeout, exchanges))
    second, _ = completion_of(post(url, body(model, DETERMINISM_PROMPT), timeout, exchanges))
    if not first.strip():
        return False, "empty reply -- nothing to compare"
    if first != second:
        # Report WHERE they diverged; "not identical" sends the reader to diff
        # two blobs by eye.
        at = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))
        return False, f"diverged at char {at}: {first[at:at + 40]!r} vs {second[at:at + 40]!r}"
    problems = degeneration_signals(first)
    if finish == "length":
        problems.append(
            "truncated_bounded_answer: finish_reason 'length' on a prompt that asked for five numbers")
    if problems:
        return False, f"reproduced exactly but incoherent -- {'; '.join(problems)}"
    return True, f"{len(first)} chars reproduced exactly, no degeneration signals"


def check_toolcall(url, model, timeout, exchanges=None):
    """A tool call must arrive as a tool call, with arguments that parse."""
    r = post(url, body(model, TOOLCALL_PROMPT, tools=TOOL_SCHEMA, tool_choice="auto"), timeout, exchanges)
    choice = choice_of(r)
    finish = choice.get("finish_reason")
    calls = (choice.get("message") or {}).get("tool_calls") or choice.get("tool_calls") or []
    if finish != "tool_calls":
        return False, f"finish_reason was {finish!r}, not 'tool_calls'"
    if not calls:
        return False, "finish_reason said tool_calls but none were returned"
    if not isinstance(calls, list):
        return False, "tool_calls must be an array"
    schema = TOOL_SCHEMA[0]["function"]
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            return False, "each tool call must be a function object"
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != schema["name"]:
            return False, f"tool name must be {schema['name']}"
        raw = function.get("arguments")
        if not isinstance(raw, str):
            return False, f"arguments were {type(raw).__name__}, not a JSON string"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            return False, f"arguments are not JSON ({e}): {raw[:200]!r}"
        if not isinstance(args, dict):
            return False, f"arguments parsed to {type(args).__name__}, not an object"
        for key in schema["parameters"]["required"]:
            if key not in args:
                return False, f"missing required argument: {key}"
        for key, definition in schema["parameters"]["properties"].items():
            if key not in args:
                continue
            expected = {"string": str, "integer": int}[definition["type"]]
            # bool is an int subclass in Python, but not a JSON integer.
            if type(args[key]) is not expected:
                return False, f"{key} must be {definition['type']}"
    return True, f"{len(calls)} {schema['name']} call(s), required argument types valid"


def check_think_leak(url, model, timeout, exchanges=None):
    """Thinking is off; the scratchpad must not be in the reply."""
    text = content_of(post(url, body(model, THINK_PROMPT), timeout, exchanges))
    if not text.strip():
        return False, "empty reply -- a leak check over no text proves nothing"
    signals = degeneration_signals(text)
    if signals:
        return False, "; ".join(signals)
    return True, f"{len(text)} chars, no leak signals"


# ── the complete-answer matcher ──
#
# A substring test cannot tell a number from the digits inside a larger one:
# `"391" in "1391"` and `"391" in "3910"` are both true, so a reply stating
# either was certified as 17*23. The same hole passed `Tokyoto` for `Tokyo`.
# So the line is TOKENISED and the expected answer has to be a whole token.
#
# Numbers and words need different tokenisers, and the expected answer says
# which: `1,391` is ONE number (a word tokeniser would split it into `1` and
# `391` and let the near miss back in), while `rotaregirfer.` is one word with
# a full stop attached. NUMBER accepts a digit-and-comma run with an optional
# decimal tail; WORD accepts a run of alphanumerics, which drops the `**`,
# `=`, commas and full stops a model wraps a stated answer in.
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[^\W_]+")


def _answer_tokens(text, numeric):
    if numeric:
        return {m.group(0).replace(",", "") for m in NUMBER.finditer(text)}
    return {m.group(0).lower() for m in WORD.finditer(text)}


def states_answer(text, expect):
    """Is `expect` a COMPLETE token of `text`, rather than a substring of one?"""
    numeric = bool(NUMBER.fullmatch(expect.strip()))
    want = expect.strip().replace(",", "") if numeric else expect.strip().lower()
    return want in _answer_tokens(text, numeric)


def judge_known_answer(text, expect):
    """`bench/agentic/coherence_check.py`'s judgement, plus the loop detector.

    That script's rule, kept because changing it would make the two probes
    disagree about the same reply: the expected value must appear in the FIRST
    or LAST non-empty line -- a direct answer, or an explicit final answer.
    `expect in out` once passed a reply that opened with "271" for 17*23 and
    only reached 391 inside its working; that is reported here as WORKING-ONLY,
    separately, rather than silently passed or silently failed.

    What IS changed is "appear": `states_answer` requires a complete token, so
    `1391`, `3910` and `Tokyoto` no longer answer a probe whose answers are
    `391` and `Tokyo`, while `391.`, `**391**`, `= 391` and `rotaregirfer.`
    still do.

    Returns (status, detail) where status is "OK", "WORKING-ONLY" or "FAIL".
    Only "OK" passes the gate; WORKING-ONLY is a distinct diagnosis, not a
    softer pass.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "FAIL", "empty reply"
    stated = states_answer(lines[0], expect) or states_answer(lines[-1], expect)
    buried = (not stated) and states_answer(text, expect)
    # Garbled output is the aliasing signature even when the expected token
    # happens to appear, so the text checks come before the answer verdict.
    printable = sum(c.isprintable() or c.isspace() for c in text)
    ratio = printable / len(text)
    if ratio <= PRINTABLE_RATIO_FLOOR:
        return "FAIL", f"printable ratio {ratio:.3f} is not above {PRINTABLE_RATIO_FLOOR}"
    signals = degeneration_signals(text)
    if signals:
        return "FAIL", "; ".join(signals)
    if stated:
        return "OK", ""
    if buried:
        return "WORKING-ONLY", "the answer appears only inside the working"
    return "FAIL", f"answer not stated: {text[:80]!r}"


def check_known_answer(url, model, timeout, exchanges=None):
    """Three questions whose answers are known must come back answered.

    Determinism, a parseable tool call and a clean scratchpad are all
    properties of the SHAPE of a reply; none of them looks at whether the reply
    is right. A quantisation bug that turns 17*23 into 271 produces a
    reproducible, well-formed, leak-free wrong answer, and the other three
    gates certify it.

    Same pinned sampling as every other request this gate makes, which is NOT
    what coherence_check.py uses (temp 0.6, 300 tokens): a gate that sampled
    differently than the ladder would be certifying a server nobody
    benchmarked. Only the prompts and the judgement are borrowed.
    """
    verdicts, ok = [], True
    for prompt, expect in KNOWN_ANSWER_CASES:
        payload = body(model, prompt, messages=[
            {"role": "system", "content": KNOWN_ANSWER_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        text, _ = completion_of(post(url, payload, timeout, exchanges))
        status, detail = judge_known_answer(text, expect)
        if status != "OK":
            ok = False
        verdicts.append(f"{expect!r} {status}" + (f" ({detail})" if detail else ""))
    return ok, "; ".join(verdicts)


def run(url, model, timeout):
    checks = (
        ("determinism_ok", check_determinism),
        ("toolcall_ok", check_toolcall),
        ("think_leak_ok", check_think_leak),
        ("known_answer_ok", check_known_answer),
    )
    out = {"schema": 1, "url": url, "model": model,
           "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "details": {}, "http_exchanges": []}
    for key, fn in checks:
        exchanges = []
        try:
            ok, detail = fn(url, model, timeout, exchanges)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError, ValueError, KeyError) as e:
            # A transport failure is a FAILED check, never a skipped one: the
            # gate's whole job is to refuse to certify what it could not see.
            ok, detail = False, f"{type(e).__name__}: {e}"
        out[key] = ok
        out["details"][key] = detail
        out["http_exchanges"].extend({"check": key, **e} for e in exchanges)
    out["passed"] = all(out[k] for k, _ in checks)
    return out


# ── selftest ──

STUB = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = sys.argv[2]
FIXTURES = sys.argv[3]
PRIME_HITS = 0
DEGENERATE = open(FIXTURES + "/degenerate_primes.txt").read().rstrip("\n")

# One entry per case in bench/agentic/coherence_check.py::CASES, keyed by a
# substring of that case's prompt. GOOD states the answer on the first line;
# WRONG states a different one and never contains the expected string anywhere;
# BURIED reaches the answer only in the middle, which is the WORKING-ONLY
# verdict -- the case that motivated that script's first/last-line judgement.
GOOD = {
    "17 * 23": "391\n\nWorking: 17 * 20 = 340, and 17 * 3 = 51. Adding the two partial products gives 391.",
    "Japan": ("Tokyo\n\nTokyo is the capital of Japan and the seat of its national government. "
              "It grew out of the castle town of Edo and now spreads around the head of its bay. "
              "The metropolitan area is the largest in the world by population."),
    "refrigerator": ("rotaregirfer\n\nI read the letters of the word from the last one back to the "
                     "first and wrote them down in that order."),
}
WRONG = {
    "17 * 23": "271\n\nMultiplying 17 by 23 gives 271 in my working.",
    "Japan": "Kyoto\n\nKyoto was the imperial seat for centuries and remains the capital.",
    "refrigerator": "rotaregirf\n\nI dropped a letter on the way back through the word.",
}
# One near miss per red case in the review of this gate: a substring test
# accepted 1391 and 3910 for a probe whose answer is 391, and Tokyoto for one
# whose answer is Tokyo. Each entry replaces exactly ONE probe's reply; the
# other two stay GOOD, so the gate has to fail on the answer rather than on
# anything else about the exchange.
NEAR_MISS = {
    "answer-1391": ("17 * 23", "1391\n\nMultiplying 17 by 23 gives 1391 in my working."),
    "answer-3910": ("17 * 23", "3910\n\nMultiplying 17 by 23 gives 3910 in my working."),
    "answer-tokyoto": ("Japan", "Tokyoto\n\nTokyoto is the capital of Japan and the seat "
                                "of its national government. It grew out of the castle town "
                                "of Edo and now spreads around the head of its bay."),
}
BURIED = {
    "17 * 23": "Let me work through it.\n17 * 20 = 340 and 17 * 3 = 51, so the product is 391.\nThat is how the multiplication goes.",
    "Japan": "Let me think about the geography.\nThe seat of government moved to Tokyo in 1868.\nThat is the answer to the question.",
    "refrigerator": "Let me reverse the letters one at a time.\nReading backwards the word becomes rotaregirfer in full.\nThat completes the reversal.",
}

def reply(text, finish="stop", calls=None):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}]}

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        global PRIME_HITS
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        # The LAST message: the known-answer probes send a system message first.
        prompt = req["messages"][-1]["content"]
        if req.get("tools"):
            body = reply("", "tool_calls", [{"id": "call_0", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": json.dumps({"city": "Reykjavik", "days": 3})}}])
            call = body["choices"][0]["message"]["tool_calls"][0]
            if MODE == "missing-args":
                call["function"]["arguments"] = "{}"
            elif MODE == "wrong-types":
                call["function"]["arguments"] = json.dumps({"city": 4, "days": True})
            elif MODE == "wrong-name":
                call["function"]["name"] = "delete_files"
            elif MODE == "wrong-call-type":
                call["type"] = "custom"
            elif MODE == "extra-bad-call":
                body["choices"][0]["message"]["tool_calls"].append({"type": "function", "function": {"name": "get_weather", "arguments": "{}"}})
        elif "prime" in prompt:
            PRIME_HITS += 1
            if MODE == "degenerate-primes":
                body = reply(DEGENERATE, "length")
            elif MODE == "length-capped":
                body = reply("101, 103, 107, 109, 113", "length")
            else:
                body = reply("101, 103, 107, 109, 113" if MODE != "nondeterministic" or PRIME_HITS == 1 else "127")
        elif any(key in prompt for key in GOOD):
            key = next(k for k in GOOD if k in prompt)
            table = {"wrong-answer": WRONG, "working-only": BURIED}.get(MODE, GOOD)
            text = table[key]
            near = NEAR_MISS.get(MODE)
            if near and near[0] == key:
                text = near[1]
            body = reply(text)
        elif MODE == "leak":
            body = reply("<think>the user wants a definition</think> To measure a system.")
        else:
            body = reply("To measure a system under a fixed workload.")
        if MODE == "empty":
            body = reply("")
        elif MODE == "malformed":
            body = {"choices": [7]}
        if MODE in ("http500", "error200"):
            body = {"error": "known failure"}
        raw = b"not json" if MODE == "invalid" else json.dumps(body).encode()
        self.send_response(500 if MODE == "http500" else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw) + (100 if MODE == "truncated" else 0)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _await_bind(port):
    import socket
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise SystemExit("stub never bound")


def selftest():
    """Exercise each gate against clean and known-bad HTTP responses.

    Tool-name/type/schema failures, nondeterminism, empty replies, malformed
    envelopes, leaked thinking, a reproducible decode loop, a bounded answer
    cut off at the token cap, and wrong or buried known answers must all fail
    rather than certify or crash.
    """
    with tempfile.TemporaryDirectory() as d:
        stub = pathlib.Path(d) / "stub.py"
        stub.write_text(STUB)
        results = {}
        for mode in ("clean", "leak", "missing-args", "wrong-types", "wrong-name", "wrong-call-type", "extra-bad-call", "nondeterministic", "empty", "malformed", "truncated", "invalid", "http500", "error200", "degenerate-primes", "length-capped", "wrong-answer", "working-only", "answer-1391", "answer-3910", "answer-tokyoto"):
            port = _free_port()
            proc = subprocess.Popen([sys.executable, str(stub), str(port), mode, str(FIXTURES)])
            try:
                _await_bind(port)
                try:
                    results[mode] = run(f"http://127.0.0.1:{port}", "stub-model", 30)
                except Exception as exc:
                    results[mode] = {"crashed": f"{type(exc).__name__}: {exc}"}
            finally:
                proc.terminate()
                proc.wait(timeout=10)

    print(json.dumps(results, indent=2))
    _assert_judgement()
    _assert_stub_results(results)
    # Validate the selftest's own exception sentinel against a known bad case.
    crashed = {**results, "clean": {"passed": True, "crashed": "known clean-path failure"}}
    try:
        _assert_stub_results(crashed)
    except AssertionError:
        pass
    else:
        raise AssertionError("a clean-stub crash must fail the selftest")
    print("SELFTEST OK: clean passes; every known-bad response -- including the recorded GB10 "
          "repetition loop -- and a clean-stub crash fail")


# (reply, expected answer, verdict). The near misses are the review's red
# cases; the OK rows are the formats a model actually uses to state a final
# answer, which the completeness rule must not start rejecting.
JUDGEMENT_CASES = (
    ("1391\n\nMultiplying 17 by 23 gives 1391.", "391", "FAIL"),
    ("3910\n\nMultiplying 17 by 23 gives 3910.", "391", "FAIL"),
    ("1,391\n\nMultiplying 17 by 23 gives 1,391.", "391", "FAIL"),
    ("3.91\n\nMultiplying 17 by 23 gives 3.91.", "391", "FAIL"),
    ("Tokyoto\n\nTokyoto is the capital.", "Tokyo", "FAIL"),
    ("rotaregirferx\n\nThat is the word backwards.", "rotaregirfer", "FAIL"),
    ("391.\n\nThat is 17 * 23.", "391", "OK"),
    ("**391**\n\nThat is 17 * 23.", "391", "OK"),
    ("The product is\n= 391", "391", "OK"),
    ("391, as it happens.\n\nThat is 17 * 23.", "391", "OK"),
    ("The word backwards is\nrotaregirfer.", "rotaregirfer", "OK"),
    ("Tokyo, on Honshu.\n\nIt is the seat of government.", "Tokyo", "OK"),
    ("tokyo\n\nIt is the seat of government.", "Tokyo", "OK"),
)


def _assert_judgement():
    """`judge_known_answer` on the formats, without a server in the way."""
    wrong = [f"{text!r} for {expect!r}: expected {want}, got {judge_known_answer(text, expect)[0]}"
             for text, expect, want in JUDGEMENT_CASES
             if judge_known_answer(text, expect)[0] != want]
    assert not wrong, "\n".join(wrong)


def _assert_stub_results(results):
    for mode, result in results.items():
        assert "crashed" not in result, f"{mode} stub crashed: {result}"
    clean, leak = results["clean"], results["leak"]
    failures = []
    for mode in ("missing-args", "wrong-types", "wrong-name", "wrong-call-type", "extra-bad-call"):
        if results[mode]["toolcall_ok"] or results[mode]["passed"]:
            failures.append(f"{mode} must fail the tool-call gate")
    for mode, key in (("nondeterministic", "determinism_ok"), ("empty", "determinism_ok"), ("malformed", "determinism_ok"), ("truncated", "determinism_ok"), ("invalid", "determinism_ok"), ("http500", "determinism_ok"), ("error200", "determinism_ok")):
        if results[mode][key] or results[mode]["passed"]:
            failures.append(f"{mode} must fail {key}")
    prime = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "101, 103, 107, 109, 113"}, "finish_reason": "stop"}]}
    # Seven requests when the transport works: two determinism, one tool call,
    # one think probe, three known-answer probes. A mode that breaks the
    # transport fails each check on its FIRST request, so four.
    broken_transport = ("malformed", "truncated", "invalid", "http500", "error200")
    for mode in ("clean", "nondeterministic", "empty", "malformed", "truncated", "invalid", "http500", "error200"):
        exchanges = results[mode].get("http_exchanges", [])
        count = 4 if mode in broken_transport else 7
        if len(exchanges) != count:
            failures.append(f"{mode}: expected {count} retained HTTP exchanges, got {len(exchanges)}")
            continue
        expected = {"malformed": '{"choices": [7]}', "invalid": "not json",
                    "http500": '{"error": "known failure"}', "error200": '{"error": "known failure"}'}
        if mode == "empty":
            empty = {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}
            expected_body = json.dumps(empty)
        else:
            expected_body = expected.get(mode, json.dumps(prime))
        first = exchanges[0]
        if (first.get("check") != "determinism_ok" or first.get("response_body") != expected_body
                or first.get("response_status") != (500 if mode == "http500" else 200)
                or first.get("response_complete") != (mode != "truncated")
                or first.get("request_json") != json.dumps(body("stub-model", DETERMINISM_PROMPT))):
            failures.append(f"{mode}: exact request/response JSON, HTTP status and completeness must be retained")
        if mode in ("clean", "nondeterministic"):
            second_body = expected_body if mode == "clean" else expected_body.replace("101, 103, 107, 109, 113", "127")
            if exchanges[1].get("response_body") != second_body or exchanges[1].get("check") != "determinism_ok":
                failures.append(f"{mode}: the second determinism body must remain separately inspectable")
    # ── the GB10 defect, and the three checks added because of it ──
    #
    # Each of these four modes must fail EXACTLY ONE gate. The gates are
    # independent oracles or they are one gate wearing four names, and the
    # reply in fixtures/degenerate_primes.txt is the proof that mattered: it
    # was caught by the tool-call check (finish_reason "length") while the
    # determinism check, looking straight at it, certified it.
    degenerate_text = (FIXTURES / "degenerate_primes.txt").read_text().rstrip("\n")
    isolated = {"degenerate-primes": "determinism_ok", "length-capped": "determinism_ok",
                "wrong-answer": "known_answer_ok", "working-only": "known_answer_ok",
                "answer-1391": "known_answer_ok", "answer-3910": "known_answer_ok",
                "answer-tokyoto": "known_answer_ok"}
    for mode, key in isolated.items():
        result = results[mode]
        if result[key] or result["passed"]:
            failures.append(f"{mode} must fail {key}")
        for other in ("determinism_ok", "toolcall_ok", "think_leak_ok", "known_answer_ok"):
            if other != key and not result[other]:
                failures.append(f"{mode} must fail {key} ALONE, but {other} also failed: {result['details'][other]}")
        if len(result.get("http_exchanges", [])) != 7:
            failures.append(f"{mode}: expected 7 retained HTTP exchanges, got {len(result.get('http_exchanges', []))}")

    detail = results["degenerate-primes"]["details"]["determinism_ok"]
    for signal in ("token_repeat", "trigram_collapse", "segment_repeat", "truncated_bounded_answer"):
        if signal not in detail:
            failures.append(f"the loop detail must name {signal}: {detail}")
    if "reproduced exactly" not in detail:
        failures.append(f"the loop detail must say the two runs still matched: {detail}")
    body_seen = results["degenerate-primes"]["http_exchanges"][0].get("response_body", "")
    if degenerate_text not in body_seen:
        failures.append("the degenerate reply must remain inspectable in the retained exchange")

    capped = results["length-capped"]["details"]["determinism_ok"]
    if "truncated_bounded_answer" not in capped:
        failures.append(f"a bounded answer cut off at the token cap must say so: {capped}")
    if any(s in capped for s in ("token_repeat", "trigram_collapse", "segment_repeat")):
        failures.append(f"a short, clean, truncated reply must not be reported as a loop: {capped}")

    if not clean["known_answer_ok"]:
        failures.append(f"the clean stub must answer all three probes: {clean['details']['known_answer_ok']}")
    for expect in ("391", "Tokyo", "rotaregirfer"):
        if f"{expect!r} OK" not in clean["details"]["known_answer_ok"]:
            failures.append(f"the clean stub must report {expect} OK: {clean['details']['known_answer_ok']}")
    if "FAIL" not in results["wrong-answer"]["details"]["known_answer_ok"]:
        failures.append(f"a wrong known answer must be FAIL: {results['wrong-answer']['details']['known_answer_ok']}")
    # The three near misses. Each must be FAIL, not WORKING-ONLY: 1391 does not
    # contain the answer anywhere, it merely contains its digits.
    for mode, expect in (("answer-1391", "391"), ("answer-3910", "391"),
                         ("answer-tokyoto", "Tokyo")):
        detail = results[mode]["details"]["known_answer_ok"]
        if f"{expect!r} FAIL" not in detail:
            failures.append(f"{mode}: {expect} must be FAIL, got: {detail}")

    working_only = results["working-only"]["details"]["known_answer_ok"]
    if "WORKING-ONLY" not in working_only:
        failures.append(f"an answer reached only in the working must be WORKING-ONLY: {working_only}")
    if "FAIL" in working_only:
        failures.append(f"WORKING-ONLY is its own verdict, not FAIL: {working_only}")

    assert not failures, "\n".join(failures)
    assert clean["passed"], f"the clean stub must pass: {clean}"
    assert not leak["passed"], "a <think> leak must FAIL the gate"
    assert leak["determinism_ok"], f"the leak must not disturb determinism: {leak}"
    assert leak["toolcall_ok"], f"the leak must not disturb tool calls: {leak}"
    assert leak["known_answer_ok"], f"the leak must not disturb the known answers: {leak}"
    assert not leak["think_leak_ok"], leak
    assert "think" in leak["details"]["think_leak_ok"], leak["details"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return 0
    if not a.url or not a.model:
        ap.error("--url and --model are required (or pass --selftest)")

    out = run(a.url, a.model, a.timeout)
    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        pathlib.Path(a.out).write_text(text + "\n")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
