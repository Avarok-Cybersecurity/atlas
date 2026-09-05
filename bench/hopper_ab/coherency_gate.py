#!/usr/bin/env python3
"""Coherency gate for one leg of the Hopper A/B.

Three claims, each a PRD gate. A leg that fails any of them has no comparable
numbers, however fast it was -- an engine that is nondeterministic at temp 0,
or that cannot emit a parseable tool call, or that leaks its scratchpad into
the reply, is not serving the same workload as the engine it is being compared
against.

  determinism  the same prompt twice at temperature 0 must be byte-identical
  toolcall     finish_reason == "tool_calls" and json.loads(arguments) succeeds
  think_leak   no <think>/</think> in content when thinking is off

The leak check reuses `_has_degeneration` from `scripts/test_coherence.py`
(defined at line 813 there) rather than re-deriving the signal list. That
function is the repo's existing answer to "does this reply look wrong", it
already covers the two think tags plus raw <tool_call> and script mixing, and
two copies of a heuristic drift apart. It is imported by path because
`scripts/` is not a package; the import is hard -- a missing source is a broken
gate, not a reason to fall back to a private copy that says something else.

Usage:
  coherency_gate.py --url URL --model MODEL [--out FILE] [--timeout 300]
  coherency_gate.py --selftest

Exits non-zero if any check fails, so a driver can `set -e` around it.
"""

import argparse
import importlib.util
import http.client
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

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


def content_of(response):
    content = choice_of(response)["message"].get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("completion content must be a string or null")
    return content or ""


def check_determinism(url, model, timeout, exchanges=None):
    """Two identical requests at temp 0 must return identical bytes.

    Not "similar": at temperature 0 the sampler is argmax, so any difference is
    a difference in the compute -- a batching-dependent reduction order, a
    leaked cache entry, an uninitialised buffer. A/B numbers measured on a
    server that cannot reproduce itself describe nothing repeatable.
    """
    first = content_of(post(url, body(model, DETERMINISM_PROMPT), timeout, exchanges))
    second = content_of(post(url, body(model, DETERMINISM_PROMPT), timeout, exchanges))
    if first == second and first.strip():
        return True, f"{len(first)} chars reproduced exactly"
    if not first.strip():
        return False, "empty reply -- nothing to compare"
    # Report WHERE they diverged; "not identical" sends the reader to diff two
    # blobs by eye.
    at = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))
    return False, f"diverged at char {at}: {first[at:at + 40]!r} vs {second[at:at + 40]!r}"


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
    degenerate, detail = HAS_DEGENERATION(text)
    if degenerate:
        return False, detail
    return True, f"{len(text)} chars, no leak signals"


def run(url, model, timeout):
    checks = (
        ("determinism_ok", check_determinism),
        ("toolcall_ok", check_toolcall),
        ("think_leak_ok", check_think_leak),
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
PRIME_HITS = 0

def reply(text, finish="stop", calls=None):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}]}

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        global PRIME_HITS
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        prompt = req["messages"][0]["content"]
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
            body = reply("101, 103, 107, 109, 113" if MODE != "nondeterministic" or PRIME_HITS == 1 else "127")
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
    envelopes and leaked thinking must fail rather than certify or crash.
    """
    with tempfile.TemporaryDirectory() as d:
        stub = pathlib.Path(d) / "stub.py"
        stub.write_text(STUB)
        results = {}
        for mode in ("clean", "leak", "missing-args", "wrong-types", "wrong-name", "wrong-call-type", "extra-bad-call", "nondeterministic", "empty", "malformed", "truncated", "invalid", "http500", "error200"):
            port = _free_port()
            proc = subprocess.Popen([sys.executable, str(stub), str(port), mode])
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
    _assert_stub_results(results)
    # Validate the selftest's own exception sentinel against a known bad case.
    crashed = {**results, "clean": {"passed": True, "crashed": "known clean-path failure"}}
    try:
        _assert_stub_results(crashed)
    except AssertionError:
        pass
    else:
        raise AssertionError("a clean-stub crash must fail the selftest")
    print("SELFTEST OK: clean passes; every known-bad response and a clean-stub crash fail")


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
    for mode in ("clean", "nondeterministic", "empty", "malformed", "truncated", "invalid", "http500", "error200"):
        exchanges = results[mode].get("http_exchanges", [])
        count = 4 if mode in ("clean", "nondeterministic", "empty") else 3
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
    assert not failures, "\n".join(failures)
    assert clean["passed"], f"the clean stub must pass: {clean}"
    assert not leak["passed"], "a <think> leak must FAIL the gate"
    assert leak["determinism_ok"], f"the leak must not disturb determinism: {leak}"
    assert leak["toolcall_ok"], f"the leak must not disturb tool calls: {leak}"
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
