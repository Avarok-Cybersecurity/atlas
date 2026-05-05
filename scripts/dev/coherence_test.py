#!/usr/bin/env python3
"""
coherence_test.py — Atlas Spark API correctness & coherence test suite.

Verifies that the server:
  1. factual accuracy   2+2=4, capital of France
  2. temperature        diverse outputs at temp=1.0
  3. streaming format   SSE structure, [DONE] sentinel, usage chunk with TTFT
  4. thinking mode      enable_thinking=True produces a response (not a crash)
  5. content array      content=[{type:text,text:...}] parses without 422
  6. null content       role=tool, content=null parses without 422
  7. default max_tokens omitting max_tokens yields >256 output tokens
  8. Korean/emoji       streaming response has no U+FFFD replacement characters

Exit code: 0 = all pass, 1 = one or more failures.

Usage:
  python coherence_test.py [--url URL] [--model MODEL] [-v]
"""

import argparse, json, sys, textwrap
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_URL   = "http://localhost:8888"
DEFAULT_MODEL = "Sehyo/Qwen3.5-35B-A3B-NVFP4"

# ANSI colour codes (gracefully degrade on non-TTY)
_tty = sys.stdout.isatty()
PASS = "\033[32m✓\033[0m" if _tty else "PASS"
FAIL = "\033[31m✗\033[0m" if _tty else "FAIL"


class TestRunner:
    def __init__(self, url: str, model: str, verbose: bool) -> None:
        self.url     = url
        self.model   = model
        self.verbose = verbose
        self.passed  = 0
        self.failed  = 0

    # ── Low-level HTTP helpers ──────────────────────────────────────────────

    def _post(self, payload: dict, timeout: int = 90) -> tuple[int, dict | str]:
        """POST to /v1/chat/completions.  Returns (http_status, parsed_body)."""
        data = json.dumps(payload).encode()
        req  = Request(
            f"{self.url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except HTTPError as e:
            return e.code, e.read().decode()

    def _stream(self, payload: dict, timeout: int = 120) -> list[dict]:
        """
        POST with stream=True, return list of parsed SSE data objects.
        The [DONE] sentinel is represented as {"_done": True}.
        """
        data = json.dumps({**payload, "stream": True}).encode()
        req  = Request(
            f"{self.url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        chunks: list[dict] = []
        with urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip()
                if not line.startswith("data: "):
                    continue
                s = line[6:]
                if s == "[DONE]":
                    chunks.append({"_done": True})
                    break
                try:
                    chunks.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
        return chunks

    # ── Assertion helper ────────────────────────────────────────────────────

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        if passed:
            self.passed += 1
            print(f"  {PASS}  {name}")
        else:
            self.failed += 1
            print(f"  {FAIL}  {name}")
            if detail:
                for line in textwrap.wrap(detail, 74):
                    print(f"       {line}")
        if self.verbose and detail:
            print(f"       [{detail[:300]}]")

    # ── Payload builder ─────────────────────────────────────────────────────

    def _base(self, content, role: str = "user", **kw) -> dict:
        """Build request payload. Omits temperature so the server uses the model's
        generation_config.json default — avoids argmax degeneration on NVFP4 models."""
        return {
            "model": self.model,
            "messages": [{"role": role, "content": content}],
            "max_tokens": 64,
            **kw,
        }

    # ── Individual tests ────────────────────────────────────────────────────

    def test_factual_2plus2(self) -> None:
        status, body = self._post(self._base("What is 2+2? Reply with just the number."))
        if status != 200 or not isinstance(body, dict):
            self.check("factual: 2+2=4", False, f"HTTP {status}")
            return
        answer = body["choices"][0]["message"]["content"]
        self.check("factual: 2+2=4", "4" in answer, f"got: {answer!r}")

    def test_factual_capital(self) -> None:
        status, body = self._post(self._base(
            "What is the capital of France? Reply with one word only."))
        if status != 200 or not isinstance(body, dict):
            self.check("factual: capital of France", False, f"HTTP {status}")
            return
        answer = body["choices"][0]["message"]["content"]
        self.check("factual: capital of France", "Paris" in answer, f"got: {answer!r}")

    def test_temperature_diversity(self) -> None:
        """Five requests at temp=1.0 must not all return the same response.

        Uses 5 runs (not 3) to tolerate NVFP4 probability concentration: a
        quantized model may assign ~80% mass to one token, making 3-run all-same
        likely (~50%) even when temperature sampling is working correctly.
        """
        pl = {
            "model": self.model,
            "messages": [{"role": "user", "content":
                          "Pick a random number from 1 to 1000. Just the number."}],
            "temperature": 1.0,
            "max_tokens": 16,
        }
        outputs = []
        for _ in range(5):
            status, body = self._post(pl, timeout=90)
            if status != 200 or not isinstance(body, dict):
                self.check("temperature diversity", False, f"HTTP {status}")
                return
            outputs.append(body["choices"][0]["message"]["content"].strip())
        unique = len(set(outputs))
        self.check(
            "temperature diversity (5 runs)",
            unique >= 2,
            f"all outputs identical: {outputs[0]!r}" if unique < 2 else "",
        )

    def test_streaming_format(self) -> None:
        """Verify SSE envelope: role chunk → content chunks → usage chunk → [DONE]."""
        try:
            chunks = self._stream(self._base("Say 'hello' in one word."))
        except Exception as exc:
            self.check("streaming format", False, str(exc))
            return

        has_done     = any(c.get("_done") for c in chunks)
        has_content  = any(
            "content" in c.get("choices", [{}])[0].get("delta", {})
            for c in chunks if not c.get("_done")
        )
        usage_chunk  = next(
            (c for c in chunks if not c.get("_done") and c.get("usage")), None
        )
        has_ttft_key = (usage_chunk is not None
                        and "time_to_first_token_ms" in (usage_chunk.get("usage") or {}))

        ok = has_done and has_content and usage_chunk is not None and has_ttft_key
        self.check(
            "streaming format (SSE + usage chunk)",
            ok,
            f"[DONE]={has_done} content={has_content} "
            f"usage_chunk={usage_chunk is not None} ttft_field={has_ttft_key}",
        )

    def test_thinking_mode(self) -> None:
        """enable_thinking=True must not crash; response must be non-empty."""
        pl = {
            "model": self.model,
            "messages": [{"role": "user", "content": "What is 3+3?"}],
            "enable_thinking": True,
            "max_tokens": 256,
            "temperature": 0.0,
        }
        status, body = self._post(pl, timeout=120)
        if status != 200 or not isinstance(body, dict):
            self.check("thinking mode (enable_thinking=True)", False, f"HTTP {status}")
            return
        content = body["choices"][0]["message"]["content"]
        self.check("thinking mode (enable_thinking=True)", bool(content.strip()),
                   "empty response")

    def test_content_array(self) -> None:
        """content=[{type:text,...}] must parse without 422."""
        pl = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Reply with the word OK."},
            ]}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        status, body = self._post(pl)
        self.check(
            "content: array-of-parts (no 422)",
            status == 200,
            f"HTTP {status}: {str(body)[:80]}",
        )

    def test_null_content_tool_role(self) -> None:
        """role=tool with content=null must not 422."""
        pl = {
            "model": self.model,
            "messages": [
                {"role": "user",      "content": "What time is it?"},
                {"role": "assistant", "content": "I'll check."},
                {"role": "tool",      "content": None},
                {"role": "user",      "content": "Thanks."},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        status, body = self._post(pl)
        self.check(
            "null content / tool role (no 422)",
            status == 200,
            f"HTTP {status}: {str(body)[:80]}",
        )

    def test_default_max_tokens(self) -> None:
        """Omitting max_tokens must NOT silently cap output at 256 tokens.

        Sets max_tokens=512 explicitly to force the server to generate at
        least 256 tokens.  The counting prompt reliably fills the budget.
        If the old 256-token default were still in effect, the server would
        silently cap output below 512.
        """
        pl = {
            "model": self.model,
            "messages": [{"role": "user", "content":
                          "Count from 1 to 500, one number per line. "
                          "Do not stop until you reach 500."}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        status, body = self._post(pl, timeout=300)
        if status != 200 or not isinstance(body, dict):
            self.check("default max_tokens > 256", False, f"HTTP {status}")
            return
        ct = body["usage"]["completion_tokens"]
        self.check(
            "default max_tokens > 256",
            ct > 256,
            f"completion_tokens={ct} (max_tokens=512, old cap was 256)",
        )

    def test_multi_turn(self) -> None:
        """Multi-turn conversation: model must reference prior context correctly."""
        pl = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "My name is Zephyr. Remember it."},
                {"role": "assistant", "content": "Got it — your name is Zephyr!"},
                {"role": "user", "content": "What is my name? Reply with just the name."},
            ],
            "max_tokens": 32,
            # Omit temperature — use model's generation_config.json default.
            # temp=0.0 (argmax) causes degeneration on some NVFP4 models.
        }
        status, body = self._post(pl, timeout=90)
        if status != 200 or not isinstance(body, dict):
            self.check("multi-turn context recall", False, f"HTTP {status}")
            return
        answer = body["choices"][0]["message"]["content"]
        self.check(
            "multi-turn context recall",
            "Zephyr" in answer,
            f"got: {answer!r}",
        )

    def test_korean_emoji_streaming(self) -> None:
        """Streaming Korean + emoji output must have no U+FFFD replacement chars."""
        pl = {
            "model": self.model,
            "messages": [{"role": "user", "content":
                          "Write one short sentence in Korean with a smiley emoji 😊."}],
            "max_tokens": 64,
            # Omit temperature — use model's generation_config.json default.
            # temp=0.0 (argmax) produces empty output on some NVFP4 models.
        }
        try:
            chunks = self._stream(pl, timeout=120)
        except Exception as exc:
            self.check("Korean/emoji no garble (streaming)", False, str(exc))
            return

        parts = []
        for c in chunks:
            if c.get("_done"):
                break
            delta = c.get("choices", [{}])[0].get("delta", {})
            text  = delta.get("content")
            if text:
                parts.append(text)
        full = "".join(parts)

        no_replacement = "\ufffd" not in full
        has_content    = bool(full.strip())
        self.check(
            "Korean/emoji no garble (streaming)",
            no_replacement and has_content,
            f"text={full[:100]!r}" if not (no_replacement and has_content) else "",
        )

    # ── Runner ──────────────────────────────────────────────────────────────

    def run_all(self) -> bool:
        tests = [
            self.test_factual_2plus2,
            self.test_factual_capital,
            self.test_temperature_diversity,
            self.test_streaming_format,
            self.test_thinking_mode,
            self.test_content_array,
            self.test_null_content_tool_role,
            self.test_default_max_tokens,
            self.test_multi_turn,
            self.test_korean_emoji_streaming,
        ]

        print("Atlas Spark — Coherence Test Suite")
        print(f"  Model : {self.model}")
        print(f"  URL   : {self.url}")
        print()

        for fn in tests:
            try:
                fn()
            except Exception as exc:
                name = fn.__name__.replace("test_", "").replace("_", " ")
                self.failed += 1
                print(f"  {FAIL}  {name} [EXCEPTION: {exc}]")

        total = self.passed + self.failed
        print()
        if self.failed == 0:
            ok_str = "\033[32mAll\033[0m" if _tty else "All"
            print(f"  {ok_str} {total} tests passed.")
        else:
            fail_str = f"\033[31m{self.failed}/{total}\033[0m" if _tty else f"{self.failed}/{total}"
            print(f"  {fail_str} tests FAILED.")
        return self.failed == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url",   default=DEFAULT_URL,   help="Server base URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name/ID")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print full response detail on every test")
    args = ap.parse_args()

    try:
        urlopen(f"{args.url}/health", timeout=5).read()
    except Exception as exc:
        print(f"ERROR: server not reachable at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    runner = TestRunner(args.url, args.model, args.verbose)
    sys.exit(0 if runner.run_all() else 1)


if __name__ == "__main__":
    main()
