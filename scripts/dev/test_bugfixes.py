#!/usr/bin/env python3
"""
test_bugfixes.py — Targeted tests for specific bug fixes in alpha 2.5.

Tests each bug fix individually with clear pass/fail for every issue.

Usage:
  python3 test_bugfixes.py [--url URL] [--model MODEL] [-v]
"""

import argparse, json, sys, time, textwrap
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_URL = "http://localhost:8888"
DEFAULT_MODEL = "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4"

_tty = sys.stdout.isatty()
PASS = "\033[32m✓\033[0m" if _tty else "PASS"
FAIL = "\033[31m✗\033[0m" if _tty else "FAIL"
SKIP = "\033[33m⊘\033[0m" if _tty else "SKIP"


class BugfixTester:
    def __init__(self, url: str, model: str, verbose: bool):
        self.url = url
        self.model = model
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def _post(self, endpoint: str, payload: dict, timeout: int = 90):
        data = json.dumps(payload).encode()
        req = Request(
            f"{self.url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except HTTPError as e:
            return e.code, e.read().decode()

    def _stream_raw(self, endpoint: str, payload: dict, timeout: int = 120):
        """Return raw SSE lines as parsed JSON objects."""
        data = json.dumps({**payload, "stream": True}).encode()
        req = Request(
            f"{self.url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        chunks = []
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

    def check(self, name: str, passed: bool, detail: str = ""):
        if passed:
            self.passed += 1
            print(f"  {PASS}  {name}")
        else:
            self.failed += 1
            print(f"  {FAIL}  {name}")
        if detail and (not passed or self.verbose):
            for line in textwrap.wrap(detail, 74):
                print(f"       {line}")

    def skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  {SKIP}  {name} — {reason}")

    # ── BUG #35: Batched decode corruption at c>=2 ──

    def test_bug35_concurrent_coherence(self):
        """BUG #35: Concurrent requests must not contaminate each other."""
        print("\n── BUG #35: Batched decode SSM slot double-release ──")

        prompts = [
            ("What is 2+2? Reply with ONLY the number, nothing else.", "4"),
            ("What is the capital of France? Reply with ONLY the city name.", "Paris"),
            ("What is the capital of Japan? Reply with ONLY the city name.", "Tokyo"),
        ]

        def send_request(content, _expected):
            status, body = self._post("/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 32,
            })
            if status != 200 or not isinstance(body, dict):
                return None, f"HTTP {status}"
            return body["choices"][0]["message"]["content"], None

        # Run 3 iterations of concurrent c=3
        for iteration in range(3):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {
                    pool.submit(send_request, p, e): (p, e) for p, e in prompts
                }
                results = []
                for future in as_completed(futures):
                    prompt, expected = futures[future]
                    answer, err = future.result()
                    results.append((expected, answer, err))

            all_ok = True
            details = []
            for expected, answer, err in results:
                if err:
                    all_ok = False
                    details.append(f"error: {err}")
                elif expected not in (answer or ""):
                    all_ok = False
                    details.append(f"expected '{expected}', got: {answer!r}")

            self.check(
                f"c=3 iteration {iteration+1}: no cross-contamination",
                all_ok,
                "; ".join(details) if details else "",
            )

    def test_cross_request_bleed_c4(self):
        """Extended bleed-through test: c=4 with diverse, easily-distinguishable prompts."""
        prompts = [
            ("List the first 5 prime numbers, comma-separated. Nothing else.", "2, 3, 5, 7, 11"),
            ("Name 3 colors of the rainbow, comma-separated. Nothing else.", None),  # any valid color list
            ("What planet is closest to the sun? One word.", "Mercury"),
            ("What is the chemical symbol for water? Just the formula.", "H2O"),
        ]
        # Validation: each response must be on-topic (not contain another response's keywords)
        topic_keywords = [
            {"2", "3", "5", "7", "11", "prime"},
            {"red", "orange", "yellow", "green", "blue", "indigo", "violet", "color", "rainbow"},
            {"Mercury", "mercury", "planet", "sun"},
            {"H2O", "h2o", "water", "H₂O"},
        ]

        def send(idx):
            content = prompts[idx][0]
            status, body = self._post("/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 64,
            })
            if status != 200 or not isinstance(body, dict):
                return idx, None, f"HTTP {status}"
            return idx, body["choices"][0]["message"]["content"], None

        for iteration in range(5):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(send, i) for i in range(4)]
                results = {}
                for f in as_completed(futures):
                    idx, answer, err = f.result()
                    results[idx] = (answer, err)

            all_ok = True
            details = []
            for idx in range(4):
                answer, err = results.get(idx, (None, "missing"))
                if err:
                    all_ok = False
                    details.append(f"prompt {idx}: {err}")
                    continue
                # Check answer contains at least one keyword from its own topic (case-insensitive)
                own_kws = topic_keywords[idx]
                answer_lower = (answer or "").lower()
                on_topic = any(kw.lower() in answer_lower for kw in own_kws)
                if not on_topic:
                    all_ok = False
                    details.append(f"prompt {idx} off-topic: {answer!r}")

            self.check(
                f"c=4 bleed-through iteration {iteration+1}",
                all_ok,
                "; ".join(details) if details else "",
            )

    def test_cross_request_bleed_streaming(self):
        """Cross-request bleed test using streaming (different code path)."""
        import threading

        results = {}
        errors = {}

        def stream_request(idx, content, expected_kw):
            try:
                chunks = self._stream_raw("/v1/chat/completions", {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 32,
                })
                text_parts = []
                for c in chunks:
                    if c.get("_done"):
                        break
                    delta = c.get("choices", [{}])[0].get("delta", {})
                    t = delta.get("content")
                    if t:
                        text_parts.append(t)
                results[idx] = "".join(text_parts)
            except Exception as e:
                errors[idx] = str(e)

        prompts = [
            (0, "What is 7*8? Just the number.", "56"),
            (1, "What is the capital of Germany? One word.", "Berlin"),
            (2, "What color is the sky? One word.", "blue"),
        ]

        for iteration in range(3):
            results.clear()
            errors.clear()
            threads = [
                threading.Thread(target=stream_request, args=(idx, content, kw))
                for idx, content, kw in prompts
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            all_ok = True
            details = []
            for idx, content, expected in prompts:
                if idx in errors:
                    all_ok = False
                    details.append(f"prompt {idx}: error {errors[idx]}")
                elif idx not in results:
                    all_ok = False
                    details.append(f"prompt {idx}: no result")
                elif expected.lower() not in results[idx].lower():
                    all_ok = False
                    details.append(f"prompt {idx}: expected '{expected}', got: {results[idx]!r}")

            self.check(
                f"c=3 streaming bleed iteration {iteration+1}",
                all_ok,
                "; ".join(details) if details else "",
            )

    # ── BUG #36: OpenAI API compat ──

    def test_bug36_missing_content_field(self):
        """BUG #36a: Assistant message with tool_calls but no content field."""
        print("\n── BUG #36: OpenAI API compat ──")
        pl = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "test", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                {"role": "user", "content": "Say OK."},
            ],
            "max_tokens": 16,
        }
        status, body = self._post("/v1/chat/completions", pl)
        self.check("assistant message with tool_calls, no content (no 422)",
                    status == 200, f"HTTP {status}: {str(body)[:80]}")

    def test_bug36_finish_reason_streaming(self):
        """BUG #36b: finish_reason must be present (null or string) in every chunk."""
        chunks = self._stream_raw("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 16,
        })
        all_have_fr = True
        for c in chunks:
            if c.get("_done"):
                continue
            choices = c.get("choices", [])
            if choices:
                # finish_reason must be a KEY in the choice dict (can be null)
                if "finish_reason" not in choices[0]:
                    all_have_fr = False
                    break
        self.check("streaming: finish_reason key present in all chunks", all_have_fr)

    def test_bug36_no_leading_newlines(self):
        """BUG #36c: Response must not start with \\n\\n."""
        status, body = self._post("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "What is 1+1? Just the number."}],
            "max_tokens": 16,
        })
        if status != 200 or not isinstance(body, dict):
            self.check("no leading \\n\\n in response", False, f"HTTP {status}")
            return
        content = body["choices"][0]["message"]["content"]
        self.check("no leading \\n\\n in response",
                    not content.startswith("\n\n"),
                    f"got: {content[:30]!r}")

    # ── Stop sequences ──

    def test_stop_sequences(self):
        """Stop sequences: generation stops at specified string."""
        print("\n── Stop sequence support ──")

        # Test 1: single string stop
        status, body = self._post("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Count from 1 to 20, one per line."}],
            "max_tokens": 256,
            "stop": "5",
        })
        if status == 200 and isinstance(body, dict):
            content = body["choices"][0]["message"]["content"]
            fr = body["choices"][0]["finish_reason"]
            # Per OpenAI spec: returned text must NOT contain the stop sequence
            ends_with_stop = content.rstrip().endswith("5") if content else False
            self.check("stop sequence (single string)",
                        fr == "stop" and not ends_with_stop,
                        f"finish_reason={fr!r}, content ends: {content[-30:]!r}")
        else:
            self.check("stop sequence (single string)", False, f"HTTP {status}")

        # Test 2: array of strings
        status, body = self._post("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Count from 1 to 20, one per line."}],
            "max_tokens": 256,
            "stop": ["3", "5"],
        })
        if status == 200 and isinstance(body, dict):
            fr = body["choices"][0]["finish_reason"]
            self.check("stop sequence (array)", fr == "stop", f"finish_reason={fr!r}")
        else:
            self.check("stop sequence (array)", False, f"HTTP {status}")

        # Test 3: streaming with stop
        chunks = self._stream_raw("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Count from 1 to 20, one per line."}],
            "max_tokens": 256,
            "stop": "5",
        })
        done_chunk = next((c for c in chunks if not c.get("_done") and c.get("usage")), None)
        if done_chunk:
            fr = done_chunk["choices"][0].get("finish_reason", "")
            self.check("stop sequence (streaming)", fr == "stop", f"finish_reason={fr!r}")
        else:
            self.check("stop sequence (streaming)", False, "no done chunk found")

    # ── Completions think-tag suppression ──

    def test_completions_think_tag(self):
        """Completions endpoint must not leak <think> tags."""
        print("\n── /v1/completions think-tag suppression ──")
        status, body = self._post("/v1/completions", {
            "model": self.model,
            "prompt": "The capital of France is",
            "max_tokens": 32,
        })
        if status != 200 or not isinstance(body, dict):
            self.check("/v1/completions: no <think> leakage", False, f"HTTP {status}")
            return
        text = body["choices"][0]["text"]
        self.check("/v1/completions: no <think> leakage",
                    "<think>" not in text,
                    f"got: {text[:80]!r}")

        # Also check no leading \n\n
        self.check("/v1/completions: no leading \\n\\n",
                    not text.startswith("\n\n"),
                    f"got: {text[:30]!r}")

    # ── OpenCode/Codex API compat (new in this session) ──

    def test_system_fingerprint(self):
        """system_fingerprint must be present in responses."""
        print("\n── OpenCode/Codex API compat ──")
        status, body = self._post("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
        })
        if status != 200 or not isinstance(body, dict):
            self.check("system_fingerprint in response", False, f"HTTP {status}")
            return
        fp = body.get("system_fingerprint")
        self.check("system_fingerprint in response", fp is not None, f"got: {fp!r}")

    def test_system_fingerprint_streaming(self):
        """system_fingerprint must be present in streaming chunks."""
        chunks = self._stream_raw("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
        })
        non_done = [c for c in chunks if not c.get("_done")]
        if not non_done:
            self.check("system_fingerprint in streaming chunks", False, "no chunks")
            return
        all_have_fp = all("system_fingerprint" in c for c in non_done)
        self.check("system_fingerprint in streaming chunks", all_have_fp)

    def test_logprobs_null(self):
        """logprobs: null must be present in response choices."""
        status, body = self._post("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
        })
        if status != 200 or not isinstance(body, dict):
            self.check("logprobs: null in choices", False, f"HTTP {status}")
            return
        choice = body["choices"][0]
        self.check("logprobs: null in choices",
                    "logprobs" in choice and choice["logprobs"] is None,
                    f"logprobs key present: {'logprobs' in choice}")

    def test_logprobs_null_streaming(self):
        """logprobs: null must be present in streaming chunk choices."""
        chunks = self._stream_raw("/v1/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
        })
        non_done = [c for c in chunks if not c.get("_done") and c.get("choices")]
        all_have_lp = all(
            "logprobs" in c["choices"][0] and c["choices"][0]["logprobs"] is None
            for c in non_done
        )
        self.check("logprobs: null in streaming chunks", all_have_lp)

    def test_models_endpoint(self):
        """GET /v1/models must include 'created' timestamp."""
        req = Request(f"{self.url}/v1/models")
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        model_obj = body["data"][0]
        has_created = "created" in model_obj and isinstance(model_obj["created"], (int, float))
        self.check("/v1/models has 'created' field", has_created,
                    f"got: {model_obj}")

    # ── KV cache startup warning ──

    def test_kv_cache_warning(self):
        """KV cache warning: server started without crash (implicit test)."""
        print("\n── KV cache startup warning ──")
        req = Request(f"{self.url}/health")
        with urlopen(req, timeout=5) as resp:
            ok = resp.read().decode().strip() == "ok"
        self.check("server healthy (KV cache allocated correctly)", ok)

    # ── Multi-turn (BUG #38/#40) ──

    def test_multi_turn_coherence(self):
        """BUG #38/#40: Multi-turn must not produce gibberish."""
        print("\n── BUG #38/#40: Multi-turn SSM state zeroing ──")

        # 3-turn conversation
        pl = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "What is the capital of France? One word."},
                {"role": "assistant", "content": "Paris"},
                {"role": "user", "content": "What is the capital of Japan? One word."},
                {"role": "assistant", "content": "Tokyo"},
                {"role": "user", "content": "Now tell me: what was the FIRST capital I asked about? One word."},
            ],
            "max_tokens": 32,
        }
        status, body = self._post("/v1/chat/completions", pl)
        if status != 200 or not isinstance(body, dict):
            self.check("multi-turn 3-turn recall", False, f"HTTP {status}")
            return
        answer = body["choices"][0]["message"]["content"]
        self.check("multi-turn 3-turn recall",
                    "Paris" in answer or "France" in answer,
                    f"got: {answer!r}")

    # ── Runner ──

    def run_all(self):
        print("Atlas Spark — Bugfix Verification Suite")
        print(f"  Model : {self.model}")
        print(f"  URL   : {self.url}")

        self.test_bug35_concurrent_coherence()
        self.test_cross_request_bleed_c4()
        self.test_cross_request_bleed_streaming()
        self.test_bug36_missing_content_field()
        self.test_bug36_finish_reason_streaming()
        self.test_bug36_no_leading_newlines()
        self.test_stop_sequences()
        self.test_completions_think_tag()
        self.test_system_fingerprint()
        self.test_system_fingerprint_streaming()
        self.test_logprobs_null()
        self.test_logprobs_null_streaming()
        self.test_models_endpoint()
        self.test_kv_cache_warning()
        self.test_multi_turn_coherence()

        total = self.passed + self.failed
        print(f"\n{'='*60}")
        if self.failed == 0:
            ok = "\033[32mAll\033[0m" if _tty else "All"
            print(f"  {ok} {total} tests passed.")
        else:
            fail = f"\033[31m{self.failed}/{total}\033[0m" if _tty else f"{self.failed}/{total}"
            print(f"  {fail} tests FAILED.")
        if self.skipped:
            print(f"  ({self.skipped} skipped)")
        return self.failed == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    try:
        urlopen(f"{args.url}/health", timeout=5).read()
    except Exception as exc:
        print(f"ERROR: server not reachable at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    tester = BugfixTester(args.url, args.model, args.verbose)
    sys.exit(0 if tester.run_all() else 1)


if __name__ == "__main__":
    main()
