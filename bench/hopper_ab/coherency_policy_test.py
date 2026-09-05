#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU HTTP oracles for the coherency gate's requested thinking policy."""

import contextlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer


SOURCE = pathlib.Path(__file__).with_name("coherency_gate.py")
SPEC = importlib.util.spec_from_file_location("coherency_gate", SOURCE)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


@contextlib.contextmanager
def endpoint(mode):
    """A real HTTP endpoint whose think-on path can fail independently."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            prompt = request["messages"][-1]["content"]
            finish = "stop"
            message = {"role": "assistant", "content": "To measure a system under a fixed workload."}
            if request.get("tools"):
                finish = "tool_calls"
                message.update(content="", tool_calls=[{
                    "id": "weather", "type": "function", "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Reykjavik", "days": 3}),
                    },
                }])
            elif prompt == GATE.DETERMINISM_PROMPT:
                message["content"] = "101, 103, 107, 109, 113"
            else:
                for question, answer in GATE.KNOWN_ANSWER_CASES:
                    if prompt == question:
                        message["content"] = "Wrong answer" if mode == "wrong-answer" else answer
            if mode == "broken-on" and request["chat_template_kwargs"]["enable_thinking"]:
                message = {"role": "assistant", "content": ""}
                finish = "stop"
            if mode == "leak-off" and prompt == GATE.THINK_PROMPT:
                message["content"] = "<think>private working</think> To measure a system."
            raw = json.dumps({"choices": [{"index": 0, "message": message, "finish_reason": finish}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


class CoherencyPolicyTest(unittest.TestCase):
    def invoke(self, mode, think=None):
        with endpoint(mode) as (url, requests):
            command = [sys.executable, str(SOURCE), "--url", url, "--model", "policy-stub", "--timeout", "5"]
            if think is not None:
                command += ["--think", think]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertIn(completed.returncode, (0, 1), completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 0 if result["passed"] else 1)
        return result, requests

    def assert_policy(self, result, requests, think):
        policy = {"think": think, "chat_template_kwargs": {"enable_thinking": think == "on"}}
        self.assertEqual(result["request_policy"], policy)
        self.assertEqual(len(requests), 7)
        expected = [think == "on"] * 7
        expected[3] = False  # The dedicated leakage oracle always disables thinking.
        self.assertEqual([request["chat_template_kwargs"]["enable_thinking"] for request in requests], expected)
        self.assertEqual(len(result["http_exchanges"]), 7)
        for request, exchange in zip(requests, result["http_exchanges"]):
            self.assertEqual(json.loads(exchange["request_json"]), request)
            check_policy = result["check_request_policy"][exchange["check"]]
            self.assertEqual(check_policy["chat_template_kwargs"], request["chat_template_kwargs"])
            self.assertEqual(check_policy["think"], "on" if request["chat_template_kwargs"]["enable_thinking"] else "off")

    def test_broken_think_on_path_refuses(self):
        result, requests = self.invoke("broken-on", "on")
        self.assertFalse(result["passed"])
        for key in ("determinism_ok", "toolcall_ok", "known_answer_ok"):
            self.assertFalse(result[key], key)
        self.assertTrue(result["think_leak_ok"])
        self.assert_policy(result, requests, "on")

    def test_known_wrong_answers_refuse_in_both_modes(self):
        for think in ("off", "on"):
            with self.subTest(think=think):
                result, requests = self.invoke("wrong-answer", think)
                self.assertFalse(result["passed"])
                self.assertFalse(result["known_answer_ok"])
                for key in ("determinism_ok", "toolcall_ok", "think_leak_ok"):
                    self.assertTrue(result[key], key)
                self.assert_policy(result, requests, think)

    def test_clean_both_modes_pass(self):
        for think in ("off", "on"):
            with self.subTest(think=think):
                result, requests = self.invoke("clean", think)
                self.assertTrue(result["passed"])
                self.assert_policy(result, requests, think)

    def test_default_remains_think_off(self):
        result, requests = self.invoke("broken-on")
        self.assertTrue(result["passed"])
        self.assert_policy(result, requests, "off")

    def test_dedicated_think_off_leak_refuses_in_both_modes(self):
        for think in ("off", "on"):
            with self.subTest(think=think):
                result, requests = self.invoke("leak-off", think)
                self.assertFalse(result["passed"])
                self.assertFalse(result["think_leak_ok"])
                for key in ("determinism_ok", "toolcall_ok", "known_answer_ok"):
                    self.assertTrue(result[key], key)
                self.assert_policy(result, requests, think)


if __name__ == "__main__":
    unittest.main(verbosity=2)
