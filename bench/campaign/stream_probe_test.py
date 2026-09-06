#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Prove complete streaming evidence against malformed and split input."""

import base64
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from stream_probe import StreamAudit

HERE = pathlib.Path(__file__).resolve().parent


def event(delta=None, finish=None, usage=None):
    doc = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        doc["usage"] = usage
    return ("data: " + json.dumps(doc, ensure_ascii=False) + "\n\n").encode()


USAGE = {"prompt_tokens": 12, "completion_tokens": 2}
BODY = (event({"role": "assistant"}) + event({"content": "héllo"})
        + event(finish="stop", usage=USAGE) + b"data: [DONE]\n\n")


class StreamTests(unittest.TestCase):
    def audit(self, chunks):
        result = StreamAudit()
        for stamp, chunk in chunks:
            result.feed(chunk, stamp)
        return result.finish()

    def test_first_role_is_not_first_content(self):
        report = self.audit([(0.01, event({"role": "assistant"})),
            (0.10, event({"reasoning_content": "reason"})),
            (0.15, event({"content": "answer"})),
            (0.20, event(finish="stop", usage=USAGE) + b"data: [DONE]\n\n")])
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["first_role_s"], 0.01)
        self.assertEqual(report["first_reasoning_s"], 0.10)
        self.assertEqual(report["first_content_s"], 0.15)

    def test_utf8_and_crlf_split_at_every_byte(self):
        body = BODY.replace(b"\n", b"\r\n")
        report = self.audit([(i / 1000, bytes([value])) for i, value in enumerate(body)])
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["content"], "héllo")

    def test_incomplete_and_malformed_streams_refuse(self):
        cases = {
            "missing_done": BODY.replace(b"data: [DONE]\n\n", b""),
            "missing_finish": event({"content": "A"}, usage=USAGE) + b"data: [DONE]\n\n",
            "missing_usage": event({"content": "A"}, finish="stop") + b"data: [DONE]\n\n",
            "malformed": b"data: not-json\n\n" + BODY,
            "empty": event({"role": "assistant"}, finish="stop", usage=USAGE) + b"data: [DONE]\n\n",
            "zero_tokens": event({"content": "A"}, finish="stop", usage={**USAGE, "completion_tokens": 0}) + b"data: [DONE]\n\n",
            "boolean_tokens": event({"content": "A"}, finish="stop", usage={**USAGE, "completion_tokens": True}) + b"data: [DONE]\n\n",
            "invalid_utf8": b"data: \xff\n\n" + BODY,
            "unfinished_event": BODY + b"data:",
            "data_after_done": BODY + event({"content": "late"}),
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                report = self.audit([(0.1, body)])
                self.assertFalse(report["passed"], name)
                self.assertTrue(report["errors"], name)

    def test_tool_arguments_must_form_json(self):
        def tool(args):
            return {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "weather", "arguments": args}}]}
        good = event(tool('{"city":"Oslo"}'), finish="tool_calls", usage=USAGE)
        bad = event(tool('{"city":'), finish="tool_calls", usage=USAGE)
        self.assertTrue(self.audit([(0.1, good + b"data: [DONE]\n\n")])["passed"])
        self.assertFalse(self.audit([(0.1, bad + b"data: [DONE]\n\n")])["passed"])

    def test_multitoken_content_is_not_counted_as_one_token(self):
        report = self.audit([(0.1, BODY)])
        self.assertEqual(report["usage"], USAGE)
        self.assertEqual(report["content_events"], 1)
        self.assertNotIn("tpot_ms", report)

    def test_real_http_retains_exact_request_and_wire_bytes(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="atlas-stream-probe-") as temp:
                root = pathlib.Path(temp)
                request = root / "request.json"
                payload = b'{"model":"fixture","messages":[],"stream":true,"stream_options":{"include_usage":true}}'
                request.write_bytes(payload)
                out = root / "evidence"
                command = [sys.executable, str(HERE / "stream_probe.py"), "--url",
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    "--request-json", str(request), "--out", str(out), "--timeout-s", "3"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(captured, [payload])
                records = [json.loads(line) for line in (out / "chunks.jsonl").read_text().splitlines()]
                self.assertEqual(b"".join(base64.b64decode(row["data_base64"]) for row in records), BODY)
                self.assertTrue(json.loads((out / "report.json").read_text())["passed"])
                repeat = subprocess.run(command, capture_output=True, text=True, timeout=8)
                self.assertNotEqual(repeat.returncode, 0, "existing evidence was overwritten")
                self.assertEqual(len(captured), 1, "refused output still sent a request")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_total_deadline_rejects_trickling_headers(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    for byte in b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n":
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="atlas-stream-deadline-") as temp:
                root = pathlib.Path(temp)
                request = root / "request.json"
                request.write_text(json.dumps({"stream": True, "stream_options": {"include_usage": True}}))
                out = root / "evidence"
                result = subprocess.run([sys.executable, str(HERE / "stream_probe.py"),
                    "--url", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    "--request-json", str(request), "--out", str(out), "--timeout-s", "0.15"],
                    capture_output=True, text=True, timeout=2)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                report = json.loads((out / "report.json").read_text())
                self.assertIn("total probe deadline exceeded", report["errors"])
                self.assertLess(report["elapsed_s"], 0.8)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
