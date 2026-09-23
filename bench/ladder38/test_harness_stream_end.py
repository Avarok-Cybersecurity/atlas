# SPDX-License-Identifier: AGPL-3.0-only
"""A streamed request the server FAILED is an error, never a 0-token success.

Drives the harness's real one_request() against a scripted SSE body (stdlib
only: aiohttp is stubbed to the one attribute one_request touches).
Run: python3 -m unittest discover -s bench/ladder38 -p 'test_harness_*.py'
"""
import asyncio
import importlib.util
import pathlib
import types
import unittest

_PATH = pathlib.Path(__file__).with_name("harness_w56_conc_ladder.py")
_spec = importlib.util.spec_from_file_location("harness_w56", _PATH)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
h.aiohttp = types.SimpleNamespace(ClientTimeout=lambda total: None)


class _Content:
    def __init__(self, body):
        self._body = body.encode()

    async def iter_any(self):
        yield self._body


class _Resp:
    status = 200

    def __init__(self, body):
        self.content = _Content(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, body):
        self._body = body

    def post(self, url, json=None, timeout=None):
        return _Resp(self._body)


def run(body):
    return asyncio.run(h.one_request(_Session(body), "http://x", "m", "p", 8))


def delta(text):
    return 'data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % text


class StreamEnd(unittest.TestCase):
    def test_chat_error_frame_is_an_error(self):
        out = run(delta("a") + 'data: {"error":{"message":"prefill_chunk failed: 901",'
                  '"type":"server_error","code":500}}\n\ndata: [DONE]\n\n')
        self.assertIn("error", out)
        self.assertIn("prefill_chunk failed: 901", out["error"])

    def test_legacy_completions_error_frame_is_an_error(self):
        out = run('data: {"error":"Scheduler queue closed"}\n\ndata: [DONE]\n\n')
        self.assertIn("Scheduler queue closed", out.get("error", ""))

    def test_truncated_stream_is_an_error(self):
        self.assertIn("truncated", run(delta("a") + delta("b")).get("error", ""))
        self.assertIn("error", run(""))

    def test_completed_streams_still_succeed(self):
        # NEGATIVE CONTROLS: [DONE], or a finish_reason chunk then EOF.
        out = run(delta("a") + "data: [DONE]\n\n")
        self.assertNotIn("error", out)
        self.assertEqual(out["sse_deltas"], 1)
        out = run(delta("b") + 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        self.assertNotIn("error", out)
        self.assertEqual(out["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
