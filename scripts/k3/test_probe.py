# SPDX-License-Identifier: AGPL-3.0-only
import copy
import unittest
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from probe import validate


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.good = {'choices': [{'text': 'hello world', 'finish_reason': 'length'}],
                     'usage': {'completion_tokens': 2}}

    def test_accepts_expected_generation(self):
        self.assertEqual(validate(self.good, 'hello'), 'hello world')

    def test_rejects_broken_generation(self):
        for change in ('empty', 'finish', 'count', 'nan', 'choices', 'prefix'):
            with self.subTest(change=change):
                result = copy.deepcopy(self.good)
                if change == 'empty':
                    result['choices'][0]['text'] = '  '
                elif change == 'finish':
                    result['choices'][0]['finish_reason'] = None
                elif change == 'count':
                    result['usage']['completion_tokens'] = 0
                elif change == 'nan':
                    result['choices'][0]['logprobs'] = [float('nan')]
                elif change == 'choices':
                    result['choices'] = []
                with self.assertRaises(ValueError):
                    validate(result, 'wrong' if change == 'prefix' else None)


class EndpointTests(unittest.TestCase):
    def test_request_receipt_and_total_deadline(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            slow = False

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                if self.slow:
                    time.sleep(1)
                body = json.dumps({'choices': [{'text': 'hello', 'finish_reason': 'stop'}],
                                   'usage': {'completion_tokens': 1}}).encode()
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                command = [sys.executable, str(Path(__file__).with_name('probe.py')),
                           '--endpoint', f'http://127.0.0.1:{server.server_port}',
                           '--model', 'fixture', '--prompt', 'test', '--expected-prefix', 'hello']
                success = Path(directory) / 'success.json'
                result = subprocess.run(command + ['--output', str(success)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(success.read_text())['status'], 'passed')
                Handler.slow = True
                failure = Path(directory) / 'timeout.json'
                result = subprocess.run(command + ['--deadline', '0.2', '--output', str(failure)],
                                        capture_output=True, timeout=3)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(json.loads(failure.read_text())['status'], 'failed')
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
