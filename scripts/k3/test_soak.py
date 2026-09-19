# SPDX-License-Identifier: AGPL-3.0-only
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soak


def event(text='', finish=None):
    return 'data: ' + json.dumps({'model': 'tiny', 'choices': [{'index': 0, 'text': text, 'finish_reason': finish}]}) + '\n\n'


class SoakTests(unittest.TestCase):
    def parse(self, value, cancel=False):
        return soak.read_sse(io.BytesIO(value.encode()), 'tiny', cancel)

    def test_valid_stream(self):
        result = self.parse(': heartbeat\n\n' + event('hello') + event(finish='length') + 'data: [DONE]\n\n')
        self.assertEqual(result['text'], 'hello')
        self.assertEqual(result['finish_reason'], 'length')

    def test_terminal_required(self):
        for value in (event('hello'), event('hello') + 'data: [DONE]\n\n', event('hello') + event(finish='stop')):
            with self.assertRaises(ValueError):
                self.parse(value)

    def test_error_and_invalid_events(self):
        for value in ('data: {bad}\n\n', 'data: {"error":"bad"}\n\n', event('hello').replace('tiny', 'wrong'), event('hello') + event(finish='stop') + event('late')):
            with self.assertRaises(ValueError):
                self.parse(value)

    def test_cancel_is_distinct(self):
        result = self.parse(event('hello'), True)
        self.assertEqual(result['status'], 'client_cancelled')
        self.assertNotIn('finish_reason', result)

    def test_cancel_after_terminal_is_not_counted(self):
        with self.assertRaises(ValueError):
            self.parse(event('hello', 'stop'), True)

    def test_wrong_output_fails(self):
        with self.assertRaises(ValueError):
            soak.match({'text': 'bad', 'finish_reason': 'stop'}, {'text': 'good', 'finish_reason': 'stop'})

    def test_counts_ignore_timing_but_check_trusted_counts(self):
        expected = {'text': 'hello', 'finish_reason': 'stop',
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3,
                              'time_to_first_token_ms': 100}}
        actual = dict(expected, usage=dict(expected['usage'], time_to_first_token_ms=200))
        soak.match(actual, expected)
        trusted = {'text': 'hello', 'finish_reason': 'stop', 'prompt_tokens': 2, 'completion_tokens': 1}
        soak.match(actual, trusted)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            wrong = dict(actual, usage=dict(actual['usage']))
            wrong['usage'][key] += 1
            with self.assertRaises(ValueError):
                soak.match(wrong, expected)
        trusted['completion_tokens'] = 9
        with self.assertRaises(ValueError):
            soak.match(actual, trusted)

    def test_trickle_deadline_and_cancel_close(self):
        disconnected = {'cancel': threading.Event(), 'timeout': threading.Event()}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    if payload['prompt'] == 'cancel':
                        self.wfile.write(event('first').encode())
                        self.wfile.flush()
                    for _ in range(100):
                        self.wfile.write(b': heartbeat\n\n')
                        self.wfile.flush()
                        time.sleep(.03)
                except (BrokenPipeError, ConnectionResetError):
                    disconnected[payload['prompt']].set()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = 'http://127.0.0.1:' + str(server.server_port)
        payload = {'model': 'tiny', 'prompt': 'timeout', 'max_tokens': 8, 'temperature': 0}
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                soak.bounded_request(endpoint, payload, 'stream', .25)
            self.assertLess(time.monotonic() - started, 1.5)
            payload['prompt'] = 'cancel'
            self.assertEqual(soak.bounded_request(endpoint, payload, 'cancel', 2)['status'], 'client_cancelled')
            self.assertTrue(disconnected['cancel'].wait(1))
        finally:
            server.shutdown()
            server.server_close()


    def test_cli_receipts_failure_and_exclusive_output(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                text = 'hello'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if payload['stream'] else 'application/json')
                self.end_headers()
                if payload['stream']:
                    self.wfile.write((event(text) + event(finish='length') + 'data: [DONE]\n\n').encode())
                else:
                    self.wfile.write(json.dumps({'model': 'tiny', 'choices': [{'text': text, 'finish_reason': 'length'}],
                                                'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cases = root / 'cases.json'
                cases.write_text(json.dumps([{'id': 'one', 'prompt': 'hi', 'max_tokens': 1}]))
                output = root / 'run'
                command = [sys.executable, str(Path(soak.__file__)), '--endpoint',
                           'http://127.0.0.1:' + str(server.server_port), '--model', 'tiny',
                           '--cases', str(cases), '--cycles', '1', '--deadline', '10', '--output', str(output)]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=12)
                self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['requests'], 7)
                self.assertEqual(summary['cycles_completed'], 1)
                saved = (output / 'receipts.jsonl').read_text()
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
                self.assertEqual((output / 'receipts.jsonl').read_text(), saved)
                trusted = root / 'trusted.json'
                trusted.write_text(json.dumps({'baseline_cases': [{'id': 'one', 'text': 'wrong', 'finish_reason': 'length'}]}))
                command[-1] = str(root / 'bad')
                completed = subprocess.run(command + ['--baseline-receipt', str(trusted)], capture_output=True, timeout=12)
                self.assertEqual(completed.returncode, 1)
                summary = json.loads((root / 'bad' / 'summary.json').read_text())
                self.assertEqual(summary['status'], 'failed')
                self.assertEqual(summary['requests'], 1)
                self.assertEqual(summary['cycles_completed'], 0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
