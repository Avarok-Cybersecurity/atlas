#!/usr/bin/env python3
"""HTTP boundary regressions invoked by time_to_ready.sh --selftest."""
import json
import pathlib
import subprocess
import threading
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def selftest():
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            observed.append(body)
            mode = self.path.split('/')[1]
            code = 500 if mode == 'http500' else 200
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            if mode == 'slow':
                time.sleep(0.7)
            if mode == 'invalid':
                raw = b'not json'
            elif mode in ('http500', 'error200'):
                raw = json.dumps({'error': 'known failure'}).encode()
            else:
                content = '' if mode == 'empty' else 'x'
                raw = json.dumps({'choices': [{'message': {'content': content},
                                               'finish_reason': 'length'}]}).encode()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = pathlib.Path(__file__).with_name('time_to_ready.sh')
    failures = []
    results = {}
    try:
        for mode in ('clean', 'http500', 'error200', 'empty', 'invalid', 'expired', 'slow'):
            start = time.time() - (2 if mode == 'expired' else 0)
            timeout = 0.4 if mode in ('expired', 'slow') else 5
            cmd = ['bash', str(script), '--url', f'http://127.0.0.1:{server.server_port}/{mode}',
                   '--model', 'stub-model', '--engine', 'selftest', '--start-epoch', str(start),
                   '--timeout-s', str(timeout)]
            wall_start = time.monotonic()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            elapsed = time.monotonic() - wall_start
            result = json.loads(proc.stdout)
            results[mode] = {'exit_code': proc.returncode, 'artifact': result}
            expected = mode == 'clean'
            if (proc.returncode == 0) != expected or (result['status'] == 'ready') != expected:
                failures.append(f'{mode}: expected pass={expected}, got exit {proc.returncode}, status {result["status"]}')
            if mode == 'slow' and elapsed > 0.65:
                failures.append(f'slow: first-token request exceeded whole-boot deadline ({elapsed:.3f}s)')
        with tempfile.TemporaryDirectory() as tmp:
            command = ['bash', str(script), '--url', f'http://127.0.0.1:{server.server_port}/clean',
                       '--model', 'stub-model', '--out', str(pathlib.Path(tmp) / 'missing' / 'boot.json')]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
            results['unwritable-artifact'] = {'exit_code': proc.returncode, 'stderr': proc.stderr}
            if proc.returncode == 0:
                failures.append('unwritable-artifact: output failure must make the command fail')
        for key, expected in {'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                              'chat_template_kwargs': {'enable_thinking': False}}.items():
            if observed[0].get(key) != expected:
                failures.append(f'first-token request did not pin {key}={expected}')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    print(json.dumps(results, indent=2))
    assert not failures, '\n'.join(failures)
    print('SELFTEST OK: HTTP500, error JSON, empty/invalid replies, expired and slow boots refused; sampling pinned')


if __name__ == '__main__':
    selftest()
