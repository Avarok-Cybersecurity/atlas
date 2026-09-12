#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU oracles for process exit while real readiness HTTP requests are pending."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / 'hopper_ab/time_to_ready.sh'
FIXTURE = r'''
import json, os, pathlib, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
mode, root = sys.argv[1], pathlib.Path(sys.argv[2])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def stall(self):
        # A surviving owned worker holds the accepted socket open after the
        # API leader exits, so curl itself cannot notice EOF and save the gate.
        if os.fork() == 0:
            time.sleep(60)
            os._exit(0)
        def leave():
            time.sleep(0.2)
            os._exit(17)
        threading.Thread(target=leave, daemon=True).start()
        time.sleep(60)
    def do_GET(self):
        if mode == 'health-stall': return self.stall()
        self.reply({'status': 'ready'})
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        if mode == 'completion-stall': return self.stall()
        self.reply({'choices': [{'message': {'content': 'x'}}]})
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
(root / 'port').write_text(str(server.server_port))
server.serve_forever()
'''


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open')
                     and shutil.which('curl'), 'requires Linux pidfds and curl')
class ProcessReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='atlas-readiness-')
        self.root = Path(self.temp.name)
        self.record = self.root / 'owner.json'
        self.original = None

    def manager(self, *args):
        return subprocess.run([sys.executable, str(HERE / 'process_launch.py'), *args],
                              capture_output=True, text=True, timeout=10)

    def start(self, mode='ready'):
        fixture = self.root / 'server.py'
        fixture.write_text(FIXTURE)
        argv = self.root / 'argv.json'
        argv.write_text(json.dumps([sys.executable, str(fixture), mode, str(self.root)]))
        result = self.manager('start', '--record', str(self.record), '--evidence',
                              str(self.root / 'launch.json'), '--log', str(self.root / 'server.log'),
                              '--argv-json', str(argv))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.original = self.record.read_text()
        self.url = 'http://127.0.0.1:' + (self.root / 'port').read_text()

    def tearDown(self):
        if self.original:
            self.record.write_text(self.original)
            result = self.manager('stop', '--record', str(self.record), '--timeout', '0.1')
            self.assertEqual(result.returncode, 0, result.stderr)
        self.temp.cleanup()

    def probe(self, status):
        start = time.monotonic()
        out = self.root / 'boot.json'
        command = ['bash', str(SCRIPT), '--url', self.url, '--model', 'fixture',
                   '--timeout-s', '1800', '--process-owner', str(self.record), '--out', str(out)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)  # Only this test's new probe session.
            process.communicate(timeout=5)
            self.fail('readiness waited on HTTP after the owned API leader exited')
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        self.assertEqual(result.returncode, 0 if status == 'ready' else 1,
                         result.stdout + result.stderr)
        doc = json.loads(out.read_text())
        self.assertEqual(doc['status'], status, doc)
        self.assertEqual(doc['passed'], status == 'ready', doc)
        print(json.dumps({'oracle': self.id(), 'status': status,
                          'elapsed_s': round(time.monotonic() - start, 3)}), flush=True)
        return doc

    def test_live_owned_server_passes(self):
        self.start()
        self.probe('ready')

    def test_exit_during_health_request_is_prompt(self):
        self.start('health-stall')
        self.probe('process-exited')

    def test_exit_during_first_completion_is_prompt(self):
        self.start('completion-stall')
        self.probe('process-exited')

    def test_already_gone_server_refuses(self):
        self.start()
        result = self.manager('stop', '--record', str(self.record), '--timeout', '0.1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.probe('process-exited')

    def test_reused_pid_and_foreign_or_stale_owner_refuse_without_signals(self):
        self.start()
        for field, value in (('start_ticks', 1), ('boot_id', 'stale'),
                             ('run_marker', 'foreign')):
            with self.subTest(field=field):
                owner = json.loads(self.original)
                owner[field] = value
                if field == 'run_marker':
                    owner['environment']['ATLAS_CAMPAIGN_RUN_TOKEN'] = value
                self.record.write_text(json.dumps(owner))
                self.probe('process-ownership-unproven')
        for bad in ('invalid JSON', '[]', 'null'):
            with self.subTest(record=bad):
                self.record.write_text(bad)
                self.probe('process-ownership-unproven')
        self.record.write_text(self.original)
        self.probe('ready')  # Invalid records did not signal the live server.


if __name__ == '__main__':
    unittest.main()
