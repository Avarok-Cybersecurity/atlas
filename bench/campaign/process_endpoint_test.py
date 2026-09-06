#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Linux CPU oracles for process endpoint ownership; real sockets and processes."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
MANAGER = HERE / 'process_launch.py'
ENDPOINT = HERE / 'process_endpoint.py'

SERVER = r'''
import socket, sys, threading, time
family, address, port, dual = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = socket.socket(family)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
if family == socket.AF_INET6:
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0 if dual else 1)
s.bind((address, port)); s.listen()
print(s.getsockname()[1], flush=True)
def handle(c):
    try:
        c.settimeout(3)
        c.recv(4096)
    except OSError:
        pass
    finally:
        c.close()
while True:
    c, _ = s.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
'''


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'),
                     'requires Linux /proc and pidfds')
class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='atlas-endpoint-test-')
        self.base = Path(self.temp.name)
        self.owner = self.base / 'owner.json'
        self.launch = self.base / 'launch.json'
        self.proof = self.base / 'endpoint.json'
        self.log = self.base / 'server.log'
        self.sockets = []
        self.foreign = []

    def tearDown(self):
        if self.owner.exists():
            subprocess.run([sys.executable, str(MANAGER), 'stop', '--record', str(self.owner),
                            '--timeout', '0.1'], capture_output=True, timeout=8)
        for proc in self.foreign:
            proc.terminate()
            proc.wait(timeout=5)
        for sock in self.sockets:
            sock.close()
        self.temp.cleanup()

    def call(self, mode, port, address='127.0.0.1', record=True):
        host = '[' + address + ']' if ':' in address else address
        cmd = [sys.executable, str(ENDPOINT), mode, '--url', f'http://{host}:{port}',
               '--out', str(self.proof)]
        if mode == 'owned' and record:
            cmd += ['--record', str(self.owner)]
        return subprocess.run(cmd, text=True, capture_output=True, timeout=8)

    def start(self, address='127.0.0.1', family=socket.AF_INET, dual=False,
              code=SERVER, port=0):
        argv = self.base / 'argv.json'
        argv.write_text(json.dumps([sys.executable, '-u', '-c', code,
                                   str(family), address, str(port), str(int(dual))]))
        result = subprocess.run([sys.executable, str(MANAGER), 'start', '--record', str(self.owner),
                                 '--evidence', str(self.launch), '--log', str(self.log),
                                 '--argv-json', str(argv)], text=True, capture_output=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            text = self.log.read_text().strip()
            if text:
                return int(text.splitlines()[0])
            time.sleep(0.01)
        self.fail('CPU server did not report a listening port')

    def listener(self, address='127.0.0.1', family=socket.AF_INET, dual=False, port=0):
        sock = socket.socket(family)
        self.sockets.append(sock)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0 if dual else 1)
        sock.bind((address, port))
        sock.listen()
        return sock.getsockname()[1]

    def test_occupied_endpoint_precedes_engine_audit_and_preserves_foreign_listener(self):
        from process_runner_test import ProcessRunnerTests
        fixture = ProcessRunnerTests(methodName='test_term_during_boot_stops_owned_process_and_retains_environment')
        fixture.setUp()
        port = self.listener()
        fixture.port = port
        try:
            fixture.start_runner()
            deadline = time.monotonic() + 8
            while fixture.runner.poll() is None and time.monotonic() < deadline:
                if (fixture.output / 'check-kernels.txt').exists():
                    break
                time.sleep(0.02)
            self.assertFalse((fixture.output / 'check-kernels.txt').exists(),
                             'occupied endpoint reached the engine kernel audit')
            self.assertEqual(fixture.runner.wait(timeout=8), 1, fixture.log.read_text())
            self.assertFalse(list(fixture.output.glob('process-*/owner.json')))
            self.assertEqual(self.sockets[0].getsockname()[1], port)
            artifact = json.loads((fixture.output / 'artifact.json').read_text())
            self.assertEqual(artifact['failing_stage'], 'serve')
            self.assertIn('endpoint', artifact['notes'])
        finally:
            fixture.tearDown()

    def test_real_runner_records_owned_endpoint_after_successful_boot(self):
        from process_runner_test import ProcessRunnerTests, SPARK_SOURCE, running
        fixture = ProcessRunnerTests(methodName='test_term_during_boot_stops_owned_process_and_retains_environment')
        fixture.setUp()
        try:
            body = '{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}'
            response = ('HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(body)) +
                        '\r\nConnection: close\r\n\r\n' + body)
            lines = SPARK_SOURCE.splitlines()
            source = '\n'.join('            const char *reply = ' + json.dumps(response) + ';'
                               if 'const char *reply =' in line else line for line in lines)
            (fixture.base / 'spark.c').write_text(source)
            subprocess.run(['cc', '-O0', str(fixture.base / 'spark.c'), '-o', str(fixture.spark)],
                           check=True, capture_output=True, timeout=20)
            fixture.start_runner()
            self.assertEqual(fixture.runner.wait(timeout=20), 1, fixture.log.read_text())
            proof = json.loads((fixture.output / 'endpoint-owned.json').read_text())
            artifact = json.loads((fixture.output / 'artifact.json').read_text())
            self.assertEqual(proof['status'], 'owned')
            self.assertEqual(artifact['failing_stage'], 'coherency', fixture.log.read_text())
            self.assertFalse(running(proof['pid']))
            self.assertFalse((fixture.output / 'ladder.json').exists())
            print(json.dumps({'oracle': 'real runner post-boot endpoint hook',
                              'endpoint_status': proof['status'], 'owned_pid': proof['pid'],
                              'failing_stage': artifact['failing_stage'],
                              'accepted_socket': proof['accepted_socket']}), flush=True)
        finally:
            fixture.tearDown()

    def test_free_endpoint_passes_then_occupied_listener_refuses_unchanged(self):
        port = self.listener()
        self.assertNotEqual(self.call('free', port).returncode, 0)
        self.assertEqual(self.sockets[0].getsockname()[1], port)
        self.sockets[0].close()
        result = self.call('free', port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.proof.read_text())['status'], 'free')

    def test_owned_ipv4_listener_and_accepted_socket_pass(self):
        port = self.start()
        result = self.call('owned', port)
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads(self.proof.read_text())
        self.assertEqual(proof['status'], 'owned')
        self.assertTrue(proof['listeners'])
        self.assertTrue(proof['accepted_socket']['inode'])
        self.assertEqual(proof['accepted_socket']['pid'], json.loads(self.owner.read_text())['pid'])
        self.assertTrue(proof['network_namespace'].startswith('net:['))

    def test_owned_group_child_can_serve_while_parent_remains_alive(self):
        child = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-u","-c",'
                 + repr(SERVER) + '] + sys.argv[1:]); time.sleep(300)')
        port = self.start(code=child)
        result = self.call('owned', port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(json.loads(self.proof.read_text())['accepted_socket']['pid'],
                            json.loads(self.owner.read_text())['pid'])

    def test_foreign_listener_refuses_even_while_owned_parent_is_alive(self):
        foreign_port = self.listener()
        self.start(code='import time; print(1,flush=True); time.sleep(300)')
        original = self.owner.read_bytes()
        result = self.call('owned', foreign_port)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('owned', result.stderr)
        self.assertEqual(self.owner.read_bytes(), original)
        os.kill(json.loads(original)['pid'], 0)
        self.assertEqual(self.sockets[0].getsockname()[1], foreign_port)

    def test_wildcard_ipv4_ipv6_and_mapped_ipv6_listeners(self):
        for address, family, url_address, dual in (
                ('0.0.0.0', socket.AF_INET, '127.0.0.1', False),
                ('::', socket.AF_INET6, '::1', False),
                ('::', socket.AF_INET6, '127.0.0.1', True),
                ('::ffff:127.0.0.1', socket.AF_INET6, '127.0.0.1', True)):
            with self.subTest(address=address, dual=dual):
                port = self.listener(address, family, dual)
                result = self.call('free', port, url_address)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('occupied', result.stderr)
                self.sockets[-1].close()

    def test_owned_ipv6_and_dual_stack_acceptance(self):
        for address, url_address, dual in (('::1', '::1', False), ('::', '127.0.0.1', True)):
            with self.subTest(address=address):
                port = self.start(address=address, family=socket.AF_INET6, dual=dual)
                result = self.call('owned', port, url_address)
                self.assertEqual(result.returncode, 0, result.stderr)
                subprocess.run([sys.executable, str(MANAGER), 'stop', '--record', str(self.owner),
                                '--timeout', '0.1'], check=True, capture_output=True, timeout=8)
                self.owner.unlink()
                self.log.unlink()

    def test_reuseport_foreign_listener_refuses_all_candidates(self):
        port = self.start()
        self.listener(port=port)
        result = self.call('owned', port)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('owned', result.stderr)
        os.kill(json.loads(self.owner.read_text())['pid'], 0)

    def test_foreign_acceptor_inheriting_owned_listener_is_refused(self):
        acceptor = ("import socket,sys,time; s=socket.socket(fileno=int(sys.argv[1])); "
                    "c,_=s.accept(); time.sleep(300)")
        parent = ("import os,socket,subprocess,sys,time; s=socket.socket(); "
                  "s.bind(('127.0.0.1',0)); s.listen(); "
                  "p=subprocess.Popen([sys.executable,'-c'," + repr(acceptor) +
                  ",str(s.fileno())],pass_fds=(s.fileno(),),start_new_session=True,"
                  "env=dict(os.environ,ATLAS_CAMPAIGN_RUN_TOKEN='foreign-acceptor')); "
                  "print(s.getsockname()[1],flush=True); print(p.pid,flush=True); time.sleep(300)")
        port = self.start(code=parent)
        deadline = time.monotonic() + 3
        while len(self.log.read_text().splitlines()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int(self.log.read_text().splitlines()[1])
        descriptor = os.pidfd_open(pid)
        try:
            result = self.call('owned', port)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('accepted outside the owned group', result.stderr)
            os.kill(pid, 0)  # The probe must not signal the unrelated acceptor.
            os.kill(json.loads(self.owner.read_text())['pid'], 0)
        finally:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            os.close(descriptor)

    def test_unreadable_proc_boundary_overwrites_prior_green_with_refusal(self):
        import process_endpoint
        self.proof.write_text('{"status":"owned"}')
        argv = [str(ENDPOINT), 'free', '--url', 'http://127.0.0.1:12345',
                '--out', str(self.proof)]
        with patch.object(sys, 'argv', argv), patch.object(
                process_endpoint, 'tcp_rows', side_effect=PermissionError('proc table unreadable')):
            self.assertEqual(process_endpoint.main(), 2)
        proof = json.loads(self.proof.read_text())
        self.assertEqual(proof['status'], 'refused')
        self.assertIn('proc table unreadable', proof['error'])

    def test_missing_stale_or_unreadable_owner_proof_cannot_reuse_green(self):
        port = self.start()
        result = self.call('owned', port)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = self.owner.read_bytes()
        bad = json.loads(original)
        bad['start_ticks'] += 1
        for data in (json.dumps(bad).encode(), b'not JSON'):
            self.owner.write_bytes(data)
            result = self.call('owned', port)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(self.proof.read_text())['status'], 'refused')
        self.owner.unlink()
        self.owner.mkdir()  # deterministically unreadable as a record, even as root
        self.assertNotEqual(self.call('owned', port).returncode, 0)
        self.owner.rmdir()
        self.assertNotEqual(self.call('owned', port).returncode, 0)
        self.owner.write_bytes(original)

    def test_non_loopback_or_credentialed_url_is_refused(self):
        for url in ('http://example.com:8000', 'http://192.0.2.1:8000',
                    'http://user:password@127.0.0.1:8000', 'https://127.0.0.1:8000'):
            result = subprocess.run([sys.executable, str(ENDPOINT), 'free', '--url', url,
                                     '--out', str(self.proof)], text=True,
                                    capture_output=True, timeout=8)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(self.proof.read_text())['status'], 'refused')


if __name__ == '__main__':
    unittest.main()
