# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import launch
from launch_process import SystemIO, listener_inodes, terminate, parse_gpu_inventory


class LocalIO(SystemIO):
    """Only GPU/procfs observations are replaced; rank/HTTP/probe run for real."""
    def admit_host(self):
        pass

    def inventory(self):
        return ([{'index': i, 'uuid': f'GPU-test{i}', 'compute_cap': '10.3'}
                 for i in range(2)], [], 'fake GPU inventory, not hardware evidence')

    def owns_endpoint(self, port, processes):
        return all(p.poll() is None for p in processes)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binary = self.root / "spark"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o700)
        (self.root / "config.json").write_text('{}')
        self.config = dict(schema=2, binary=str(self.binary),
            binary_sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            compiled_arch="sm_103a", model_dir=str(self.root), model_name="twin",
            world_size=1, tp_size=1, ep_size=1,
            ranks=[dict(rank=0, device=0, uuid="GPU-test")], master_addr="127.0.0.1",
            master_port=29500, port_base=18888, env={}, max_seq_len=1024,
            max_batch_size=1, max_num_seqs=1, gpu_memory_utilization=0.8,
            max_prefill_tokens=32, kv_cache_dtype="bf16", enable_prefix_caching=False,
            boot_timeout=5, probe_timeout=3, cleanup_timeout=1,
            prompt="2+2=", expected_prefix="4", max_tokens=4)

    def test_valid_and_invalid_manifest(self):
        launch.validate(self.config)
        for key, value in [("world_size", 2), ("binary_sha256", "a"*64),
                           ("model_dir", str(self.root/"absent")),
                           ("boot_timeout", float("nan")),
                           ("env", {"HF_TOKEN": "not-a-real-token"})]:
            bad = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate(bad)

    def test_explicit_prefill_and_cache_settings_are_forwarded(self):
        for dtype in ('bf16', 'fp8', 'nvfp4'):
            for enabled in (False, True):
                for cap in (0, 32):
                    config = dict(self.config, kv_cache_dtype=dtype,
                                  enable_prefix_caching=enabled, max_prefill_tokens=cap)
                    launch.validate(config)
                    argv = launch.command(config, config['ranks'][0])
                    for flag, expected in [('--max-prefill-tokens', str(cap)),
                                           ('--kv-cache-dtype', dtype),
                                           ('--enable-prefix-caching', str(enabled).lower())]:
                        self.assertEqual(argv[argv.index(flag)+1], expected)
        for field, value in [('schema', 1), ('max_prefill_tokens', -1),
                             ('max_prefill_tokens', True), ('kv_cache_dtype', 'auto'),
                             ('enable_prefix_caching', 'false')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.validate(dict(self.config, **{field:value}))
        old = dict(self.config)
        del old['max_prefill_tokens']
        with self.assertRaises(ValueError):
            launch.validate(old)

    def test_ports_are_checked(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            with self.assertRaises(ValueError):
                launch.free_ports([occupied.getsockname()[1]])

    def test_ipv4_ownership_does_not_require_optional_ipv6_table(self):
        net = self.root/'net'
        net.mkdir()
        with self.assertRaises(FileNotFoundError):
            listener_inodes(18888, net)
        (net/'tcp').write_text('header\n0: 0100007F:49C8 00000000:0000 0A 0 0 0 0 0 123\n')
        self.assertEqual(listener_inodes(18888, net), {'123'})
        (net/'tcp6').mkdir()
        with self.assertRaises(IsADirectoryError):
            listener_inodes(18888, net)

    def test_arch_and_uuid_must_match(self):
        rows = [{"index": 0, "uuid": "GPU-test", "compute_cap": "10.3"}]
        launch.check_devices(self.config, rows, [])
        for field, value in [("compute_cap", "10.0"), ("uuid", "GPU-other")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.check_devices(self.config, [dict(rows[0], **{field: value})], [])
        with self.assertRaises(ValueError):
            launch.check_devices(self.config, rows, ["GPU-test"])

    def test_command_pins_rank_without_masking_devices(self):
        command = launch.command(self.config, self.config["ranks"][0])
        self.assertEqual(command[0], str(self.binary))
        self.assertEqual(command[command.index("--gpu-ordinal")+1], "0")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", launch.environment(self.config))
        self.config['env'] = {'K3_ALLOW_MXFP4': '1', 'K3_CUDA_KDA': '0', 'K3_CUDA_MLA': '1'}
        self.assertEqual(launch.environment(self.config)['K3_ALLOW_MXFP4'], '1')
        self.config['env']['K3_CUDA_KDA'] = 'maybe'
        with self.assertRaises(ValueError):
            launch.environment(self.config)

    def configure_process(self, source):
        self.binary.write_text(f'#!{sys.executable}\n'+source)
        self.config['binary_sha256'] = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.config['ranks'][0]['uuid'] = 'GPU-test0'
        # Find a free adjacent port pair for API/worker, distinct from bootstrap.
        for _ in range(100):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            try:
                launch.free_ports([port, port+1, self.config['master_port']])
                self.config['port_base'] = port
                break
            except ValueError:
                continue
        else:
            self.fail('could not reserve local test port range')

    def test_real_http_probe_and_teardown(self):
        self.configure_process('''
import http.server, json, sys
port = int(sys.argv[sys.argv.index('--port')+1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        data = json.dumps({'model':'twin','choices':[{'text':'4','finish_reason':'stop'}],
                           'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', port), Handler).serve_forever()
''')
        output = self.root/'passed'
        self.assertEqual(launch.run(self.config, output, LocalIO()), 0)
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['status'], 'passed')
        self.assertTrue(all(v is not None for v in summary['process_exit_codes'].values()))
        self.assertEqual(json.loads((output/'probe.json').read_text())['status'], 'passed')
        with socket.socket() as sock:
            self.assertNotEqual(sock.connect_ex(('127.0.0.1', self.config['port_base'])), 0)

    def test_dead_worker_stops_head_and_retains_failure(self):
        self.configure_process('''
import sys, time
if sys.argv[sys.argv.index('--rank')+1] == '1': raise SystemExit(7)
time.sleep(60)
''')
        self.config.update(world_size=2, tp_size=2)
        self.config['ranks'].append(dict(rank=1, device=1, uuid='GPU-test1'))
        output = self.root/'failed'
        self.assertEqual(launch.run(self.config, output, LocalIO()), 1)
        summary = json.loads((output/'summary.json').read_text())
        self.assertIn('exited with 7', summary['error'])
        self.assertTrue(all(v is not None for v in summary['process_exit_codes'].values()))
        self.assertFalse((output/'probe.json').exists())

    def test_boot_timeout_stops_only_owned_process(self):
        self.configure_process('import time; time.sleep(60)\n')
        self.config['boot_timeout'] = 0.15
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                     start_new_session=True)
        try:
            output = self.root/'timedout'
            self.assertEqual(launch.run(self.config, output, LocalIO()), 1)
            summary = json.loads((output/'summary.json').read_text())
            self.assertIn('boot deadline', summary['error'])
            self.assertIsNone(unrelated.poll())
            self.assertTrue(all(v is not None for v in summary['process_exit_codes'].values()))
        finally:
            terminate([unrelated], 1)

    def test_timeout_kills_descendant_after_leader_exits(self):
        marker = self.root/'descendant-survived'
        child = f"import time; from pathlib import Path; time.sleep(0.6); Path({str(marker)!r}).write_text('bad')"
        self.configure_process(f'import subprocess, sys, time\n'
                               f'subprocess.Popen([sys.executable, "-c", {child!r}])\n'
                               'time.sleep(60)\n')
        self.config['boot_timeout'] = 0.15
        self.config['cleanup_timeout'] = 0.1
        self.assertEqual(launch.run(self.config, self.root/'descendant', LocalIO()), 1)
        time.sleep(0.65)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()

class InventoryTests(unittest.TestCase):
    def test_spark_unified_memory_is_unknown_not_zero(self):
        rows=parse_gpu_inventory("0, GPU-test, 12.1, [N/A]\n")
        self.assertIsNone(rows[0]['memory_mib'])
        self.assertEqual(parse_gpu_inventory("0, GPU-test, 10.3, 288000\n")[0]['memory_mib'],288000)
