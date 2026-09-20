# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded single-host Atlas rank canary. All ranks are stopped after the probe."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from launch_process import SystemIO, terminate, wait_alive
from probe_payload import read_payload, validate_prompt, validate_payload

FIELDS = set("schema binary binary_sha256 compiled_arch model_dir model_name world_size "
             "tp_size ep_size ranks master_addr master_port port_base env max_seq_len "
             "max_batch_size max_num_seqs gpu_memory_utilization boot_timeout probe_timeout "
             "cleanup_timeout prompt expected_prefix max_tokens max_prefill_tokens "
             "kv_cache_dtype enable_prefix_caching".split())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def validate(c):
    if set(c) not in (FIELDS, (FIELDS-{'prompt'}) | {'request_file'}) or type(c['schema']) is not int or c['schema'] != 2:
        raise ValueError(f'manifest fields must be exactly {sorted(FIELDS)}; schema=2')
    for field in ('world_size', 'tp_size', 'ep_size', 'master_port', 'port_base',
                  'max_seq_len', 'max_batch_size', 'max_num_seqs', 'max_tokens'):
        if type(c[field]) is not int or c[field] <= 0:
            raise ValueError(f'{field} must be a positive integer')
    if type(c['max_prefill_tokens']) is not int or c['max_prefill_tokens'] < 0:
        raise ValueError('max_prefill_tokens must be a nonnegative integer')
    if c['kv_cache_dtype'] not in ('bf16', 'fp8', 'nvfp4'):
        raise ValueError('kv_cache_dtype must be bf16, fp8 or nvfp4')
    if type(c['enable_prefix_caching']) is not bool:
        raise ValueError('enable_prefix_caching must be a boolean')
    w, tp, ep = c['world_size'], c['tp_size'], c['ep_size']
    if not (w == tp*ep or w == tp == ep):
        raise ValueError('world_size must equal tp_size*ep_size or tp_size=ep_size')
    if not isinstance(c['ranks'], list) or len(c['ranks']) != w:
        raise ValueError('all local ranks must be explicitly configured')
    for rank in c['ranks']:
        if set(rank) != {'rank', 'device', 'uuid'}:
            raise ValueError('rank requires rank, device, uuid')
        if any(type(rank[k]) is not int or rank[k] < 0 for k in ('rank', 'device')):
            raise ValueError('rank/device must be nonnegative integers')
        if not isinstance(rank['uuid'], str) or not rank['uuid'].startswith('GPU-'):
            raise ValueError('each device needs its NVIDIA GPU UUID')
    if sorted(r['rank'] for r in c['ranks']) != list(range(w)):
        raise ValueError('rank IDs must cover 0..world_size-1 exactly once')
    if len({r['device'] for r in c['ranks']}) != w or len({r['uuid'] for r in c['ranks']}) != w:
        raise ValueError('one distinct GPU per rank is required')
    if c['master_addr'] != '127.0.0.1':
        raise ValueError('this controller supports a single local host only')
    ports = [c['master_port']] + list(range(c['port_base'], c['port_base']+w))
    if max(ports) > 65535 or len(set(ports)) != len(ports):
        raise ValueError('rank/API and bootstrap ports must be distinct valid ports')
    for field in ('boot_timeout', 'probe_timeout', 'cleanup_timeout'):
        if not isinstance(c[field], (int, float)) or not math.isfinite(c[field]) or c[field] <= 0:
            raise ValueError(f'{field} must be positive and finite')
    if not isinstance(c['gpu_memory_utilization'], (float, int)) or not 0 < c['gpu_memory_utilization'] < 1:
        raise ValueError('gpu_memory_utilization must be between zero and one')
    for field in ('model_name', 'expected_prefix'):
        if not isinstance(c[field], str) or not c[field].strip():
            raise ValueError(f'{field} must be nonempty text')
    if not re.fullmatch(r'sm_[0-9]{2,3}[af]?', c['compiled_arch']):
        raise ValueError('compiled_arch must be the architecture from the build receipt')
    binary = Path(c['binary'])
    model = Path(c['model_dir'])
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('binary must be an absolute executable file')
    if not re.fullmatch(r'[0-9a-f]{64}', c['binary_sha256']) or digest(binary) != c['binary_sha256']:
        raise ValueError('binary SHA256 differs from build receipt')
    if not model.is_absolute() or not (model/'config.json').is_file():
        raise ValueError('model_dir must contain config.json')
    if not isinstance(json.loads((model/'config.json').read_text()), dict):
        raise ValueError('model config must be an object')
    environment(c)
    probe_payload(c)
    return c


def probe_payload(c):
    if 'request_file' in c:
        path = c['request_file']
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError('request_file must be an absolute path')
        payload = read_payload(path)
        if payload['model'] != c['model_name'] or payload['max_tokens'] != c['max_tokens']:
            raise ValueError('prepared request model/max_tokens differs from manifest')
        return payload
    return validate_payload(dict(model=c['model_name'], prompt=validate_prompt(c['prompt']),
                                 max_tokens=c['max_tokens'], temperature=0, stream=False))


def environment(c):
    if not isinstance(c['env'], dict):
        raise ValueError('env must be an explicit object')
    result = {'PATH': os.defpath, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
    for key, value in c['env'].items():
        if (not re.fullmatch(r'(NCCL_[A-Z0-9_]+|AVAROK_[A-Z0-9_]+|RUST_LOG|LD_LIBRARY_PATH|CUDA_CACHE_PATH|K3_ALLOW_MXFP4|K3_CUDA_KDA|K3_CUDA_MLA|K3_CUDA_DENSE)', key)
                or re.search(r'TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY', key)
                or not isinstance(value, str) or '\0' in value):
            raise ValueError(f'environment key/value not allowed: {key}')
        if key.startswith('K3_') and value not in ('0', '1'):
            raise ValueError(f'{key} requires explicit 0 or 1')
        if key == 'CUDA_CACHE_PATH':
            cache = Path(value)
            if not cache.is_absolute() or not cache.is_dir() or not os.access(cache, os.W_OK | os.X_OK):
                raise ValueError('CUDA_CACHE_PATH must be an absolute writable existing directory')
        result[key] = value
    return result


def command(c, rank):
    pairs = {'model-from-path': c['model_dir'], 'model-name': c['model_name'],
             'bind': '127.0.0.1', 'port': c['port_base']+rank['rank'],
             'rank': rank['rank'], 'world-size': c['world_size'],
             'tp-size': c['tp_size'], 'ep-size': c['ep_size'],
             'gpu-ordinal': rank['device'], 'master-addr': c['master_addr'],
             'master-port': c['master_port'], 'max-seq-len': c['max_seq_len'],
             'max-batch-size': c['max_batch_size'],
             'max-num-seqs': c['max_num_seqs'],
             'max-prefill-tokens': c['max_prefill_tokens'],
             'kv-cache-dtype': c['kv_cache_dtype'],
             'enable-prefix-caching': str(c['enable_prefix_caching']).lower(),
             'gpu-memory-utilization': c['gpu_memory_utilization']}
    argv = [c['binary'], 'serve']
    for key, value in pairs.items():
        argv.extend([f'--{key}', str(value)])
    return argv


def free_ports(ports):
    held = []
    try:
        for port in ports:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            held.append(sock)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise ValueError(f'port {port} is occupied or unavailable') from exc
    finally:
        for sock in held:
            sock.close()


def check_devices(c, rows, occupied):
    expected = re.fullmatch(r'sm_([0-9]+)[af]?', c['compiled_arch']).group(1)
    expected_cc = f'{int(expected)//10}.{int(expected)%10}'
    for rank in c['ranks']:
        matches = [row for row in rows if row['index'] == rank['device']]
        if len(matches) != 1 or matches[0]['uuid'] != rank['uuid']:
            raise ValueError(f'device ordinal/UUID mismatch for rank {rank["rank"]}')
        if matches[0]['compute_cap'] != expected_cc:
            raise ValueError(f'compiled architecture/device CC mismatch for rank {rank["rank"]}')
        if rank['uuid'] in occupied:
            raise ValueError(f'GPU for rank {rank["rank"]} already has a compute process')


def run(c, output, io=None):
    io = io or SystemIO()
    validate(c)
    payload = probe_payload(c)
    output.mkdir(parents=True, exist_ok=False)
    request_file = output/'probe-request.json'
    request_file.write_text(json.dumps(payload, allow_nan=False)+'\n')
    processes, logs = [], []
    leader = None
    summary = {'schema': 1, 'status': 'failed', 'manifest': c, 'ranks': []}
    started = time.monotonic()
    try:
        io.admit_host()
        rows, occupied, topology = io.inventory()
        summary['devices'] = rows
        (output/'topology.txt').write_text(topology)
        check_devices(c, rows, occupied)
        free_ports([c['master_port']] + list(range(c['port_base'], c['port_base']+c['world_size'])))
        for rank in sorted(c['ranks'], key=lambda r: (r['rank'] == 0, r['rank'])):
            argv = command(c, rank)
            log = (output/f'rank-{rank["rank"]}.log').open('xb')
            logs.append(log)
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=environment(c),
                                       start_new_session=True)
            processes.append(process)
            if rank['rank'] == 0:
                leader = process
            summary['ranks'].append({'rank': rank['rank'], 'pid': process.pid, 'argv': argv})
        deadline = time.monotonic() + c['boot_timeout']
        endpoint = f'http://127.0.0.1:{c["port_base"]}'
        while True:
            wait_alive(processes)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('boot deadline exceeded')
            if io.healthy(endpoint, min(remaining, 0.5)):
                if not io.owns_endpoint(c['port_base'], processes):
                    raise ValueError('ready endpoint is not owned by this rank launch')
                break
            time.sleep(min(0.1, remaining))
        probe_cmd = [sys.executable, str(Path(__file__).with_name('probe.py')),
                     '--endpoint', endpoint, '--model', c['model_name'],
                     '--request-file', str(request_file), '--expected-prefix', c['expected_prefix'],
                     '--max-tokens', str(c['max_tokens']), '--deadline', str(c['probe_timeout']),
                     '--output', str(output/'probe.json')]
        log = (output/'probe.log').open('xb')
        logs.append(log)
        probe = subprocess.Popen(probe_cmd, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        processes.append(probe)
        deadline = time.monotonic() + c['probe_timeout'] + 2
        while probe.poll() is None:
            wait_alive(processes[:-1])
            if time.monotonic() >= deadline:
                raise TimeoutError('generation probe deadline exceeded')
            time.sleep(0.05)
        if probe.returncode != 0:
            raise ValueError('real-generation probe failed; inspect probe.json/probe.log')
        wait_alive(processes[:-1])
        if not io.owns_endpoint(c['port_base'], processes[:-1]):
            raise ValueError('endpoint ownership changed during the probe')
        summary['status'] = 'passed'
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        summary['error'] = str(exc) or 'interrupted'
    finally:
        failures = terminate(processes, c['cleanup_timeout'], leader=leader)
        if failures:
            summary['status'] = 'failed'
            summary['cleanup_errors'] = failures
        summary['process_exit_codes'] = {str(p.pid): p.poll() for p in processes}
        summary['elapsed_seconds'] = time.monotonic()-started
        for log in logs:
            log.close()
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return 0 if summary['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    c = validate(json.loads(args.manifest.read_text()))
    if args.dry_run:
        print(json.dumps({'validated': True, 'hardware_checked': False,
                          'commands': [command(c, rank) for rank in c['ranks']]}, indent=2))
        return 0
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    return run(c, args.output)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'launch refused: {error}', file=sys.stderr)
        sys.exit(1)
