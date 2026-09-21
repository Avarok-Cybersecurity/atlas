# SPDX-License-Identifier: AGPL-3.0-only
"""System observations and process-group lifecycle for the rental launcher."""
import csv
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_alive(processes):
    for process in processes:
        if process.poll() is not None:
            raise ValueError(f'owned rank PID {process.pid} exited with {process.returncode}')


def listener_inodes(port, net_root):
    listeners = set()
    for table in ('tcp', 'tcp6'):
        try:
            rows = (net_root/table).read_text().splitlines()[1:]
        except FileNotFoundError:
            if table == 'tcp6':
                continue  # Linux without the IPv6 module has no optional table.
            raise
        for line in rows:
            fields = line.split()
            if int(fields[1].split(':')[1], 16) == port and fields[3] == '0A':
                listeners.add(fields[9])
    return listeners


def terminate(processes, grace, leader=None):
    errors = []
    end = time.monotonic() + grace

    def send(group, sig):
        for process in group:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'PID {process.pid}: {error}')

    def wait_until(deadline):
        while time.monotonic() < deadline and any(p.poll() is None for p in processes):
            time.sleep(0.02)

    # Workers must remain alive to receive the leader's shutdown collective.
    # Reserve half the one shared grace period for fallback group termination.
    if leader is not None and leader.poll() is None:
        send([leader], signal.SIGTERM)
        wait_until(time.monotonic() + max(0, end-time.monotonic())/2)
    send(processes, signal.SIGTERM)
    wait_until(end)
    # Signal groups even if their original parent exited: descendants may remain.
    send(processes, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            errors.append(f'PID {process.pid}: exit not confirmed')
    return errors


def parse_gpu_inventory(raw):
    rows = []
    for values in csv.reader(raw.splitlines()):
        if len(values) != 4:
            raise ValueError('unrecognized nvidia-smi inventory')
        index, uuid, cc, memory = [v.strip() for v in values]
        rows.append({'index': int(index), 'uuid': uuid, 'compute_cap': cc,
                     'memory_mib': None if memory in ('[N/A]', 'N/A') else float(memory)})
    return rows


class SystemIO:
    def admit_host(self):
        if sys.platform != 'linux' or not Path('/proc/net/tcp').is_file():
            raise ValueError('live rank launch requires Linux procfs endpoint ownership')

    def inventory(self):
        def query(args):
            return subprocess.run(['nvidia-smi', *args], check=True, capture_output=True,
                                  text=True, timeout=15).stdout
        raw = query(['--query-gpu=index,uuid,compute_cap,memory.total', '--format=csv,noheader,nounits'])
        rows = parse_gpu_inventory(raw)
        occupied = query(['--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'])
        occupied = [value.strip() for value in occupied.splitlines() if value.strip()]
        if any(not re.fullmatch(r'GPU-[0-9A-Fa-f-]+', value) for value in occupied):
            raise ValueError('unrecognized GPU process inventory; occupancy is unknown')
        return rows, occupied, query(['topo', '-m'])

    def healthy(self, endpoint, timeout):
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     '_health', endpoint], timeout=timeout,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def owns_endpoint(self, port, processes):
        groups = {p.pid for p in processes if p.poll() is None}
        sockets = set()
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) not in groups:
                    continue
                for fd in (proc/'fd').iterdir():
                    link = os.readlink(fd)
                    if link.startswith('socket:['):
                        sockets.add(link[8:-1])
            except (FileNotFoundError, ProcessLookupError):
                continue
        listeners = listener_inodes(port, Path('/proc/net'))
        return bool(listeners) and listeners <= sockets


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '_health':
        sys.exit(2)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(sys.argv[2]+'/health', timeout=1) as response:
            sys.exit(0 if response.status == 200 else 1)
    except (OSError, urllib.error.URLError):
        sys.exit(1)
