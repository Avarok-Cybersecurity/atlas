#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Prove a local process endpoint using current-netns TCP inodes and owned FDs.

/proc/PID/net describes a network namespace, NOT sockets owned by that PID.
Read /proc/self/net and correlate inodes with actual /proc/PID/fd links, using
process_launch_proc's ownership policy. tcp6 prints four native-endian u32s:
https://docs.kernel.org/networking/proc_net_tcp.html
https://man7.org/linux/man-pages/man5/proc_pid_net.5.html
https://github.com/torvalds/linux/blob/v6.17/net/ipv6/tcp_ipv6.c#L2076-L2092

Admission is a snapshot, not a port reservation. The owned check also connects
and proves which group accepted that connection. Neither proves every future
connection; rerun after readiness and immediately before measured requests.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit

from process_launch_proc import (atomic_json, capture, owned_members, process_stat,
                                 read_owner, require_linux, timestamp)


def normalized(address):
    ip = ipaddress.ip_address(address)
    return ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip


def endpoint(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.username is not None or parsed.password is not None
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment
            or not parsed.hostname or '%' in parsed.hostname):
        raise ValueError('endpoint must be a plain numeric loopback HTTP URL')
    address = normalized(parsed.hostname)
    if not address.is_loopback or parsed.port is None or not 1 <= parsed.port <= 65535:
        raise ValueError('endpoint requires an explicit port and numeric loopback address')
    return address, parsed.port


def decode_address(value):
    encoded, port = value.split(':')
    if len(encoded) not in (8, 32):
        raise ValueError('invalid address in kernel TCP table')
    raw = b''.join(int(encoded[i:i + 8], 16).to_bytes(4, sys.byteorder)
                   for i in range(0, len(encoded), 8))
    return normalized(ipaddress.ip_address(raw)), int(port, 16)


def tcp_rows():
    rows = []
    for table in ('tcp', 'tcp6'):
        # Missing or unreadable tables cannot establish that a port is free.
        lines = (Path('/proc/self/net') / table).read_text().splitlines()
        if not lines or 'local_address' not in lines[0] or 'inode' not in lines[0]:
            raise ValueError('unrecognized kernel TCP table: ' + table)
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10:
                raise ValueError('incomplete kernel TCP table: ' + table)
            local, port = decode_address(fields[1])
            remote, remote_port = decode_address(fields[2])
            rows.append({'address': str(local), 'port': port, 'remote': str(remote),
                         'remote_port': remote_port, 'state': fields[3],
                         'inode': int(fields[9]), 'table': table})
    return rows


def listeners(rows, address, port):
    found = []
    for row in rows:
        if row['state'] != '0A' or row['port'] != port:
            continue
        bound = normalized(row['address'])
        # An IPv6 wildcard may accept IPv4 too. proc does not expose V6ONLY;
        # conservatively include it, even if a separate IPv6-only service
        # could coexist. An ambiguous endpoint is not positive ownership proof.
        if bound == address or (bound.is_unspecified and
                                (bound.version == address.version or bound.version == 6)):
            if row['inode'] <= 0:
                raise ValueError('listener has no verifiable socket inode')
            found.append(row)
    return found


def namespace(pid='self'):
    return os.readlink(f'/proc/{pid}/ns/net')


def group_sockets(owner, netns):
    sockets = {}
    for member in owned_members(owner):
        pid = member['pid']
        if namespace(pid) != netns:
            raise ValueError('owned process group is in a different network namespace')
        directory = Path(f'/proc/{pid}/fd')
        try:
            for fd in directory.iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue  # A concurrently closed FD cannot contribute proof.
                if target.startswith('socket:[') and target.endswith(']'):
                    inode = int(target[8:-1])
                    sockets.setdefault(inode, []).append({
                        'pid': pid, 'start_ticks': member['start_ticks'], 'fd': int(fd.name)})
            after = process_stat(pid)
        except (FileNotFoundError, ProcessLookupError):
            raise ValueError('owned process disappeared during endpoint proof') from None
        if (any(after[key] != member[key] for key in ('pid', 'start_ticks', 'pgid', 'sid'))
                or after['state'] in ('Z', 'X') or namespace(pid) != netns):
            raise ValueError('owned process changed during endpoint proof')
    return sockets


def require_owned_listeners(rows, address, port, sockets):
    found = listeners(rows, address, port)
    if not found:
        raise ValueError('endpoint has no listener to prove owned')
    if any(row['inode'] not in sockets for row in found):
        raise ValueError('endpoint listener is not in the verified owned process group')
    return found


def prove_owned(address, port, record, netns):
    owner = read_owner(record)
    capture(owner)  # Exact original launch, environment, boot ID and live leader.
    before = require_owned_listeners(tcp_rows(), address, port, group_sockets(owner, netns))
    # The accepted server socket proves who handles this connection, including
    # when a listener FD was inherited outside the owned group. No HTTP request
    # is sent, and closing the probe connection releases it immediately.
    with socket.create_connection((str(address), port), timeout=2) as connection:
        client_address, client_port = connection.getsockname()[:2]
        client_address = normalized(client_address)
        until = time.monotonic() + 2
        accepted = None
        while time.monotonic() < until:
            rows = tcp_rows()
            sockets = group_sockets(owner, netns)
            current = require_owned_listeners(rows, address, port, sockets)
            if {r['inode'] for r in current} != {r['inode'] for r in before}:
                raise ValueError('endpoint listeners changed during ownership proof')
            established = [r for r in rows if r['state'] == '01' and
                           normalized(r['address']) == address and r['port'] == port and
                           normalized(r['remote']) == client_address and
                           r['remote_port'] == client_port]
            for row in established:
                if row['inode'] > 0:
                    if row['inode'] not in sockets:
                        raise ValueError('endpoint connection was accepted outside the owned group')
                    accepted = dict(row, **sockets[row['inode']][0])
            if accepted:
                break
            time.sleep(0.02)
        if accepted is None:
            raise ValueError('accepted endpoint socket ownership could not be observed')
        capture(owner)
        final_sockets = group_sockets(owner, netns)
        final = require_owned_listeners(tcp_rows(), address, port, final_sockets)
        if (namespace() != netns or accepted['inode'] not in final_sockets or
                {r['inode'] for r in final} != {r['inode'] for r in before}):
            raise ValueError('endpoint ownership changed before proof completed')
    return {'status': 'owned', 'pid': owner['pid'], 'start_ticks': owner['start_ticks'],
            'boot_id': owner['boot_id'], 'run_marker': owner['run_marker'],
            'pgid': owner['pgid'], 'sid': owner['sid'], 'listeners': before,
            'accepted_socket': accepted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('free', 'owned'))
    parser.add_argument('--url', required=True)
    parser.add_argument('--record')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = {'schema': 1, 'kind': 'linux-proc-endpoint', 'operation': args.operation,
              'checked_at': timestamp(), 'status': 'refused'}
    try:
        require_linux()
        address, port = endpoint(args.url)
        netns = namespace()
        result.update(address=str(address), port=port, network_namespace=netns)
        if args.operation == 'free':
            if listeners(tcp_rows(), address, port):
                raise ValueError('endpoint is occupied before engine launch')
            if namespace() != netns:
                raise ValueError('network namespace changed during admission')
            result.update(status='free', listeners=[])
        else:
            if not args.record:
                raise ValueError('owned endpoint proof requires the process owner record')
            result.update(prove_owned(address, port, args.record, netns))
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        result.update(status='refused', error=str(error))
    # Replace a previous green result on every refusal; callers must also check
    # this invocation's exit status. An unwritable output itself exits nonzero.
    atomic_json(args.out, result)
    print(json.dumps(result), file=sys.stdout if result['status'] != 'refused' else sys.stderr)
    return 2 if result['status'] == 'refused' else 0


if __name__ == '__main__':
    sys.exit(main())
