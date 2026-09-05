#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Start, capture and stop a campaign-owned Linux server process group."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import uuid

from process_launch_proc import (ENV_KEYS, TOKEN_KEY, atomic_json, capture,
                                 boot_id, process_stat, read_owner, require_linux,
                                 snapshot, stop, timestamp)


def launch_environment(args):
    supplied = {key: os.environ[key] for key in
                ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH")
                if key in os.environ}
    if args.env_json:
        explicit = json.loads(Path(args.env_json).read_text())
        if not isinstance(explicit, dict):
            raise ValueError("environment JSON must contain an explicit key/value map")
        supplied.update(explicit)
    for key in args.env:
        if key not in os.environ:
            raise ValueError("requested environment key is missing: " + key)
        if key in supplied and supplied[key] != os.environ[key]:
            raise ValueError("conflicting environment values for " + key)
        supplied[key] = os.environ[key]
    for key, value in supplied.items():
        if key not in ENV_KEYS:
            raise ValueError("environment key is not allowed: " + key)
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("environment value must be a NUL-free string: " + key)
    return supplied


def launch_argv(args):
    if args.argv_json:
        argv = json.loads(Path(args.argv_json).read_text())
    else:
        data = Path(args.argv_nul).read_bytes()
        if not data.endswith(b"\0"):
            raise ValueError("argv NUL file must end in NUL")
        argv = [item.decode() for item in data[:-1].split(b"\0")]
    if (not isinstance(argv, list) or not argv or
            any(not isinstance(item, str) or "\0" in item for item in argv) or not argv[0]):
        raise ValueError("argv must be a nonempty array of NUL-free strings")
    return argv


def _start(args):
    argv = launch_argv(args)
    environment = launch_environment(args)
    marker = uuid.uuid4().hex
    environment[TOKEN_KEY] = marker
    created_at = timestamp()
    record = Path(args.record)
    descriptor = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"schema": 1, "kind": "linux-proc-owner", "status": "starting",
                   "run_marker": marker}, stream)
    process = None
    emergency_owner = None
    try:
        with Path(args.log).open("ab", buffering=0) as log:
            process = subprocess.Popen(argv, cwd=args.cwd, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        initial = process_stat(process.pid)
        emergency_owner = dict(initial, boot_id=boot_id(), run_marker=marker)
        # exec has completed when Popen returns. This catches immediate refusals.
        time.sleep(0.05)
        if process.poll() is not None:
            raise ValueError("server exited during launch: " + str(process.returncode))
        proof = snapshot(process.pid, environment)
        if proof["run_marker"] != marker:
            raise ValueError("new process did not retain its ownership marker")
        owner = {key: proof[key] for key in ("schema", "pid", "start_ticks", "boot_id",
                 "run_marker", "pgid", "sid", "executable", "executable_sha256",
                 "argv", "environment")}
        owner.update(kind="linux-proc-owner", created_at=created_at)
        atomic_json(record, owner)
        atomic_json(args.evidence, proof)
        return proof
    except Exception:
        if emergency_owner is not None:
            stop(emergency_owner, 0.2)
        # Reap the direct child after its group was stopped.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        atomic_json(record, {"schema": 1, "kind": "linux-proc-owner",
                            "status": "failed", "run_marker": marker})
        raise


def start(args):
    # A server starts in a new session, so terminating this short-lived manager
    # must not leave a child behind before its owner record has been published.
    interrupted = []

    def defer(signum, _frame):
        interrupted.append(signum)

    previous = {number: signal.signal(number, defer)
                for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        result = _start(args)
        if interrupted:
            error = "launch interrupted by " + signal.Signals(interrupted[0]).name
            stop(read_owner(args.record), 0.2)
            atomic_json(args.evidence, {"schema": 1, "kind": "linux-proc",
                                       "running": False, "error": error})
            raise ValueError(error)
        return result
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="operation", required=True)
    launch = sub.add_parser("start")
    launch.add_argument("--record", required=True)
    launch.add_argument("--evidence", required=True)
    launch.add_argument("--log", required=True)
    argv = launch.add_mutually_exclusive_group(required=True)
    argv.add_argument("--argv-json")
    argv.add_argument("--argv-nul")
    launch.add_argument("--env-json")
    launch.add_argument("--env", action="append", default=[])
    launch.add_argument("--cwd")
    inspect = sub.add_parser("capture")
    inspect.add_argument("--record", required=True)
    inspect.add_argument("--evidence", required=True)
    shutdown = sub.add_parser("stop")
    shutdown.add_argument("--record", required=True)
    shutdown.add_argument("--timeout", type=float, default=15)
    return cli


def main():
    args = parser().parse_args()
    try:
        require_linux()
        if args.operation == "start":
            result = start(args)
        elif args.operation == "capture":
            result = capture(read_owner(args.record))
            atomic_json(args.evidence, result)
        else:
            if not 0 <= args.timeout <= 120:
                raise ValueError("stop timeout must be between 0 and 120 seconds")
            result = stop(read_owner(args.record), args.timeout)
        print(json.dumps(result))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as error:
        if args.operation == "capture":
            atomic_json(args.evidence, {"schema": 1, "kind": "linux-proc",
                                       "running": False, "error": str(error)})
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
