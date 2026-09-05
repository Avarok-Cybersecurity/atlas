#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Linux /proc boundary for campaign-owned processes."""
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

TOKEN_KEY = "ATLAS_CAMPAIGN_RUN_TOKEN"
ENV_KEYS = frozenset("""
PATH HOME LANG LC_ALL LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES
HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE RUST_LOG OMP_NUM_THREADS SPT_NOENV
ATLAS_DECODE_TIMING ATLAS_DENSE_FP8
GLOO_SOCKET_IFNAME NCCL_CUMEM_ENABLE NCCL_SOCKET_IFNAME PYTORCH_CUDA_ALLOC_CONF
VLLM_ALLREDUCE_USE_FLASHINFER VLLM_ENGINE_READY_TIMEOUT_S
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS VLLM_FLASHINFER_ALLREDUCE_BACKEND
VLLM_FLOAT32_MATMUL_PRECISION VLLM_PLE_CPU_OFFLOAD VLLM_USE_DEEP_GEMM
VLLM_USE_RUST_FRONTEND VLLM_USE_V2_MODEL_RUNNER
""".split())


def require_linux():
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        raise ValueError("owned process launch requires Linux /proc and pidfds")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def process_stat(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": fields[0], "pgid": int(fields[2]),
            "sid": int(fields[3]), "start_ticks": int(fields[19])}


def process_environment(pid):
    fields = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return dict(part.decode(errors="surrogateescape").split("=", 1)
                for part in fields if b"=" in part)


def marker_matches(pid, marker):
    return process_environment(pid).get(TOKEN_KEY) == marker


def snapshot(pid, environment_keys):
    before = process_stat(pid)
    if before["state"] in ("Z", "X"):
        raise ValueError("owned process is no longer running")
    proc = Path(f"/proc/{pid}")
    executable = os.readlink(proc / "exe")
    if not executable.startswith("/") or executable.endswith(" (deleted)"):
        raise ValueError("process executable is not a stable absolute path")
    digest = hashlib.sha256()
    with (proc / "exe").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    command = (proc / "cmdline").read_bytes()
    if not command.endswith(b"\0"):
        raise ValueError("process command line is missing its NUL terminator")
    argv = [part.decode(errors="surrogateescape") for part in command[:-1].split(b"\0")]
    environment = process_environment(pid)
    selected = {key: environment[key] for key in environment_keys if key in environment}
    current = process_stat(pid)
    if (any(current[key] != before[key] for key in ("pid", "start_ticks", "pgid", "sid"))
            or current["state"] in ("Z", "X") or not argv):
        raise ValueError("process changed while capturing launch evidence")
    return {"schema": 1, "kind": "linux-proc", "pid": pid,
            "start_ticks": before["start_ticks"], "pgid": before["pgid"],
            "sid": before["sid"], "boot_id": boot_id(),
            "run_marker": environment.get(TOKEN_KEY), "executable": executable,
            "executable_sha256": digest.hexdigest(), "argv": argv,
            "environment": selected, "running": True,
            "captured_pid": pid, "captured_start_ticks": before["start_ticks"],
            "captured_boot_id": boot_id(), "captured_at": timestamp()}


def read_owner(path):
    owner = json.loads(Path(path).read_text())
    if owner.get("schema") != 1 or owner.get("kind") != "linux-proc-owner":
        raise ValueError("invalid process ownership record")
    for key in ("pid", "start_ticks", "pgid", "sid"):
        if type(owner.get(key)) is not int or owner[key] <= 0:
            raise ValueError("invalid process ownership field: " + key)
    if owner["pid"] != owner["pgid"] or owner["pid"] != owner["sid"]:
        raise ValueError("owned process must lead its isolated process group and session")
    if owner.get("boot_id") != boot_id() or not owner.get("run_marker"):
        raise ValueError("stale boot or missing process ownership marker")
    environment = owner.get("environment")
    if not isinstance(environment, dict) or set(environment) - ENV_KEYS - {TOKEN_KEY}:
        raise ValueError("invalid environment evidence in process ownership record")
    if environment.get(TOKEN_KEY) != owner["run_marker"]:
        raise ValueError("process ownership marker differs from recorded environment")
    return owner


def capture(owner):
    proof = snapshot(owner["pid"], owner["environment"])
    for key in ("pid", "start_ticks", "boot_id", "run_marker", "pgid", "sid",
                "argv", "executable", "executable_sha256", "environment"):
        if proof[key] != owner.get(key):
            raise ValueError("process ownership or launch changed: " + key)
    return proof


def owned_members(owner):
    # Refuse a reused leader PID before looking for descendants.
    try:
        leader = process_stat(owner["pid"])
        if leader["start_ticks"] != owner["start_ticks"]:
            raise ValueError("stale process ownership: leader PID was reused")
        if leader["state"] not in ("Z", "X") and not marker_matches(
                owner["pid"], owner["run_marker"]):
            raise ValueError("foreign process ownership marker")
    except (FileNotFoundError, ProcessLookupError):
        pass
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = process_stat(int(entry.name))
            if stat["pgid"] != owner["pgid"] or stat["state"] in ("Z", "X"):
                continue
            if (stat["sid"] != owner["sid"] or
                    stat["start_ticks"] < owner["start_ticks"] or
                    not marker_matches(stat["pid"], owner["run_marker"])):
                raise ValueError("foreign member in owned process group; refusing signals")
            members.append(stat)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def signal_member(member, owner, signum):
    try:
        descriptor = os.pidfd_open(member["pid"])
    except ProcessLookupError:
        return
    try:
        current = process_stat(member["pid"])
        if current["state"] in ("Z", "X"):
            return
        if any(current[key] != member[key] for key in ("pid", "start_ticks", "pgid", "sid")):
            raise ValueError("process changed before signal; refusing reused PID")
        if not marker_matches(member["pid"], owner["run_marker"]):
            raise ValueError("foreign process marker before signal")
        signal.pidfd_send_signal(descriptor, signum)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(descriptor)


def stop(owner, timeout):
    deadline = time.monotonic() + timeout
    final_deadline = deadline + 2
    signalled = set()
    while True:
        members = owned_members(owner)
        if not members:
            return {"status": "stopped", "pid": owner["pid"],
                    "signalled_pids": sorted(signalled)}
        now = time.monotonic()
        if now >= final_deadline:
            raise ValueError("owned process group did not exit after SIGKILL")
        signum = signal.SIGTERM if now < deadline else signal.SIGKILL
        # Signal children first; pidfds pin the exact processes across PID reuse.
        for member in sorted(members, key=lambda m: m["pid"] == owner["pid"]):
            signal_member(member, owner, signum)
            signalled.add(member["pid"])
        time.sleep(0.03)
