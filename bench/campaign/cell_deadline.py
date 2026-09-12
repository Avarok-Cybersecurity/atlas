#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bound a Linux runner; engine cleanup remains the runner's responsibility.

Start watch as the runner's direct background child, then wait-armed before
work. Cancel and reap the watchdog only after finalization. Expiry records its
reason before TERM; a runner exceeding cleanup grace is killed through its
pidfd, with cleanup explicitly unconfirmed. No engine/group signals are sent.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import time
import uuid


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("deadline seconds must be finite and positive")
    return number


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def start_ticks(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        raise ProcessLookupError("process has exited")
    return int(fields[19])


def pinned_pidfd(pid, expected):
    if type(pid) is not int or pid <= 1 or type(expected) is not int:
        raise ValueError("invalid process identity")
    descriptor = os.pidfd_open(pid)
    try:
        if start_ticks(pid) != expected:
            raise ValueError("process identity changed; refusing signal")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path, doc, *, exclusive=False):
    doc["updated_utc"] = timestamp()
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(doc, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(doc, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_receipt(path):
    doc = json.loads(path.read_text())
    if doc.get("schema") != 1 or doc.get("kind") != "cell-deadline":
        raise ValueError("invalid deadline receipt")
    if doc.get("boot_id") != boot_id():
        raise ValueError("stale deadline receipt from another boot")
    return doc


def watch(args):
    if args.pid != os.getppid():
        raise ValueError("watch target must be the watchdog's direct parent")
    target_ticks = start_ticks(args.pid)
    descriptor = pinned_pidfd(args.pid, target_ticks)
    cancelled = []

    def cancel_request(signum, frame):
        cancelled.append(signum)

    signal.signal(signal.SIGTERM, cancel_request)
    signal.signal(signal.SIGINT, cancel_request)
    began = time.monotonic()
    deadline = began + args.timeout_s
    doc = {"schema": 1, "kind": "cell-deadline", "status": "armed",
           "target_pid": args.pid, "target_start_ticks": target_ticks,
           "watchdog_pid": os.getpid(), "watchdog_start_ticks": start_ticks(os.getpid()),
           "boot_id": boot_id(), "armed_utc": timestamp(),
           "timeout_s": args.timeout_s, "grace_s": args.grace_s,
           "deadline_exceeded": False, "cleanup_unconfirmed": False}
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    try:
        write_receipt(args.receipt, doc, exclusive=True)
        while True:
            elapsed = time.monotonic() - began
            if cancelled:
                doc.update(status="cancelled", elapsed_s=elapsed)
                write_receipt(args.receipt, doc)
                return 0
            if poller.poll(0):
                doc.update(status="target_exited", elapsed_s=elapsed)
                write_receipt(args.receipt, doc)
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not doc["deadline_exceeded"]:
                    doc.update(status="deadline_exceeded", deadline_exceeded=True,
                               elapsed_s=elapsed, deadline_utc=timestamp())
                    write_receipt(args.receipt, doc)
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                    deadline = time.monotonic() + args.grace_s
                else:
                    doc.update(status="grace_exceeded", cleanup_unconfirmed=True,
                               elapsed_s=elapsed,
                               detail="runner did not exit within cleanup grace; artifact and engine cleanup may be incomplete")
                    write_receipt(args.receipt, doc)
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    return 2
            poller.poll(max(1, min(50, int(max(0, deadline - time.monotonic()) * 1000))))
    except ProcessLookupError:
        doc.update(status="target_exited", elapsed_s=time.monotonic() - began)
        write_receipt(args.receipt, doc)
        return 0
    finally:
        os.close(descriptor)


def wait_armed(args):
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        try:
            doc = read_receipt(args.receipt)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
            continue
        if doc.get("target_pid") != os.getppid():
            raise ValueError("deadline receipt does not identify this runner")
        if doc.get("target_start_ticks") != start_ticks(os.getppid()):
            raise ValueError("deadline receipt identifies another runner start")
        if doc.get("status") != "armed":
            raise ValueError("watchdog is not armed: " + str(doc.get("status")))
        descriptor = pinned_pidfd(doc.get("watchdog_pid"), doc.get("watchdog_start_ticks"))
        os.close(descriptor)
        return 0
    raise ValueError("watchdog did not arm before startup deadline")


def cancel(args):
    unarmed = False
    try:
        doc = read_receipt(args.receipt)
    except FileNotFoundError:
        if not args.watchdog_pid:
            raise
        # Arming can fail before a receipt exists. Only the calling runner's
        # direct child executing this exact watch command may be cancelled.
        unarmed = True
        doc = {"target_pid": os.getppid(), "target_start_ticks": start_ticks(os.getppid()),
               "watchdog_pid": args.watchdog_pid,
               "watchdog_start_ticks": start_ticks(args.watchdog_pid)}
    if args.watchdog_pid and doc.get("watchdog_pid") != args.watchdog_pid:
        raise ValueError("deadline receipt names another watchdog")
    try:
        descriptor = pinned_pidfd(doc.get("watchdog_pid"), doc.get("watchdog_start_ticks"))
    except (ProcessLookupError, FileNotFoundError):
        if doc.get("status") in ("cancelled", "target_exited", "grace_exceeded"):
            return 0 if doc.get("status") != "grace_exceeded" else 2
        raise ValueError("watchdog exited without a terminal receipt")
    try:
        target = pinned_pidfd(doc.get("target_pid"), doc.get("target_start_ticks"))
        os.close(target)
        proc = Path(f"/proc/{doc['watchdog_pid']}")
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        argv = [part.decode() for part in argv]
        if (int(stat[1]) != doc["target_pid"] or len(argv) != 11
                or Path(argv[1]).resolve() != Path(__file__).resolve() or argv[2] != "watch"):
            raise ValueError("process is not the recorded runner's deadline watchdog")
        options = dict(zip(argv[3::2], argv[4::2]))
        if (set(options) != {"--pid", "--timeout-s", "--grace-s", "--receipt"}
                or options["--pid"] != str(doc["target_pid"])
                or Path(options["--receipt"]).resolve() != args.receipt.resolve()):
            raise ValueError("watchdog command does not own this deadline receipt")
        if not unarmed and (float(options["--timeout-s"]) != doc.get("timeout_s")
                            or float(options["--grace-s"]) != doc.get("grace_s")):
            raise ValueError("watchdog budget differs from its receipt")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        if not poller.poll(2000):
            raise ValueError("watchdog did not acknowledge cancellation")
    finally:
        os.close(descriptor)
    if unarmed and not args.receipt.exists():
        return 0
    after = read_receipt(args.receipt)
    if after.get("status") not in ("cancelled", "target_exited"):
        raise ValueError("watchdog cleanup failed: " + str(after.get("status")))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    watching = sub.add_parser("watch")
    watching.add_argument("--pid", type=int, required=True)
    watching.add_argument("--timeout-s", type=positive, required=True)
    watching.add_argument("--grace-s", type=positive, required=True)
    waiting = sub.add_parser("wait-armed")
    waiting.add_argument("--timeout-s", type=positive, required=True)
    stopping = sub.add_parser("cancel")
    stopping.add_argument("--watchdog-pid", type=int,
                          help="runner's recorded child PID, also permits cancellation before arming")
    for command in (watching, waiting, stopping):
        command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
            raise ValueError("cell deadline requires Linux pidfds")
        return {"watch": watch, "wait-armed": wait_armed, "cancel": cancel}[args.operation](args)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("cell deadline refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
