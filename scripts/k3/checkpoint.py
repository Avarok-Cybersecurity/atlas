# SPDX-License-Identifier: AGPL-3.0-only
"""Pin, stage, and verify a checkpoint before spending inference GPU time.

Only `manifest` reads remote metadata; only `download` transfers model files.
The huggingface_hub dependency is loaded only for those operations. Credentials
use its normal environment/login mechanisms and are never written to receipts.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time


def validate(doc):
    if doc.get("schema") != 1:
        raise ValueError("unsupported manifest schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", doc.get("repo", "")):
        raise ValueError("repo must be owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", doc.get("revision", "")):
        raise ValueError("revision must be an immutable 40-character commit SHA")
    if not isinstance(doc.get("files"), list) or not doc["files"]:
        raise ValueError("manifest must contain files")
    seen = set()
    for item in doc["files"]:
        name = item["path"]
        path = PurePosixPath(name)
        if (not name or not path.parts or path.is_absolute() or ".." in path.parts or
                "\\" in name or str(path) != name or name in seen or
                path.parts[0] in (".cache", ".k3-pin.json", ".k3-download.lock")):
            raise ValueError(f"unsafe or duplicate file path: {name}")
        seen.add(name)
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError(f"invalid file size: {name}")
        width = {"sha256": 64, "git-sha1": 40}.get(item["algorithm"])
        if not width or not re.fullmatch(rf"[0-9a-f]{{{width}}}", item["digest"]):
            raise ValueError(f"invalid digest: {name}")
    return doc


def safe_path(root, name):
    root = root.resolve()
    candidate = root / name
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"snapshot contains a symlink: {name}")
        cursor = cursor.parent
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"path escapes snapshot: {name}")
    return candidate


def check_file(root, item):
    path = safe_path(root, item["path"])
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
    if item["algorithm"] == "git-sha1":
        digest.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == item["digest"]


def metadata(repo, revision):
    # Validate the pin before any network call; mutable refs are not accepted.
    validate({"schema": 1, "repo": repo, "revision": revision,
              "files": [{"path": "probe", "size": 0, "algorithm": "sha256",
                         "digest": "0" * 64}]})
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=60)
    if info.sha != revision:
        raise ValueError("remote resolved revision differs from requested pin")
    files = []
    for entry in info.siblings:
        lfs = entry.lfs
        files.append({"path": entry.rfilename, "size": entry.size,
                      "algorithm": "sha256" if lfs else "git-sha1",
                      "digest": lfs.sha256 if lfs else entry.blob_id})
    return validate({"schema": 1, "repo": repo, "revision": revision,
                     "files": sorted(files, key=lambda item: item["path"])})


def bounded_run(argv, seconds):
    """Own a new process group; a timeout terminates only that group."""
    if os.name != "posix":
        raise ValueError("bounded download requires a POSIX process group")
    with subprocess.Popen(argv, start_new_session=True) as process:
        try:
            return process.wait(timeout=seconds)
        except BaseException:
            # Even if the leader just exited, workers may still hold the group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


@contextmanager
def snapshot_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock = safe_path(root, ".k3-download.lock")
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another download owns this snapshot") from error
        yield


def download(doc, root, reserve, attempts, timeout, total_timeout):
    validate(doc)
    with snapshot_lock(root):
        _download(doc, root, reserve, attempts, timeout, total_timeout)


def _download(doc, root, reserve, attempts, timeout, total_timeout):
    if (min(reserve, attempts, timeout, total_timeout) <= 0 or
            not math.isfinite(timeout) or not math.isfinite(total_timeout)):
        raise ValueError("reserve, attempts, and deadlines must be positive")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    pin = root / ".k3-pin.json"
    identity = {"repo": doc["repo"], "revision": doc["revision"]}
    if pin.is_symlink():
        raise ValueError("pin file is a symlink")
    if pin.exists():
        if json.loads(pin.read_text()) != identity:
            raise ValueError("snapshot belongs to another checkpoint revision")
    else:
        # Never overwrite another owner's pin, even during concurrent startup.
        with pin.open("x") as destination:
            json.dump(identity, destination)
    started = time.monotonic()
    pending = [item for item in doc["files"] if not check_file(root, item)]
    needed = sum(item["size"] for item in pending)
    if shutil.disk_usage(root).free < needed + reserve:
        raise ValueError(f"insufficient free disk: need {needed} download bytes plus {reserve} reserve")
    for item in pending:
        if time.monotonic() - started >= total_timeout:
            raise TimeoutError("total download deadline expired")
        path = safe_path(root, item["path"])
        # A full extra file plus caller's reserve is conservative even when HF
        # has an existing partial transfer. Never delete a mismatching file.
        if shutil.disk_usage(root).free < item["size"] + reserve:
            raise ValueError(f"insufficient free disk plus reserve for {item['path']}")
        for attempt in range(attempts):
            remaining = total_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("total download deadline expired")
            argv = [sys.executable, str(Path(__file__).resolve()), "_fetch",
                    "--repo", doc["repo"], "--revision", doc["revision"],
                    "--root", str(root), "--file", item["path"]]
            if path.exists():
                # HF's own completion metadata must not bless corrupted bytes.
                argv.append("--force")
            try:
                result = bounded_run(argv, min(timeout, remaining))
            except subprocess.TimeoutExpired:
                result = 124
            if result == 0 and check_file(root, item):
                print(f"verified {item['path']}", flush=True)
                break
            print(f"attempt {attempt + 1}/{attempts} failed: {item['path']}", file=sys.stderr)
        else:
            raise ValueError(f"download or verification failed: {item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = commands.add_parser("manifest", help="metadata only; JSON on stdout")
    manifest_parser.add_argument("--repo", required=True)
    manifest_parser.add_argument("--revision", required=True)
    for command in ("verify", "download"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--root", type=Path, required=True)
        if command == "download":
            sub.add_argument("--reserve-bytes", type=int, required=True)
            sub.add_argument("--attempts", type=int, required=True)
            sub.add_argument("--file-timeout", type=float, required=True)
            sub.add_argument("--total-timeout", type=float, required=True)
    fetch = commands.add_parser("_fetch", help=argparse.SUPPRESS)
    for field in ("repo", "revision", "root", "file"):
        fetch.add_argument(f"--{field}", required=True)
    fetch.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "manifest":
        print(json.dumps(metadata(args.repo, args.revision), indent=2))
    elif args.command == "_fetch":
        from huggingface_hub import hf_hub_download
        hf_hub_download(args.repo, args.file, revision=args.revision,
                        local_dir=args.root, force_download=args.force)
    else:
        doc = validate(json.loads(args.manifest.read_text()))
        if args.command == "download":
            download(doc, args.root, args.reserve_bytes, args.attempts,
                     args.file_timeout, args.total_timeout)
        else:
            bad = [item["path"] for item in doc["files"] if not check_file(args.root, item)]
            print(json.dumps({"repo": doc["repo"], "revision": doc["revision"],
                              "verified": not bad, "missing_or_corrupt": bad,
                              "total_bytes": sum(item["size"] for item in doc["files"])}))
            return 1 if bad else 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, TimeoutError, ImportError) as error:
        print(f"checkpoint: {error}", file=sys.stderr)
        sys.exit(1)
