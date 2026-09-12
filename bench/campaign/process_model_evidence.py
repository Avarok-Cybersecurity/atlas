#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify owned Linux process evidence without consulting recipe revisions."""

import datetime
import hashlib
import json
import math
import pathlib
import re
import uuid

from model_launch_evidence import command_revision


def read_record(path, kind):
    if not path:
        raise ValueError(f"missing {kind} evidence")
    try:
        raw = pathlib.Path(path).read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {kind} evidence: {error}") from error
    if not isinstance(record, dict) or record.get("schema") != 1 or record.get("kind") != kind:
        raise ValueError(f"expected a schema 1 {kind} object")
    return record, hashlib.sha256(raw).hexdigest()


def timestamp(value, field):
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.year < 1970:
            raise ValueError("unset timestamp")
        return parsed.timestamp()
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"process evidence has no usable {field}") from error


def normalized_command(argv, executable, engine):
    """Interpret supported actual exec forms; Docker's argv contract stays separate."""
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(token, str) and token for token in argv)):
        raise ValueError("Linux /proc cmdline must be a nonempty argv array")
    if not isinstance(executable, str) or not executable.startswith("/"):
        raise ValueError("Linux /proc executable must be an absolute path")
    binary = pathlib.PurePosixPath(executable).name
    first = pathlib.PurePosixPath(argv[0]).name
    if engine == "atlas":
        if binary != "spark" or first != "spark":
            raise ValueError("actual Atlas process executable and argv must name spark")
        return argv
    if binary == "vllm" and first == "vllm":
        return argv
    python_name = r"python(?:[0-9]+(?:\.[0-9]+)*)?"
    if not re.fullmatch(python_name, binary) or not re.fullmatch(python_name, first):
        raise ValueError("actual vLLM process is neither vllm nor its Python interpreter")
    if len(argv) > 2 and argv[1] == "-m" and argv[2] == "vllm.entrypoints.cli.main":
        return ["vllm"] + argv[3:]
    if (len(argv) > 1 and argv[1].startswith("/")
            and pathlib.PurePosixPath(argv[1]).name == "vllm"):
        return ["vllm"] + argv[2:]
    raise ValueError("actual Python process does not invoke the supported vLLM CLI")


def launched_process_revision(path, owner_path, *, engine, hf_id, boot):
    """Return a launched model pin only after an owned process and matching boot."""
    observed, digest = read_record(path, "linux-proc")
    owner, owner_digest = read_record(owner_path, "linux-proc-owner")
    for key in ("pid", "start_ticks", "pgid", "sid"):
        if type(owner.get(key)) is not int or owner[key] <= 0:
            raise ValueError(f"process owner has no positive integer {key}")
    try:
        if not isinstance(owner.get("boot_id"), str) or not uuid.UUID(owner["boot_id"]).int:
            raise ValueError("empty boot UUID")
    except (ValueError, AttributeError) as error:
        raise ValueError("process owner has no usable Linux boot_id") from error
    marker = owner.get("run_marker")
    if not isinstance(marker, str) or not marker or any(c.isspace() for c in marker):
        raise ValueError("process owner has no unique run marker")
    environment = observed.get("environment")
    if not isinstance(environment, dict) or environment.get("ATLAS_CAMPAIGN_RUN_TOKEN") != marker:
        raise ValueError("actual process environment does not carry this run's marker")
    for key in ("pid", "start_ticks", "boot_id", "pgid", "sid", "run_marker", "argv",
                "executable", "executable_sha256", "environment"):
        if type(observed.get(key)) is not type(owner.get(key)) or observed.get(key) != owner.get(key):
            raise ValueError(f"actual process {key} differs from the owned launch record")
    if observed.get("running") is not True:
        raise ValueError("actual process was not running at capture")
    for key in ("pid", "start_ticks", "boot_id"):
        if observed.get("captured_" + key) != owner[key]:
            raise ValueError(f"process {key} changed during capture")
    binary_hash = observed.get("executable_sha256")
    if not isinstance(binary_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", binary_hash):
        raise ValueError("process evidence has no complete executable SHA256")
    created = timestamp(owner.get("created_at"), "created_at")
    captured = timestamp(observed.get("captured_at"), "captured_at")
    if captured < created:
        raise ValueError("process snapshot predates its owned launch record")
    argv = normalized_command(observed.get("argv"), observed.get("executable"), engine)
    revision = command_revision(argv, engine, hf_id)
    provenance = (f"actual model launch evidence {path} sha256={digest}; "
                  f"owner {owner_path} sha256={owner_digest}; owned Linux /proc executable, "
                  "cmdline, PID/start_ticks/boot_id and run marker; "
                  "model launch identity, not weight bytes or an engine build identity")
    if not isinstance(boot, dict) or boot.get("passed") is not True:
        return None, "model.revision is null: the boot gate did not pass; " + provenance
    if boot.get("engine") != engine or boot.get("model") not in (hf_id, argv[2]):
        raise ValueError("boot evidence does not identify this engine and model")
    for key in ("start_epoch", "total_s"):
        value = boot.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"boot evidence has no usable {key} for process freshness")
    if captured + 1 < boot["start_epoch"] + boot["total_s"]:
        raise ValueError("process snapshot predates successful boot completion")
    if revision is None:
        return None, "model.revision is null: actual launch used no immutable model pin; " + provenance
    return revision, provenance
