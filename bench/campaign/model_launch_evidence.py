#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Read a model pin from an owned, running container's actual launch command.

Recipe and dry-render data are not accepted here. The caller supplies the ID
and unique run label returned by its own create, plus selected actual Docker
inspect fields captured before teardown: ID, exact argv, ownership label and
runtime state. Other fields, including environment values, are unnecessary.
A successful boot is also required before the launched pin is usable. This
records launch identity, not a weight hash.
"""

import datetime
import hashlib
import json
import pathlib
import re

REVISION = re.compile(r"[0-9a-f]{40}")


def command_revision(argv, engine, hf_id):
    executable = "spark" if engine == "atlas" else "vllm"
    if (len(argv) < 3 or pathlib.PurePosixPath(argv[0]).name != executable
            or argv[1] != "serve"):
        raise ValueError(f"actual command is not {executable} serve MODEL")
    model = argv[2]
    snapshot = re.fullmatch(r"/.*/models--([^/]+)/snapshots/([0-9a-f]{40})/?", model)
    snapshot_revision = None
    if snapshot and snapshot[1] == hf_id.replace("/", "--"):
        snapshot_revision = snapshot[2]
    elif model != hf_id:
        raise ValueError("actual model differs from the cell's HF repository")

    revisions = []
    for index, token in enumerate(argv[3:], 3):
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0].replace("_", "-")
        if flag == "--revision":
            value = token.split("=", 1)[1] if "=" in token else (
                argv[index + 1] if index + 1 < len(argv) else "")
            revisions.append(value)
        elif any(protected.startswith(flag) or flag == protected
                 for protected in ("--revision", "--model", "--tokenizer",
                                   "--tokenizer-revision", "--code-revision", "--config")):
            raise ValueError(f"actual command carries competing identity option {flag}")
    if revisions and (len(revisions) != 1 or not REVISION.fullmatch(revisions[0])):
        raise ValueError("actual command must carry one full --revision SHA")
    if engine == "atlas" and revisions:
        raise ValueError("Atlas has no --revision option; use an explicit HF snapshot path")
    if snapshot_revision and revisions and revisions[0] != snapshot_revision:
        raise ValueError("actual snapshot path and --revision disagree")
    return snapshot_revision or (revisions[0] if revisions else None)


def launched_revision(path, *, engine, hf_id, container_id, run_label, boot):
    """Return (revision-or-null, provenance note); reject contradictory evidence."""
    if not path:
        return None, "model.revision is null: no actual model launch evidence was captured"
    try:
        raw = pathlib.Path(path).read_bytes()
        objects = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read Docker inspect evidence: {error}") from error
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError("Docker inspect evidence must contain exactly one container object")
    observed = objects[0]
    if not container_id or observed.get("Id") != container_id:
        raise ValueError("Docker inspect ID does not match this run's created container ID")
    config = observed.get("Config")
    if not isinstance(config, dict):
        raise ValueError("Docker inspect has no launch Config object")
    labels = config.get("Labels")
    key, separator, value = (run_label or "").partition("=")
    if (not key or not separator or not value or not isinstance(labels, dict)
            or labels.get(key) != value):
        raise ValueError("Docker inspect label does not match this run's ownership label")
    state = observed.get("State")
    if (not isinstance(state, dict) or state.get("Running") is not True
            or type(state.get("Pid")) is not int or state["Pid"] <= 0):
        raise ValueError("Docker inspect does not record a running container process")
    try:
        started = datetime.datetime.fromisoformat(state["StartedAt"].replace("Z", "+00:00"))
        if started.tzinfo is None or started.year < 1970:
            raise ValueError("unset StartedAt")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("Docker inspect does not record a usable StartedAt") from error
    entrypoint, command = config.get("Entrypoint") or [], config.get("Cmd")
    if (not isinstance(entrypoint, list) or not isinstance(command, list)
            or not all(isinstance(token, str) for token in entrypoint + command)):
        raise ValueError("Docker Config Entrypoint/Cmd must be argv arrays")
    revision = command_revision(entrypoint + command, engine, hf_id)
    provenance = (f"actual model launch evidence {path} sha256={hashlib.sha256(raw).hexdigest()}; "
                  "owned Docker Config argv, not recipe data or a weight-file hash")
    if not isinstance(boot, dict) or boot.get("passed") is not True:
        return None, "model.revision is null: the boot gate did not pass; " + provenance
    if boot.get("engine") != engine or boot.get("model") not in (hf_id, (entrypoint + command)[2]):
        raise ValueError("boot evidence does not identify this engine and model")
    if revision is None:
        return None, "model.revision is null: actual launch used no immutable model pin; " + provenance
    return revision, provenance
