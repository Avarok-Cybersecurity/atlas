#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Retain model identity fields from Docker inspect without Config.Env secrets."""

import argparse
import json
import pathlib
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--container-id", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    try:
        objects = json.load(sys.stdin)
        if not isinstance(objects, list) or len(objects) != 1:
            raise ValueError("expected one Docker inspect object")
        observed = objects[0]
        if not isinstance(observed, dict) or observed.get("Id") != args.container_id:
            raise ValueError("inspect identity differs from the created container")
        config, state = observed.get("Config"), observed.get("State")
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError("inspect has no Config or State object")
        key, separator, value = args.label.partition("=")
        labels = config.get("Labels")
        if (not separator or not key or not value or not isinstance(labels, dict)
                or labels.get(key) != value):
            raise ValueError("inspect ownership differs from this run")
        projection = [{
            "Id": observed["Id"],
            "Config": {"Entrypoint": config.get("Entrypoint"), "Cmd": config.get("Cmd"),
                       "Labels": {key: labels[key]}},
            "State": {field: state.get(field) for field in ("Running", "Pid", "StartedAt")},
        }]
        destination = pathlib.Path(args.out)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(projection, indent=2) + "\n")
        temporary.replace(destination)
    except (ValueError, TypeError, OSError) as error:
        print(f"model launch capture unavailable: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
