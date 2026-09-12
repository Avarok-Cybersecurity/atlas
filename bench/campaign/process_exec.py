#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Exec an audit with the same validated environment used for process serving."""

import argparse
import os
import sys

from process_launch import launch_argv, launch_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--argv-nul", required=True)
    parser.add_argument("--env-json", required=True)
    args = parser.parse_args()
    args.argv_json = None
    args.env = []
    try:
        argv = launch_argv(args)
        environment = launch_environment(args)
        os.execvpe(argv[0], argv, environment)
    except (OSError, ValueError) as error:
        print(f"audit exec refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
