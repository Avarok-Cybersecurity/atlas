#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare AVAROK_COMM_DIAGNOSTICS host submissions. Never proves completion."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FIELD = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^\s]+)')
MARKER = "NCCL host submission"
WIDTH = {"Bf16": 2, "U8": 1, "F32": 4}
OPS = {"all_reduce", "all_reduce_async", "all_gather", "reduce_scatter", "broadcast", "barrier", "send", "recv"}


@dataclass(frozen=True)
class Submission:
    rank: int
    world: int
    sequence: int
    op: str
    dtype: str
    count: int
    peer: int | None

    def signature(self):
        return self.op, self.dtype, self.count, self.peer


def integer(value, field):
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"invalid {field}: {value!r}")
    return int(value)


def parse_line(line):
    """Return None for ordinary logs; malformed diagnostic events fail closed."""
    line = ANSI.sub("", line).strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if MARKER in line or "avarok::comm" in line:
                raise ValueError(f"truncated/invalid diagnostic JSON: {exc}") from exc
            return None
        fields = event.get("fields", event)
        if not isinstance(fields, dict) or MARKER not in str(fields.get("message", "")):
            return None
    else:
        if MARKER not in line:
            return None
        fields = {}
        for key, value in FIELD.findall(line):
            if key in fields:
                raise ValueError(f"duplicate diagnostic field {key}")
            fields[key] = json.loads(value) if value.startswith('"') else value
    required = {"rank", "world_size", "sequence", "op", "dtype", "count", "bytes", "peer", "phase"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"truncated diagnostic event: missing {sorted(missing)}")
    if fields["phase"] != "submit" or fields["op"] not in OPS or fields["dtype"] not in WIDTH:
        raise ValueError("unrecognized submission phase, operation or dtype")
    rank, world, seq, count, size = [integer(fields[k], k) for k in ("rank", "world_size", "sequence", "count", "bytes")]
    if world == 0 or rank >= world or size != count * WIDTH[fields["dtype"]]:
        raise ValueError("invalid rank/world or dtype/count/bytes contract")
    peer_value = fields["peer"]
    if peer_value is None or peer_value == "None":
        peer = None
    else:
        match = re.fullmatch(r"Some\((\d+)\)", str(peer_value))
        peer = integer(match[1] if match else peer_value, "peer")
        if peer >= world:
            raise ValueError(f"peer/root {peer} outside world {world}")
    needs_peer = fields["op"] in {"broadcast", "send", "recv"}
    if needs_peer != (peer is not None):
        raise ValueError(f"unexpected/missing peer/root for {fields['op']}")
    return Submission(rank, world, seq, fields["op"], fields["dtype"], count, peer)


def read_rank(path, rank, world, expected=None):
    rows = []
    with Path(path).open(encoding="utf-8", errors="strict") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = parse_line(line)
                if row is None:
                    continue
                if row.rank != rank or row.world != world:
                    raise ValueError(f"event rank/world {row.rank}/{row.world} differs from declared {rank}/{world}")
                if row.sequence != len(rows):
                    raise ValueError(f"expected sequence={len(rows)}, found {row.sequence} (missing, duplicated or reordered event)")
                rows.append(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"rank {rank}: no submission diagnostics in {path}")
    if expected is not None and len(rows) != expected:
        raise ValueError(f"rank {rank}: expected {expected} submissions, found {len(rows)} (possibly truncated)")
    return rows


def compare(logs, world, expected=None):
    if world < 1 or set(logs) != set(range(world)):
        raise ValueError(f"need exactly rank logs 0..{world - 1}; received {sorted(logs)}")
    ranks = {rank: read_rank(logs[rank], rank, world, expected) for rank in range(world)}
    collective = {rank: [r for r in rows if r.op not in {"send", "recv"}] for rank, rows in ranks.items()}
    reference = collective[0]
    for rank in range(1, world):
        other = collective[rank]
        for index, (a, b) in enumerate(zip(reference, other)):
            if a.signature() != b.signature():
                raise ValueError(f"collective {index}: rank 0 sequence {a.sequence} {a.signature()} != rank {rank} sequence {b.sequence} {b.signature()}")
        if len(other) != len(reference):
            raise ValueError(f"rank {rank}: {len(other)} collectives vs rank 0's {len(reference)} (truncated or divergent log)")
    sends, receives = defaultdict(list), defaultdict(list)
    for rank, rows in ranks.items():
        for row in rows:
            if row.op == "send":
                sends[(rank, row.peer)].append(row)
            elif row.op == "recv":
                receives[(row.peer, rank)].append(row)
    for channel in sorted(sends.keys() | receives.keys()):
        sent, received = sends[channel], receives[channel]
        if len(sent) != len(received):
            raise ValueError(f"peer channel {channel[0]}->{channel[1]}: {len(sent)} sends vs {len(received)} receives (missing/truncated peer)")
        for index, (a, b) in enumerate(zip(sent, received)):
            if (a.dtype, a.count) != (b.dtype, b.count):
                raise ValueError(f"peer channel {channel[0]}->{channel[1]} transfer {index}: send sequence {a.sequence} {a.dtype}/{a.count} != receive sequence {b.sequence} {b.dtype}/{b.count}")
    return {"status": "MATCH_SUBMISSIONS", "world_size": world,
            "submissions_per_rank": {str(rank): len(rows) for rank, rows in ranks.items()},
            "collectives_per_rank": len(reference), "peer_transfers": sum(map(len, sends.values())),
            "expected_submissions_per_rank": expected,
            "limitation": "Submission agreement only; not GPU completion or deadlock freedom. Equal truncation is undetectable without an external expected count; graph replays are not traced."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--rank-log", required=True, action="append", metavar="RANK=PATH")
    parser.add_argument("--expected-submissions", type=int, help="Known per-rank submission count from the harness; detects equally truncated tails")
    args = parser.parse_args()
    try:
        logs = {}
        for assignment in args.rank_log:
            rank_text, path = assignment.split("=", 1)
            rank = integer(rank_text, "rank")
            if rank in logs:
                raise ValueError(f"duplicate --rank-log for rank {rank}")
            logs[rank] = path
        if args.expected_submissions is not None and args.expected_submissions < 1:
            raise ValueError("--expected-submissions must be positive")
        print(json.dumps(compare(logs, args.world_size, args.expected_submissions), indent=2))
        return 0
    except (ValueError, OSError, UnicodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
