#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path
import tempfile
import unittest

from check_collectives import compare, parse_line


def event(rank, seq, op="all_reduce", count=8, peer=None, dtype="Bf16", json_mode=False):
    fields = dict(message="NCCL host submission (not completion)", rank=rank, world_size=2,
                  sequence=seq, op=op, dtype=dtype, count=count,
                  bytes=count * {"Bf16": 2, "U8": 1, "F32": 4}[dtype],
                  stream=100 + rank, peer="None" if peer is None else f"Some({peer})", phase="submit")
    if json_mode:
        return json.dumps(dict(target="avarok::comm", fields=fields)) + "\n"
    return "2026-09-19T00:00:00Z \x1b[32mINFO\x1b[0m avarok::comm: " + fields.pop("message") + " " + " ".join(f"{k}={json.dumps(v) if k in {'op', 'phase'} else v}" for k, v in fields.items()) + "\n"


class CheckCollectivesTests(unittest.TestCase):
    def check(self, first, second, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            logs = {rank: Path(directory) / f"rank{rank}.log" for rank in range(2)}
            logs[0].write_text(first)
            logs[1].write_text(second)
            return compare(logs, 2, expected)

    def test_ansi_text_and_tracing_json_match_ignoring_stream(self):
        result = self.check("ordinary startup log\n" + event(0, 0), event(1, 0, json_mode=True))
        self.assertEqual(result["status"], "MATCH_SUBMISSIONS")
        self.assertEqual(result["submissions_per_rank"], {"0": 1, "1": 1})

    def test_deliberate_count_mismatch_names_rank_and_sequence(self):
        with self.assertRaisesRegex(ValueError, r"rank 1 sequence 0"):
            self.check(event(0, 0), event(1, 0, count=9))

    def test_truncated_tail_and_interior_gap_fail(self):
        with self.assertRaisesRegex(ValueError, "truncated or divergent"):
            self.check(event(0, 0) + event(0, 1), event(1, 0))
        with self.assertRaisesRegex(ValueError, "expected sequence=1, found 2"):
            self.check(event(0, 0) + event(0, 2), event(1, 0))

    def test_equal_truncation_requires_expected_count(self):
        with self.assertRaisesRegex(ValueError, "expected 2 submissions, found 1"):
            self.check(event(0, 0), event(1, 0), expected=2)

    def test_empty_duplicate_or_wrong_rank_logs_fail(self):
        with self.assertRaisesRegex(ValueError, "no submission diagnostics"):
            self.check(event(0, 0), "startup only\n")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.check(event(0, 0) * 2, event(1, 0))
        with self.assertRaisesRegex(ValueError, "differs from declared"):
            self.check(event(0, 0), event(0, 0))
        with self.assertRaisesRegex(ValueError, "exactly rank logs"):
            compare({}, 2)

    def test_truncated_event_and_bad_payload_contract_fail(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_line(event(0, 0).split("count=")[0])
        with self.assertRaisesRegex(ValueError, "count/bytes"):
            parse_line(event(0, 0).replace("bytes=16", "bytes=15"))
        with self.assertRaisesRegex(ValueError, "invalid diagnostic JSON"):
            parse_line(event(0, 0, json_mode=True).strip()[:-2])

    def test_broadcast_root_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "collective 0"):
            self.check(event(0, 0, "broadcast", peer=0, dtype="U8"),
                       event(1, 0, "broadcast", peer=1, dtype="U8"))

    def test_send_and_receive_are_paired_not_compared_as_collectives(self):
        result = self.check(event(0, 0, "send", peer=1, dtype="U8") + event(0, 1),
                            event(1, 0, "recv", peer=0, dtype="U8") + event(1, 1))
        self.assertEqual(result["peer_transfers"], 1)
        self.assertEqual(result["collectives_per_rank"], 1)
        with self.assertRaisesRegex(ValueError, "sends vs 0 receives"):
            self.check(event(0, 0, "send", peer=1, dtype="U8"), event(1, 0, "send", peer=0, dtype="U8"))
        with self.assertRaisesRegex(ValueError, "peer channel 0->1 transfer 0"):
            self.check(event(0, 0, "send", peer=1, dtype="U8"),
                       event(1, 0, "recv", count=9, peer=0, dtype="U8"))


if __name__ == "__main__":
    unittest.main()
