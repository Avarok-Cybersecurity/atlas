# SPDX-License-Identifier: AGPL-3.0-only
"""Offline counterexamples for the paid-rental checkpoint boundary."""
import hashlib
import json
from unittest.mock import patch
import tempfile
import subprocess
import sys
import time
import unittest
from pathlib import Path

import checkpoint as cp


def manifest(data=b"weights", name="model.safetensors"):
    return {"schema": 1, "repo": "moonshotai/Kimi-K3", "revision": "a" * 40,
            "files": [{"path": name, "size": len(data), "algorithm": "sha256",
                       "digest": hashlib.sha256(data).hexdigest()}]}


class CheckpointTests(unittest.TestCase):
    def test_pin_and_path_counterexamples(self):
        for key, value in [("revision", "main"), ("repo", "../escape")]:
            bad = manifest()
            bad[key] = value
            with self.assertRaises(ValueError):
                cp.validate(bad)
        for name in [".", "../escape", "/absolute", "x/../escape", "x\\escape", "x//y"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                cp.validate(manifest(name=name))

    def test_full_hash_catches_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.safetensors"
            self.assertFalse(cp.check_file(root, manifest()["files"][0]))
            for content in [b"we", b"WEIGHTS"]:
                target.write_bytes(content)
                self.assertFalse(cp.check_file(root, manifest()["files"][0]))
            target.write_bytes(b"weights")
            self.assertTrue(cp.check_file(root, manifest()["files"][0]))

    def test_git_blob_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"{}"
            item = {"path": "config.json", "size": 2, "algorithm": "git-sha1",
                    "digest": hashlib.sha1(b"blob 2\0" + data).hexdigest()}
            (root / item["path"]).write_bytes(data)
            self.assertTrue(cp.check_file(root, item))
            item["digest"] = hashlib.sha1(data).hexdigest()
            self.assertFalse(cp.check_file(root, item))

    def test_symlink_escape_refuses_even_existing_valid_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshot"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.write_bytes(b"weights")
            (root / "model.safetensors").symlink_to(outside)
            with self.assertRaises(ValueError):
                cp.check_file(root, manifest()["files"][0])

    def test_duplicate_files_and_wrong_hash_refuse(self):
        bad = manifest()
        bad["files"] *= 2
        with self.assertRaises(ValueError):
            cp.validate(bad)
        bad = manifest()
        bad["files"][0]["digest"] = "bad"
        with self.assertRaises(ValueError):
            cp.validate(bad)

    def test_timeout_and_exit_are_not_success(self):
        self.assertEqual(cp.bounded_run([sys.executable, "-c", "raise SystemExit(7)"], 5), 7)
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            cp.bounded_run([sys.executable, "-c", "import time; time.sleep(60)"], 0.1)
        self.assertLess(time.monotonic() - started, 6)

    def test_download_reuses_verified_files_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            cp.download(manifest(), root, 1, 1, 1, 5)
            other = manifest()
            other["revision"] = "b" * 40
            with self.assertRaisesRegex(ValueError, "another checkpoint"):
                cp.download(other, root, 1, 1, 1, 5)

    def test_space_admission_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "free disk"):
                cp.download(manifest(), Path(directory), 10**30, 1, 1, 5)

    def test_second_download_cannot_share_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with cp.snapshot_lock(root):
                with self.assertRaisesRegex(ValueError, "another download owns"):
                    cp.download(manifest(), root, 1, 1, 1, 5)

    def test_progress_skips_verified_bytes_without_claiming_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'snapshot'
            root.mkdir()
            (root/'model.safetensors').write_bytes(b'weights')
            log = Path(directory)/'progress.jsonl'
            cp.download(manifest(), root, 1, 1, 1, 5, progress_log=log)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'complete')
            self.assertEqual(rows[-1]['existing_verified_bytes'], 7)
            self.assertEqual(rows[-1]['newly_verified_bytes'], 0)
            self.assertIsNone(rows[-1]['verified_bytes_per_second'])
            self.assertEqual(rows[-1]['estimated_remaining_seconds'], 0)
            self.assertIn('verified_existing', [row['event'] for row in rows])
            self.assertTrue(all(row['timestamp_utc'].endswith('Z') for row in rows))
            with self.assertRaises(FileExistsError):
                cp.download(manifest(), root, 1, 1, 1, 5, progress_log=log)

    def test_progress_counts_only_verified_completed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'snapshot'
            log = Path(directory)/'progress.jsonl'
            doc = manifest()
            doc['files'].append(dict(doc['files'][0], path='second.safetensors'))
            calls = []
            def transfer(argv, seconds):
                calls.append(argv)
                (root/argv[argv.index('--file')+1]).write_bytes(b'CORRUPT' if len(calls)==1 else b'weights')
                return 0
            with patch('checkpoint.bounded_run', transfer):
                cp.download(doc, root, 1, 2, 1, 5, progress_log=log)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            failed = next(row for row in rows if row['event']=='attempt_failed')
            self.assertEqual(failed['newly_verified_bytes'], 0)
            self.assertIsNone(failed['estimated_remaining_seconds'])
            first = next(row for row in rows if row['event']=='verified_download')
            self.assertEqual(first['remaining_unverified_bytes'], 7)
            self.assertGreater(first['estimated_remaining_seconds'], 0)
            self.assertEqual(rows[-1]['newly_verified_bytes'], 14)
            self.assertGreater(rows[-1]['verified_bytes_per_second'], 0)
            self.assertIn('--force', calls[1])

    def test_progress_failure_is_preserved_without_overwriting_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'snapshot'
            with self.assertRaisesRegex(ValueError, 'outside'):
                cp.download(manifest(), root, 1, 1, 1, 5, progress_log=root/'model.safetensors')
            log = Path(directory)/'progress.jsonl'
            with self.assertRaisesRegex(ValueError, 'free disk'):
                cp.download(manifest(), root, 10**30, 1, 1, 5, progress_log=log)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'failed')
            self.assertEqual(rows[-1]['error_type'], 'ValueError')

    def test_nonfinite_deadline_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            for deadline in [float("nan"), float("inf"), -1]:
                with self.assertRaises(ValueError):
                    cp.download(manifest(), Path(directory), 1, 1, deadline, 5)


if __name__ == "__main__":
    unittest.main()
