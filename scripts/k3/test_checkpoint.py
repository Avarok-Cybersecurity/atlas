# SPDX-License-Identifier: AGPL-3.0-only
"""Offline counterexamples for the paid-rental checkpoint boundary."""
import hashlib
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

    def test_nonfinite_deadline_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            for deadline in [float("nan"), float("inf"), -1]:
                with self.assertRaises(ValueError):
                    cp.download(manifest(), Path(directory), 1, 1, deadline, 5)


if __name__ == "__main__":
    unittest.main()
