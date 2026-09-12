#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU regression: a nested renderer refusal must survive the cell finalizer."""

import os
import pathlib
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


class DryRunFailureTest(unittest.TestCase):
    def cell(self, engine, spec="off", env=None):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "cell"
            result = subprocess.run([
                "bash", str(HERE / "run_cell.sh"), "--engine", engine,
                "--model", "nemotron-3-nano-fp8", "--sku", "h100",
                "--workload", "lat", "--concurrency", "1", "--spec", spec,
                "--think", "off", "--out", str(out), "--dry-run",
            ], capture_output=True, text=True, timeout=30,
                env={**os.environ, **(env or {})})
            self.assertFalse(out.exists(), result.stdout + result.stderr)
            return result

    def test_a_vllm_spec_refusal_survives_finalizer(self):
        direct = subprocess.run([
            "bash", str(HERE / "vllm_control.sh"), "nemotron-3-nano-fp8",
            "h100", "--spec", "on", "--dry-run",
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(direct.returncode, 4, direct.stdout + direct.stderr)
        wrapped = self.cell("vllm", spec="on")
        self.assertEqual(wrapped.returncode, direct.returncode, wrapped.stdout + wrapped.stderr)
        self.assertIn("no speculative", wrapped.stdout + wrapped.stderr)
        self.assertNotIn("stage 3/7", wrapped.stdout)
        self.assertNotIn("stage 4/7", wrapped.stdout)
        self.assertNotIn("stage 5/7", wrapped.stdout)

    def test_b_atlas_launcher_error_survives_finalizer(self):
        # Invalid port syntax is a real launcher error, before any launch.
        result = self.cell("atlas", env={"ATLAS_PORT": "invalid+"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("stage 3/7", result.stdout)

    def test_c_valid_cells_still_render(self):
        for engine in ("atlas", "vllm"):
            with self.subTest(engine=engine):
                result = self.cell(engine)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("stage 5/7", result.stdout)
                self.assertIn("dry-run: nothing launched, nothing written", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
