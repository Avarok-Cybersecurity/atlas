#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU oracles for explicit process mode, pin refusal and recipe preservation."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / "bench/campaign/run_cell.sh"
REV = "95a723d08a9490559dae23d0cff1d9466213d989"
SNAPSHOT = f"/cache/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/{REV}"


class ProcessModeTests(unittest.TestCase):
    def render(self, engine="vllm", path=SNAPSHOT, model="qwen3.6-35b-a3b-fp8"):
        env = dict(os.environ, SPARK_BIN="/prepared/spark", VLLM_BIN="/prepared/vllm")
        return subprocess.run(
            ["bash", str(RUN), "--engine", engine, "--model", model,
             "--sku", "h100", "--workload", "lat", "--concurrency", "1",
             "--spec", "off", "--think", "off", "--out", "/not-created/process-cell",
             "--process", "--model-path", path, "--dry-run"],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_a_wrong_snapshot_refuses_before_launch(self):
        result = self.render(path=SNAPSHOT.replace(REV, "a" * 40))
        self.assertEqual(result.returncode, 8, result.stdout)
        self.assertIn("snapshot revision", result.stdout)
        self.assertNotIn("stage 3/7", result.stdout)

    def test_b_foreign_model_snapshot_refuses(self):
        result = self.render(path=SNAPSHOT.replace("models--Qwen--", "models--Other--"))
        self.assertEqual(result.returncode, 8, result.stdout)

    def test_c_vllm_keeps_recipe_argv_and_explicit_identity(self):
        result = self.render()
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next(x for x in result.stdout.splitlines() if x.startswith("process_argv: "))
        argv = json.loads(line.partition(": ")[2])
        self.assertEqual(argv[:3], ["/prepared/vllm", "serve", SNAPSHOT])
        self.assertIn("qwen3_xml", argv)
        self.assertEqual(argv[argv.index("--revision") + 1], REV)
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "Qwen/Qwen3.6-35B-A3B-FP8")
        self.assertNotIn("docker run", result.stdout)
        self.assertNotIn("--speculative-config", argv)

    def test_d_atlas_keeps_launcher_and_recipe_flags(self):
        result = self.render(engine="atlas")
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next(x for x in result.stdout.splitlines() if x.startswith("process_argv: "))
        argv = json.loads(line.partition(": ")[2])
        self.assertEqual(argv[:3], ["/prepared/spark", "serve", SNAPSHOT])
        self.assertEqual(argv[argv.index("--fp8-kv-calibration-tokens") + 1], "256")
        self.assertEqual(argv[argv.index("--world-size") + 1], "1")
        self.assertIn("--disable-thinking", argv)
        self.assertNotIn("docker run", result.stdout)

    def test_e_multi_rank_atlas_is_explicitly_refused(self):
        result = self.render(engine="atlas", model="nemotron-3-super-fp8")
        self.assertEqual(result.returncode, 6, result.stdout)
        self.assertIn("single-rank", result.stdout)

    def test_f_audit_uses_declared_environment_without_ambient_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            env_path = folder / "env.json"
            env_path.write_text(json.dumps({"HF_HUB_OFFLINE": "1", "RUST_LOG": "info"}))
            argv_path = folder / "audit.argv"
            argv = [sys.executable, "-c", "import json,os;print(json.dumps(dict(os.environ)))"]
            argv_path.write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
            result = subprocess.run(
                [sys.executable, str(ROOT / "bench/campaign/process_exec.py"),
                 "--argv-nul", str(argv_path), "--env-json", str(env_path)],
                env=dict(os.environ, HF_TOKEN="synthetic-test-credential", RUST_LOG="error"),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout)
            self.assertEqual(observed["HF_HUB_OFFLINE"], "1")
            self.assertEqual(observed["RUST_LOG"], "info")
            self.assertEqual(observed["SPT_NOENV"], "1")
            self.assertNotIn("HF_TOKEN", observed)


if __name__ == "__main__":
    unittest.main()
