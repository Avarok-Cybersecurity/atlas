#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""PRD oracles: required/unsupported thinking modes refuse before launch."""

import json
import pathlib
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


class ThinkingPolicyTest(unittest.TestCase):
    def cell(self, engine, model, think, dry=True, sku="h200"):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "cell"
            result = subprocess.run([
                "bash", str(HERE / "run_cell.sh"), "--engine", engine,
                "--model", model, "--sku", sku, "--workload", "lat",
                "--concurrency", "1", "--spec", "off", "--think", think,
                "--out", str(out), "--dry-run" if dry else "--yes",
            ], capture_output=True, text=True, timeout=30)
            self.assertFalse(out.exists(), result.stdout + result.stderr)
            return result

    def test_a_forbidden_modes_refuse_before_stages(self):
        # PRD sections 4 and 6.1 are the oracle, not current rendered flags.
        for engine, model, think in (
            ("vllm", "glm-5.3", "off"),
            ("vllm", "glm-5.3-flash", "off"),
            ("vllm", "qwen3-next-80b-fp8", "on"),
            ("atlas", "qwen3-next-80b-fp8", "on"),
        ):
            with self.subTest(engine=engine, model=model, think=think):
                result = self.cell(engine, model, think)
                self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
                self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
                self.assertIn(model, result.stderr)
                self.assertIn("--think " + think, result.stderr)
                self.assertNotIn("stage 1/7", result.stdout)

    def test_b_atlas_renderer_cannot_bypass_policy(self):
        result = subprocess.run([
            "python3", str(HERE / "atlas_render.py"), "--recipes",
            str(HERE / "atlas_recipes.json"), "--model", "qwen3-next-80b-fp8",
            "--sku", "h200", "--spec", "off", "--think", "on", "--extra-args",
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_c_permitted_modes_render(self):
        for engine, model, think in (
            ("vllm", "glm-5.3", "on"),
            ("vllm", "glm-5.3-flash", "on"),
            ("vllm", "qwen3-next-80b-fp8", "off"),
            ("atlas", "qwen3-next-80b-fp8", "off"),
            ("atlas", "nemotron-3-nano-fp8", "on"),
            ("atlas", "nemotron-3-nano-fp8", "off"),
        ):
            with self.subTest(engine=engine, model=model, think=think):
                result = self.cell(engine, model, think)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("stage 5/7", result.stdout)

    def test_d_absent_recipe_still_names_missing_pair(self):
        for engine, model, think, sku in (
            ("atlas", "glm-5.3", "off", "h200"),
            ("atlas", "qwen3-next-80b-fp8", "on", "h100"),
            ("vllm", "future-model", "on", "h200"),
        ):
            with self.subTest(engine=engine, model=model, sku=sku):
                result = self.cell(engine, model, think, sku=sku)
                self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
                self.assertIn("no rendered profile", result.stdout + result.stderr)

    def test_e_catalog_covers_exact_recipe_models(self):
        policy = json.loads((HERE / "campaign_policy.json").read_text())
        models = {e["model_key"] for name in ("atlas_recipes.json", "vllm_recipes.json")
                  for e in json.loads((HERE / name).read_text())["entries"]}
        self.assertEqual(set(policy["models"]), models)

    def test_f_real_refusal_also_precedes_resource_creation(self):
        # The dry-run negative above established the defect; --yes must take
        # the same early-refusal branch, with no output dir or GPU work.
        result = self.cell("vllm", "glm-5.3", "off", dry=False)
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
        self.assertNotIn("stage 1/7", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
