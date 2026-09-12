#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU artifact oracles: passing booleans cannot certify another request mode."""

import copy
import json
import pathlib
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKS = ("determinism_ok", "toolcall_ok", "think_leak_ok", "known_answer_ok")


def fixture(name):
    return json.loads((HERE / "fixtures" / name).read_text())


def policy(think):
    return {"think": think, "chat_template_kwargs": {"enable_thinking": think == "on"}}


def coherency(think):
    doc = fixture("stub_coherency.json")
    doc["request_policy"] = policy(think)
    doc["check_request_policy"] = {
        key: policy("off" if key == "think_leak_ok" else think) for key in CHECKS}
    return doc


class CoherencyEvidenceTest(unittest.TestCase):
    def assemble(self, think, coh, ladder):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            paired = fixture("stub_pair_vllm_cell.json")
            paired["think"] = think
            for name, data in (("coh", coh), ("ladder", ladder), ("pair", paired)):
                (directory / (name + ".json")).write_text(json.dumps(data))
            binary = directory / "fake-spark"
            binary.write_text("fixture identity only; never executed\n")
            artifact = directory / "artifact.json"
            command = [
                "python3", str(HERE / "cell_assemble.py"), "--engine", "atlas",
                "--model-key", "nemotron-3-super-fp8", "--sku", "h200",
                "--workload", "lat", "--concurrency", "16", "--spec", "off",
                "--think", think, "--out", str(artifact), "--workloads",
                str(ROOT / "bench/hopper_ab/workloads.json"), "--atlas-recipes",
                str(HERE / "atlas_recipes.json"), "--vllm-recipes", str(HERE / "vllm_recipes.json"),
                "--client", str(ROOT / "bench/ladder38/harness_w55_conc_ladder.py"),
                "--nvidia-smi-q", str(HERE / "fixtures/stub_nvidia_smi_q.txt"),
                "--boot-json", str(HERE / "fixtures/stub_boot.json"),
                "--coherency-json", str(directory / "coh.json"),
                "--ladder-json", str(directory / "ladder.json"),
                "--paired-artifact", str(directory / "pair.json"), "--binary", str(binary),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(["python3", str(HERE / "validate_artifact.py"), str(artifact)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(artifact.read_text())

    def ladder(self, think):
        doc = fixture("stub_ladder_c16.json")
        doc["chat_template_kwargs"] = {"enable_thinking": think == "on"}
        return doc

    def test_a_opposite_coherency_mode_is_no_go(self):
        for think, other in (("on", "off"), ("off", "on")):
            with self.subTest(think=think):
                result = self.assemble(think, coherency(other), self.ladder(think))
                self.assertEqual((result["verdict"], result["failing_stage"]), ("NO-GO", "coherency"))
                self.assertIn("request policy", result["notes"])

    def test_b_legacy_coherency_cannot_certify_think_on(self):
        result = self.assemble("on", fixture("stub_coherency.json"), self.ladder("on"))
        self.assertEqual(result["failing_stage"], "coherency")

    def test_c_inconsistent_or_missing_per_check_policy_refuses(self):
        for key in CHECKS:
            doc = coherency("on")
            doc["check_request_policy"][key] = policy("on" if key == "think_leak_ok" else "off")
            with self.subTest(check=key):
                result = self.assemble("on", doc, self.ladder("on"))
                self.assertEqual(result["failing_stage"], "coherency")
        doc = coherency("on")
        del doc["check_request_policy"]
        self.assertEqual(self.assemble("on", doc, self.ladder("on"))["failing_stage"], "coherency")

    def test_d_malformed_policy_does_not_become_true(self):
        for bad in (None, {}, "on", {"think": "on", "chat_template_kwargs": {"enable_thinking": 1}}):
            with self.subTest(bad=bad):
                doc = coherency("on")
                doc["request_policy"] = copy.deepcopy(bad)
                self.assertEqual(self.assemble("on", doc, self.ladder("on"))["failing_stage"], "coherency")

    def test_e_opposite_ladder_mode_is_partial(self):
        for think, other in (("on", "off"), ("off", "on")):
            with self.subTest(think=think):
                result = self.assemble(think, coherency(think), self.ladder(other))
                self.assertEqual((result["verdict"], result["failing_stage"]), ("PARTIAL", "ladder"))

    def test_e_malformed_ladder_policy_is_partial(self):
        for bad in (None, "on", {}, {"enable_thinking": 1}):
            with self.subTest(bad=bad):
                doc = self.ladder("on")
                doc["chat_template_kwargs"] = bad
                self.assertEqual(self.assemble("on", coherency("on"), doc)["failing_stage"], "ladder")

    def test_f_matched_modes_and_legacy_off_still_certify(self):
        for think in ("on", "off"):
            with self.subTest(think=think):
                self.assertEqual(self.assemble(think, coherency(think), self.ladder(think))["verdict"], "CERTIFIED")
        self.assertEqual(self.assemble("off", fixture("stub_coherency.json"), self.ladder("off"))["verdict"], "CERTIFIED")

    def test_g_driver_passes_policy_to_gate(self):
        for engine in ("atlas", "vllm"):
            for think in ("on", "off"):
                with self.subTest(engine=engine, think=think), tempfile.TemporaryDirectory() as tmp:
                    result = subprocess.run([
                        "bash", str(HERE / "run_cell.sh"), "--engine", engine,
                        "--model", "nemotron-3-super-fp8", "--sku", "h200",
                        "--workload", "lat", "--concurrency", "16", "--spec", "off",
                        "--think", think, "--out", str(pathlib.Path(tmp) / "cell"), "--dry-run",
                    ], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    line = next(line for line in result.stdout.splitlines() if "coherency_gate.py --url" in line)
                    self.assertIn("--think " + think, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
