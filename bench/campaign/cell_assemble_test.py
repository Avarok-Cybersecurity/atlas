#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU CLI oracles for assembly and model identity from actual launch evidence."""

import copy
import json
import pathlib
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HF_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
REVISION = "b" * 40
CONTAINER_ID = "a" * 64
RUN_LABEL = "atlas-cell.run=launch-proof-fixture"


def fixture(name):
    return json.loads((HERE / "fixtures" / name).read_text())


def launch(engine="vllm"):
    model = HF_ID if engine == "vllm" else (
        "/root/.cache/huggingface/hub/models--" + HF_ID.replace("/", "--")
        + "/snapshots/" + REVISION)
    command = ["serve", model]
    if engine == "vllm":
        command += ["--revision", REVISION]
    return [{"Id": CONTAINER_ID,
             "Config": {"Entrypoint": ["vllm" if engine == "vllm" else "/usr/local/bin/spark"],
                        "Cmd": command, "Labels": dict([RUN_LABEL.split("=", 1)])},
             "State": {"Running": True, "Pid": 1234,
                       "StartedAt": "2026-09-05T11:00:00.123456789Z"}}]


class AssemblyTest(unittest.TestCase):
    def assemble(self, evidence=None, *, engine="vllm", boot=None, coh=None,
                 process=None, owner=None, boot_engine=None, boot_model=None):
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            paired = fixture("stub_pair_vllm_cell.json")
            paired["engine"] = "atlas" if engine == "vllm" else "vllm"
            paired["cell_id"] = paired["engine"] + paired["cell_id"][4:]
            boot = copy.deepcopy(boot if boot is not None else fixture("stub_boot.json"))
            boot["engine"] = engine if boot_engine is None else boot_engine
            boot["model"] = HF_ID if boot_model is None else boot_model
            coh = coh if coh is not None else fixture("stub_coherency.json")
            for name, data in (("pair", paired), ("boot", boot), ("coh", coh)):
                (folder / (name + ".json")).write_text(json.dumps(data))
            output = folder / "artifact.json"
            args = ["python3", str(HERE / "cell_assemble.py"), "--engine", engine,
                    "--model-key", "nemotron-3-super-fp8", "--sku", "h200",
                    "--workload", "lat", "--concurrency", "16", "--spec", "off",
                    "--think", "off", "--out", str(output), "--workloads",
                    str(ROOT / "bench/hopper_ab/workloads.json"), "--atlas-recipes",
                    str(HERE / "atlas_recipes.json"), "--vllm-recipes", str(HERE / "vllm_recipes.json"),
                    "--client", str(ROOT / "bench/ladder38/harness_w55_conc_ladder.py"),
                    "--nvidia-smi-q", str(HERE / "fixtures/stub_nvidia_smi_q.txt"),
                    "--image-digest", "sha256:" + "c" * 64,
                    "--boot-json", str(folder / "boot.json"), "--coherency-json", str(folder / "coh.json"),
                    "--ladder-json", str(HERE / "fixtures/stub_ladder_c16.json"),
                    "--paired-artifact", str(folder / "pair.json")]
            if evidence is not None:
                path = folder / "model-launch.json"
                path.write_text(json.dumps(evidence))
                args += ["--model-launch-json", str(path),
                         "--model-launch-container-id", CONTAINER_ID,
                         "--model-launch-label", RUN_LABEL]
            for name, data in (("process", process), ("process-owner", owner)):
                if data is not None:
                    path = folder / (name + ".json")
                    path.write_text(json.dumps(data))
                    args += ["--model-launch-" + name + "-json", str(path)]
            result = subprocess.run(args, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            validation = subprocess.run(["python3", str(HERE / "validate_artifact.py"), str(output)],
                                        text=True, capture_output=True, timeout=20)
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
            return json.loads(output.read_text())

    def test_a_unowned_launch_cannot_certify(self):
        for field in ("id", "label"):
            evidence = launch()
            if field == "id":
                evidence[0]["Id"] = "d" * 64
            else:
                evidence[0]["Config"]["Labels"] = {"atlas-cell.run": "another-run"}
            with self.subTest(field=field):
                result = self.assemble(evidence)
                self.assertEqual((result["verdict"], result["failing_stage"]), ("NO-GO", "serve"))
                self.assertIsNone(result["model"]["revision"])
                self.assertIn("model launch evidence", result["notes"])

    def test_b_unstarted_or_malformed_launch_cannot_certify(self):
        bad = [[], {}, [7]]
        for field, value in (("Running", False), ("Pid", 0), ("StartedAt", "0001-01-01T00:00:00Z")):
            evidence = launch()
            evidence[0]["State"][field] = value
            bad.append(evidence)
        for evidence in bad:
            with self.subTest(evidence=evidence):
                result = self.assemble(evidence)
                self.assertEqual(result["failing_stage"], "serve")
                self.assertIsNone(result["model"]["revision"])

    def test_c_wrong_model_or_competing_revision_cannot_certify(self):
        bad = [["serve", "another/model", "--revision", REVISION],
               ["serve", HF_ID, "--revision", "main"],
               ["serve", HF_ID, "--revision", REVISION, "--revision", "d" * 40],
               ["serve", HF_ID, "--revision", REVISION, "--model", "another/model"]]
        for command in bad:
            evidence = launch()
            evidence[0]["Config"]["Cmd"] = command
            with self.subTest(command=command):
                result = self.assemble(evidence)
                self.assertEqual(result["failing_stage"], "serve")
                self.assertIsNone(result["model"]["revision"])

    def test_d_failed_boot_does_not_prove_loaded_revision(self):
        result = self.assemble(launch(), boot=fixture("stub_boot_timeout.json"))
        self.assertEqual(result["failing_stage"], "boot")
        self.assertIsNone(result["model"]["revision"])

    def test_e_recipe_pins_alone_are_not_launch_proof(self):
        result = self.assemble()
        self.assertEqual(result["verdict"], "CERTIFIED")
        self.assertIsNone(result["model"]["revision"])

    def test_f_owned_launched_pin_and_boot_prove_revision(self):
        for engine in ("vllm", "atlas"):
            with self.subTest(engine=engine):
                result = self.assemble(launch(engine), engine=engine)
                self.assertEqual(result["verdict"], "CERTIFIED")
                self.assertEqual(result["model"]["revision"], REVISION)
                self.assertIn("model-launch.json", result["notes"])

    def test_g_floating_launch_keeps_revision_null(self):
        for engine in ("vllm", "atlas"):
            evidence = launch(engine)
            evidence[0]["Config"]["Cmd"] = ["serve", HF_ID]
            with self.subTest(engine=engine):
                result = self.assemble(evidence, engine=engine)
                self.assertEqual(result["verdict"], "CERTIFIED")
                self.assertIsNone(result["model"]["revision"])

    def test_h_failed_known_answer_remains_no_go(self):
        result = self.assemble(coh=fixture("stub_coherency_wrong_answer.json"))
        self.assertEqual((result["verdict"], result["failing_stage"]), ("NO-GO", "coherency"))


def selftest():
    from process_model_evidence_test import ProcessAssemblyTest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AssemblyTest)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ProcessAssemblyTest))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print(f"cell_assemble selftest: {result.testsRun}/{result.testsRun} test methods passed")
    else:
        print(f"cell_assemble selftest: {len(result.failures)} failures and "
              f"{len(result.errors)} errors in {result.testsRun} test methods")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
