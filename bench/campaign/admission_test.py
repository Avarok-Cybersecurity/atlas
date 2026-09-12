#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU subprocess oracles: failed preflight must never dispatch an engine."""

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
CASES = (
    ("atlas-process", "atlas", True, False),
    ("atlas-binary", "atlas", False, False),
    ("atlas-container", "atlas", False, True),
    ("vllm-container", "vllm", False, False),
)


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


class AdmissionTests(unittest.TestCase):
    def cell(self, case, preflight_ok):
        name, engine, process, container = case
        with tempfile.TemporaryDirectory(prefix="atlas-admission-") as temporary:
            base = pathlib.Path(temporary)
            binaries = base / "bin"
            binaries.mkdir()
            calls = base / "calls.jsonl"
            calls.touch()
            # Embedded paths survive the process manager's restricted environment.
            recorder = (
                "import json, pathlib, sys\n"
                f"with pathlib.Path({str(calls)!r}).open('a') as stream:\n"
                "    stream.write(json.dumps(sys.argv) + '\\n')\n"
            )
            executable(binaries / "spark", recorder + "raise SystemExit(17)\n")
            executable(binaries / "docker", recorder + (
                "if sys.argv[1:2] in (['run'], ['create'], ['start']):\n"
                "    raise SystemExit(125)\n"
            ))
            executable(binaries / "nvidia-smi", (
                "import sys\n"
                "print('Driver Version : 999.00\\nCUDA Version : 99.9\\n'\n"
                "      'Attached GPUs : 1\\nProduct Name : CPU admission stub')\n"
                + ("raise SystemExit(0)\n" if preflight_ok else
                   "print('admission fixture: nvidia-smi failed', file=sys.stderr)\n"
                   "raise SystemExit(13)\n")
            ))
            snapshot = base / "hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots" / REVISION
            snapshot.mkdir(parents=True)
            output = base / "cell"
            env = {key: value for key, value in os.environ.items()
                   if key in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT", "LANG", "LC_ALL")}
            env.update(PATH=str(binaries) + os.pathsep + env.get("PATH", os.defpath),
                       SPARK_BIN=str(binaries / "spark"), DOCKER=str(binaries / "docker"),
                       CUDA_VISIBLE_DEVICES="-1", HF_CACHE=str(base / "hub"),
                       VLLM_IMAGE_DIGEST="sha256:" + "a" * 64)
            with socket.socket() as port_probe:
                port_probe.bind(("127.0.0.1", 0))
                port = str(port_probe.getsockname()[1])
            env.update(ATLAS_PORT=port, VLLM_PORT=port)
            if container:
                env["IMAGE"] = "admission-test:never-contacted"
            command = ["bash", str(HERE / "run_cell.sh"), "--engine", engine,
                       "--model", "qwen3.6-35b-a3b-fp8", "--sku", "h100",
                       "--workload", "lat", "--concurrency", "1", "--spec", "off",
                       "--think", "off", "--out", str(output), "--yes"]
            if process:
                command += ["--process", "--model-path", str(snapshot)]
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    capture_output=True, timeout=30)
            observed = [json.loads(line) for line in calls.read_text().splitlines()]
            launches = [argv for argv in observed if pathlib.Path(argv[0]).name == "spark"
                        or argv[1:2] in (["run"], ["create"], ["start"])]
            artifact_path = output / "artifact.json"
            artifact = json.loads(artifact_path.read_text()) if artifact_path.exists() else {}
            summary = {"case": name, "preflight_ok": preflight_ok,
                       "runner_exit": result.returncode, "engine_launches": launches,
                       "verdict": artifact.get("verdict"),
                       "failing_stage": artifact.get("failing_stage")}
            print(json.dumps(summary), flush=True)
            evidence = os.environ.get("CAMPAIGN_ADMISSION_EVIDENCE")
            if evidence:
                destination = pathlib.Path(evidence) / (name + ("-pass" if preflight_ok else "-fail"))
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                (destination / "stdout.txt").write_text(result.stdout)
                (destination / "stderr.txt").write_text(result.stderr)
                (destination / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
                shutil.copy2(calls, destination / calls.name)
                shutil.copytree(output, destination / "cell", dirs_exist_ok=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual((artifact.get("verdict"), artifact.get("failing_stage")),
                             ("NO-GO", "serve" if preflight_ok else "preflight"))
            validated = subprocess.run([sys.executable, str(HERE / "validate_artifact.py"),
                                        str(artifact_path)], capture_output=True, text=True,
                                       timeout=10)
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            if preflight_ok:
                self.assertEqual(len(launches), 1, observed)
                if engine == "atlas":
                    self.assertIn("--check-kernels", launches[0])
            else:
                self.assertEqual(launches, [], "failed preflight reached engine launch")
                self.assertIn("admission fixture: nvidia-smi failed",
                              (output / "nvidia-smi-q.err").read_text())
                self.assertFalse((output / "serve.argv").exists())
                self.assertEqual(list(output.glob("process-*")), [])
                self.assertEqual(list(output.glob("node-ep-*")), [])

    def test_failed_preflight_never_launches_and_writes_no_go(self):
        for case in CASES:
            with self.subTest(case=case[0]):
                self.cell(case, preflight_ok=False)

    def test_successful_preflight_reaches_selected_serve_path(self):
        for case in CASES[1:]:
            with self.subTest(case=case[0]):
                self.cell(case, preflight_ok=True)

    @unittest.skipUnless(sys.platform == "linux", "process endpoint admission requires Linux /proc")
    def test_successful_process_preflight_reaches_engine_audit(self):
        self.cell(CASES[0], preflight_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
