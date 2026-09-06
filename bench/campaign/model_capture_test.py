#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the real cell runner against CPU process shims; inspect before teardown."""

import json
import os
import pathlib
import signal
import subprocess
import tempfile
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
CID = "a" * 64
SECRET = "MODEL_CAPTURE_TEST_SECRET_DO_NOT_RETAIN"

DOCKER = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
base = pathlib.Path(os.environ['MODEL_CAPTURE_FIXTURES'])
args = sys.argv[1:]
with (base / 'calls.jsonl').open('a') as out:
    out.write(json.dumps(args) + '\n')
state_path = base / 'state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else None
if args and args[0] == 'run':
    position = args.index('--entrypoint')
    labels = dict(args[i+1].split('=', 1) for i, arg in enumerate(args[:-1]) if arg == '--label')
    state = {'Id': os.environ['MODEL_CAPTURE_CID'],
             'Name': args[args.index('--name')+1],
             'Image': 'sha256:' + 'b' * 64,
             'Config': {'Entrypoint': [args[position+1]], 'Cmd': args[position+3:],
                        'Labels': labels, 'Env': [os.environ['MODEL_CAPTURE_SECRET']],
                        'Image': args[position+2]},
             'State': {'Running': True, 'Pid': 2345, 'StartedAt': '2026-09-05T12:00:00Z'}}
    state_path.write_text(json.dumps(state))
    (base / 'created.json').write_text(json.dumps(state))
    print(state['Id'])
elif args[:2] == ['image', 'inspect']:
    print('<no value>')
elif args and args[0] == 'inspect':
    if '--format' in args:
        form = args[args.index('--format')+1]
        if form == '{{.Image}}':
            print(state['Image'])
        elif form == '{{.Config.Image}}':
            print(state['Config']['Image'])
        else:
            sys.exit('unexpected format in Docker fixture: ' + form)
    else:
        if os.environ['MODEL_CAPTURE_MODE'] == 'failed':
            sys.exit('known Docker inspect failure')
        if os.environ['MODEL_CAPTURE_MODE'] == 'unknown':
            print('[]')
        elif not state or state.get('removed') or args[-1] != state['Id']:
            sys.exit('known container is absent')
        else:
            print(json.dumps([state]))
elif args and args[0] == 'ps':
    if state and not state.get('removed'):
        print(state['Id'])
elif args and args[0] in ('stop', 'rm'):
    if not state or args[-1] != state['Id']:
        sys.exit('refusing a foreign fixture container')
    state['State']['Running'] = False
    if args[0] == 'rm':
        state['removed'] = True
    state_path.write_text(json.dumps(state))
'''

CURL = r'''#!/usr/bin/env python3
import os, pathlib
(pathlib.Path(os.environ['MODEL_CAPTURE_FIXTURES']) / 'boot-poll-seen').touch()
print('{"status":"loading"}\n503')
'''


class ModelCaptureTest(unittest.TestCase):
    def run_cell(self, mode):
        with tempfile.TemporaryDirectory(prefix="atlas-model-capture-") as temporary:
            base = pathlib.Path(temporary)
            binary = base / "bin"
            binary.mkdir()
            for name, source in (("docker", DOCKER), ("curl", CURL), (
                    "nvidia-smi", "#!/bin/sh\nprintf 'Driver Version : 999.00\\nCUDA Version : 99.9\\nAttached GPUs : 1\\nProduct Name : Stub GPU\\n'\n")):
                path = binary / name
                path.write_text(source)
                path.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=str(binary) + os.pathsep + env["PATH"], DOCKER=str(binary / "docker"),
                       VLLM_IMAGE_DIGEST="sha256:" + "b" * 64,
                       MODEL_CAPTURE_FIXTURES=str(base), MODEL_CAPTURE_CID=CID,
                       MODEL_CAPTURE_SECRET=SECRET, MODEL_CAPTURE_MODE=mode)
            output = base / "cell"
            command = ["bash", str(HERE / "run_cell.sh"), "--engine", "vllm",
                       "--model", "nemotron-3-nano-fp8", "--sku", "h100",
                       "--workload", "lat", "--concurrency", "1", "--spec", "off",
                       "--think", "off", "--out", str(output), "--yes"]
            log_path = base / "runner.log"
            with log_path.open("w") as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    deadline = time.monotonic() + 20
                    while not (base / "boot-poll-seen").exists():
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail("runner did not enter boot: " + log_path.read_text())
                        time.sleep(0.05)
                    process.send_signal(signal.SIGTERM)
                    status = process.wait(timeout=20)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
            log = log_path.read_text()
            self.assertEqual(status, 143, log)
            artifact = json.loads((output / "artifact.json").read_text())
            self.assertEqual((artifact["verdict"], artifact["failing_stage"]), ("NO-GO", "boot"), log)
            self.assertIsNone(artifact["model"]["revision"], "unfinished boot cannot prove model loading")
            calls = [json.loads(line) for line in (base / "calls.jsonl").read_text().splitlines()]
            self.assertIn(["stop", CID], calls)
            self.assertIn(["rm", CID], calls)
            self.assertTrue(json.loads((base / "state.json").read_text())["removed"])
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertNotIn(SECRET.encode(), path.read_bytes(), str(path))
            self.assertNotIn(SECRET, log)
            proof_path = output / "model-launch.json"
            proof = json.loads(proof_path.read_text()) if proof_path.exists() else None
            return artifact, calls, proof, json.loads((base / "created.json").read_text())

    def test_a_owned_capture_precedes_teardown_and_reaches_assembler(self):
        artifact, calls, proof, created = self.run_cell("success")
        self.assertIsNotNone(proof, "runner discarded actual launch identity before teardown")
        self.assertEqual(len(proof), 1)
        observed = proof[0]
        self.assertEqual(observed["Id"], CID)
        self.assertEqual(observed["Config"]["Entrypoint"], created["Config"]["Entrypoint"])
        self.assertEqual(observed["Config"]["Cmd"], created["Config"]["Cmd"])
        self.assertEqual(observed["Config"]["Labels"], created["Config"]["Labels"])
        self.assertNotIn("Env", observed["Config"])
        self.assertTrue(observed["State"]["Running"])
        inspection = calls.index(["inspect", CID])
        self.assertLess(inspection, calls.index(["stop", CID]))
        self.assertLess(inspection, calls.index(["rm", CID]))
        self.assertIn("actual model launch evidence", artifact["notes"])
        self.assertNotIn("invalid model launch evidence", artifact["notes"])

    def test_b_failed_inspect_does_not_forge_proof(self):
        artifact, calls, proof, _ = self.run_cell("failed")
        self.assertIn(["inspect", CID], calls, "runner did not attempt actual identity capture")
        self.assertIsNone(proof)
        self.assertIsNone(artifact["model"]["revision"])

    def test_c_unknown_inspect_does_not_forge_proof(self):
        artifact, calls, proof, _ = self.run_cell("unknown")
        self.assertIn(["inspect", CID], calls, "runner did not attempt actual identity capture")
        self.assertIsNone(proof)
        self.assertIsNone(artifact["model"]["revision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
