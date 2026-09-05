#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Linux CPU integration oracles for real process-mode ownership and teardown."""

import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANAGER = HERE / "process_launch.py"
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

SPARK_SOURCE = r'''// SPDX-License-Identifier: AGPL-3.0-only
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--check-kernels")) {
            const char *offline = getenv("HF_HUB_OFFLINE");
            const char *level = getenv("RUST_LOG");
            if (!offline || strcmp(offline, "1") || !level || strcmp(level, "info")) {
                fputs("audit environment differs from declared serve environment\n", stderr);
                return 19;
            }
            puts("CPU fixture kernel audit ran");
            return 0;
        }
    }
    int port = 0;
    for (int i = 1; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--port")) port = atoi(argv[i + 1]);
    int listener = socket(AF_INET, SOCK_STREAM, 0), reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener, (struct sockaddr *)&addr, sizeof(addr)) || listen(listener, 16)) {
        perror("owned CPU fixture listener");
        return 18;
    }
    for (;;) {
        int client = accept(listener, NULL, NULL);
        if (client < 0) continue;
        char input[4096];
        if (read(client, input, sizeof(input)) > 0) {
            const char *reply = "HTTP/1.1 503 Loading\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"status\":\"loading\"}";
            write(client, reply, strlen(reply));
        }
        close(client);
    }
}
'''

SMI = r'''#!/usr/bin/env python3
import os, pathlib, time
blocked = os.environ.get('PROCESS_TEST_PREFLIGHT')
if blocked:
    root = pathlib.Path(blocked)
    (root / 'entered').touch()
    deadline = time.monotonic() + 30
    while not (root / 'release').exists():
        if time.monotonic() > deadline:
            raise SystemExit('test did not release preflight')
        time.sleep(0.02)
print('Driver Version : 999.00\nCUDA Version : 99.9\nAttached GPUs : 1\nProduct Name : CPU test stub')
'''


def running(pid):
    try:
        state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X")
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open") and shutil.which("cc"),
                     "requires Linux /proc, pidfds and a CPU C compiler")
class ProcessRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="atlas-process-runner-")
        self.base = pathlib.Path(self.temp.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        source = self.base / "spark.c"
        source.write_text(SPARK_SOURCE)
        self.spark = self.bin / "spark"
        subprocess.run(["cc", "-O0", str(source), "-o", str(self.spark)], check=True,
                       capture_output=True, text=True, timeout=20)
        smi = self.bin / "nvidia-smi"
        smi.write_text(SMI)
        smi.chmod(0o755)
        self.snapshot = self.base / "hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots" / REVISION
        self.snapshot.mkdir(parents=True)
        self.output = self.base / "cell"
        self.runner = None
        self.stream = None
        self.foreign_record = None
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]

    def loading(self):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.2).close()
        except urllib.error.HTTPError as error:
            return error.code == 503
        except (OSError, urllib.error.URLError):
            return False
        return False

    def tearDown(self):
        if self.runner is not None and self.runner.poll() is None:
            self.runner.send_signal(signal.SIGTERM)
            try:
                self.runner.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.runner.pid, signal.SIGKILL)
                self.runner.wait(timeout=5)
        records = list(self.output.glob("process-*/owner.json"))
        if self.foreign_record:
            records.append(self.foreign_record)
        for record in records:
            subprocess.run([sys.executable, str(MANAGER), "stop", "--record", str(record),
                            "--timeout", "0.1"], capture_output=True, timeout=5)
        if self.stream:
            self.stream.close()
        self.temp.cleanup()

    def start_runner(self, blocked=False):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   SPARK_BIN=str(self.spark), ATLAS_PORT=str(self.port),
                   CUDA_VISIBLE_DEVICES="-1", RUST_LOG="error")
        env.pop("IMAGE", None)
        env.pop("ATLAS_NODE_RUN_DIR", None)
        if blocked:
            env["PROCESS_TEST_PREFLIGHT"] = str(self.base)
        self.log = self.base / "runner.log"
        self.stream = self.log.open("w")
        command = ["bash", str(HERE / "run_cell.sh"), "--engine", "atlas", "--model",
                   "qwen3.6-35b-a3b-fp8", "--sku", "h100", "--workload", "lat",
                   "--concurrency", "1", "--spec", "off", "--think", "off",
                   "--out", str(self.output), "--process", "--model-path", str(self.snapshot), "--yes"]
        self.runner = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        return self.output / f"process-{self.runner.pid}" / "owner.json"

    def wait_for(self, condition, description):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if condition():
                return
            if self.runner is not None and self.runner.poll() is not None:
                self.fail(f"runner exited before {description}: " + self.log.read_text())
            time.sleep(0.03)
        self.fail(f"timed out waiting for {description}: " + self.log.read_text())

    def read_artifact(self, stage):
        path = self.output / "artifact.json"
        validation = subprocess.run([sys.executable, str(HERE / "validate_artifact.py"), str(path)],
                                    text=True, capture_output=True, timeout=10)
        self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
        artifact = json.loads(path.read_text())
        self.assertEqual((artifact["verdict"], artifact["failing_stage"]), ("NO-GO", stage))
        self.assertIsNone(artifact["model"]["revision"])
        return artifact

    def test_server_exit_during_boot_fails_promptly_with_boot_json(self):
        record = self.start_runner()
        self.wait_for(self.loading, "the owned boot health response")
        self.wait_for(lambda: record.exists() and 'pid' in json.loads(record.read_text()),
                      "the completed owned process record")
        self.wait_for(lambda: 'stage 3/7 boot gate' in self.log.read_text(),
                      "the boot gate")
        stopped = subprocess.run([sys.executable, str(MANAGER), "stop", "--record",
                                  str(record), "--timeout", "0.1"],
                                 capture_output=True, text=True, timeout=5)
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        started = time.monotonic()
        try:
            rc = self.runner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.fail("boot gate kept polling after its owned server exited")
        self.assertNotEqual(rc, 0)
        boot = json.loads((self.output / "boot.json").read_text())
        self.assertEqual(boot["status"], "process-exited", boot)
        self.assertFalse(boot["passed"])
        self.read_artifact("boot")
        print(json.dumps({"oracle": "owned server exit ends boot before the cap",
                          "exit": rc, "elapsed_s": time.monotonic() - started,
                          "boot_status": boot["status"]}), flush=True)

    def test_term_during_boot_stops_owned_process_and_retains_environment(self):
        record = self.start_runner()

        def owned():
            try:
                return "pid" in json.loads(record.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return False

        self.wait_for(owned, "the completed owned process record")
        self.wait_for(self.loading, "the owned boot health response")
        owner = json.loads(record.read_text())
        self.runner.send_signal(signal.SIGTERM)
        self.assertEqual(self.runner.wait(timeout=25), 143, self.log.read_text())
        self.assertFalse(running(owner["pid"]), "owned server survived teardown")
        artifact = self.read_artifact("boot")
        self.assertEqual(artifact["serve_env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(artifact["serve_env"]["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(artifact["serve_env"]["RUST_LOG"], "info")
        self.assertIn("owned Linux /proc", artifact["notes"])
        self.assertNotIn("invalid model launch evidence", artifact["notes"])
        proof = json.loads((self.output / "process-launch.json").read_text())
        self.assertEqual(proof["pid"], owner["pid"])
        self.assertTrue(proof["running"], "capture must precede stopping the owned process")
        print(json.dumps({"oracle": "SIGTERM during boot", "exit": 143,
                          "owned_pid": owner["pid"], "alive_after": False,
                          "verdict": artifact["verdict"], "stage": artifact["failing_stage"],
                          "serve_env": artifact["serve_env"]}), flush=True)

    def test_preexisting_process_directory_cannot_capture_or_stop_foreign_owner(self):
        self.foreign_record = self.base / "foreign-owner.json"
        argv = self.base / "foreign-argv.json"
        argv.write_text(json.dumps([str(self.spark), "serve", str(self.snapshot)]))
        started = subprocess.run([sys.executable, str(MANAGER), "start", "--record",
                                  str(self.foreign_record), "--evidence", str(self.base / "foreign-proof.json"),
                                  "--argv-json", str(argv), "--log", str(self.base / "foreign.log")],
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(started.returncode, 0, started.stderr)
        foreign = json.loads(self.foreign_record.read_text())
        record = self.start_runner(blocked=True)
        self.wait_for(lambda: (self.base / "entered").exists(), "blocked preflight")
        record.parent.mkdir()
        original = self.foreign_record.read_bytes()
        record.write_bytes(original)
        (self.base / "release").touch()
        self.assertEqual(self.runner.wait(timeout=25), 1, self.log.read_text())
        self.assertTrue(running(foreign["pid"]), "refused runner stopped the foreign owner")
        self.assertEqual(record.read_bytes(), original, "refused runner changed the foreign record")
        self.assertFalse((self.output / "process-capture.log").exists())
        self.assertFalse((self.output / "process-stop.log").exists())
        artifact = self.read_artifact("serve")
        self.assertIn("process run directory already exists", artifact["notes"])
        print(json.dumps({"oracle": "preexisting process directory", "exit": 1,
                          "foreign_pid": foreign["pid"], "alive_after": True,
                          "owner_unchanged": True, "verdict": artifact["verdict"],
                          "stage": artifact["failing_stage"]}), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
