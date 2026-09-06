#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Linux CPU tests: a whole-cell deadline must supervise existing cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HELPER = Path(__file__).with_name("cell_deadline.py")
VICTIM = r'''
import http.server, json, os, pathlib, signal, subprocess, sys, threading, time, urllib.request
helper, folder, mode = sys.argv[1:]
folder = pathlib.Path(folder)
receipt = folder / 'deadline.json'
if mode == 'stubborn':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    def terminate(signum, frame):
        raise InterruptedError('runner interrupted')
    signal.signal(signal.SIGTERM, terminate)
watch = subprocess.Popen([sys.executable, helper, 'watch', '--pid', str(os.getpid()),
    '--timeout-s', '0.35', '--grace-s', '1.5', '--receipt', str(receipt)])
subprocess.run([sys.executable, helper, 'wait-armed', '--receipt', str(receipt),
                '--timeout-s', '2'], check=True)
(folder / 'started.json').write_text(json.dumps({'pid': os.getpid(), 'watchdog_pid': watch.pid}))
stage = 'coherency' if mode == 'trickle' else 'ladder'
try:
    if mode == 'normal':
        pass
    elif mode == 'trickle':
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', '1000000')
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b' '); self.wfile.flush(); time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError): pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with urllib.request.urlopen('http://127.0.0.1:' + str(server.server_port), timeout=0.1) as response:
            response.read()
    else:
        while True: time.sleep(1)
except InterruptedError:
    doc = json.loads(receipt.read_text())
    (folder / 'artifact.json').write_text(json.dumps({'failing_stage': stage,
        'notes': 'deadline exceeded' if doc.get('deadline_exceeded') else 'interrupted',
        'cleanup_ran': True}))
    subprocess.run([sys.executable, helper, 'cancel', '--receipt', str(receipt)], check=True)
    watch.wait(timeout=3)
    raise SystemExit(143)
else:
    subprocess.run([sys.executable, helper, 'cancel', '--receipt', str(receipt)], check=True)
    watch.wait(timeout=3)
'''


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open"),
                     "requires Linux pidfds; no GPU needed")
class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        self.temp.cleanup()

    def call(self, *args):
        return subprocess.run([sys.executable, str(HELPER), *args],
                              text=True, capture_output=True, timeout=5)

    def victim(self, mode):
        process = subprocess.Popen([sys.executable, "-c", VICTIM,
                                    str(HELPER), str(self.folder), mode],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        self.processes.append(process)
        return process

    def wait_started(self, process):
        deadline = time.monotonic() + 3
        path = self.folder / "started.json"
        while not path.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.fail(f"runner exited before armed: {stdout} {stderr}")
            time.sleep(0.01)
        self.assertTrue(path.exists(), "watchdog must acknowledge arming before work")
        return json.loads(path.read_text())

    def assert_not_running(self, pid):
        path = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 2
        while path.exists() and time.monotonic() < deadline:
            if path.read_text().rsplit(")", 1)[1].split()[0] in ("Z", "X"):
                return
            time.sleep(0.01)
        self.assertFalse(path.exists(), f"watchdog {pid} still running")

    def test_deadline_interrupts_nonterminating_stage(self):
        started = time.monotonic()
        process = self.victim("hang")
        identity = self.wait_started(process)
        stdout, stderr = process.communicate(timeout=4)
        self.assertEqual(process.returncode, 143, (stdout, stderr))
        self.assertLess(time.monotonic() - started, 3)
        artifact = json.loads((self.folder / "artifact.json").read_text())
        self.assertEqual(artifact["failing_stage"], "ladder")
        self.assertEqual(artifact["notes"], "deadline exceeded")
        self.assertTrue(artifact["cleanup_ran"])
        self.assert_not_running(identity["watchdog_pid"])

    def test_deadline_bounds_a_trickling_http_body(self):
        process = self.victim("trickle")
        self.wait_started(process)
        stdout, stderr = process.communicate(timeout=4)
        self.assertEqual(process.returncode, 143, (stdout, stderr))
        artifact = json.loads((self.folder / "artifact.json").read_text())
        self.assertEqual(artifact["failing_stage"], "coherency")
        self.assertEqual(artifact["notes"], "deadline exceeded")

    def test_normal_completion_leaves_no_watchdog(self):
        process = self.victim("normal")
        identity = self.wait_started(process)
        stdout, stderr = process.communicate(timeout=4)
        self.assertEqual(process.returncode, 0, (stdout, stderr))
        doc = json.loads((self.folder / "deadline.json").read_text())
        self.assertEqual(doc["status"], "cancelled")
        self.assertFalse(doc["deadline_exceeded"])
        self.assert_not_running(identity["watchdog_pid"])

    def test_operator_interrupt_cancels_watchdog(self):
        process = self.victim("hang")
        identity = self.wait_started(process)
        process.terminate()
        stdout, stderr = process.communicate(timeout=4)
        self.assertEqual(process.returncode, 143, (stdout, stderr))
        self.assertEqual(json.loads((self.folder / "artifact.json").read_text())["notes"], "interrupted")
        self.assert_not_running(identity["watchdog_pid"])

    def test_hard_grace_is_explicit_cleanup_failure(self):
        process = self.victim("stubborn")
        identity = self.wait_started(process)
        process.communicate(timeout=4)
        self.assertEqual(process.returncode, -9)
        doc = json.loads((self.folder / "deadline.json").read_text())
        self.assertEqual(doc["status"], "grace_exceeded")
        self.assertTrue(doc["cleanup_unconfirmed"])
        self.assertTrue(doc["deadline_exceeded"])
        self.assert_not_running(identity["watchdog_pid"])

    def test_invalid_budgets_are_refused(self):
        for value in ("0", "-1", "nan", "inf"):
            result = self.call("watch", "--pid", str(os.getpid()), "--timeout-s", value,
                               "--grace-s", "1", "--receipt", str(self.folder / "deadline.json"))
            self.assertEqual(result.returncode, 2)
            self.assertFalse((self.folder / "deadline.json").exists())

    def test_watch_refuses_a_target_other_than_its_parent(self):
        result = self.call("watch", "--pid", "1", "--timeout-s", "1", "--grace-s", "1",
                           "--receipt", str(self.folder / "deadline.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("direct parent", result.stderr)

    def test_cancel_cannot_signal_a_reused_or_unrelated_pid(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        self.processes.append(sleeper)
        receipt = self.folder / "deadline.json"
        fields = Path(f"/proc/{sleeper.pid}/stat").read_text().rsplit(")", 1)[1].split()
        receipt.write_text(json.dumps({"schema": 1, "kind": "cell-deadline",
            "watchdog_pid": sleeper.pid, "watchdog_start_ticks": int(fields[19]) + 1,
            "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
        result = self.call("cancel", "--receipt", str(receipt))
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(sleeper.poll(), "a stale receipt must never signal another process")

    def test_cancel_refuses_unrelated_pid_even_with_correct_start(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        self.processes.append(sleeper)
        receipt = self.folder / "deadline.json"
        def ticks(pid):
            return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        receipt.write_text(json.dumps({"schema": 1, "kind": "cell-deadline", "status": "armed",
            "target_pid": os.getpid(), "target_start_ticks": ticks(os.getpid()),
            "watchdog_pid": sleeper.pid, "watchdog_start_ticks": ticks(sleeper.pid),
            "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
        result = self.call("cancel", "--receipt", str(receipt))
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(sleeper.poll(), "correct PID and ticks are insufficient watchdog proof")

    def test_unarmed_watchdog_can_be_cancelled_without_waiting_for_budget(self):
        helper = self.folder / "cell_deadline.py"
        helper.write_text(HELPER.read_text().replace("def watch(args):\n",
                                                  "def watch(args):\n    time.sleep(10)\n"))
        receipt = self.folder / "deadline.json"
        process = subprocess.Popen([sys.executable, str(helper), "watch", "--pid", str(os.getpid()),
                                    "--timeout-s", "30", "--grace-s", "1", "--receipt", str(receipt)])
        self.processes.append(process)
        started = time.monotonic()
        result = subprocess.run([sys.executable, str(helper), "cancel", "--receipt", str(receipt),
                                 "--watchdog-pid", str(process.pid)], text=True,
                                capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
        self.assertEqual(process.wait(timeout=2), -15)
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(receipt.exists())

    def test_deadline_during_real_finalizer_preserves_cleanup(self):
        self.run_finalizer_case(False)

    def test_deadline_during_operator_cleanup_preserves_original_stage(self):
        self.run_finalizer_case(True)

    def run_finalizer_case(self, operator_interrupt):
        source = Path(__file__).with_name("cell_finalize.sh")
        self.assertTrue(source.exists(), "copy the real finalizer beside this test")
        (self.folder / "cell_deadline.py").symlink_to(HELPER)
        (self.folder / "validate_artifact.py").write_text("raise SystemExit(0)\n")
        (self.folder / "assemble.py").write_text(
            "import json,pathlib,sys\npathlib.Path(sys.argv[1]).write_text(json.dumps("
            "{'failing_stage':sys.argv[2],'notes':sys.argv[3]}))\n")
        script = r'''
set -uo pipefail
HERE="$1"; OUT="$1"; ARTIFACT="$1/artifact.json"; SOURCE="$2"
FINALIZED=0; MAIN_DONE=1; INTERRUPT_SIG=""; CURRENT_STAGE=ladder
FAILING_STAGE=""; EXTRA_NOTE=""; DRY_RUN=0; CELL_ID=test
CELL_DEADLINE_PID=""; CELL_DEADLINE_RECEIPT="$OUT/deadline.json"
note_fail() { [ -n "$FAILING_STAGE" ] || FAILING_STAGE="$1"; }
add_note() { EXTRA_NOTE="${EXTRA_NOTE:+$EXTRA_NOTE; }$1"; }
step() { :; }; show() { :; }; stage() { CURRENT_STAGE="$1"; }
capture_model_launch() { :; }; capture_engine_identity() { :; }
teardown_owned() {
  sleep 0.6 & child=$!
  while kill -0 "$child" 2>/dev/null; do wait "$child" || true; done
  touch "$OUT/cleanup-complete"
}
build_assemble() { ASSEMBLE=(python3 "$HERE/assemble.py" "$ARTIFACT" "$FAILING_STAGE" "$EXTRA_NOTE"); }
source "$SOURCE"
python3 "$HERE/cell_deadline.py" watch --pid "$$" --timeout-s 0.3 --grace-s 2 --receipt "$CELL_DEADLINE_RECEIPT" &
CELL_DEADLINE_PID=$!
python3 "$HERE/cell_deadline.py" wait-armed --receipt "$CELL_DEADLINE_RECEIPT" --timeout-s 2
if [ "$3" = operator ]; then CURRENT_STAGE=coherency; kill -TERM "$$"; fi
finalize
'''
        mode = "operator" if operator_interrupt else "normal"
        result = subprocess.run(["bash", "-c", script, "fixture", str(self.folder), str(source), mode],
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 143, (result.stdout, result.stderr))
        self.assertTrue((self.folder / "cleanup-complete").exists(), result.stdout)
        doc = json.loads((self.folder / "artifact.json").read_text())
        self.assertEqual(doc["failing_stage"], "coherency" if operator_interrupt else "teardown")
        self.assertIn("deadline exceeded", doc["notes"])
        self.assertIn("deadline.json", doc["notes"])


if __name__ == "__main__":
    unittest.main()
