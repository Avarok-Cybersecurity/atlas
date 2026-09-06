#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Real Linux process ownership oracles; no GPU or engine dependency."""
import json
import errno
import importlib.util
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import process_launch
import process_launch_proc

MANAGER = Path(__file__).with_name("process_launch.py")


class ProcessEnvironmentTests(unittest.TestCase):
    def environment(self, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.json"
            path.write_text(json.dumps(values))
            return process_launch.launch_environment(SimpleNamespace(env_json=str(path), env=[]))

    def test_multisequence_profile_is_explicit_and_preserved(self):
        for key in ("ATLAS_MS_PROFILE", "ATLAS_SSM_MS_PROFILE", "ATLAS_SSM_DETAIL_PROFILE"):
            for value in ("0", "1"):
                with self.subTest(key=key, value=value):
                    self.assertEqual(self.environment({key: value})[key], value)

    def test_unknown_diagnostic_still_refuses(self):
        with self.assertRaisesRegex(ValueError, "environment key is not allowed: UNLISTED_DIAGNOSTIC"):
            self.environment({"UNLISTED_DIAGNOSTIC": "1", "ATLAS_MS_PROFILE": "1"})


@unittest.skipUnless(sys.platform == "linux", "requires Linux /proc and pidfds")
class ProcessLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.record = self.root / "owner.json"
        self.evidence = self.root / "process.json"
        self.log = self.root / "server.log"
        self.argv_file = self.root / "argv.json"
        self.owner = None

    def tearDown(self):
        if self.owner:
            self.record.write_text(json.dumps(self.owner))
            self.call("stop", "--timeout", "0.15")
        self.temp.cleanup()

    def call(self, operation, *extra):
        command = [sys.executable, str(MANAGER), operation,
                   "--record", str(self.record)]
        if operation in ("start", "capture"):
            command += ["--evidence", str(self.evidence)]
        if operation == "start":
            command += ["--log", str(self.log)]
            if "--argv-nul" not in extra:
                command += ["--argv-json", str(self.argv_file)]
        environment = dict(os.environ, CAMPAIGN_SECRET_SENTINEL="must-never-be-retained",
                           HF_TOKEN="must-never-reach-child")
        return subprocess.run(command + list(extra), text=True, capture_output=True,
                              env=environment, timeout=15)

    def start(self, code="import time; time.sleep(300)", *extra):
        self.argv_file.write_text(json.dumps([sys.executable, "-u", "-c", code]))
        result = self.call("start", *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.owner = json.loads(self.record.read_text())
        return json.loads(self.evidence.read_text())

    @staticmethod
    def running(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False

    def test_actual_snapshot_and_environment_are_owned_and_minimal(self):
        env_file = self.root / "env.json"
        env_file.write_text(json.dumps({"RUST_LOG": "info", "HF_HUB_OFFLINE": "1"}))
        snapshot = self.start("import time; time.sleep(300)", "--env-json", str(env_file))
        for key in ("pid", "start_ticks", "boot_id", "run_marker", "pgid", "sid",
                    "argv", "executable", "executable_sha256"):
            self.assertEqual(snapshot[key], self.owner[key], key)
        self.assertEqual(snapshot["environment"]["RUST_LOG"], "info")
        self.assertEqual(snapshot["environment"]["ATLAS_CAMPAIGN_RUN_TOKEN"],
                         self.owner["run_marker"])
        self.assertTrue(Path(snapshot["executable"]).is_absolute())
        self.assertEqual(len(snapshot["executable_sha256"]), 64)
        actual_environment = Path(f"/proc/{snapshot['pid']}/environ").read_bytes()
        self.assertNotIn(b"must-never-reach-child", actual_environment)
        self.assertEqual(self.call("capture").returncode, 0)
        for path in (self.record, self.evidence, self.log):
            self.assertNotIn("must-never-be-retained", path.read_text())
            self.assertNotIn("CAMPAIGN_SECRET_SENTINEL", path.read_text())

    def test_stale_pid_and_foreign_token_refuse_without_signalling(self):
        snapshot = self.start()
        for key, bad in (("start_ticks", snapshot["start_ticks"] + 1),
                         ("run_marker", "foreign-run-token")):
            changed = dict(self.owner, **{key: bad})
            self.record.write_text(json.dumps(changed))
            self.assertNotEqual(self.call("capture").returncode, 0)
            self.assertFalse(json.loads(self.evidence.read_text())["running"])
            self.assertNotEqual(self.call("stop", "--timeout", "0.1").returncode, 0)
            self.assertTrue(self.running(snapshot["pid"]))
        self.record.write_text(json.dumps(self.owner))
        self.assertEqual(self.call("capture").returncode, 0)

    def test_recorded_diagnostics_are_explicit_and_preserved_in_actual_process_evidence(self):
        for key in ("ATLAS_DECODE_TIMING", "ATLAS_DENSE_FP8", "ATLAS_MS_PROFILE",
                    "ATLAS_SSM_MS_PROFILE", "ATLAS_SSM_DETAIL_PROFILE"):
            env_file = self.root / "env.json"
            self.argv_file.write_text(json.dumps([sys.executable, "-c", "import time; time.sleep(300)"]))
            env_file.write_text(json.dumps({key: "1", "UNLISTED_DIAGNOSTIC": "1"}))
            refused = self.call("start", "--env-json", str(env_file))
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("not allowed", refused.stderr)
            self.assertFalse(self.record.exists(), "unknown keys must refuse before process creation")

            env_file.write_text(json.dumps({key: "1"}))
            snapshot = self.start("import time; time.sleep(300)", "--env-json", str(env_file))
            actual = Path(f"/proc/{snapshot['pid']}/environ").read_bytes().split(b"\0")
            self.assertIn((key + "=1").encode(), actual)
            for proof in (snapshot, self.owner):
                self.assertEqual(proof["environment"][key], "1")
            captured = self.call("capture")
            self.assertEqual(captured.returncode, 0, captured.stderr)
            self.assertEqual(json.loads(self.evidence.read_text())["environment"][key], "1")
            stopped = self.call("stop", "--timeout", "0.2")
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertFalse(self.running(snapshot["pid"]))
            self.owner = None
            self.record.unlink()


    def test_stop_reaps_owned_group_children_even_when_they_ignore_term(self):
        code = ("import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)']); "
                "print(p.pid,flush=True); time.sleep(300)")
        snapshot = self.start(code)
        until = time.monotonic() + 3
        while not self.log.read_text().strip() and time.monotonic() < until:
            time.sleep(0.02)
        child_pid = int(self.log.read_text().strip())
        result = self.call("stop", "--timeout", "0.2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.running(snapshot["pid"]))
        self.assertFalse(self.running(child_pid))
        self.assertEqual(self.call("stop").returncode, 0)

    def test_leader_exit_between_stat_and_environment_read_is_absent(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 start_new_session=True,
                                 env=dict(os.environ, ATLAS_CAMPAIGN_RUN_TOKEN='race-fixture'))
        owner = dict(process_launch_proc.process_stat(child.pid), run_marker='race-fixture')
        original = process_launch_proc.marker_matches
        observed = []

        def disappear(pid, marker):
            if pid != child.pid or observed:
                return original(pid, marker)
            # Resolve the proc directory while live, then let the exact owned
            # child exit/reap before the kernel opens its environ entry.
            directory = os.open(f'/proc/{pid}', os.O_RDONLY | os.O_DIRECTORY)
            try:
                child.kill()
                child.wait(timeout=3)
                try:
                    descriptor = os.open('environ', os.O_RDONLY, dir_fd=directory)
                except ProcessLookupError as error:
                    observed.append(error.errno)
                    raise
                else:
                    os.close(descriptor)
                    self.fail('kernel did not produce the expected procfs ESRCH race')
            finally:
                os.close(directory)

        try:
            with patch.object(process_launch_proc, 'marker_matches', side_effect=disappear):
                self.assertEqual(process_launch_proc.owned_members(owner), [])
            self.assertEqual(observed, [errno.ESRCH])
            self.assertEqual(process_launch_proc.stop(owner, 0.1)['status'], 'stopped')
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)

    def test_leader_environment_permission_and_io_errors_still_refuse(self):
        snapshot = self.start()
        for error in (PermissionError(errno.EACCES, 'denied'), OSError(errno.EIO, 'I/O error')):
            with self.subTest(errno=error.errno):
                with patch.object(process_launch_proc, 'marker_matches', side_effect=error):
                    with self.assertRaises(type(error)):
                        process_launch_proc.owned_members(self.owner)
                self.assertTrue(self.running(snapshot['pid']))

    def test_failed_launch_and_unknown_environment_refuse(self):
        self.argv_file.write_text(json.dumps(["/no/such/campaign-engine"]))
        failed = self.call("start")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("No such file", failed.stderr)
        self.record.unlink(missing_ok=True)
        env_file = self.root / "env.json"
        env_file.write_text(json.dumps({"HF_TOKEN": "never-pass-this"}))
        refused = self.call("start", "--env-json", str(env_file))
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("not allowed", refused.stderr)
        self.assertNotIn("never-pass-this", refused.stderr)

    def test_duplicate_record_refuses_before_launch(self):
        snapshot = self.start()
        duplicate = self.call("start")
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertEqual(json.loads(self.record.read_text()), self.owner)
        self.assertTrue(self.running(snapshot["pid"]))

    def test_nul_argv_preserves_empty_arguments(self):
        argv = [sys.executable, "-u", "-c", "import time; time.sleep(300)", ""]
        nul = self.root / "argv.nul"
        nul.write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
        result = self.call("start", "--argv-nul", str(nul))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.owner = json.loads(self.record.read_text())
        self.assertEqual(self.owner["argv"], argv)

    def test_foreign_group_member_refuses_before_signalling_any_process(self):
        code = ("import subprocess,sys,time,os; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],"
                "env=dict(os.environ,ATLAS_CAMPAIGN_RUN_TOKEN='foreign-child')); "
                "print(p.pid,flush=True); time.sleep(300)")
        snapshot = self.start(code)
        until = time.monotonic() + 3
        while not self.log.read_text().strip() and time.monotonic() < until:
            time.sleep(0.02)
        child = int(self.log.read_text().strip())
        descriptor = os.pidfd_open(child)
        try:
            result = self.call("stop", "--timeout", "0.1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("foreign member", result.stderr)
            self.assertTrue(self.running(snapshot["pid"]))
            self.assertTrue(self.running(child))
        finally:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            os.close(descriptor)

    def test_immediate_failure_cleans_up_children(self):
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
                "print(p.pid,flush=True); sys.exit(7)")
        self.argv_file.write_text(json.dumps([sys.executable, "-u", "-c", code]))
        failed = self.call("start")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("server exited during launch: 7", failed.stderr)
        child = int(self.log.read_text().strip())
        self.assertFalse(self.running(child))
        self.assertEqual(json.loads(self.record.read_text())["status"], "failed")

    def test_signal_during_start_cannot_orphan_the_new_session(self):
        code = "import os,time; print(os.getpid(),flush=True); time.sleep(300)"
        self.argv_file.write_text(json.dumps([sys.executable, "-u", "-c", code]))
        command = [sys.executable, str(MANAGER), "start", "--record", str(self.record),
                   "--evidence", str(self.evidence), "--log", str(self.log),
                   "--argv-json", str(self.argv_file)]
        manager = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        descriptor = None
        try:
            until = time.monotonic() + 3
            while time.monotonic() < until:
                if self.log.exists() and self.log.read_text().strip():
                    break
                time.sleep(0.001)
            child = int(self.log.read_text().strip())
            descriptor = os.pidfd_open(child)
            manager.send_signal(signal.SIGTERM)
            manager.communicate(timeout=5)
            self.assertNotEqual(manager.returncode, 0)
            until = time.monotonic() + 1
            while self.running(child) and time.monotonic() < until:
                time.sleep(0.01)
            self.assertFalse(self.running(child), "SIGTERM orphaned the new server session")
        finally:
            if descriptor is not None:
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.close(descriptor)
            if manager.poll() is None:
                manager.kill()
                manager.wait()

    @unittest.skipUnless(importlib.util.find_spec("setproctitle"),
                         "requires the optional real setproctitle package")
    def test_renamed_child_retains_marker_and_can_be_stopped(self):
        child_code = ("import os,time,setproctitle; "
                      "setproctitle.setproctitle('VLLM::EngineCore_DP0'); "
                      "print(os.getpid(),flush=True); time.sleep(300)")
        parent_code = ("import subprocess,sys,time; "
                       "subprocess.Popen([sys.executable,'-c'," + repr(child_code) + "]); "
                       "time.sleep(300)")
        original = self.start(parent_code)
        until = time.monotonic() + 3
        while not self.log.read_text().strip() and time.monotonic() < until:
            time.sleep(0.01)
        child = int(self.log.read_text().strip())
        descriptor = os.pidfd_open(child)
        try:
            self.assertIn(b"VLLM::EngineCore_DP0", Path(f"/proc/{child}/cmdline").read_bytes())
            marker = ("ATLAS_CAMPAIGN_RUN_TOKEN=" + self.owner["run_marker"]).encode()
            self.assertIn(marker, Path(f"/proc/{child}/environ").read_bytes(),
                          "setproctitle erased the child's ownership marker")
            self.assertEqual(original["environment"]["SPT_NOENV"], "1")
            self.assertEqual(self.call("capture").returncode, 0)
            self.assertEqual(json.loads(self.evidence.read_text())["argv"], original["argv"])
            result = self.call("stop", "--timeout", "0.2")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(self.running(child))
        finally:
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.close(descriptor)

    def test_conflicting_process_title_environment_refuses(self):
        env_file = self.root / "env.json"
        env_file.write_text(json.dumps({"SPT_NOENV": ""}))
        self.argv_file.write_text(json.dumps([sys.executable, "-c", "pass"]))
        result = self.call("start", "--env-json", str(env_file))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SPT_NOENV must be 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
