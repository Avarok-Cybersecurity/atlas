# SPDX-License-Identifier: AGPL-3.0-only
"""Real subprocess checks for cooperative rank shutdown."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from launch_process import terminate


class ShutdownTests(unittest.TestCase):
    def test_leader_can_notify_worker_before_worker_is_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = subprocess.Popen([sys.executable, '-c', '''
import signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(9))
(root/'worker-ready').touch()
while not (root/'shutdown').exists(): time.sleep(.01)
''', directory], start_new_session=True)
            leader = subprocess.Popen([sys.executable, '-c', '''
import signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
def stop(*_):
    (root/'shutdown').touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(root/'leader-ready').touch()
while True: time.sleep(.01)
''', directory], start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not all((root/name).exists() for name in ['worker-ready', 'leader-ready']):
                    if time.monotonic() > deadline:
                        self.fail('subprocess readiness deadline')
                    time.sleep(.01)
                self.assertEqual(terminate([worker, leader], 1, leader=leader), [])
                self.assertEqual(leader.returncode, 0)
                self.assertEqual(worker.returncode, 0)
            finally:
                terminate([worker, leader], .1)

    def test_unresponsive_leader_and_worker_share_one_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = "import signal,time,sys; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(sys.argv[1]).touch(); time.sleep(60)"
            processes = [subprocess.Popen([sys.executable, '-c', script, str(root/str(i))],
                                          start_new_session=True) for i in range(2)]
            try:
                deadline = time.monotonic() + 5
                while not all((root/str(i)).exists() for i in range(2)):
                    if time.monotonic() > deadline:
                        self.fail('subprocess readiness deadline')
                    time.sleep(.01)
                started = time.monotonic()
                self.assertEqual(terminate(processes, .2, leader=processes[1]), [])
                self.assertLess(time.monotonic()-started, 1.5)
                self.assertTrue(all(p.returncode == -9 for p in processes))
            finally:
                terminate(processes, .1)


if __name__ == '__main__':
    unittest.main()
