#!/usr/bin/env python3
"""Host-only subprocess checks for serve-matrix artifact freshness.

The real orchestrator launches a tiny Python fixture instead of a model suite.
No container, server, network request, checkpoint, or GPU is used.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_results as G
import run_all_models as M
from test_gate_results import _model_result


class OrchestratorEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "a.json"
        self.suite = self.root / "fixture_suite.py"
        self.spec = M.TestSpec("a", "M/a")
        self.roster = [("a", "M/a")]
        for name, value in (
            ("RESULTS_DIR", self.tmp.name),
            ("MANIFEST_PATH", str(self.root / G.MANIFEST_NAME)),
            ("SUITE", str(self.suite)),
            ("ROUNDS", [[("head", self.spec)]]),
        ):
            patcher = patch.object(M, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_old_result(self):
        self.output.write_text(json.dumps(_model_result("M/a", tps=(42.0,))))

    def _launch(self, *, exit_code, write_output):
        # The fixture understands the same process arguments as the real
        # single_gpu_suite. Keep a distinct TPS value so reuse is observable.
        fresh = _model_result("M/a", tps=(63.0,))
        self.suite.write_text(
            "import argparse, json, pathlib, sys\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--base-url')\n"
            "p.add_argument('--model')\n"
            "p.add_argument('--output')\n"
            "a = p.parse_args()\n"
            "assert a.model == 'M/a'\n"
            + (f"pathlib.Path(a.output).write_text({json.dumps(fresh)!r})\n"
               if write_output else "")
            + f"sys.exit({exit_code})\n"
        )
        job = M.run_suite("head", self.spec, 8888)
        with contextlib.redirect_stdout(io.StringIO()):
            result = M.wait_and_read(job)
        return result, fresh

    def _gate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return G.gate_manifest(self.tmp.name, self.roster,
                                   str(self.root / "baselines"))

    def _write_manifest(self):
        with contextlib.redirect_stdout(io.StringIO()):
            M.write_manifest(run_singlegpu=True, skip=set(), only_round=None,
                             run_ep2=False, run_tp2=False, run_tpep=False, run_ep4=False)

    def test_new_manifest_invalidates_planned_results_before_any_boot(self):
        self._write_old_result()
        unrelated = self.root / "unplanned.json"
        unrelated.write_text("preserve unrelated run artifact")
        self._write_manifest()
        self.assertEqual(G.load_manifest(self.tmp.name, None), self.roster)
        self.assertEqual(self._gate(), (0, 1, [("M/a", ["no-result"])]))
        self.assertEqual(unrelated.read_text(), "preserve unrelated run artifact")

    def test_failed_subprocess_cannot_reuse_a_previous_result(self):
        self._write_old_result()
        result, _ = self._launch(exit_code=7, write_output=False)
        self.assertIsNone(result)
        self.assertEqual(self._gate(), (0, 1, [("M/a", ["no-result"])]))

    def test_failed_subprocess_cannot_leave_a_gateable_result(self):
        result, _ = self._launch(exit_code=7, write_output=True)
        self.assertIsNone(result)
        self.assertEqual(self._gate(), (0, 1, [("M/a", ["no-result"])]))

    def test_success_without_output_cannot_reuse_a_previous_result(self):
        self._write_old_result()
        result, _ = self._launch(exit_code=0, write_output=False)
        self.assertIsNone(result)
        self.assertEqual(self._gate(), (0, 1, [("M/a", ["no-result"])]))

    def test_successful_subprocess_supplies_fresh_passing_evidence(self):
        self._write_old_result()
        result, fresh = self._launch(exit_code=0, write_output=True)
        self.assertEqual(result, fresh)
        self.assertEqual(self._gate(), (1, 1, []))

    def test_all_labels_are_validated_before_cleanup(self):
        for label in ("../outside", "nested/label", "nested\\label", "_manifest", ""):
            with self.subTest(label=label):
                self._write_old_result()
                old_bytes = self.output.read_bytes()
                M.ROUNDS = [[("head", self.spec), ("head", M.TestSpec(label, "M/b"))]]
                with self.assertRaisesRegex(ValueError, "invalid result label"):
                    self._write_manifest()
                self.assertEqual(self.output.read_bytes(), old_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
