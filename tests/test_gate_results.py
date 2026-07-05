#!/usr/bin/env python3
"""Self-test for tests/gate_results.py — the serve-matrix release gate.

Pure stdlib (unittest); no server, no GPU. Validates the two properties that
matter: the per-model bars, and — the reason this gate exists — that a planned
model which never produced a result FAILS coverage instead of being invisible.

    python3 -m unittest tests.test_gate_results        # from repo root
    python3 tests/test_gate_results.py                 # direct
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_results as G  # noqa: E402


def _model_result(model, *, coh=("PASS", "PASS", "PASS"),
                  fib=("PASS",), tools=("PASS",), tps=(42.0,)):
    return {
        "model": model,
        "coherence": [{"status": s} for s in coh],
        "fibonacci": [{"status": s} for s in fib],
        "tool_calls": [{"status": s} for s in tools],
        "tps": [{"tps": v} for v in tps],
    }


class VerdictBars(unittest.TestCase):
    def test_clean_pass(self):
        self.assertEqual(G.verdict(_model_result("m")), [])

    def test_creative_probe_may_miss(self):
        # 2/3 coherence still passes (tolerates the temp>0 creative probe).
        self.assertEqual(G.verdict(_model_result("m", coh=("PASS", "PASS", "FAIL"))), [])

    def test_coherence_below_bar(self):
        self.assertIn("coherence(1/3)", G.verdict(_model_result("m", coh=("PASS", "FAIL", "FAIL"))))

    def test_fibonacci_must_exec(self):
        self.assertIn("fibonacci", G.verdict(_model_result("m", fib=("FAIL",))))

    def test_known_gap_tools_all_na_pass(self):
        # A parser that is a known gap scores every tool test N/A — must not fail.
        self.assertEqual(G.verdict(_model_result("m", tools=("N/A", "N/A"))), [])

    def test_supported_parser_all_fail_is_regression(self):
        self.assertIn("tool_calls", G.verdict(_model_result("m", tools=("FAIL", "WARN"))))

    def test_zero_tps_fails(self):
        self.assertIn("tps(0)", G.verdict(_model_result("m", tps=(0.0,))))

    def test_no_baseline_liveness_only_passes(self):
        # Positive tps, no baseline, not required -> pass (liveness only).
        self.assertEqual(G.verdict(_model_result("m", tps=(30.0,)), baseline=None), [])

    def test_no_baseline_required_fails(self):
        v = G.verdict(_model_result("m", tps=(30.0,)), baseline=None, require_baseline=True)
        self.assertIn("tps(no-baseline)", v)

    def test_tps_within_tolerance_passes(self):
        # 46 vs baseline 50 -> 8% down, inside the 10% band.
        self.assertEqual(G.verdict(_model_result("m", tps=(46.0,)), baseline={"tps": 50.0}), [])

    def test_tps_regression_fails(self):
        # 30 vs baseline 50 -> 40% down, a real regression tps(0) can't catch.
        v = G.verdict(_model_result("m", tps=(30.0,)), baseline={"tps": 50.0})
        self.assertTrue(any(b.startswith("tps(") and "<" in b for b in v), v)

    def test_dead_server_beats_baseline_check(self):
        # avg<=0 is tps(0) even when a baseline exists — liveness first.
        self.assertIn("tps(0)", G.verdict(_model_result("m", tps=(0.0,)), baseline={"tps": 50.0}))


class Coverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, label, result):
        with open(os.path.join(self.tmp, f"{label}.json"), "w") as f:
            json.dump(result, f)

    def _manifest(self, labels):
        man = {"generated_by": "test", "labels": [{"label": l, "model": m} for l, m in labels]}
        with open(os.path.join(self.tmp, G.MANIFEST_NAME), "w") as f:
            json.dump(man, f)

    def test_full_coverage_passes(self):
        self._manifest([("a", "M/a"), ("b", "M/b")])
        self._write("a", _model_result("M/a"))
        self._write("b", _model_result("M/b"))
        roster = G.load_manifest(self.tmp, None)
        verified, total, failed = G.gate_manifest(self.tmp, roster)
        self.assertEqual((verified, total, failed), (2, 2, []))

    def test_missing_model_is_a_failure(self):
        # 'b' was planned but never booted -> no b.json. This MUST fail.
        self._manifest([("a", "M/a"), ("b", "M/b")])
        self._write("a", _model_result("M/a"))
        roster = G.load_manifest(self.tmp, None)
        verified, total, failed = G.gate_manifest(self.tmp, roster)
        self.assertEqual(total, 2)
        self.assertEqual(verified, 1)
        self.assertEqual([f[0] for f in failed], ["M/b"])
        self.assertIn("no-result", failed[0][1])

    def test_present_but_below_bar_fails(self):
        self._manifest([("a", "M/a")])
        self._write("a", _model_result("M/a", fib=("FAIL",)))
        roster = G.load_manifest(self.tmp, None)
        verified, total, failed = G.gate_manifest(self.tmp, roster)
        self.assertEqual((verified, total), (0, 1))
        self.assertIn("fibonacci", failed[0][1])

    def test_no_manifest_returns_none(self):
        self.assertIsNone(G.load_manifest(self.tmp, None))


class Baselines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bdir = tempfile.mkdtemp()

    def _write(self, label, result):
        with open(os.path.join(self.tmp, f"{label}.json"), "w") as f:
            json.dump(result, f)

    def _manifest(self, labels):
        man = {"labels": [{"label": l, "model": m} for l, m in labels]}
        with open(os.path.join(self.tmp, G.MANIFEST_NAME), "w") as f:
            json.dump(man, f)

    def test_update_writes_avg_and_gate_then_passes(self):
        self._manifest([("a", "M/a")])
        self._write("a", _model_result("M/a", tps=(40.0, 60.0)))  # avg 50
        roster = G.load_manifest(self.tmp, None)
        written = G.update_baselines(self.tmp, roster, self.bdir)
        self.assertEqual(written, [("a", 50.0)])
        self.assertEqual(G.load_baseline("a", self.bdir), {"tps": 50.0})
        # Re-run the SAME result against the fresh baseline -> clean pass.
        _, _, failed = G.gate_manifest(self.tmp, roster, self.bdir)
        self.assertEqual(failed, [])

    def test_update_skips_dead_server(self):
        self._manifest([("a", "M/a")])
        self._write("a", _model_result("M/a", tps=(0.0,)))
        roster = G.load_manifest(self.tmp, None)
        self.assertEqual(G.update_baselines(self.tmp, roster, self.bdir), [])

    def test_regression_caught_by_gate(self):
        self._manifest([("a", "M/a")])
        G.write_baseline("a", {"tps": 50.0}, self.bdir)
        self._write("a", _model_result("M/a", tps=(30.0,)))  # 40% down
        roster = G.load_manifest(self.tmp, None)
        _, _, failed = G.gate_manifest(self.tmp, roster, self.bdir)
        self.assertEqual([f[0] for f in failed], ["M/a"])

    def test_require_baselines_blocks_unblessed_model(self):
        self._manifest([("a", "M/a")])
        self._write("a", _model_result("M/a", tps=(50.0,)))  # healthy but no baseline
        roster = G.load_manifest(self.tmp, None)
        _, _, ok = G.gate_manifest(self.tmp, roster, self.bdir, require_baseline=False)
        self.assertEqual(ok, [])                       # default: liveness-only pass
        _, _, blocked = G.gate_manifest(self.tmp, roster, self.bdir, require_baseline=True)
        self.assertEqual([f[0] for f in blocked], ["M/a"])


class MainExit(unittest.TestCase):
    """Exercise the process-level contract: exit 0 ships, exit 1 blocks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _run(self, extra=()):
        argv = sys.argv
        sys.argv = ["gate_results.py", "--results-dir", self.tmp, *extra]
        try:
            return G.main()
        finally:
            sys.argv = argv

    def test_no_manifest_blocks_by_default(self):
        self.assertEqual(self._run(), 1)  # PCND: no silent pass without coverage

    def test_missing_results_dir(self):
        argv = sys.argv
        sys.argv = ["gate_results.py", "--results-dir", os.path.join(self.tmp, "nope")]
        try:
            self.assertEqual(G.main(), 1)
        finally:
            sys.argv = argv

    def test_full_pass_ships(self):
        man = {"labels": [{"label": "a", "model": "M/a"}]}
        with open(os.path.join(self.tmp, G.MANIFEST_NAME), "w") as f:
            json.dump(man, f)
        with open(os.path.join(self.tmp, "a.json"), "w") as f:
            json.dump(_model_result("M/a"), f)
        self.assertEqual(self._run(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
