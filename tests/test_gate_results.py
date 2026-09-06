#!/usr/bin/env python3
"""Self-test for tests/gate_results.py — the serve-matrix release gate.

Pure stdlib (unittest); no server, no GPU. Checks the per-model bars, missing
and invalid evidence, declared checkpoint identity, baseline eligibility, and
the process exit status that the release workflow consumes.

    python3 -m unittest tests.test_gate_results        # from repo root
    python3 tests/test_gate_results.py                 # direct
"""

import contextlib
import io
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


class EvidenceContracts(unittest.TestCase):
    """Reject missing or misleading evidence, not just explicit FAIL rows."""

    def test_every_required_probe_must_supply_evidence(self):
        for key in ("coherence", "fibonacci", "tool_calls", "tps"):
            for kind in ("absent", "empty", "null"):
                with self.subTest(probe=key, evidence=kind):
                    result = _model_result("m")
                    if kind == "absent":
                        del result[key]
                    else:
                        result[key] = [] if kind == "empty" else None
                    self.assertEqual(G.verdict(result), [f"{key}(no-evidence)"])

    def test_status_reason_cannot_masquerade_as_a_pass_or_waiver(self):
        cases = (
            ("coherence", "FAIL (expected PASS)", ["coherence(0/3)"]),
            ("fibonacci", "BYPASS", ["fibonacci"]),
            ("tool_calls", "FAIL (expected N/A)", ["tool_calls"]),
            ("tool_calls", "NOT PASS", ["tool_calls"]),
        )
        for key, status, expected in cases:
            with self.subTest(probe=key, status=status):
                result = _model_result("m")
                for row in result[key]:
                    row["status"] = status
                self.assertEqual(G.verdict(result), expected)

    def test_malformed_probe_collections_are_rejected_by_name(self):
        for key in ("coherence", "fibonacci", "tool_calls", "tps"):
            for rows in ({}, "PASS", [None], ["PASS"]):
                with self.subTest(probe=key, rows=rows):
                    result = _model_result("m")
                    result[key] = rows
                    self.assertEqual(G.verdict(result), [f"{key}(invalid)"])

    def test_status_annotations_emitted_by_the_suite_remain_valid(self):
        self.assertEqual(G.verdict(_model_result(
            "m", fib=("PASS (plain-text)",),
            tools=("N/A (parser not supported)",))), [])

    def test_invalid_tps_cannot_hide_alongside_a_valid_sample(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "42", None,
                      10 ** 400):
            with self.subTest(value=value):
                self.assertEqual(G.verdict(_model_result("m", tps=(42.0, value))),
                                 ["tps(invalid)"])

    def test_existing_invalid_baseline_cannot_disable_the_regression_bar(self):
        for baseline in ({}, [], {"tps": 0}, {"tps": -1}, {"tps": True},
                         {"tps": float("nan")}, {"tps": float("inf")}, {"tps": "50"}):
            with self.subTest(baseline=baseline):
                self.assertEqual(G.verdict(_model_result("m"), baseline=baseline),
                                 ["tps(invalid-baseline)"])

    def test_tps_tolerance_boundary_is_inclusive(self):
        self.assertEqual(G.verdict(_model_result("m", tps=(45.0,)),
                                   baseline={"tps": 50.0}), [])
        self.assertEqual(G.verdict(_model_result("m", tps=(44.0,)),
                                   baseline={"tps": 50.0}), ["tps(44.0<45.0)"])


class ArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bdir = os.path.join(self.tmp.name, "baselines")

    def _write(self, result):
        with open(os.path.join(self.tmp.name, "a.json"), "w") as f:
            json.dump(result, f)

    def test_manifest_model_must_match_the_observed_result(self):
        self._write(_model_result("M/wrong-checkpoint"))
        with contextlib.redirect_stdout(io.StringIO()):
            got = G.gate_manifest(self.tmp.name, [("a", "M/expected")], self.bdir)
        self.assertEqual(got, (0, 1, [("M/expected", ["model-mismatch"])]))

    def test_baseline_update_requires_all_correctness_bars_and_identity(self):
        cases = (
            _model_result("M/a", coh=("FAIL",)),
            _model_result("M/a", fib=("FAIL",)),
            _model_result("M/a", tools=("FAIL",)),
            _model_result("M/a", tps=(float("nan"),)),
            _model_result("M/a", tps=(float("inf"),)),
            _model_result("M/wrong-checkpoint"),
        )
        for result in cases:
            with self.subTest(result=result):
                G.write_baseline("a", {"tps": 50.0}, self.bdir)
                self._write(result)
                self.assertEqual(G.update_baselines(
                    self.tmp.name, [("a", "M/a")], self.bdir), [])
                self.assertEqual(G.load_baseline("a", self.bdir), {"tps": 50.0})

    def test_malformed_baseline_is_present_but_invalid(self):
        os.makedirs(self.bdir)
        self._write(_model_result("M/a"))
        for content in ("{broken json", "null", "[]"):
            with self.subTest(content=content):
                with open(os.path.join(self.bdir, "a.json"), "w") as f:
                    f.write(content)
                with contextlib.redirect_stdout(io.StringIO()):
                    got = G.gate_manifest(self.tmp.name, [("a", "M/a")], self.bdir)
                self.assertEqual(got, (0, 1, [("M/a", ["tps(invalid-baseline)"])]))

    def test_cli_blocks_a_partial_artifact(self):
        with open(os.path.join(self.tmp.name, G.MANIFEST_NAME), "w") as f:
            json.dump({"labels": [{"label": "a", "model": "M/a"}]}, f)
        self._write({"model": "M/a", "coherence": []})
        # Use the actual entry point: a gate that only prints FAIL is unsafe
        # for the documented release command sequence.
        import subprocess
        result = subprocess.run(
            [sys.executable, G.__file__, "--results-dir", self.tmp.name,
             "--baseline-dir", self.bdir], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        for bar in ("coherence", "fibonacci", "tool_calls", "tps"):
            self.assertIn(f"{bar}(no-evidence)", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
