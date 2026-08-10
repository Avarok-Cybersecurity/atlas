#!/usr/bin/env python3
"""SSOT for the serve-matrix harness layout, rooted at THIS checkout.

`tests/run_all_models.py` (writer) and `tests/gate_results.py` (reader) must
agree on where results live. They previously each carried their own answer —
the orchestrator held an absolute `/workspace/atlas/tests/...` path into a
DIFFERENT working copy while the gate resolved `dirname(__file__)`. The two
disagreed, so the gate could grade months-old JSON from another checkout as if
it were the current run's output.

Every path here derives from this file's own location, so a worktree, a clone,
or a container copy each get their own results — no absolute path is baked in.
CLI overrides (`--results-dir`, `--baseline-dir`) still win; these are only the
defaults they start from.
"""

import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-model `<label>.json` + `<label>.log` written by run_all_models.py, plus
# the `_manifest.json` coverage roster the gate enforces. Gitignored: it is a
# run artifact, not source.
RESULTS_DIR = os.path.join(TESTS_DIR, "all_models_results")

# Committed per-label blessed tokens/sec ({"tps": N}); the gate's regression bar.
BASELINE_DIR = os.path.join(TESTS_DIR, "baselines")

# The per-model probe suite the orchestrator shells out to.
SUITE_PATH = os.path.join(TESTS_DIR, "single_gpu_suite.py")
