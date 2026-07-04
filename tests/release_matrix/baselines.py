"""Snapshot-baseline helpers: load/write per-label tps baselines under
tests/baselines/<label>.json, used by test_tps_no_regression."""

import json
import os

import pytest

from release_matrix.specs import BASELINE_DIR


def _baseline_path(label):
    return os.path.join(BASELINE_DIR, f"{label}.json")


def _load_baseline(label):
    path = _baseline_path(label)
    if not os.path.isfile(path):
        pytest.fail(f"no baseline for {label} at {path} — run with --update-baselines once, "
                    f"review the diff, and commit it")
    with open(path) as f:
        return json.load(f)


def _maybe_update_baseline(request, label, key, value):
    if request.config.getoption("--update-baselines"):
        os.makedirs(BASELINE_DIR, exist_ok=True)
        path = _baseline_path(label)
        data = json.load(open(path)) if os.path.isfile(path) else {}
        data[key] = value
        json.dump(data, open(path, "w"), indent=2)


# NOTE: --update-baselines and the `multirank` marker are registered in the
# sibling tests/conftest.py (pytest only loads pytest_addoption/
# pytest_configure hooks from conftest.py / plugins, not from a plain test
# module — verified: defining them here means `--update-baselines` is
# rejected as an unrecognized argument).
