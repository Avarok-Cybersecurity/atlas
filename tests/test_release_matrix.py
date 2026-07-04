#!/usr/bin/env python3
"""
Release-matrix as real pytest tests, replacing the
run_all_models.py + single_gpu_suite.py + gate_results.py orchestrator/suite/
gate trio with a single pytest test suite.

Run (single-GPU + multi-rank groups in parallel across the two physical GB10
nodes, tests within a group serialized):
    pytest --dist=loadgroup -n 2 tests/test_release_matrix.py

Multi-rank (EP=2 / TP=2 / TP+EP) rounds use BOTH physical GPUs at once, so
they cannot run concurrently with a single-GPU "head" or "worker" round.
Run them as a separate pass:
    pytest --dist=loadgroup -n 2 tests/test_release_matrix.py -m "not multirank"
    pytest tests/test_release_matrix.py -m multirank

Point debugging a single model:
    pytest -k "27B-dense-nvfp4 and test_tps_no_regression" tests/test_release_matrix.py

Update baselines (only right before merge, then review+commit the diff):
    pytest tests/test_release_matrix.py --update-baselines

NOTE: this is a straight behavioral port of tests/run_all_models.py +
tests/single_gpu_suite.py. tests/gate_results.py's zero-floor `verdict()` is
replaced by the per-label baseline comparison in `_load_baseline` /
`test_tps_no_regression`. The old three scripts are left untouched for now;
this file is the new implementation, to be swapped in as a separate change.

Fixtures/test bodies live here; supporting code is split by functional axis
under tests/release_matrix/ (specs, docker lifecycle, HTTP-API helpers,
assertion bodies, baseline helpers) since the combined file grew too long
for one module.
"""

import subprocess

import pytest

from release_matrix.api import (
    SEARCH_TOOL,
    WEATHER_TOOL,
    _tool_calls_supported,
)
from release_matrix.assertions import (
    _assert_coherence,
    _measure_tps,
    _run_fibonacci,
    _run_long_context,
    _run_tool_call,
)
from release_matrix.baselines import _load_baseline, _maybe_update_baseline
from release_matrix.docker import (
    build_ep2_serve_cmd,
    docker_on,
    settle,
    start_container,
    stop_container,
    wait_ep2_ready,
    wait_listening,
    warmup_request,
)
from release_matrix.specs import (
    HEAD_PORT,
    HF_CACHE_HEAD,
    HF_CACHE_WORKER,
    IMAGE,
    MULTIRANK_SPECS,
    NCCL_ENV,
    RDMA_FLAGS,
    SPECS,
    STARTUP_TIMEOUT,
    TPS_TOLERANCE,
    WORKER_IP,
)


def _xdist_group(name):
    return pytest.mark.xdist_group(name=name)


# ─── Fixture: single-GPU deploy ─────────────────────────────────────────

@pytest.fixture(scope="module", params=SPECS, ids=lambda s: s.label)
def deployed_atlas(request):
    spec = request.param
    name = f"atlas-test-{spec.label}"

    try:
        start_container(spec.host, spec, spec.port)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{spec.label}: failed to start container: {e}")

    if not wait_listening(spec.host, name):
        stop_container(spec.host, name)
        pytest.fail(f"{spec.label}: container never reached 'Listening on' "
                    f"within {STARTUP_TIMEOUT}s (or exited early)")

    base_url = f"http://{'localhost' if spec.host == 'head' else WORKER_IP}:{spec.port}/v1"
    warmup_request(spec.host, spec, spec.port)

    yield spec, base_url

    stop_container(spec.host, name)
    settle()


# ─── Fixture: multi-rank (EP=2 / TP=2 / TP+EP) deploy ──────────────────

@pytest.fixture(scope="module", params=MULTIRANK_SPECS, ids=lambda s: s.label)
def deployed_atlas_ep2(request):
    """Brings up rank 0 on head + rank 1 on worker over NCCL/RDMA, same
    launch semantics as run_all_models.py's start_ep2/wait_ep2_ready/
    run_ep2_round (topology-agnostic: reads spec.tp_size/spec.ep_size)."""
    spec = request.param
    rank0_name = f"atlas-ep0-{spec.label}"
    rank1_name = f"atlas-ep1-{spec.label}"

    docker_on("head", f"rm -f {rank0_name}", check=False, capture=True)
    docker_on("worker", f"rm -f {rank1_name}", check=False, capture=True)

    try:
        rank0_serve = build_ep2_serve_cmd(spec, rank=0)
        docker0 = (
            f"run -d --name {rank0_name} --gpus all --ipc=host --network host "
            f"{RDMA_FLAGS}{NCCL_ENV} -e RUST_LOG=info "
            f"-v {HF_CACHE_HEAD}:/root/.cache/huggingface {IMAGE} {rank0_serve}"
        )
        docker_on("head", docker0, check=True, capture=True)

        rank1_serve = build_ep2_serve_cmd(spec, rank=1)
        docker1 = (
            f"run -d --name {rank1_name} --gpus all --ipc=host --network host "
            f"{RDMA_FLAGS}{NCCL_ENV} -e RUST_LOG=info "
            f"-v {HF_CACHE_WORKER}:/root/.cache/huggingface {IMAGE} {rank1_serve}"
        )
        docker_on("worker", docker1, check=True, capture=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{spec.label}: failed to start EP=2 ranks: {e}")

    if not wait_ep2_ready(rank0_name, rank1_name):
        stop_container("head", rank0_name)
        stop_container("worker", rank1_name)
        pytest.fail(f"{spec.label}: EP=2/TP=2 ranks never reached ready state")

    base_url = f"http://localhost:{HEAD_PORT}/v1"
    warmup_request("head", spec, HEAD_PORT)

    yield spec, base_url

    stop_container("head", rank0_name)
    stop_container("worker", rank1_name)
    settle()


# ─── Single-GPU tests ────────────────────────────────────────────────────

class TestReleaseMatrix:

    def test_coherence(self, deployed_atlas, request):
        spec, base_url = deployed_atlas
        request.node.add_marker(_xdist_group(spec.host))
        _assert_coherence(base_url, spec.model, spec.label)

    def test_fibonacci(self, deployed_atlas, request):
        spec, base_url = deployed_atlas
        request.node.add_marker(_xdist_group(spec.host))
        status, detail = _run_fibonacci(base_url, spec.model)
        assert status.startswith("PASS"), f"{spec.label}: fibonacci {status} — {detail}"

    def test_tool_calls(self, deployed_atlas, request):
        spec, base_url = deployed_atlas
        request.node.add_marker(_xdist_group(spec.host))

        if not _tool_calls_supported(spec.model):
            pytest.skip(f"{spec.label}: tool-call parser not wired up for this model (N/A)")

        for name, prompt, tools in [
            ("Weather", "What is the weather in Paris?", [WEATHER_TOOL]),
            ("Search", "Search for the latest NVIDIA GPU benchmarks", [SEARCH_TOOL]),
        ]:
            status, detail = _run_tool_call(base_url, spec.model, name, prompt, tools)
            # Only a structured-but-invalid call is a hard failure. A model
            # answering in plain text ("WARN (no structured tool call)") is
            # not treated as a regression, matching single_gpu_suite.py.
            assert not status.startswith("FAIL"), f"{spec.label}: tool call {name} {status} — {detail}"

    def test_tps_no_regression(self, deployed_atlas, request):
        spec, base_url = deployed_atlas
        request.node.add_marker(_xdist_group(spec.host))

        measurements = _measure_tps(base_url, spec.model)
        for m in measurements:
            assert m["ok"], f"{spec.label}: tps run at max_tokens={m['max_tokens']} failed — {m['preview']!r}"

        avg_tps = sum(m["tps"] for m in measurements) / len(measurements)
        _maybe_update_baseline(request, spec.label, "tps", avg_tps)
        if request.config.getoption("--update-baselines"):
            pytest.skip(f"{spec.label}: baseline updated to {avg_tps:.1f} tok/s, not asserting")

        baseline = _load_baseline(spec.label)["tps"]
        floor = baseline * (1 - TPS_TOLERANCE)
        assert avg_tps >= floor, (
            f"{spec.label}: tps regressed — {avg_tps:.1f} < baseline {baseline:.1f} "
            f"(floor {floor:.1f}, tolerance {TPS_TOLERANCE:.0%})"
        )

    def test_long_context(self, deployed_atlas, request):
        spec, base_url = deployed_atlas
        request.node.add_marker(_xdist_group(spec.host))

        if spec.skip_longctx:
            pytest.skip(f"{spec.label}: long-context marked skip_longctx")

        results = _run_long_context(base_url, spec.model)
        failures = [r for r in results if not r["status"].startswith(("PASS", "OOM"))]
        assert not failures, (
            f"{spec.label}: long-context failures: "
            + "; ".join(f"{r['target']}: {r['status']} ({r['preview']!r})" for r in failures)
        )


# ─── Multi-rank (EP=2 / TP=2 / TP+EP) tests ─────────────────────────────
#
# Same assertion bodies as the single-GPU class, driven off deployed_atlas_ep2
# instead of deployed_atlas. Grouped under a single xdist group ("multirank")
# since these rounds occupy BOTH physical GPUs and cannot run alongside a
# single-GPU "head" or "worker" round; run them as a separate `-m multirank`
# pass (see module docstring).

@pytest.mark.multirank
class TestReleaseMatrixMultiRank:

    def test_coherence(self, deployed_atlas_ep2, request):
        spec, base_url = deployed_atlas_ep2
        request.node.add_marker(_xdist_group("multirank"))
        _assert_coherence(base_url, spec.model, spec.label)

    def test_fibonacci(self, deployed_atlas_ep2, request):
        spec, base_url = deployed_atlas_ep2
        request.node.add_marker(_xdist_group("multirank"))
        status, detail = _run_fibonacci(base_url, spec.model)
        assert status.startswith("PASS"), f"{spec.label}: fibonacci {status} — {detail}"

    def test_tool_calls(self, deployed_atlas_ep2, request):
        spec, base_url = deployed_atlas_ep2
        request.node.add_marker(_xdist_group("multirank"))

        if not _tool_calls_supported(spec.model):
            pytest.skip(f"{spec.label}: tool-call parser not wired up for this model (N/A)")

        for name, prompt, tools in [
            ("Weather", "What is the weather in Paris?", [WEATHER_TOOL]),
            ("Search", "Search for the latest NVIDIA GPU benchmarks", [SEARCH_TOOL]),
        ]:
            status, detail = _run_tool_call(base_url, spec.model, name, prompt, tools)
            assert not status.startswith("FAIL"), f"{spec.label}: tool call {name} {status} — {detail}"

    def test_tps_no_regression(self, deployed_atlas_ep2, request):
        spec, base_url = deployed_atlas_ep2
        request.node.add_marker(_xdist_group("multirank"))

        measurements = _measure_tps(base_url, spec.model)
        for m in measurements:
            assert m["ok"], f"{spec.label}: tps run at max_tokens={m['max_tokens']} failed — {m['preview']!r}"

        avg_tps = sum(m["tps"] for m in measurements) / len(measurements)
        _maybe_update_baseline(request, spec.label, "tps", avg_tps)
        if request.config.getoption("--update-baselines"):
            pytest.skip(f"{spec.label}: baseline updated to {avg_tps:.1f} tok/s, not asserting")

        baseline = _load_baseline(spec.label)["tps"]
        floor = baseline * (1 - TPS_TOLERANCE)
        assert avg_tps >= floor, (
            f"{spec.label}: tps regressed — {avg_tps:.1f} < baseline {baseline:.1f} "
            f"(floor {floor:.1f}, tolerance {TPS_TOLERANCE:.0%})"
        )

    def test_long_context(self, deployed_atlas_ep2, request):
        spec, base_url = deployed_atlas_ep2
        request.node.add_marker(_xdist_group("multirank"))

        if spec.skip_longctx:
            pytest.skip(f"{spec.label}: long-context marked skip_longctx")

        results = _run_long_context(base_url, spec.model)
        failures = [r for r in results if not r["status"].startswith(("PASS", "OOM"))]
        assert not failures, (
            f"{spec.label}: long-context failures: "
            + "; ".join(f"{r['target']}: {r['status']} ({r['preview']!r})" for r in failures)
        )
