#!/usr/bin/env python3
"""
Atlas multi-model test orchestrator.

Pairs models across a head node and an optional worker node, runs
`single_gpu_suite.py` against each, collects JSON results, and writes a
Markdown table to `tests/all_models_results.md`.

Design:
- Each "round" has up to 2 TestSpecs: one for `head`, one for `worker`.
- For each round: start both containers, wait for "Listening on", run the
  full suite against both in parallel, stop containers, move on.
- EP=2 phase runs sequentially at the end, uses both nodes cooperatively.
- Individual failures are captured but do not abort the run.

Configuration via env vars (all optional, sensible single-node defaults):
  ATLAS_IMAGE        Docker image tag (default: atlas-gb10:latest)
  ATLAS_HEAD_IP      IP of head node (default: 127.0.0.1)
  ATLAS_WORKER_IP    IP of worker node (default: 127.0.0.1; same as head for single-node)
  ATLAS_HF_CACHE     HuggingFace cache path (default: ~/.cache/huggingface)

Run: python3 tests/run_all_models.py 2>&1 | tee /tmp/atlas-full-run.log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Configuration ─────────────────────────────────────────────────────

IMAGE = os.environ.get("ATLAS_IMAGE", "atlas-gb10:latest")
HEAD_IP = os.environ.get("ATLAS_HEAD_IP", "127.0.0.1")
WORKER_IP = os.environ.get("ATLAS_WORKER_IP", "127.0.0.1")
HEAD_PORT = int(os.environ.get("ATLAS_HEAD_PORT", "8888"))
WORKER_PORT = int(os.environ.get("ATLAS_WORKER_PORT", "8888"))
# HF cache may live in different paths on each node. Override per-host
# via ATLAS_HF_CACHE_HEAD / ATLAS_HF_CACHE_WORKER if needed; otherwise
# both default to the user's standard ~/.cache/huggingface.
_default_hf_cache = os.path.expanduser("~/.cache/huggingface")
HF_CACHE_HEAD = os.environ.get("ATLAS_HF_CACHE_HEAD", _default_hf_cache)
HF_CACHE_WORKER = os.environ.get("ATLAS_HF_CACHE_WORKER", _default_hf_cache)
RESULTS_DIR = "/workspace/atlas/tests/all_models_results"
SUITE = "/workspace/atlas/tests/single_gpu_suite.py"
STARTUP_TIMEOUT = 600  # seconds


def hf_cache_for(host: str) -> str:
    return HF_CACHE_HEAD if host == "head" else HF_CACHE_WORKER

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Spec ──────────────────────────────────────────────────────────────

@dataclass
class TestSpec:
    label: str         # short name, used for container + result filenames
    model: str         # HF ID
    mtp: bool = False
    quant: str = ""    # "fp8" or "" (nvfp4 implied)
    kv_dtype: str = "" # override default KV dtype
    extra_args: List[str] = field(default_factory=list)
    # If the suite takes too long or longctx is not meaningful, skip it:
    skip_longctx: bool = False
    # ── Multi-rank (head + worker) parallelism ──
    # Set tp_size and/or ep_size > 1 to run as a 2-rank multi-node round.
    # Default (1, 1) = single-GPU. For a 2-rank GB10 cluster the supported
    # configurations are:
    #   tp_size=1, ep_size=2 → pure EP=2 (legacy EP2_ROUNDS path)
    #   tp_size=2, ep_size=1 → pure TP=2 attention shard, no expert sharding
    #   tp_size=2, ep_size=2 → TP+EP overlapping (both groups share comm)
    # Atlas auto-derives world_size from tp×ep when world_size<=1; on the
    # overlapping topology the two groups share the same NCCL comm.
    tp_size: int = 1
    ep_size: int = 1


# ─── Plan ──────────────────────────────────────────────────────────────

# Each round = list of (host, spec). A single round runs its specs in parallel.
#
# Cached-model constraints (verified 2026-04-10):
#   head-only:   27B-dense, Qwen3-VL-30B, Gemma-4-31B, Gemma-4-26B, 80B-NVFP4
#   worker-only: Sehyo/Qwen3.5-35B-A3B-NVFP4
#   both:        Nemotron-Nano, Mistral-Small-4, 35B-FP8, Coder-FP8,
#                122B-NVFP4, 122B-FP8, Nemotron-Super-120B
ROUNDS: List[List[tuple]] = [
    # Round 1: 27B dense (head-only) + Sehyo 35B NVFP4 baseline (worker-only)
    [
        ("head",   TestSpec("27B-dense-nvfp4", "Kbenkhaled/Qwen3.5-27B-NVFP4")),
        ("worker", TestSpec("35B-nvfp4", "Sehyo/Qwen3.5-35B-A3B-NVFP4")),
    ],
    # Round 2: VL-30B + Sehyo 35B NVFP4 MTP
    [
        ("head",   TestSpec("qwen3-vl-30B", "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4")),
        ("worker", TestSpec("35B-nvfp4-mtp", "Sehyo/Qwen3.5-35B-A3B-NVFP4", mtp=True)),
    ],
    # Round 3: Gemma-4-31B + Nemotron-Nano 30B
    # Gemma-4-26B's MODEL.toml sets default_kv_dtype="bf16" (quality
    # requirement for the MoE variant). Mirror that for both Gemma variants.
    #
    # Gemma "crashes" at --max-seq-len 32768 were actually the KV-budget
    # preflight bail: at max_batch_size=16 × 2048 blocks/seq, the KV pool
    # of 8379 blocks only fits 4 concurrent sequences. Drop to
    # --max-batch-size 1 so 32k fits for the long-context test.
    [
        ("head",   TestSpec("gemma-4-31B", "nvidia/Gemma-4-31B-IT-NVFP4",
                            kv_dtype="bf16",
                            extra_args=["--max-batch-size", "1"])),
        ("worker", TestSpec("nemotron-nano-30B",
                            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")),
    ],
    # Round 4: Gemma-4-26B + 35B FP8 baseline
    [
        ("head",   TestSpec("gemma-4-26B",
                            "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16",
                            kv_dtype="bf16",
                            extra_args=["--max-batch-size", "1"])),
        ("worker", TestSpec("35B-fp8", "Qwen/Qwen3.5-35B-A3B-FP8", quant="fp8")),
    ],
    # Round 5: 35B FP8 MTP (head) + Coder-Next FP8 baseline (worker)
    # Coder-Next gets the full 32k context budget — pass 1's
    # 16k override was from an outdated memory note.
    [
        ("head",   TestSpec("35B-fp8-mtp", "Qwen/Qwen3.5-35B-A3B-FP8",
                            mtp=True, quant="fp8")),
        ("worker", TestSpec("coder-next-fp8", "Qwen/Qwen3-Coder-Next-FP8",
                            quant="fp8")),
    ],
    # Round 6: Mistral-Small-4 (worker) — Coder-Next FP8 has NO MTP head
    # weights in its checkpoint (verified: 0 mtp-* keys in the safetensors
    # index), so `--speculative` silently falls back to single-token decode
    # and the "+MTP" variant produces the same throughput as baseline.
    # Dropped to avoid reporting a spurious +0% MTP row in the final table.
    [
        ("worker", TestSpec("mistral-small-4",
                            "mistralai/Mistral-Small-4-119B-2603-NVFP4",
                            kv_dtype="bf16")),
                            # NOTE: 2026-05-01 sweep showed the 16K LC test
                            # hangs for 857s with actual_input=0 on default
                            # max-prefill-tokens=8192. Capping to 4096 fixed
                            # the hang but introduced silent corruption on
                            # the chunked-prefill MLA path: 8K LC collapses
                            # to "The\nThe\nThe..." and 16K to
                            # "TheIt's aIt's a..." (the latter only "passes"
                            # because the harness n-gram detector doesn't
                            # trip on the irregular tokenization). Net
                            # regression — reverted to default. Real fix
                            # belongs in MLA chunked-prefill correctness,
                            # not the harness spec.
        ("head",   TestSpec("35B-qwen36-fp8", "Qwen/Qwen3.6-35B-A3B-FP8",
                            quant="fp8")),
    ],
    # Round 7: 80B NVFP4 baseline (head) — 122B FP8 can't fit single-GPU
    # (118 GB weights > 121 GB GPU), so it moves to the EP=2 phase below.
    [
        ("head",   TestSpec("80B-nvfp4",
                            "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")),
        ("head",   None),
    ],
    # Round 8: 80B NVFP4 MTP (head)
    [
        ("head",   TestSpec("80B-nvfp4-mtp",
                            "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", mtp=True)),
        ("head",   None),
    ],
    # Round 9: 122B NVFP4 baseline single-GPU (head) — tight fit without MTP
    [
        ("head",   TestSpec("122B-nvfp4", "Sehyo/Qwen3.5-122B-A10B-NVFP4",
                            extra_args=["--max-batch-size", "1"])),
        ("head",   None),
    ],
]

# EP=2 rounds run sequentially at the end, on both DGXs together.
# Each spec launches rank 0 on head (HTTP + scheduler) and rank 1 on
# worker (EP worker loop), both using the same RDMA/NCCL config as
# scripts/start-ep2.sh.
EP2_ROUNDS = [
    # 122B NVFP4 + MTP — the canonical EP=2 workhorse
    TestSpec("122B-nvfp4-ep2-mtp", "Sehyo/Qwen3.5-122B-A10B-NVFP4", mtp=True),
    # 122B FP8 — can ONLY fit on EP=2 (118 GB weights > 121 GB single GPU).
    # Baseline first, then MTP variant.
    TestSpec("122B-fp8-ep2", "Qwen/Qwen3.5-122B-A10B-FP8", quant="fp8"),
    TestSpec("122B-fp8-ep2-mtp", "Qwen/Qwen3.5-122B-A10B-FP8",
             mtp=True, quant="fp8"),
    # 80B NVFP4 + MTP — requires weights on both DGXs (rsync'd in pass 2)
    TestSpec("80B-nvfp4-ep2-mtp", "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
             mtp=True),
    # Nemotron Super 120B — user asked to force EP=2
    TestSpec("nemotron-super-120B-ep2",
             "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"),
]


# Pure TP=2 rounds — attention sharded across ranks, experts replicated.
#
# WARNING: MiniMax M2.7-NVFP4 doesn't fit pure-TP=2 on 2-rank GB10. The
# 229B-param MoE has ~125 GB of expert weights; pure TP=2 only shards
# the (small) attention path, so each rank still tries to load the full
# 125 GB → 127 GB peak vs 119 GB free → OOM pre-flight bail. Pure TP=2
# is only useful on dense models where attention dominates memory; on
# this 2-GB10 cluster the only model that fits is one where
# (weights / 2) ≤ ~110 GB AFTER reserves — i.e. nothing currently in
# the cache. Left as an empty list so the phase is a no-op until a
# pure-dense TP-aware checkpoint lands.
TP2_ROUNDS: List[TestSpec] = [
    # Disabled: minimax-m27 OOMs in pure TP=2 (only TP+EP composition fits).
    # TestSpec("minimax-m27-tp2",
    #          "lukealonso/MiniMax-M2.7-NVFP4",
    #          tp_size=2, ep_size=1,
    #          extra_args=["--max-seq-len", "16384"]),
]


# Mixed TP=2 + EP=2 (overlapping topology on 2 ranks). Both groups share
# the same NCCL communicator — TP shards attention across {0,1} while EP
# also routes experts across {0,1}. Atlas's TP Phase-1 work (commit 8bd91ba
# on master-rewrite) verified MiniMax M2.7 stays coherent here, with a
# documented ~12% cold TTFT win at 4096 tok over EP-only.
TPEP_ROUNDS = [
    TestSpec("minimax-m27-tp2-ep2",
             "lukealonso/MiniMax-M2.7-NVFP4",
             tp_size=2, ep_size=2,
             extra_args=["--max-seq-len", "16384"]),
]


# ─── Docker helpers ────────────────────────────────────────────────────

def sh(cmd, check=True, capture=False, timeout=None):
    """Run a local shell command."""
    if isinstance(cmd, str):
        cmd = ["bash", "-lc", cmd]
    return subprocess.run(
        cmd, check=check,
        capture_output=capture, text=True, timeout=timeout,
    )


def ssh_worker(cmd, check=True, capture=False, timeout=None):
    """Run a remote command on the worker via SSH."""
    full = ["ssh", "-o", "BatchMode=yes", WORKER_IP, cmd]
    return subprocess.run(
        full, check=check,
        capture_output=capture, text=True, timeout=timeout,
    )


def docker_on(host, args, check=True, capture=False, timeout=None):
    """Run `sudo docker ARGS` on the given host."""
    cmd = "sudo docker " + args
    if host == "head":
        return sh(cmd, check=check, capture=capture, timeout=timeout)
    return ssh_worker(cmd, check=check, capture=capture, timeout=timeout)


def build_serve_cmd(spec: TestSpec, port: int) -> str:
    """Build the `serve ...` command-line tail for a container."""
    args = [
        "serve", spec.model,
        "--port", str(port),
        "--scheduling-policy", "slai",
    ]
    # Default long context; individual specs override via extra_args.
    has_max_seq = any(a == "--max-seq-len" for a in spec.extra_args)
    if not has_max_seq:
        args += ["--max-seq-len", "32768"]
    # KV dtype policy for the benchmark only — we never change model
    # defaults in MODEL.toml. Historic alpha-2 benchmarks used NVFP4 KV
    # for NVFP4 models, so match that here to get comparable throughput:
    #   - spec.kv_dtype set explicitly (e.g. bf16 for Mistral MLA) → use as-is
    #   - FP8 quantized models → fp8 KV (matches checkpoint)
    #   - Everything else (NVFP4 models) → nvfp4 KV
    kv = spec.kv_dtype
    if not kv:
        kv = "fp8" if spec.quant == "fp8" else "nvfp4"
    args += ["--kv-cache-dtype", kv]
    if spec.mtp:
        # Do not force --num-drafts: CLI default is 1 (K=2), and MODEL.toml
        # [behavior].default_num_drafts may override it per model. Benchmarks
        # show K=2 beats K=3 on MoE models like Qwen3.5-35B by ~43%.
        args += ["--speculative"]
        mtp_q = "fp8" if spec.quant == "fp8" else "nvfp4"
        args += ["--mtp-quantization", mtp_q]
    args += spec.extra_args
    return " ".join(args)


def start_container(host: str, spec: TestSpec, port: int) -> str:
    """Start a serve container. Returns container name."""
    name = f"atlas-test-{spec.label}"
    # Ensure prior container is gone
    docker_on(host, f"rm -f {name}", check=False, capture=True)
    serve_cmd = build_serve_cmd(spec, port)
    cache = hf_cache_for(host)
    docker_cmd = (
        f"run -d --name {name} --gpus all --ipc=host "
        f"-p {port}:{port} "
        f"-v {cache}:/root/.cache/huggingface "
        f"{IMAGE} {serve_cmd}"
    )
    docker_on(host, docker_cmd, check=True, capture=True)
    return name


def wait_listening(host: str, name: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll docker logs until 'Listening on' appears or we time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Container might have exited
        r = docker_on(host, f"ps -q -f name={name}", check=False, capture=True)
        if not r.stdout.strip():
            print(f"    [{host}/{name}] container exited unexpectedly")
            return False
        r = docker_on(host, f"logs {name} 2>&1", check=False, capture=True)
        log = r.stdout
        if "Listening on" in log:
            return True
        if "Error:" in log and "ERROR" in log:
            print(f"    [{host}/{name}] error detected in log")
            return False
        time.sleep(10)
    print(f"    [{host}/{name}] startup timeout after {timeout}s")
    return False


def stop_container(host: str, name: str) -> None:
    docker_on(host, f"stop {name}", check=False, capture=True, timeout=60)
    docker_on(host, f"rm -f {name}", check=False, capture=True, timeout=30)


INTER_ROUND_SETTLE_SECONDS = 30
"""GB10 unified memory takes several seconds to release after a container
stops. If the next round starts too quickly, the new container's weight
load races the cleanup and hits OOM (observed: Gemma-4-31B in pass 1 failed
because Round 2's VL-30B GPU state hadn't cleared yet). A conservative
30-second settle covers both VL-30B (~100 GB released) and 122B NVFP4
(~60 GB released)."""


def settle() -> None:
    print(f"  Settling GPU memory for {INTER_ROUND_SETTLE_SECONDS}s")
    time.sleep(INTER_ROUND_SETTLE_SECONDS)


def run_suite(host: str, spec: TestSpec, port: int) -> dict:
    """Run single_gpu_suite.py against the given server, return parsed JSON."""
    if host == "head":
        base_url = f"http://localhost:{port}/v1"
    else:
        base_url = f"http://{WORKER_IP}:{port}/v1"
    out_json = os.path.join(RESULTS_DIR, f"{spec.label}.json")
    log_path = os.path.join(RESULTS_DIR, f"{spec.label}.log")
    cmd = [
        "python3", SUITE,
        "--base-url", base_url,
        "--model", spec.model,
        "--output", out_json,
    ]
    if spec.skip_longctx:
        cmd.append("--skip-longctx")
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return {"proc": proc, "out_json": out_json, "log_path": log_path}


def wait_and_read(job: dict) -> Optional[dict]:
    proc = job["proc"]
    proc.wait()
    try:
        with open(job["out_json"]) as f:
            return json.load(f)
    except Exception as e:
        print(f"    [warn] could not read {job['out_json']}: {e}")
        return None


def warmup_request(host: str, spec: TestSpec, port: int, timeout: int = 120) -> None:
    """Send one throwaway request to absorb first-call JIT costs (CUDA graph
    capture, cuBLAS workspace, FP8 calibration token, paged KV allocation).

    Result is discarded; failures are logged but not fatal — the real suite
    will report them. Runs inside the server container via `curl` so it
    doesn't depend on host-side Python HTTP libs.
    """
    base = "localhost" if host == "head" else WORKER_IP
    url = f"http://{base}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": spec.model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0,
    })
    # -s: silent, -o /dev/null: discard body, --max-time: wall cap
    cmd = [
        "curl", "-s", "-o", "/dev/null",
        "-w", "%{http_code}",
        "--max-time", str(timeout),
        "-H", "Content-Type: application/json",
        "-d", payload,
        url,
    ]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout + 10)
        code = r.stdout.strip() if r.stdout else "?"
        print(f"    [{spec.label}] warmup → HTTP {code}")
    except Exception as e:
        print(f"    [{spec.label}] warmup failed (ignored): {e}")


# ─── Round execution ───────────────────────────────────────────────────

def run_round(round_idx: int, pairs: list) -> dict:
    """Run a single round: start containers on each host, run suite in
    parallel, stop containers. Returns {label: result_json}."""
    print(f"\n{'=' * 70}\nROUND {round_idx}\n{'=' * 70}")
    started = []  # list of (host, name, spec)

    # 1. Start containers
    for host, spec in pairs:
        if spec is None:
            continue
        port = HEAD_PORT if host == "head" else WORKER_PORT
        print(f"  Starting {spec.label} on {host} ({spec.model})")
        try:
            name = start_container(host, spec, port)
            started.append((host, name, spec, port))
        except subprocess.CalledProcessError as e:
            print(f"    [FAIL] start_container: {e}")

    if not started:
        print("  [skip] round has no specs to run")
        return {}

    # 2. Wait for each container to be ready
    ready = []
    for host, name, spec, port in started:
        print(f"  Waiting for {spec.label} to listen...")
        if wait_listening(host, name):
            print(f"    [{spec.label}] ready")
            ready.append((host, name, spec, port))
        else:
            print(f"    [{spec.label}] FAILED TO START — skipping")
            stop_container(host, name)

    # 3. Warmup: one throwaway request per model to absorb first-call JIT
    # cost (CUDA graph capture, cuBLAS workspace, FP8 calibration,
    # initial KV page allocation). Without this, the first suite test
    # (Factual) measures startup tax and under-reports throughput.
    for host, name, spec, port in ready:
        print(f"  Warmup {spec.label} on {host}")
        warmup_request(host, spec, port)

    # 4. Run suites in parallel
    jobs = []
    for host, name, spec, port in ready:
        print(f"  Running suite for {spec.label}")
        job = run_suite(host, spec, port)
        job["spec"] = spec
        job["host"] = host
        job["name"] = name
        jobs.append(job)

    # 4. Wait for all jobs to complete
    results = {}
    for job in jobs:
        spec = job["spec"]
        print(f"  Waiting for suite to finish: {spec.label}")
        data = wait_and_read(job)
        if data is None:
            data = {"model": spec.model, "error": "no_json_output"}
        results[spec.label] = {"data": data, "mtp": spec.mtp}

    # 5. Stop containers
    for host, name, spec, port in ready:
        print(f"  Stopping {spec.label}")
        stop_container(host, name)

    # 6. GPU settle before next round. GB10 unified memory takes time
    # to release; see INTER_ROUND_SETTLE_SECONDS docstring.
    settle()

    return results


# ─── EP=2 round ────────────────────────────────────────────────────────

MASTER_PORT = 29500
RDMA_FLAGS = (
    "--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1"
)
NCCL_ENV = (
    " -e NCCL_SOCKET_IFNAME=enp1s0f0np0"
    " -e NCCL_IB_DISABLE=0"
    " -e NCCL_IB_HCA=rocep1s0f0"
    " -e NCCL_IB_ROCE_VERSION_NUM=2"
    " -e NCCL_IB_ADDR_FAMILY=AF_INET"
    " -e NCCL_IB_TIMEOUT=22"
    " -e NCCL_IB_RETRY_CNT=7"
    " -e NCCL_NET_GDR_LEVEL=0"
    " -e NCCL_NET_GDR_C2C=0"
    " -e NCCL_DMABUF_ENABLE=0"
    " -e NCCL_NVLS_ENABLE=0"
    " -e NCCL_CUMEM_HOST_ENABLE=0"
    " -e NCCL_PROTO=Simple"
    " -e NCCL_ALGO=Ring"
    " -e NCCL_MIN_NCHANNELS=1"
    " -e NCCL_MAX_NCHANNELS=2"
)


def build_ep2_serve_cmd(spec: TestSpec, rank: int) -> str:
    """Multi-rank serve cmd (EP=2, TP=2, or TP+EP overlapping). `--network host`
    is required for NCCL discovery. The legacy name is preserved for back-compat;
    behaviour generalises by reading `spec.tp_size` and `spec.ep_size`. When
    both default to 1, we fall back to the historical pure-EP=2 launch
    (world_size=2, ep_size=2 implied by Atlas's auto-derive logic).
    """
    # Default: pure EP=2 (back-compat for existing EP2_ROUNDS specs that
    # don't set tp_size/ep_size explicitly).
    tp = spec.tp_size if spec.tp_size > 1 or spec.ep_size > 1 else 1
    ep = spec.ep_size if spec.tp_size > 1 or spec.ep_size > 1 else 2
    # On 2-GPU GB10: tp==ep means overlapping topology (single comm), tp*ep==4
    # would be orthogonal mesh (needs 4 GPUs — out of scope). Worker rank
    # uses port 0 (no HTTP). Atlas auto-derives world_size from tp_size /
    # ep_size when world_size<=1, so we still pass --world-size 2 to keep
    # the rank-discovery channel deterministic.
    args = [
        "serve", spec.model,
        "--rank", str(rank),
        "--world-size", "2",
        "--tp-size", str(tp),
        "--ep-size", str(ep),
        "--master-addr", HEAD_IP,
        "--master-port", str(MASTER_PORT),
        "--port", str(HEAD_PORT if rank == 0 else 0),
        "--max-batch-size", "1",
        "--gpu-memory-utilization", "0.70",
        "--scheduling-policy", "slai",
    ]
    if not any(a == "--kv-cache-dtype" for a in spec.extra_args) and not spec.kv_dtype:
        args += ["--kv-cache-dtype", "nvfp4"]
    if spec.kv_dtype:
        args += ["--kv-cache-dtype", spec.kv_dtype]
    if spec.mtp:
        # Do not force --num-drafts: use CLI default (1) or MODEL.toml default.
        args += ["--speculative"]
        mtp_q = "fp8" if spec.quant == "fp8" else "nvfp4"
        args += ["--mtp-quantization", mtp_q]
    has_max_seq = any(a == "--max-seq-len" for a in spec.extra_args)
    if not has_max_seq:
        # Match single-GPU default of 32k so the 16K long-context test
        # (which sends ~16K tokens + requests output) fits under the
        # `prompt_len >= max_seq_len` check in api.rs. The previous 16k
        # ceiling caused immediate 400 rejects on all EP=2 long-ctx tests.
        args += ["--max-seq-len", "32768"]
    args += spec.extra_args
    return " ".join(args)


def start_ep2(spec: TestSpec) -> tuple:
    """Start rank 0 on head + rank 1 on worker. Returns (rank0_name, rank1_name)."""
    rank0_name = f"atlas-ep0-{spec.label}"
    rank1_name = f"atlas-ep1-{spec.label}"
    # Cleanup any stale containers
    docker_on("head", f"rm -f {rank0_name}", check=False, capture=True)
    docker_on("worker", f"rm -f {rank1_name}", check=False, capture=True)

    # Rank 0 (head)
    rank0_serve = build_ep2_serve_cmd(spec, rank=0)
    docker0 = (
        f"run -d --name {rank0_name} --gpus all --ipc=host --network host "
        f"{RDMA_FLAGS}{NCCL_ENV} "
        f"-e RUST_LOG=info "
        f"-v {HF_CACHE_HEAD}:/root/.cache/huggingface "
        f"{IMAGE} {rank0_serve}"
    )
    docker_on("head", docker0, check=True, capture=True)

    # Rank 1 (worker)
    rank1_serve = build_ep2_serve_cmd(spec, rank=1)
    docker1 = (
        f"run -d --name {rank1_name} --gpus all --ipc=host --network host "
        f"{RDMA_FLAGS}{NCCL_ENV} "
        f"-e RUST_LOG=info "
        f"-v {HF_CACHE_WORKER}:/root/.cache/huggingface "
        f"{IMAGE} {rank1_serve}"
    )
    docker_on("worker", docker1, check=True, capture=True)

    return rank0_name, rank1_name


def wait_ep2_ready(rank0_name: str, rank1_name: str, timeout: int = 900) -> bool:
    """Wait for rank 0 to be listening AND rank 1 to log EP worker ready."""
    deadline = time.time() + timeout
    rank0_ready = False
    rank1_ready = False
    while time.time() < deadline:
        if not rank0_ready:
            r = docker_on("head", f"ps -q -f name={rank0_name}",
                          check=False, capture=True)
            if not r.stdout.strip():
                print(f"    [rank0] container exited")
                return False
            r = docker_on("head", f"logs {rank0_name} 2>&1",
                          check=False, capture=True)
            if "Listening on" in r.stdout:
                rank0_ready = True
                print("    [rank0] listening")
        if not rank1_ready:
            r = docker_on("worker", f"ps -q -f name={rank1_name}",
                          check=False, capture=True)
            if not r.stdout.strip():
                print(f"    [rank1] container exited")
                return False
            r = docker_on("worker", f"logs {rank1_name} 2>&1",
                          check=False, capture=True)
            if "EP worker ready" in r.stdout or "worker ready" in r.stdout.lower():
                rank1_ready = True
                print("    [rank1] worker ready")
        if rank0_ready and rank1_ready:
            return True
        time.sleep(15)
    print(f"    [ep2] startup timeout after {timeout}s")
    return False


def run_ep2_round(spec: TestSpec) -> Optional[dict]:
    """Launch both ranks, run the suite against head, stop both."""
    print(f"\n{'=' * 70}\nEP=2 ROUND: {spec.label} ({spec.model})\n{'=' * 70}")
    try:
        rank0, rank1 = start_ep2(spec)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] start_ep2: {e}")
        return {"label": spec.label, "status": "EP2_START_FAIL",
                "model": spec.model, "error": str(e)}

    print(f"  Waiting for EP=2 ready (rank0={rank0}, rank1={rank1})")
    if not wait_ep2_ready(rank0, rank1):
        print("  [FAIL] EP=2 not ready in time")
        # Dump last 40 lines of each log for debugging
        for host, name in [("head", rank0), ("worker", rank1)]:
            r = docker_on(host, f"logs --tail 40 {name} 2>&1",
                          check=False, capture=True)
            print(f"  --- {host}/{name} log tail ---\n{r.stdout}")
        stop_container("head", rank0)
        stop_container("worker", rank1)
        return {"label": spec.label, "status": "EP2_NOT_READY",
                "model": spec.model}

    print(f"  Warmup {spec.label} against EP=2 head endpoint")
    warmup_request("head", spec, HEAD_PORT)

    print("  Running suite against EP=2 head endpoint")
    job = run_suite("head", spec, HEAD_PORT)
    job["spec"] = spec
    data = wait_and_read(job)
    if data is None:
        data = {"model": spec.model, "error": "no_json_output"}

    print(f"  Stopping EP=2 containers")
    stop_container("head", rank0)
    stop_container("worker", rank1)
    settle()

    return {"label": spec.label, "mtp": spec.mtp, "data": data, "ep2": True}


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rounds", type=str, default="",
                        help="Comma-separated round indices to skip (e.g. 1,3)")
    parser.add_argument("--only-round", type=int, default=None,
                        help="Run only the given round index")
    parser.add_argument("--skip-ep2", action="store_true")
    parser.add_argument("--skip-tp2", action="store_true",
                        help="Skip the pure-TP=2 phase (TP attention shard, EP=1)")
    parser.add_argument("--skip-tpep", action="store_true",
                        help="Skip the mixed TP=2+EP=2 (overlapping topology) phase")
    parser.add_argument("--only-tp2", action="store_true",
                        help="Only run the TP=2 phase (skip single-GPU + EP=2 + TPEP)")
    parser.add_argument("--only-tpep", action="store_true",
                        help="Only run the TP+EP overlapping phase")
    args = parser.parse_args()

    skip = set(int(s) for s in args.skip_rounds.split(",") if s.strip())
    all_results = {}

    only_phase_active = args.only_tp2 or args.only_tpep
    run_singlegpu = (args.only_round is not None) or (not only_phase_active)
    run_ep2 = (not args.skip_ep2) and (args.only_round is None) and (not only_phase_active)
    run_tp2 = (not args.skip_tp2) and (args.only_round is None) and (not args.only_tpep)
    run_tpep = (not args.skip_tpep) and (args.only_round is None) and (not args.only_tp2)

    if run_singlegpu:
        for idx, pairs in enumerate(ROUNDS, start=1):
            if args.only_round is not None and idx != args.only_round:
                continue
            if idx in skip:
                print(f"\n[skip] Round {idx} (per --skip-rounds)")
                continue
            round_res = run_round(idx, pairs)
            all_results.update(round_res)
            # Persist after every round so partial results survive crashes.
            with open(os.path.join(RESULTS_DIR, "_partial.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    if run_ep2:
        for spec in EP2_ROUNDS:
            ep2 = run_ep2_round(spec)
            if ep2:
                all_results[spec.label] = ep2
            # Persist after each EP=2 run too
            with open(os.path.join(RESULTS_DIR, "_partial.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # TP=2 (pure) — uses the same multi-rank launch path as EP=2; only the
    # tp_size/ep_size on the spec differ. The serve-cmd builder reads them
    # and passes --tp-size/--ep-size to Atlas; the run_ep2_round() driver
    # is already topology-agnostic (head HTTP + worker rank-1 join).
    if run_tp2:
        for spec in TP2_ROUNDS:
            res = run_ep2_round(spec)
            if res:
                all_results[spec.label] = res
            with open(os.path.join(RESULTS_DIR, "_partial.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Mixed TP=2 + EP=2 (overlapping topology on 2 GB10 ranks).
    if run_tpep:
        for spec in TPEP_ROUNDS:
            res = run_ep2_round(spec)
            if res:
                all_results[spec.label] = res
            with open(os.path.join(RESULTS_DIR, "_partial.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Final dump
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll done. Results in {RESULTS_DIR}/all_results.json")


if __name__ == "__main__":
    main()
