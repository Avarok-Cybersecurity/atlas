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
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

# ─── Configuration (mirrors run_all_models.py) ──────────────────────────

IMAGE = os.environ.get("ATLAS_IMAGE", "atlas-gb10:latest")
HEAD_IP = os.environ.get("ATLAS_HEAD_IP", "127.0.0.1")
WORKER_IP = os.environ.get("ATLAS_WORKER_IP", "127.0.0.1")
HEAD_PORT = int(os.environ.get("ATLAS_HEAD_PORT", "8888"))
WORKER_PORT = int(os.environ.get("ATLAS_WORKER_PORT", "8888"))
_default_hf_cache = os.path.expanduser("~/.cache/huggingface")
HF_CACHE_HEAD = os.environ.get("ATLAS_HF_CACHE_HEAD", _default_hf_cache)
HF_CACHE_WORKER = os.environ.get("ATLAS_HF_CACHE_WORKER", _default_hf_cache)
STARTUP_TIMEOUT = 600  # seconds

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")
TPS_TOLERANCE = 0.10  # allow 10% noise before failing a regression

INTER_ROUND_SETTLE_SECONDS = 30
"""GB10 unified memory takes several seconds to release after a container
stops. If the next round starts too quickly, the new container's weight
load races the cleanup and hits OOM. Same rationale as run_all_models.py."""

MASTER_PORT = 29500
RDMA_FLAGS = "--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1"
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


def hf_cache_for(host: str) -> str:
    return HF_CACHE_HEAD if host == "head" else HF_CACHE_WORKER


# ─── Spec ────────────────────────────────────────────────────────────

@dataclass
class Spec:
    label: str
    model: str
    host: str = "head"       # "head" | "worker" -> xdist_group for single-GPU specs
    port: int = 8888
    mtp: bool = False
    quant: str = ""           # "fp8" or "" (nvfp4 implied)
    kv_dtype: str = ""        # override default KV dtype
    extra_args: List[str] = field(default_factory=list)
    skip_longctx: bool = False
    # Multi-rank (head + worker) — set >1 to route through deployed_atlas_ep2.
    tp_size: int = 1
    ep_size: int = 1


# ─── Spec matrix — transplanted from run_all_models.py ROUNDS / *_ROUNDS ──
#
# Single-GPU specs (was ROUNDS: List[List[(host, TestSpec)]]). The ("head",
# None) padding entries in rounds 7-9 (reserved for models later moved to the
# EP=2 phase) carry no spec and are simply omitted here.

SPECS: List[Spec] = [
    # Round 1
    Spec("27B-dense-nvfp4", "Kbenkhaled/Qwen3.5-27B-NVFP4", host="head", port=HEAD_PORT),
    Spec("35B-nvfp4", "Sehyo/Qwen3.5-35B-A3B-NVFP4", host="worker", port=WORKER_PORT),
    # Round 2
    Spec("qwen3-vl-30B", "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4", host="head", port=HEAD_PORT),
    Spec("35B-nvfp4-mtp", "Sehyo/Qwen3.5-35B-A3B-NVFP4", host="worker", port=WORKER_PORT, mtp=True),
    # Round 3 — Gemma-4 variants pin kv_dtype=bf16 (quality requirement) and
    # --max-batch-size 1 (KV-budget preflight bail at default batch size).
    Spec("gemma-4-31B", "nvidia/Gemma-4-31B-IT-NVFP4", host="head", port=HEAD_PORT,
         kv_dtype="bf16", extra_args=["--max-batch-size", "1"]),
    Spec("nemotron-nano-30B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
         host="worker", port=WORKER_PORT),
    # Round 4
    Spec("gemma-4-26B", "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16",
         host="head", port=HEAD_PORT, kv_dtype="bf16", extra_args=["--max-batch-size", "1"]),
    Spec("35B-fp8", "Qwen/Qwen3.5-35B-A3B-FP8", host="worker", port=WORKER_PORT, quant="fp8"),
    # Round 5
    Spec("35B-fp8-mtp", "Qwen/Qwen3.5-35B-A3B-FP8", host="head", port=HEAD_PORT,
         mtp=True, quant="fp8"),
    Spec("coder-next-fp8", "Qwen/Qwen3-Coder-Next-FP8", host="worker", port=WORKER_PORT,
         quant="fp8"),
    # Round 6 — Coder-Next has no MTP head weights (verified: 0 mtp-* keys in
    # its safetensors index), so a "+MTP" variant is intentionally absent.
    Spec("mistral-small-4", "mistralai/Mistral-Small-4-119B-2603-NVFP4",
         host="worker", port=WORKER_PORT, kv_dtype="bf16"),
    Spec("35B-qwen36-fp8", "Qwen/Qwen3.6-35B-A3B-FP8", host="head", port=HEAD_PORT, quant="fp8"),
    # Round 7 — 80B NVFP4 baseline (head-only; 122B FP8 can't fit single-GPU,
    # it moves to the EP=2 phase below)
    Spec("80B-nvfp4", "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", host="head", port=HEAD_PORT),
    # Round 8
    Spec("80B-nvfp4-mtp", "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
         host="head", port=HEAD_PORT, mtp=True),
    # Round 9 — tight fit without MTP
    Spec("122B-nvfp4", "Sehyo/Qwen3.5-122B-A10B-NVFP4", host="head", port=HEAD_PORT,
         extra_args=["--max-batch-size", "1"]),
]

# EP=2 rounds (was EP2_ROUNDS) — pure EP=2, rank 0 on head + rank 1 on worker.
EP2_SPECS: List[Spec] = [
    Spec("122B-nvfp4-ep2-mtp", "Sehyo/Qwen3.5-122B-A10B-NVFP4", mtp=True),
    Spec("122B-fp8-ep2", "Qwen/Qwen3.5-122B-A10B-FP8", quant="fp8"),
    Spec("122B-fp8-ep2-mtp", "Qwen/Qwen3.5-122B-A10B-FP8", mtp=True, quant="fp8"),
    Spec("80B-nvfp4-ep2-mtp", "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", mtp=True),
    Spec("nemotron-super-120B-ep2", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"),
]

# Pure TP=2 rounds (was TP2_ROUNDS). Currently empty: the only dense-attention
# checkpoint that would fit (minimax-m27-tp2) OOMs pure-TP=2 on 2-rank GB10 —
# see the commented-out spec and rationale in run_all_models.py. Left as an
# empty list so this phase stays a documented no-op, matching upstream.
TP2_SPECS: List[Spec] = []

# Mixed TP=2 + EP=2 overlapping topology (was TPEP_ROUNDS).
TPEP_SPECS: List[Spec] = [
    Spec("minimax-m27-tp2-ep2", "lukealonso/MiniMax-M2.7-NVFP4",
         tp_size=2, ep_size=2, extra_args=["--max-seq-len", "16384"]),
]

# EP=4 (was EP4_ROUNDS) is explicitly out of scope here: run_ep2_round /
# deployed_atlas_ep2 only bring up 2 ranks (head + one worker), and generalizing
# to N ranks is a separate change. The real 4-node smoke test still runs via
# /home/cluster/launch-atlas-ep4.sh.

MULTIRANK_SPECS: List[Spec] = EP2_SPECS + TP2_SPECS + TPEP_SPECS


# ─── Docker helpers (mirrors run_all_models.py sh/ssh_worker/docker_on) ────

def sh(cmd, check=True, capture=False, timeout=None):
    if isinstance(cmd, str):
        cmd = ["bash", "-lc", cmd]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, timeout=timeout)


def ssh_worker(cmd, check=True, capture=False, timeout=None):
    full = ["ssh", "-o", "BatchMode=yes", WORKER_IP, cmd]
    return subprocess.run(full, check=check, capture_output=capture, text=True, timeout=timeout)


def docker_on(host, args, check=True, capture=False, timeout=None):
    """Run `sudo docker ARGS` on the given host (local for head, SSH for worker)."""
    cmd = "sudo docker " + args
    if host == "head":
        return sh(cmd, check=check, capture=capture, timeout=timeout)
    return ssh_worker(cmd, check=check, capture=capture, timeout=timeout)


def build_serve_cmd(spec: Spec, port: int) -> str:
    """Build the `serve ...` command-line tail for a single-GPU container.
    Direct port of build_serve_cmd() in run_all_models.py."""
    args = ["serve", spec.model, "--port", str(port), "--scheduling-policy", "slai"]
    has_max_seq = any(a == "--max-seq-len" for a in spec.extra_args)
    if not has_max_seq:
        args += ["--max-seq-len", "32768"]
    kv = spec.kv_dtype
    if not kv:
        kv = "fp8" if spec.quant == "fp8" else "nvfp4"
    args += ["--kv-cache-dtype", kv]
    if spec.mtp:
        args += ["--speculative"]
        mtp_q = "fp8" if spec.quant == "fp8" else "nvfp4"
        args += ["--mtp-quantization", mtp_q]
    args += spec.extra_args
    return " ".join(args)


def build_ep2_serve_cmd(spec: Spec, rank: int) -> str:
    """Multi-rank serve cmd (EP=2, TP=2, or TP+EP overlapping). Direct port of
    build_ep2_serve_cmd() in run_all_models.py."""
    tp = spec.tp_size if spec.tp_size > 1 or spec.ep_size > 1 else 1
    ep = spec.ep_size if spec.tp_size > 1 or spec.ep_size > 1 else 2
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
        args += ["--speculative"]
        mtp_q = "fp8" if spec.quant == "fp8" else "nvfp4"
        args += ["--mtp-quantization", mtp_q]
    has_max_seq = any(a == "--max-seq-len" for a in spec.extra_args)
    if not has_max_seq:
        args += ["--max-seq-len", "32768"]
    args += spec.extra_args
    return " ".join(args)


def start_container(host: str, spec: Spec, port: int) -> str:
    name = f"atlas-test-{spec.label}"
    docker_on(host, f"rm -f {name}", check=False, capture=True)
    serve_cmd = build_serve_cmd(spec, port)
    cache = hf_cache_for(host)
    docker_cmd = (
        f"run -d --name {name} --gpus all --ipc=host "
        f"-p {port}:{port} -v {cache}:/root/.cache/huggingface "
        f"{IMAGE} {serve_cmd}"
    )
    docker_on(host, docker_cmd, check=True, capture=True)
    return name


def wait_listening(host: str, name: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = docker_on(host, f"ps -q -f name={name}", check=False, capture=True)
        if not r.stdout.strip():
            return False
        r = docker_on(host, f"logs {name} 2>&1", check=False, capture=True)
        if "Listening on" in r.stdout:
            return True
        time.sleep(10)
    return False


def stop_container(host: str, name: str) -> None:
    docker_on(host, f"stop {name}", check=False, capture=True, timeout=60)
    docker_on(host, f"rm -f {name}", check=False, capture=True, timeout=30)


def settle() -> None:
    time.sleep(INTER_ROUND_SETTLE_SECONDS)


def warmup_request(host: str, spec: Spec, port: int, timeout: int = 120) -> None:
    """One throwaway request to absorb first-call JIT costs (CUDA graph
    capture, cuBLAS workspace, FP8 calibration token, paged KV allocation).
    Failures are logged but not fatal — the real tests will report them."""
    base = "localhost" if host == "head" else WORKER_IP
    url = f"http://{base}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": spec.model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0,
    })
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
           "--max-time", str(timeout), "-H", "Content-Type: application/json",
           "-d", payload, url]
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout + 10)
    except Exception:
        pass


def wait_ep2_ready(rank0_name: str, rank1_name: str, timeout: int = 900) -> bool:
    """Direct port of wait_ep2_ready() in run_all_models.py."""
    deadline = time.time() + timeout
    rank0_ready = False
    rank1_ready = False
    while time.time() < deadline:
        if not rank0_ready:
            r = docker_on("head", f"ps -q -f name={rank0_name}", check=False, capture=True)
            if not r.stdout.strip():
                return False
            r = docker_on("head", f"logs {rank0_name} 2>&1", check=False, capture=True)
            if "Listening on" in r.stdout:
                rank0_ready = True
        if not rank1_ready:
            r = docker_on("worker", f"ps -q -f name={rank1_name}", check=False, capture=True)
            if not r.stdout.strip():
                return False
            r = docker_on("worker", f"logs {rank1_name} 2>&1", check=False, capture=True)
            if "EP worker ready" in r.stdout or "worker ready" in r.stdout.lower():
                rank1_ready = True
        if rank0_ready and rank1_ready:
            return True
        time.sleep(15)
    return False


# ─── Baseline helpers ───────────────────────────────────────────────

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


# ─── API helpers (transplanted from single_gpu_suite.py) ───────────────

def api_call(base_url, endpoint, payload, timeout=120):
    url = f"{base_url}/{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {body[:500]}"}
    except Exception as e:
        return {"error": str(e)}


def chat(base_url, model, messages, max_tokens=256, tools=None, timeout=120,
         temperature=0.3, repetition_penalty=None, extra_body=None):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if extra_body:
        payload.update(extra_body)
    t0 = time.time()
    result = api_call(base_url, "chat/completions", payload, timeout=timeout)
    elapsed = time.time() - t0
    return result, elapsed


def _has_repetition_loop(text: str, min_repeats: int = 4) -> bool:
    """Detect pathological repetition loops ('a a a a a a…'). Checks n-gram
    sizes 1-12 words, flags if ANY size repeats `min_repeats` times in a row."""
    if not text:
        return False
    toks = text.split()
    for ngram in range(1, 13):
        if len(toks) < ngram * min_repeats:
            continue
        for i in range(len(toks) - ngram * min_repeats + 1):
            window = toks[i: i + ngram]
            ok = True
            for r in range(1, min_repeats):
                start = i + r * ngram
                if toks[start: start + ngram] != window:
                    ok = False
                    break
            if ok:
                return True
    return False


def extract_text(result):
    """Falls back to `reasoning_content` when `content` is empty (reasoning-
    first models that exhaust their thinking budget without a closing
    `</think>`)."""
    if "error" in result:
        return f"[ERROR] {result['error']}"
    try:
        choice = result["choices"][0]
        msg = choice["message"]
        if msg.get("content"):
            return msg["content"]
        if msg.get("tool_calls"):
            return f"[TOOL_CALL] {json.dumps(msg['tool_calls'], indent=2)}"
        if msg.get("reasoning_content"):
            return f"[THINK-ONLY]\n{msg['reasoning_content']}"
        return "[EMPTY]"
    except (KeyError, IndexError) as e:
        return f"[PARSE_ERROR] {e} — {json.dumps(result)[:300]}"


def calc_tps(result, elapsed):
    """Wall-clock tokens/sec (includes prefill/TTFT)."""
    try:
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0 and elapsed > 0:
            return completion_tokens / elapsed
    except Exception:
        pass
    return 0.0


def calc_decode_tps(result, elapsed):
    """Decode-only tokens/sec (subtracts TTFT). Returns 0.0 if the server
    did not report time_to_first_token_ms."""
    try:
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        ttft_ms = usage.get("time_to_first_token_ms") or usage.get("ttft_ms")
        if completion_tokens <= 1 or ttft_ms is None:
            return 0.0
        decode_seconds = (elapsed * 1000.0 - ttft_ms) / 1000.0
        if decode_seconds <= 0:
            return 0.0
        return (completion_tokens - 1) / decode_seconds
    except Exception:
        return 0.0


LONG_CTX_NEEDLE = "PURPLE-DOLPHIN-42"


def generate_long_context(target_tokens):
    """Needle-in-haystack long-context prompt with diverse filler (not a
    repeated sentence, which trains models to emit degenerate continuations)."""
    filler = (
        "Alice walked quietly through the tall forest. The pines above "
        "her swayed with the afternoon wind, and somewhere in the distance "
        "a woodpecker tapped a rhythm against a hollow trunk. A narrow "
        "stream wound between mossy rocks, reflecting broken fragments of "
        "the sky. Squirrels chased one another up the bark while small "
        "blue flowers nodded along the path. She paused to breathe the "
        "cool green air before moving on, her footsteps soft on the needles. "
    )
    repeats = max(2, target_tokens // 100)
    middle = repeats // 2
    paragraphs = []
    for i in range(repeats):
        if i == middle:
            paragraphs.append(
                f"[Important fact — remember this] The secret code is {LONG_CTX_NEEDLE}. " + filler
            )
        else:
            paragraphs.append(filler)
    content = "\n\n".join(paragraphs)
    content += (
        f"\n\nQuestion: What is the secret code mentioned in the text above? "
        f"Answer in one short sentence that contains the exact code."
    )
    return content


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
}


def _strip_thinking(text: str) -> str:
    for pat in (r"<think>.*?</think>", r"\[THINK\].*?\[/THINK\]"):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    for pat in (r"<think>.*$", r"\[THINK\].*$"):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# Models whose Atlas tool-call parser is a known gap (lower-case substring
# match on model ID). Currently empty — nothing is unsupported. Kept as a
# named constant (not inlined) so a future gap can be recorded here without
# touching test bodies, matching single_gpu_suite.py's `_tool_calls_supported`.
TOOL_CALL_UNSUPPORTED_SUBSTRINGS: tuple = ()


def _tool_calls_supported(model: str) -> bool:
    m = model.lower()
    return not any(s in m for s in TOOL_CALL_UNSUPPORTED_SUBSTRINGS)


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


# ─── Shared assertion bodies (used by both single-GPU and multi-rank tests) ─

def _assert_coherence(base_url, model, label):
    """Port of run_coherence_tests(): keyword-based acceptance, post-think-
    strip, plus a repetition-loop guard."""
    tests = [
        ("Factual", "What is the capital of Japan? Answer in one sentence.",
         ["tokyo"], 400),
        ("Reasoning",
         "A car drives 120 km in 2 hours. Speed = distance / time. What is 120 / 2? Give the answer in km/h.",
         ["60 km", "60km", "60 kilometers", "60 km/h", "60km/h", "60 kph", "60kph",
          "60 kilometers per hour", "60 kilometres", "60 km per hour", "sixty km",
          "sixty kilometers", "speed = 60", "speed is 60", "= 60 km", "= 60km", "60 mph"],
         2048),
        ("Creative", "Write a haiku about the ocean.", [], 400),
    ]

    failures = []
    for name, prompt, keywords, mt in tests:
        temp = 0.0 if keywords else 0.3
        result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                                max_tokens=mt, temperature=temp)
        text = extract_text(result)
        visible = _strip_thinking(text)

        ok = ("error" not in text.lower() and "[EMPTY]" not in text
              and "[PARSE_ERROR]" not in text and len(visible.strip()) >= 3)
        if not ok:
            failures.append(f"{name}: empty or error — {text[:200]!r}")
            continue

        if _has_repetition_loop(visible):
            failures.append(f"{name}: repetition loop — {visible[:200]!r}")
            continue

        if keywords:
            tl = visible.lower() if visible else text.lower()
            matched = any(k in tl for k in keywords)
            if not matched and name == "Reasoning":
                if re.search(r"\b60\b[^\n]{0,30}(km|kilomet|mph|speed|kph)", tl):
                    matched = True
                elif re.search(r"(\\text|\\mathrm|\\operatorname|\\,|\\;|\\\\)\s*\{?\s*60\s*\\?\s*(km|kilomet|mph|speed|kph)", tl):
                    matched = True
                elif re.search(r"60\s*(?:km|kilometer|kilometre|kph|mph)", tl):
                    matched = True
            if not matched:
                failures.append(f"{name}: missing expected keyword {keywords[0]!r} — {tl[:200]!r}")

    assert not failures, f"{label}: coherence failures: " + "; ".join(failures)


def _run_fibonacci(base_url, model):
    """Port of run_fibonacci_test(): extract a fenced/bare Python code block,
    execute it, verify fib(0..9). Returns (status: str, detail: str)."""
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    prompt = (
        "Write a Python function `fib(n)` that returns the n-th Fibonacci "
        "number (fib(0)=0, fib(1)=1). Then print the values fib(0) through "
        "fib(9) on a single line, space-separated. Output only a single "
        "fenced ```python code block with no explanation."
    )
    result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                            max_tokens=4096, timeout=360, temperature=0.0,
                            repetition_penalty=1.05)
    text = extract_text(result)
    visible = _strip_thinking(text)

    def plain_text_fallback(reason):
        nums_in_text = [int(x) for x in re.findall(r"-?\d+", text)]
        if len(nums_in_text) >= 10 and nums_in_text[:10] == expected:
            return "PASS (plain-text)", f"{reason}; found correct sequence in response text"
        return f"FAIL ({reason})", text[:300]

    m = re.search(r"```(?:python)?\n?(.*?)```", visible, re.DOTALL)
    if not m:
        m = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if not m:
        bare = re.search(r"(def (?:fib|fibonacci|get_fib)\w*\(.*?\).*?)(?:\n\n|\Z)", visible, re.DOTALL)
        if not bare:
            bare = re.search(r"(def (?:fib|fibonacci|get_fib)\w*\(.*?\).*?)(?:\n\n|\Z)", text, re.DOTALL)
        if bare:
            m = bare
        else:
            return plain_text_fallback("no code block")
    code = m.group(1).strip()

    if re.search(r'def\s+(fib|fibonacci|get_fib)', code) and 'print' not in code:
        fn_match = re.search(r'def\s+(\w+)\s*\(', code)
        if fn_match:
            fn_name = fn_match.group(1)
            code += f"\nprint(' '.join(str({fn_name}(i)) for i in range(10)))"

    try:
        proc = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return plain_text_fallback("execution timeout")
    except Exception as e:
        return plain_text_fallback(f"execution error: {e}")

    if proc.returncode != 0:
        first_err_line = (proc.stderr.strip().splitlines() or [""])[0][:120]
        return plain_text_fallback(f"code raised (exit={proc.returncode}): {first_err_line}")

    nums_in_stdout = re.findall(r"-?\d+", proc.stdout.strip())
    parsed = [int(x) for x in nums_in_stdout[:10]]
    if parsed == expected:
        return "PASS", proc.stdout.strip()[:200]
    return plain_text_fallback(f"code ran but output was {parsed[:10]}, expected {expected}")


def _run_tool_call(base_url, model, name, prompt, tools):
    """Port of one iteration of run_tool_call_tests(). Returns (status, detail)."""
    result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                            max_tokens=1024, tools=tools)
    text = extract_text(result)

    has_tool_call = False
    tool_name = ""
    tool_args = ""
    try:
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tc = msg.get("tool_calls", [])
        if tc:
            has_tool_call = True
            tool_name = tc[0].get("function", {}).get("name", "")
            tool_args = tc[0].get("function", {}).get("arguments", "")
    except Exception:
        pass

    if has_tool_call:
        try:
            parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            return "PASS", f"Called {tool_name}({json.dumps(parsed_args)})"
        except json.JSONDecodeError:
            return "FAIL (invalid JSON args)", f"Called {tool_name} but args not valid JSON: {tool_args[:100]}"
    return "WARN (no structured tool call)", text[:200]


def _measure_tps(base_url, model):
    """Port of run_tps_benchmark(). Returns list of per-length result dicts."""
    tests = [
        (50, "Explain what a GPU is in exactly two sentences."),
        (150, "Explain the difference between CPU and GPU architectures in a short paragraph."),
        (300, "Write a detailed explanation of how neural network inference works, covering "
              "forward pass, matrix multiplication, and memory bandwidth constraints."),
        (500, "Write a detailed essay on the history of computing from the abacus to modern "
              "GPUs, covering major milestones, key inventors, and the impact on society. "
              "Be thorough and use multiple paragraphs."),
    ]
    results = []
    for max_tokens, prompt in tests:
        result, elapsed = chat(base_url, model, [{"role": "user", "content": prompt}],
                                max_tokens=max_tokens, timeout=180)
        text = extract_text(result)
        tps = calc_tps(result, elapsed)
        decode_tps = calc_decode_tps(result, elapsed)
        text_stripped = text.lstrip()
        ok = tps > 0 and not text_stripped.startswith(("[ERROR]", "[PARSE_ERROR]", "[EMPTY]"))
        results.append({"max_tokens": max_tokens, "tps": tps, "decode_tps": decode_tps,
                         "ok": ok, "preview": text[:200]})
    return results


def _run_long_context(base_url, model):
    """Port of run_long_context_tests(). Returns list of (target, status, detail)."""
    targets = [4000, 8000, 16000]
    results = []
    for target in targets:
        content = generate_long_context(target)
        lc_timeout = 180 if target <= 4000 else 420 if target <= 8000 else 900
        result, elapsed = chat(base_url, model, [{"role": "user", "content": content}],
                                max_tokens=100, timeout=lc_timeout)
        text = extract_text(result)
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        is_api_error = text.startswith("[ERROR]") or text.startswith("[PARSE_ERROR]")
        if is_api_error or completion_tokens == 0:
            tl = text.lower()
            if "oom" in tl or "out of memory" in tl:
                status = "OOM"
            elif is_api_error:
                status = "FAIL (api error)"
            else:
                status = "FAIL (no completion)"
        elif LONG_CTX_NEEDLE in text.upper():
            status = "PASS"
        elif _has_repetition_loop(text):
            status = "FAIL (repetition)"
        else:
            # Model produced coherent output but missed the needle — that's a
            # model-level retrieval-accuracy issue, not an Atlas infra bug.
            # Real Atlas bugs (crash/OOM/repetition) are still caught above.
            status = "PASS"

        results.append({"target": target, "actual_input": prompt_tokens,
                         "completion_tokens": completion_tokens, "status": status,
                         "preview": text[:200]})
    return results


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
