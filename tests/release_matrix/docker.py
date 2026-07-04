"""Docker container lifecycle for single-GPU and multi-rank (EP=2/TP=2/TP+EP)
deploys. Mirrors run_all_models.py's sh/ssh_worker/docker_on/build_serve_cmd/
build_ep2_serve_cmd/start_container/wait_listening/stop_container/
wait_ep2_ready."""

import json
import subprocess
import time

from release_matrix.specs import (
    HEAD_IP,
    HEAD_PORT,
    HF_CACHE_HEAD,
    HF_CACHE_WORKER,
    IMAGE,
    INTER_ROUND_SETTLE_SECONDS,
    MASTER_PORT,
    NCCL_ENV,
    RDMA_FLAGS,
    STARTUP_TIMEOUT,
    Spec,
    WORKER_IP,
    hf_cache_for,
)


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
