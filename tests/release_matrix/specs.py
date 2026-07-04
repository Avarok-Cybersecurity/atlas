"""Spec dataclass, config constants, and the spec matrix (mirrors
run_all_models.py's TestSpec + ROUNDS/EP2_ROUNDS/TP2_ROUNDS/TPEP_ROUNDS)."""

import os
from dataclasses import dataclass, field
from typing import List

IMAGE = os.environ.get("ATLAS_IMAGE", "atlas-gb10:latest")
HEAD_IP = os.environ.get("ATLAS_HEAD_IP", "127.0.0.1")
WORKER_IP = os.environ.get("ATLAS_WORKER_IP", "127.0.0.1")
HEAD_PORT = int(os.environ.get("ATLAS_HEAD_PORT", "8888"))
WORKER_PORT = int(os.environ.get("ATLAS_WORKER_PORT", "8888"))
_default_hf_cache = os.path.expanduser("~/.cache/huggingface")
HF_CACHE_HEAD = os.environ.get("ATLAS_HF_CACHE_HEAD", _default_hf_cache)
HF_CACHE_WORKER = os.environ.get("ATLAS_HF_CACHE_WORKER", _default_hf_cache)
STARTUP_TIMEOUT = 600  # seconds

# tests/release_matrix/specs.py -> tests/baselines (one level up from this package)
BASELINE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "baselines")
)
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
