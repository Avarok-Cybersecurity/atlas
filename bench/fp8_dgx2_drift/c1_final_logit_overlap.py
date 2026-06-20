#!/usr/bin/env python3
"""C1 (2026-05-26) — top-K final-logit overlap, Atlas FP8 vs HF BF16.

Uses the existing per-layer hidden-state dumps at
/workspace/atlas-dumps/fp8native_dgx2/ to compute the FINAL token-level
logit distribution under two precision regimes and quantify divergence
at the level that actually drives generation (token selection), not
just hidden-state cosine.

Pipeline:
  h_L39 (from each source)
  -> apply final RMSNorm with HF BF16 norm weights
  -> matmul against HF BF16 lm_head.weight
  -> softmax in float32
  -> compare top-K (K=1, 5, 10, 50, 200) by Jaccard
  -> compute KL(BF16 || FP8) and top-1 agreement

Single-prompt summary (the canonical 10382-token chat probe, last-token slice).

Outputs to bench/fp8_dgx2_drift/c1_final_logit_overlap.json and prints a
short report.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

SNAP = pathlib.Path(
    "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DUMP_DIR = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
OUT = pathlib.Path(__file__).resolve().parent / "c1_final_logit_overlap.json"

# Final layer is L39 (40 layers). Both the f32 hidden vectors and the
# rms-norm/lm_head weights live on disk; we never need a model load.
HIDDEN_SIZE = 2048
LAST_LAYER = 39
RMS_NORM_EPS = 1.0e-6  # standard Qwen value; will read from config if present


def load_config_rms_eps() -> float:
    cfg = json.load(open(SNAP / "config.json"))
    tc = cfg.get("text_config", cfg)
    eps = tc.get("rms_norm_eps", 1e-6)
    return float(eps)


def load_final_norm_weight() -> np.ndarray:
    """RMS-norm gamma for the final norm. Stored in shard 26."""
    shard = SNAP / "model-00026-of-00026.safetensors"
    # safetensors numpy backend can't deserialize bfloat16; use torch then cast.
    with safe_open(str(shard), framework="pt") as f:
        for key in f.keys():
            if key.endswith("model.language_model.norm.weight"):
                t = f.get_tensor(key)
                return t.to(torch.float32).cpu().numpy()
    raise RuntimeError("final norm weight not found in shard 26")


def load_lm_head_weight() -> np.ndarray:
    """lm_head [vocab, hidden] in BF16 → cast to f32 for stable matmul."""
    shard = SNAP / "model-00026-of-00026.safetensors"
    with safe_open(str(shard), framework="pt") as f:
        for key in f.keys():
            if key == "lm_head.weight":
                t = f.get_tensor(key)
                return t.to(torch.float32).cpu().numpy()
    raise RuntimeError("lm_head.weight not found in shard 26")


def rms_norm(h: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    # h: [hidden], gamma: [hidden]. Cast everything to float32 for stability.
    h = h.astype(np.float32)
    rms = np.sqrt(np.mean(h * h) + eps)
    return (h / rms) * gamma.astype(np.float32)


def load_hidden(path: pathlib.Path) -> np.ndarray:
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype="<f4").copy()
    assert arr.size == HIDDEN_SIZE, f"{path}: expected {HIDDEN_SIZE}, got {arr.size}"
    return arr


def topk(logits: np.ndarray, k: int) -> set[int]:
    # argpartition then take top-k, no need to sort within
    if k >= logits.size:
        return set(range(logits.size))
    idx = np.argpartition(-logits, k - 1)[:k]
    return set(idx.tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def softmax_f64(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    # KL(p || q) in nats. Mask near-zero p values (they contribute ~0).
    mask = p > 1e-20
    pp = p[mask]
    qq = np.clip(q[mask], 1e-30, None)
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def main() -> None:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading rms eps + final-norm gamma + lm_head", flush=True)
    eps = load_config_rms_eps()
    gamma = load_final_norm_weight()
    lm_head = load_lm_head_weight()
    print(f"  eps={eps}", flush=True)
    print(f"  gamma.shape={gamma.shape}", flush=True)
    print(f"  lm_head.shape={lm_head.shape} (vocab x hidden, f32)", flush=True)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    h_atlas = load_hidden(DUMP_DIR / f"atlas_L{LAST_LAYER}.bin")
    h_bf16 = load_hidden(DUMP_DIR / f"hf_bf16_L{LAST_LAYER}.bin")
    print(f"\nh_atlas: norm={np.linalg.norm(h_atlas):.4f}  max={h_atlas.max():.4f}", flush=True)
    print(f"h_bf16:  norm={np.linalg.norm(h_bf16):.4f}  max={h_bf16.max():.4f}", flush=True)

    # First, residual-stream cosine (should match MASTER_DRIFT_TABLE 0.97657 at worst)
    cos = float(np.dot(h_atlas, h_bf16) / (np.linalg.norm(h_atlas) * np.linalg.norm(h_bf16)))
    print(f"residual cos(atlas, bf16) at L{LAST_LAYER}: {cos:.5f}", flush=True)

    # Apply final RMSNorm + lm_head to get logits.
    print(f"\n[{time.strftime('%H:%M:%S')}] computing logits via lm_head matmul ...", flush=True)
    z_atlas = rms_norm(h_atlas, gamma, eps)
    z_bf16 = rms_norm(h_bf16, gamma, eps)
    logits_atlas = lm_head @ z_atlas.astype(np.float32)
    logits_bf16 = lm_head @ z_bf16.astype(np.float32)
    print(f"  logits.shape={logits_atlas.shape}", flush=True)

    # Logit-level metrics
    logit_cos = float(
        np.dot(logits_atlas, logits_bf16)
        / (np.linalg.norm(logits_atlas) * np.linalg.norm(logits_bf16))
    )

    arg_atlas = int(np.argmax(logits_atlas))
    arg_bf16 = int(np.argmax(logits_bf16))
    top1_agree = arg_atlas == arg_bf16

    results = {
        "residual_cos_L39": cos,
        "logit_cos": logit_cos,
        "argmax_atlas_token": arg_atlas,
        "argmax_bf16_token": arg_bf16,
        "top1_agree": top1_agree,
        "topk_jaccard": {},
        "kl_bf16_vs_atlas": None,
        "kl_atlas_vs_bf16": None,
    }

    for k in (1, 5, 10, 50, 200, 1000):
        a = topk(logits_atlas, k)
        b = topk(logits_bf16, k)
        j = jaccard(a, b)
        results["topk_jaccard"][str(k)] = j
        print(f"  top-{k:<5d} jaccard(atlas, bf16): {j:.4f}", flush=True)

    p_atlas = softmax_f64(logits_atlas)
    p_bf16 = softmax_f64(logits_bf16)
    kl_atlas_vs_bf16 = kl_div(p_atlas, p_bf16)
    kl_bf16_vs_atlas = kl_div(p_bf16, p_atlas)
    results["kl_atlas_vs_bf16"] = kl_atlas_vs_bf16
    results["kl_bf16_vs_atlas"] = kl_bf16_vs_atlas

    print(f"\n=== summary ===", flush=True)
    print(f"  residual cos L39       : {cos:.5f}", flush=True)
    print(f"  logit cos              : {logit_cos:.5f}", flush=True)
    print(f"  argmax(atlas)          : {arg_atlas}", flush=True)
    print(f"  argmax(bf16 ref)       : {arg_bf16}", flush=True)
    print(f"  top1 agree             : {top1_agree}", flush=True)
    print(f"  KL(atlas || bf16)      : {kl_atlas_vs_bf16:.4f} nats", flush=True)
    print(f"  KL(bf16  || atlas)     : {kl_bf16_vs_atlas:.4f} nats", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
