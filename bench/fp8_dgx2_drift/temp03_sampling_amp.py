#!/usr/bin/env python3
"""Temperature-0.3 sampling amplification analysis (ANGLE study).

Apples-to-apples: recompute final-token logits from BOTH the vLLM-FP8 and
Atlas-FP8 last-layer residual dumps using the SAME BF16 final-norm + lm_head.
This isolates the question: given the (small) FP8 forward drift that produces
cos(L39)=0.988, how much does temperature-0.3 sampling amplify it at the
token-distribution level vs greedy?

We answer:
  1. greedy argmax agreement (already known: SAME token "Now")
  2. top-1 prob, top1-top2 margin (is this a low-margin position?)
  3. softmax at T in {1.0, 0.3} -> TVD, KL, top-1 prob shift, prob of sampling
     a DIFFERENT token than the other engine's top-1
  4. effective support size (1/sum p^2) at each T
  5. a Monte-Carlo estimate of P(divergent sample) per step at T=0.3 with the
     observed logit gap, and how that compounds over a 500-token generation.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from safetensors import safe_open
import torch

SNAP = pathlib.Path(
    "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")
DUMP = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
HID = 2048
EPS = 1e-6

def load_hidden(p):
    a = np.frombuffer(p.read_bytes(), dtype="<f4").copy()
    assert a.size == HID, (p, a.size)
    return a

def load_w(key_suffix):
    shard = SNAP / "model-00026-of-00026.safetensors"
    with safe_open(str(shard), framework="pt") as f:
        for k in f.keys():
            if k == key_suffix or k.endswith(key_suffix):
                return f.get_tensor(k).to(torch.float32).cpu().numpy()
    raise RuntimeError(f"{key_suffix} not found")

def rms_norm(h, gamma, eps=EPS):
    h = h.astype(np.float32)
    rms = np.sqrt(np.mean(h*h) + eps)
    return (h/rms)*gamma.astype(np.float32)

def softmax_T(logits, T):
    x = (logits.astype(np.float64))/T
    x = x - x.max()
    e = np.exp(x)
    return e/e.sum()

def tvd(p, q):
    return 0.5*np.abs(p-q).sum()

def kl(p, q):
    m = p > 1e-300
    return float(np.sum(p[m]*(np.log(p[m]) - np.log(np.clip(q[m],1e-300,None)))))

def eff_support(p):  # inverse participation ratio / collision entropy
    return 1.0/np.sum(p*p)

def main():
    gamma = load_w("model.language_model.norm.weight")
    lm = load_w("lm_head.weight")  # [vocab, hidden]
    print("lm_head", lm.shape, "gamma", gamma.shape)

    h_v = load_hidden(DUMP/"vllm_L39.bin")
    h_a = load_hidden(DUMP/"atlas_L39.bin")
    cos = float(np.dot(h_v,h_a)/(np.linalg.norm(h_v)*np.linalg.norm(h_a)))
    print(f"residual cos(vllm,atlas) L39 = {cos:.5f}")

    zv = rms_norm(h_v, gamma); za = rms_norm(h_a, gamma)
    Lv = lm @ zv.astype(np.float32)
    La = lm @ za.astype(np.float32)
    print("logit shape", Lv.shape)

    # raw logit-level stats
    dl = La - Lv
    print(f"logit L2 diff = {np.linalg.norm(dl):.4f}, max|diff| = {np.abs(dl).max():.4f}, "
          f"mean|diff| = {np.abs(dl).mean():.5f}")
    logit_cos = float(np.dot(Lv,La)/(np.linalg.norm(Lv)*np.linalg.norm(La)))
    print(f"logit cos = {logit_cos:.6f}")

    av = int(np.argmax(Lv)); aa = int(np.argmax(La))
    print(f"argmax vllm={av} atlas={aa} agree={av==aa}")

    # sorted vllm logits to characterize the margin at THIS position
    sv = np.sort(Lv)[::-1]
    print(f"\nvLLM top-5 logits: {sv[:5]}")
    print(f"vLLM top1-top2 gap = {sv[0]-sv[1]:.4f}, top1-top5 gap = {sv[0]-sv[4]:.4f}")

    out = {"residual_cos": cos, "logit_cos": logit_cos,
           "argmax_agree": av==aa, "argmax_vllm": av, "argmax_atlas": aa,
           "logit_l2_diff": float(np.linalg.norm(dl)),
           "logit_max_diff": float(np.abs(dl).max()),
           "top1_top2_gap_vllm": float(sv[0]-sv[1]),
           "per_T": {}}

    for T in (1.0, 0.6, 0.3, 0.1):
        pv = softmax_T(Lv, T); pa = softmax_T(La, T)
        t = tvd(pv, pa)
        klva = kl(pv, pa)
        # prob mass that Atlas places on vLLM's top-1 token (i.e. agreement prob)
        p_agree = float(pa[av])
        p_v_top1 = float(pv[av])
        # probability the two engines sample DIFFERENT tokens in one independent draw
        # = 1 - sum_i pv_i * pa_i  (collision probability)
        p_diff = float(1.0 - np.sum(pv*pa))
        esv = eff_support(pv); esa = eff_support(pa)
        out["per_T"][str(T)] = {
            "tvd": float(t), "kl_v_a": klva,
            "p_vllm_top1": p_v_top1, "p_atlas_on_vllm_top1": p_agree,
            "p_divergent_draw": p_diff,
            "eff_support_vllm": float(esv), "eff_support_atlas": float(esa),
        }
        print(f"\n--- T={T} ---")
        print(f"  TVD(pv,pa)            = {t:.5f}")
        print(f"  KL(pv||pa)            = {klva:.5f} nats")
        print(f"  p_vllm(top1)          = {p_v_top1:.4f}")
        print(f"  p_atlas(vllm_top1)    = {p_agree:.4f}")
        print(f"  P(divergent draw)     = {p_diff:.5f}")
        print(f"  eff_support vllm/atlas= {esv:.1f} / {esa:.1f}")

    # Compounding over a generation: if avg per-step divergence prob is d,
    # P(at least one divergence in N steps) = 1-(1-d)^N
    print("\n--- compounding over generation (using T=0.3 P(divergent draw)) ---")
    d03 = out["per_T"]["0.3"]["p_divergent_draw"]
    for N in (50, 200, 500, 1000):
        p_any = 1.0 - (1.0 - d03)**N
        print(f"  N={N:5d}: P(>=1 divergent token) = {p_any:.4f}")
    out["compound_T03"] = {str(N): float(1.0-(1.0-d03)**N) for N in (50,200,500,1000)}

    OUT = pathlib.Path(__file__).resolve().parent/"temp03_sampling_amp.json"
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
