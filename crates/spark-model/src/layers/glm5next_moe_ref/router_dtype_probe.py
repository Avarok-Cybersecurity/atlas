"""Slice 10 gate 1 — measure the HF(fp32) vs vLLM(bf16) router-dtype disagreement.

HF Glm5NextTextTopkRouter computes router logits in FP32 explicitly:
    F.linear(hidden.type(torch.float32), weight.type(torch.float32))
vLLM's GateLinear gets out_dtype=None for glm5_next (the fp32 override in
_get_moe_router_dtype only fires for model_type == "glm_moe_dsa" or an explicit
moe_router_dtype), so the gate GEMM and all of grouped_topk run in BF16.

Equations identical. Dtype ladder different. This measures the consequence.
"""
import json, struct, sys
import numpy as np, torch

_ST = {"BF16": (torch.bfloat16, np.uint16), "F32": (torch.float32, np.float32)}

def load(path):
    fh = open(path, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(n)); h.pop("__metadata__", None); base = 8 + n
    out = {}
    for k, m in h.items():
        a, b = m["data_offsets"]; fh.seek(base + a)
        arr = np.frombuffer(fh.read(b - a), dtype=_ST[m["dtype"]][1]).copy()
        t = torch.from_numpy(arr)
        if m["dtype"] == "BF16": t = t.view(torch.bfloat16)
        out[k] = t.reshape(m["shape"])
    return out

class Lcg:
    def __init__(self, s): self.s = s & 0xFFFFFFFFFFFFFFFF
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0
    def t(self, *sh):
        n = 1
        for x in sh: n *= x
        return torch.tensor([self.u() for _ in range(n)], dtype=torch.float32).reshape(sh)

TOPK, NGROUP, TOPKGROUP, NORM, SCALE = 8, 1, 1, True, 2.5

def route(h, w, bias, dt):
    """One routing pass in dtype `dt`. `dt=f32` is HF; `dt=bf16` is vLLM's ladder."""
    if dt == torch.float32:
        logits = torch.nn.functional.linear(h.float(), w.float())
    else:
        logits = torch.nn.functional.linear(h.to(dt), w.to(dt))
    scores = logits.sigmoid()
    sfc = scores + bias.to(scores.dtype)
    # n_group == topk_group == 1 -> the group mask is all-ones; asserted, not assumed.
    assert NGROUP == 1 and TOPKGROUP == 1
    idx = torch.topk(sfc, k=TOPK, dim=-1, sorted=False)[1]
    wts = scores.gather(1, idx)
    if NORM:
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
    return logits.float(), idx, (wts * SCALE).float()

for LAYER in (3, 23, 44):
    W = load(f"/w/router_layer{LAYER}.safetensors")
    w, bias = W["gate.weight"], W["gate.e_score_correction_bias"]
    print(f"\n=== layer {LAYER}  gate.weight {tuple(w.shape)} {w.dtype} · "
          f"bias {tuple(bias.shape)} {bias.dtype} ===")
    for T in (1, 64, 2048):
        h = Lcg(0x0F0F_0DE5).t(T, w.shape[1]) * 0.5
        l32, i32, w32 = route(h, w, bias, torch.float32)
        lbf, ibf, wbf = route(h, w, bias, torch.bfloat16)
        s32 = [set(r.tolist()) for r in i32]
        sbf = [set(r.tolist()) for r in ibf]
        diff = [a != b for a, b in zip(s32, sbf)]
        inter = np.array([len(a & b) for a, b in zip(s32, sbf)])
        # Downstream mixture consequence, expert-free: how much routed weight mass
        # sits on experts the other ladder did NOT pick.
        moved = []
        for r in range(T):
            common = s32[r] & sbf[r]
            m = sum(float(w32[r, j]) for j in range(TOPK)
                    if int(i32[r, j]) not in common)
            moved.append(m / (float(w32[r].sum()) + 1e-20))
        print(f"  T={T:<5} logit|Δ|max {(l32 - lbf).abs().max():.4e}  "
              f"rows with a different top-8 SET {sum(diff)}/{T} ({100*sum(diff)/T:.1f}%)  "
              f"mean |∩| {inter.mean():.3f}/8  "
              f"routed-mass on non-shared experts {np.mean(moved)*100:.2f}% "
              f"(max {np.max(moved)*100:.2f}%)")
