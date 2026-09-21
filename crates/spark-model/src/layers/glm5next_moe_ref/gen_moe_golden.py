#!/usr/bin/env python3
"""Slice 10 gates 6/7 — COMPLETE routed MoE layer golden, HF 5.16.1, real weights.

Runs the whole routed FFN for a layer:
  hidden -> router logits -> top-k ids/weights -> selected NVFP4 experts
         -> weighted routed mixture -> shared BF16 expert -> final

BOTH router dtype ladders, kept strictly apart:
  HF_FP32    canonical -- F.linear(h.float(), w.float()), sigmoid/bias/top-k/renorm in fp32
  VLLM_BF16  compatibility -- the gate GEMM and all of grouped_topk in bf16

🔴 apply_routed_scale_to_output = False: routed_scaling_factor rides on topk_weights and the
   SHARED expert is NOT multiplied by it.
🪤 The correction bias steers SELECTION only; weights come from the UNBIASED scores.
🪤 Renormalisation epsilon is 1e-20.
Also emits a `nearcut` fixture: the input (searched, not hand-tuned) whose rank-8 / rank-9
scores are closest together, so the gate exercises real routing competition rather than only
comfortable margins.
"""
import hashlib, json, struct, sys
import numpy as np, torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

LAYERS = [int(x) for x in sys.argv[1].split(",")]
OUT = "/w/moe_golden.json"
STRIDE, STRIDE_BIG = 251, 10009
_NP = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}


class Lazy:
    """Reads only the tensors asked for — a full routed layer is 3.85 GiB on disk."""
    def __init__(self, path):
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        self.h = json.loads(self.fh.read(n)); self.h.pop("__metadata__", None)
        self.base = 8 + n
    def __contains__(self, k):
        return k in self.h
    def get(self, k):
        m = self.h[k]; a, b = m["data_offsets"]
        self.fh.seek(self.base + a)
        arr = np.frombuffer(self.fh.read(b - a), dtype=_NP[m["dtype"]]).copy()
        t = torch.from_numpy(arr)
        if m["dtype"] == "BF16": t = t.view(torch.bfloat16)
        return (t.reshape(m["shape"]) if m["shape"] else t.reshape(())), m["dtype"]


class Lcg:
    def __init__(self, s): self.s = s & 0xFFFFFFFFFFFFFFFF
    def t(self, *sh):
        n = 1
        for x in sh: n *= x
        o = []
        for _ in range(n):
            self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            o.append(((self.s >> 40) / (1 << 24)) * 2.0 - 1.0)
        return torch.tensor(o, dtype=torch.float32).reshape(sh)


E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])

def e4m3(u8):
    u = u8.to(torch.int32)
    sign = torch.where((u >> 7) & 1 == 1, -1.0, 1.0)
    exp, man = (u >> 3) & 0xF, u & 0x7
    val = torch.where(exp == 0, man.float() * 2.0 ** -9,
                      (1.0 + man.float() * .125) * torch.pow(2.0, (exp - 7).float()))
    return sign * torch.where((exp == 15) & (man == 7), torch.zeros_like(val), val)

def deq(packed, scale, s2, K):
    N = packed.shape[0]; b = packed.to(torch.int32)
    lo = E2M1[(b & 0xF).flatten()].reshape(N, K // 2)
    hi = E2M1[((b >> 4) & 0xF).flatten()].reshape(N, K // 2)
    v = torch.stack([lo, hi], -1).reshape(N, K)
    return v * (e4m3(scale) * s2).repeat_interleave(16, dim=1)


cfg = Glm5NextTextConfig(**json.load(open("/w/config.json"))["text_config"])
HID, MI, LIMIT = cfg.hidden_size, cfg.moe_intermediate_size, cfg.swiglu_limit
E, K, SCALE = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.routed_scaling_factor
assert cfg.n_group == 1 and cfg.topk_group == 1, "grouped routing is NOT a no-op on this config"
assert cfg.norm_topk_prob

def swiglu(g, u):
    return torch.nn.functional.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)

def route(x, w, bias, mode):
    """mode 'hf_fp32' | 'vllm_bf16'. Equations identical; only the dtype ladder differs."""
    if mode == "hf_fp32":
        lg = torch.nn.functional.linear(x.float(), w.float())
    else:
        lg = torch.nn.functional.linear(x.to(torch.bfloat16), w.to(torch.bfloat16))
    sc = lg.sigmoid()
    idx = torch.topk(sc + bias.to(sc.dtype), k=K, dim=-1, sorted=True)[1]
    wt = sc.gather(1, idx)
    wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
    return lg.float(), idx.to(torch.int64), (wt * SCALE).float()


def emit(st):
    parts = []
    for name, t in st.items():
        t = t.flatten()
        stride = 1 if t.numel() <= 4096 else (STRIDE if t.numel() <= 4_000_000 else STRIDE_BIG)
        s = t[::stride]
        if t.dtype in (torch.int64, torch.int32):
            data = ",".join(str(int(x)) for x in s.tolist())
        else:
            data = ",".join(f"{x:.9g}" for x in s.tolist())
        f = t.to(torch.float64)
        ck = float((f * (torch.arange(f.numel(), dtype=torch.float64) + 1.0)).sum())
        parts.append('    "%s":{"n":%d,"stride":%d,"ck":%r,"data":[%s]}'
                     % (name, t.numel(), stride, ck, data))
    return "{\n" + ",\n".join(parts) + "\n   }"


real = torch.load("/w/real_router.pt")["X"].float()          # [256, HID] REAL layer-3
sections, diag, needed = [], [], {}

for L in LAYERS:
    P = Lazy(f"/w/moe_layer{L}.safetensors")
    gw, _ = P.get("gate.weight"); bias, _ = P.get("gate.e_score_correction_bias")
    sh = {p: P.get(f"shared_experts.{p}.weight")[0] for p in ("gate_proj", "up_proj", "down_proj")}

    # ── fixtures ──
    # 🪤 Every fixture is BF16-ROUNDED HERE, once. `__inputs` emits exactly these values and the
    # Atlas side feeds exactly these values, so the two sides start from an identical tensor.
    # Emitting a bf16-rounded input while ROUTING on the original fp32 vector made the fp32
    # router logits differ by ~2.6e-3 on the synthetic fixtures — a harness bug that looked
    # exactly like a kernel residual.
    def bf(x):
        return x.to(torch.bfloat16).float()
    fx = {"t1": bf(Lcg(0x0F0E_5EED).t(1, HID) * 0.5),
          "t7": bf(Lcg(0x0F0E_5EED).t(7, HID) * 0.5),
          "real32": bf(real[:32].clone())}
    # near-cutoff: search real rows for the smallest rank-8 / rank-9 gap.
    with torch.no_grad():
        sc = torch.nn.functional.linear(real.float(), gw.float()).sigmoid() + bias.float()
        top = torch.topk(sc, K + 1, dim=-1)[0]
        gap = (top[:, K - 1] - top[:, K]).abs()
        j = int(gap.argmin())
    fx["nearcut"] = bf(real[j:j + 1].clone())
    print(f"layer {L}: near-cutoff row {j}, rank8-rank9 gap {float(gap[j]):.3e} "
          f"(median {float(gap.median()):.3e})", file=sys.stderr)

    # Inputs are emitted ONCE per layer, UNSTRIDED: `real32`/`nearcut` come from the real
    # prefix and cannot be regenerated from an LCG on the Atlas side, and an input read from a
    # strided sample would not be the input at all.
    inp = {}
    for rn, x in fx.items():
        v = x.to(torch.bfloat16).float().flatten()
        inp[rn] = '[' + ",".join(f"{q:.9g}" for q in v.tolist()) + ']'
    per = {}
    for mode in ("hf_fp32", "vllm_bf16"):
        for rn, x in fx.items():
            lg, idx, wt = route(x, gw, bias, mode)
            T = x.shape[0]
            sel = sorted({int(v) for v in idx.flatten().tolist()})
            needed.setdefault(L, set()).update(sel)
            # expert outputs, per (token, slot)
            eo = torch.zeros(T, K, HID)
            cache = {}
            for e in sel:
                W = {}
                for p in ("gate_proj", "up_proj", "down_proj"):
                    pk, sk, gk = (f"experts.{e}.{p}.weight", f"experts.{e}.{p}.weight_scale",
                                  f"experts.{e}.{p}.weight_scale_2")
                    pa, dtp = P.get(pk); sa, dts = P.get(sk); s2, dt2 = P.get(gk)
                    assert (dtp, dts, dt2) == ("U8", "F8_E4M3", "F32")
                    W[p] = deq(pa, sa, float(s2.item()), HID if p != "down_proj" else MI)
                cache[e] = W
            xb = x.to(torch.bfloat16).float()
            for t in range(T):
                for k in range(K):
                    W = cache[int(idx[t, k])]
                    g = xb[t:t + 1] @ W["gate_proj"].T
                    u = xb[t:t + 1] @ W["up_proj"].T
                    eo[t, k] = (swiglu(g, u) @ W["down_proj"].T)[0]
            routed = (wt.unsqueeze(-1) * eo).sum(1)
            gs = xb @ sh["gate_proj"].float().T
            us = xb @ sh["up_proj"].float().T
            shared = swiglu(gs, us) @ sh["down_proj"].float().T
            final = routed + shared            # shared NOT routed-scaled
            st = {"router_logits": lg.flatten().clone(),
                  "topk_ids": idx.flatten().clone(),
                  "topk_weights": wt.flatten().clone(),
                  "expert_out": eo.flatten().clone(),
                  "routed_sum": routed.flatten().clone(),
                  "shared_out": shared.flatten().clone(),
                  "ffn_out": final.flatten().clone()}
            per[f"{mode}__{rn}"] = emit(st)
            diag.append((L, mode, rn, T, len(sel), float(wt.sum(-1).mean()),
                         float(final.abs().max()), float(shared.abs().max())))
            del cache
    per_s = ",\n".join(f'   "{k}":{v}' for k, v in per.items())
    inp_s = ",\n".join(f'    "{k}":{v}' for k, v in inp.items())
    sections.append(f'  "{L}":{{\n   "__inputs":{{\n{inp_s}\n   }},\n' + per_s + "\n  }")

body = ("{\n"
        f' "fixture":{{"hidden":{HID},"moe_intermediate":{MI},"num_experts":{E},"top_k":{K},'
        f'"routed_scaling_factor":{SCALE!r},"norm_topk_prob":true,"renorm_eps":1e-20,'
        f'"n_group":1,"topk_group":1,"swiglu_limit":{LIMIT!r},'
        f'"apply_routed_scale_to_output":false,"seed":"0x0F0E5EED","layers":{LAYERS},'
        f'"checkpoint":"LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3"}},\n'
        ' "by_layer":{\n' + ",\n".join(sections) + "\n }\n}\n")
open(OUT, "w").write(body)
json.dump({str(k): sorted(v) for k, v in needed.items()}, open("/w/needed_experts.json", "w"))
print("bytes", len(body), "sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"{'L':>3} {'mode':10} {'regime':8} {'T':>4} {'#exp':>5} {'sum(w)':>8} "
      f"{'|ffn|max':>10} {'|shared|max':>12}")
for r in diag: print("%3d %-10s %-8s %4d %5d %8.4f %10.5g %12.5g" % r)
