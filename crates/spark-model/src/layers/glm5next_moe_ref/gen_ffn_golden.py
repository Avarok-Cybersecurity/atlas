#!/usr/bin/env python3
"""Slice 10 gates 3/4/6 — GLM-5.3 FFN golden from REAL checkpoint weights, HF 5.16.1.

Covers:
  * dense FFN (layer 0)            — BF16 weights, intermediate 12288
  * shared expert (layer 3)        — BF16 weights, intermediate 2048
  * routed experts 0..3 (layer 3)  — ModelOpt NVFP4, production dims

🪤 swiglu_limit = 10.0 clamping is ASYMMETRIC: gate is upper-bounded only
   (`clamp(min=None, max=limit)`), up is bounded BOTH ways. Applies to the dense MLP,
   the shared expert AND every routed expert.
🪤 There are NO `input_scale` tensors — this is weight-only NVFP4 (W4A16). Activations
   stay bf16; nothing quantizes them.
🪤 Scale order is ModelOpt's: value = E2M1[nibble] * fp8_group_scale * weight_scale_2,
   with weight_scale_2 a DIRECT multiplier (not 1/global).
"""
import hashlib, json, struct, sys

import numpy as np
import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP  # noqa: E402

OUT = "/w/ffn_golden.json"
STRIDE, STRIDE_BIG = 251, 10009
_ST = {"BF16": (torch.bfloat16, np.uint16), "F32": (torch.float32, np.float32),
       "F8_E4M3": (None, np.uint8), "U8": (None, np.uint8)}


def load_packet(path):
    fh = open(path, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n)); hdr.pop("__metadata__", None); base = 8 + n
    out = {}
    for k, m in hdr.items():
        a, b = m["data_offsets"]; fh.seek(base + a)
        arr = np.frombuffer(fh.read(b - a), dtype=_ST[m["dtype"]][1]).copy()
        t = torch.from_numpy(arr)
        if m["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        out[k] = (t.reshape(m["shape"]) if m["shape"] else t.reshape(()), m["dtype"])
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


# ── ModelOpt NVFP4 dequant, written from the OCP MX / ModelOpt spec, not from Atlas ──
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def e4m3_decode(u8: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 (no infinities; 0xFF/0x7F are NaN) -> fp32. Independent of Atlas's LUT."""
    u = u8.to(torch.int32)
    sign = torch.where((u >> 7) & 1 == 1, -1.0, 1.0)
    exp = (u >> 3) & 0xF
    man = u & 0x7
    sub = man.to(torch.float32) * (2.0 ** -9)
    nor = (1.0 + man.to(torch.float32) * 0.125) * torch.pow(2.0, (exp - 7).to(torch.float32))
    val = torch.where(exp == 0, sub, nor)
    val = torch.where((exp == 15) & (man == 7), torch.zeros_like(val), val)  # NaN -> 0
    return sign * val


def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, scale2: float, K: int):
    """packed [N, K/2] u8, scale [N, K/16] e4m3, scale2 scalar -> fp32 [N, K].
    Even column = LOW nibble, odd = HIGH nibble."""
    N = packed.shape[0]
    b = packed.to(torch.int32)
    lo = E2M1[(b & 0xF).flatten()].reshape(N, K // 2)
    hi = E2M1[((b >> 4) & 0xF).flatten()].reshape(N, K // 2)
    vals = torch.stack([lo, hi], dim=-1).reshape(N, K)
    s = e4m3_decode(scale) * scale2                       # [N, K/16]
    return vals * s.repeat_interleave(16, dim=1)


raw = json.load(open("/w/config.json"))["text_config"]
cfg = Glm5NextTextConfig(**raw)
HID, LIMIT = cfg.hidden_size, cfg.swiglu_limit
assert LIMIT == 10.0 and cfg.hidden_act == "silu"


def swiglu(gate, up):
    """The GLM activation, verbatim from Glm5NextTextMLP / Glm5NextTextExperts._apply_gate."""
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    return torch.nn.functional.silu(gate) * up


def emit(st):
    parts = []
    for name, t in st.items():
        t = t.flatten()
        stride = 1 if t.numel() <= 4096 else (STRIDE if t.numel() <= 4_000_000 else STRIDE_BIG)
        s = t[::stride]
        data = ",".join(f"{x:.9g}" for x in s.tolist())
        f = t.to(torch.float64)
        ck = float((f * (torch.arange(f.numel(), dtype=torch.float64) + 1.0)).sum())
        parts.append('    "%s":{"n":%d,"stride":%d,"ck":%r,"data":[%s]}'
                     % (name, t.numel(), stride, ck, data))
    return "{\n" + ",\n".join(parts) + "\n   }"


# 🪤 The last regime exists ONLY to make the clamp fire. At scale 0.5 nothing reaches
# +/-10 and the swiglu_limit semantics are completely untested; a kernel with no clamp,
# a symmetric clamp, or a clamp on the wrong tensor all pass. At scale 8.0 `gate` exceeds
# +10 (clamped) AND drops below -10 (NOT clamped — the asymmetry), and `up` exceeds both
# bounds. `clampcount` in the diag table is the proof it actually fired.
REGIMES = [("t1", 1, 0.5), ("t7", 7, 0.5), ("t64", 64, 0.5), ("clamp64", 64, 8.0)]
sections, diag = [], []

# ── dense FFN, layer 0 ──────────────────────────────────────────────────────
D = load_packet("/w/dense_layer0.safetensors")
gate_w, up_w, down_w = (D["gate_proj.weight"][0], D["up_proj.weight"][0], D["down_proj.weight"][0])
INTER = gate_w.shape[0]
assert (INTER, gate_w.shape[1]) == (cfg.intermediate_size, HID), (INTER, gate_w.shape)
mlp = Glm5NextTextMLP(cfg).to(torch.bfloat16).eval()
mlp.load_state_dict({"gate_proj.weight": gate_w, "up_proj.weight": up_w,
                     "down_proj.weight": down_w}, strict=True)
for dt_name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
    m = Glm5NextTextMLP(cfg).to(dt).eval()
    m.load_state_dict({k: v.to(dt) for k, v in
                       {"gate_proj.weight": gate_w, "up_proj.weight": up_w,
                        "down_proj.weight": down_w}.items()}, strict=True)
    for rname, T, sc in REGIMES:
        x = (Lcg(0x0FFF5EED).t(T, HID) * sc).to(dt)
        with torch.no_grad():
            g = m.gate_proj(x); u = m.up_proj(x)
            act = swiglu(g, u)
            out = m.down_proj(act)
            ref = m(x)
        assert torch.equal(out, ref), "hand-rolled dense path diverged from Glm5NextTextMLP"
        st = {"gate_out": g[0 if T == 1 else slice(None)].float().flatten().clone(),
              "up_out": u.float().flatten().clone(),
              "act": act.float().flatten().clone(),
              "ffn_out": out.float().flatten().clone()}
        st["gate_out"] = g.float().flatten().clone()
        sections.append(f'   "dense0__{dt_name}__{rname}":' + emit(st))
        diag.append(("dense0", dt_name, rname, T, float(g.float().abs().max()),
                     float(out.float().abs().max()),
                     int((g.float() > LIMIT).sum()), int((g.float() < -LIMIT).sum()),
                     int((u.float().abs() > LIMIT).sum())))

# ── layer 3: shared expert (BF16) + routed experts 0..3 (NVFP4) ─────────────
M = load_packet("/w/moe_layer3_e0_3.safetensors")
sh = {p: M[f"shared_experts.{p}.weight"][0] for p in ("gate_proj", "up_proj", "down_proj")}
SI = sh["gate_proj"].shape[0]
assert SI == cfg.moe_intermediate_size * cfg.n_shared_experts

MI = cfg.moe_intermediate_size
for e in range(4):
    W = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        pk, sk, gk = (f"experts.{e}.{p}.weight", f"experts.{e}.{p}.weight_scale",
                      f"experts.{e}.{p}.weight_scale_2")
        packed, dtp = M[pk]; scale, dts = M[sk]; s2, dt2 = M[gk]
        assert dtp == "U8" and dts == "F8_E4M3" and dt2 == "F32", (dtp, dts, dt2)
        K = HID if p != "down_proj" else MI
        W[p] = dequant_nvfp4(packed, scale, float(s2.item()), K)
        assert W[p].shape == ((MI, HID) if p != "down_proj" else (HID, MI)), W[p].shape
    st = {}
    # The dequantised weight itself IS the layout proof: strided sample + full checksum.
    for p in ("gate_proj", "up_proj", "down_proj"):
        st[f"deq_{p}"] = W[p].flatten().clone()
    for rname, T, sc in REGIMES:
        x = Lcg(0x0FFF5EED).t(T, HID) * sc
        xb = x.to(torch.bfloat16).float()          # bf16 activation, fp32 accumulate
        g32 = x @ W["gate_proj"].T
        u32 = x @ W["up_proj"].T
        o32 = swiglu(g32, u32) @ W["down_proj"].T
        gb = xb @ W["gate_proj"].T
        ub = xb @ W["up_proj"].T
        ob = swiglu(gb, ub) @ W["down_proj"].T
        st[f"{rname}__gate_f32"] = g32.flatten().clone()
        st[f"{rname}__out_f32"] = o32.flatten().clone()
        st[f"{rname}__gate_bf16act"] = gb.flatten().clone()
        st[f"{rname}__out_bf16act"] = ob.flatten().clone()
        diag.append((f"expert{e}", "nvfp4", rname, T, float(g32.abs().max()),
                     float(o32.abs().max()), int((g32 > LIMIT).sum()), int((g32 < -LIMIT).sum()),
                     int((u32.abs() > LIMIT).sum())))
    sections.append(f'   "expert3_{e}":' + emit(st))

shst = {}
for dt_name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
    for rname, T, sc in REGIMES:
        x = (Lcg(0x0FFF5EED).t(T, HID) * sc).to(dt)
        g = torch.nn.functional.linear(x, sh["gate_proj"].to(dt))
        u = torch.nn.functional.linear(x, sh["up_proj"].to(dt))
        o = torch.nn.functional.linear(swiglu(g.float(), u.float()).to(dt), sh["down_proj"].to(dt))
        shst[f"{dt_name}__{rname}__out"] = o.float().flatten().clone()
sections.append('   "shared3":' + emit(shst))

body = ("{\n"
        f' "fixture":{{"hidden":{HID},"intermediate":{cfg.intermediate_size},'
        f'"moe_intermediate":{MI},"shared_intermediate":{SI},"swiglu_limit":{LIMIT!r},'
        f'"hidden_act":"{cfg.hidden_act}","group_size":16,"has_input_scale":false,'
        f'"seed":"0x0FFF5EED","layers":{{"dense":0,"moe":3}},'
        f'"checkpoint":"LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3"}},\n'
        + ",\n".join(sections) + "\n}\n")
open(OUT, "w").write(body)
print("transformers", __import__("transformers").__version__, "torch", torch.__version__)
print("bytes", len(body), "sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"{'what':10} {'dt':6} {'regime':8} {'T':>4} {'|gate|max':>11} {'|out|max':>11} "
      f"{'g>+10':>7} {'g<-10':>7} {'|up|>10':>8}")
for r in diag:
    print("%-10s %-6s %-8s %4d %11.5g %11.5g %7d %7d %8d" % r)
