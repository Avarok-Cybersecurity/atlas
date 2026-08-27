#!/usr/bin/env python3
"""Slice 6 — REAL-CHECKPOINT full-KDA-layer golden, HF transformers 5.16.1.

Drives the genuine `Glm5NextTextLinearAttention` module built from the real `config.json`
(`gate_lower_bound`, `rms_norm_eps`, `hidden_act` all READ) and loaded with the real layer-0
tensors of `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ 9e0d74e3, shard 1/120.

Layer 0 is KDA + dense FFN, and every layer-0 `self_attn` tensor is BF16 (A_log / dt_bias F32) —
nothing is quantised — so this golden IS the production numerics, not a dequantised approximation.

The module's `forward` is re-expressed inline here so every stage is observable. Every call is to
HF's own submodule or HF's own function; no equation is re-derived on this side.

Four regimes:
  R1 decode1            T=1, non-zero conv state, non-zero recurrent state  -> recurrent_kda
  R2 prefill4           T=4, zero state                                     -> chunk_kda
  R3 prefill7           T=7 ragged (7 % 64 != 0), zero state                -> chunk_kda
  R4 prefill7_decode1   R3, then one decode carrying BOTH states            -> the regression carrier

Two dtypes per regime: `f32` (floor A, HF/reference math) and `bf16` (floor B, the production
activation path). Weights are the same bytes; only the cast differs.

Inputs come from the integer LCG the Rust microtest reproduces bit for bit, so only outputs are
committed — bulky tensors as a prime-strided sample plus an fp64 index-weighted checksum.
"""

import hashlib
import json
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import (  # noqa: E402
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextLinearAttention,
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    l2norm,
    recurrent_kimi_delta_attention,
)

PACKET = "/w/layer0.safetensors"
CONFIG = "/w/config.json"
OUT = "/w/kda_layer_golden.json"

STRIDE_CONV = 503   # [T, conv_dim]
STRIDE = 251        # [T, qkv_dim] and [T, hidden]
STRIDE_STATE = 1009  # [H, D, D]

_ST_DT = {"BF16": (torch.bfloat16, np.uint16), "F32": (torch.float32, np.float32)}


def load_packet(path):
    fh = open(path, "rb")
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for k, m in hdr.items():
        a, b = m["data_offsets"]
        fh.seek(base + a)
        raw = fh.read(b - a)
        tdt, ndt = _ST_DT[m["dtype"]]
        arr = np.frombuffer(raw, dtype=ndt).copy()
        t = torch.from_numpy(arr)
        if m["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        out[k] = t.reshape(m["shape"])
    return out


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0

    def t(self, *shape):
        n = 1
        for s in shape:
            n *= s
        return torch.tensor([self.u() for _ in range(n)], dtype=torch.float32).reshape(shape)


# ── config: every KDA-relevant value is READ from the checkpoint ────────────────
raw_cfg = json.load(open(CONFIG))["text_config"]
cfg = Glm5NextTextConfig(**raw_cfg)
H, D = cfg.linear_num_heads, cfg.linear_head_dim
HID = cfg.hidden_size
QKV = H * D
CONV_DIM = 3 * QKV
KS = cfg.linear_conv_kernel_dim
assert (H, D, HID, KS) == (64, 128, 4096, 4), (H, D, HID, KS)
assert cfg.linear_lower_bound == -5.0 and cfg.rms_norm_eps == 1e-5 and cfg.hidden_act == "silu"

W = load_packet(PACKET)


def build(dtype):
    m = Glm5NextTextLinearAttention(cfg, 0).to(dtype).eval()
    sd = {
        "q_proj.weight": W["self_attn.q_proj.weight"],
        "k_proj.weight": W["self_attn.k_proj.weight"],
        "v_proj.weight": W["self_attn.v_proj.weight"],
        # HF fuses q|k|v into ONE depthwise conv1d of conv_dim; the checkpoint stores three
        # separate [8192,1,4] tensors. Concatenation order must match `mixed_qkv = cat([q,k,v])`.
        "conv1d.weight": torch.cat(
            [
                W["self_attn.q_conv1d.weight"],
                W["self_attn.k_conv1d.weight"],
                W["self_attn.v_conv1d.weight"],
            ],
            dim=0,
        ),
        "forget_gate.f_a_proj.weight": W["self_attn.f_a_proj.weight"],
        "forget_gate.f_b_proj.weight": W["self_attn.f_b_proj.weight"],
        "forget_gate.dt_bias": W["self_attn.dt_bias"],
        "forget_gate.A_log": W["self_attn.A_log"],
        "b_proj.weight": W["self_attn.b_proj.weight"],
        "g_a_proj.weight": W["self_attn.g_a_proj.weight"],
        "g_b_proj.weight": W["self_attn.g_b_proj.weight"],
        "o_norm.weight": W["self_attn.o_norm.weight"],
        "o_proj.weight": W["self_attn.o_proj.weight"],
    }
    # dt_bias / A_log stay F32 on disk and F32 in the module (Slice-3 kernel signatures).
    sd = {
        k: (v.to(torch.float32) if k.endswith(("dt_bias", "A_log")) else v.to(dtype))
        for k, v in sd.items()
    }
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    del missing, unexpected
    return m


def run(m, hidden, conv_state3, rec_state, dtype):
    """Inline re-expression of Glm5NextTextLinearAttention.forward, stage by stage.

    `conv_state3` is HF's `kernel_size - 1` = 3 slots, or None for a zero-started prefill.
    Returns (stages dict, final conv state [conv_dim, 3], final recurrent state [H, D, D] fp32).
    """
    T = hidden.shape[0]
    hs = hidden.to(dtype).unsqueeze(0)  # [1, T, HID]
    st = {}

    mixed = torch.cat([m.q_proj(hs), m.k_proj(hs), m.v_proj(hs)], dim=-1)  # [1,T,conv_dim]
    st["qkv_proj"] = mixed[0].float().clone()
    mixed_ct = mixed.transpose(1, 2)  # [1, conv_dim, T]

    w_conv = m.conv1d.weight.squeeze(1)
    if conv_state3 is not None and T == 1:
        cs = conv_state3.to(dtype).unsqueeze(0).clone()  # [1, conv_dim, 3]
        conv = causal_conv1d_update(mixed_ct, cs, weight=w_conv, bias=None, activation=m.activation)
        new_conv_state = cs[0].float().clone()
    else:
        assert conv_state3 is None, "prefill-continuation is out of scope for this golden"
        conv = causal_conv1d_fn(mixed_ct, weight=w_conv, bias=None, activation=m.activation)
        conv = conv[:, :, -T:]
        # zero-started prefill: final state = last (KS-1) RAW conv inputs
        new_conv_state = mixed_ct[0, :, -(KS - 1) :].float().clone()
    st["conv_out"] = conv[0].t().contiguous().float().clone()  # [T, conv_dim], post-SiLU pre-L2

    q, k, v = torch.split(conv.transpose(1, 2), [QKV] * 3, dim=-1)
    shp = (1, T, -1, D)
    q, k, v = q.view(shp), k.view(shp), v.view(shp)
    # L2 is applied INSIDE the KDA kernels (use_qk_l2norm_in_kernel=True), in fp32, on q|k only.
    st["q_l2"] = l2norm(q[0].float().reshape(-1, D), dim=-1, eps=1e-6).reshape(T, QKV).clone()
    st["k_l2"] = l2norm(k[0].float().reshape(-1, D), dim=-1, eps=1e-6).reshape(T, QKV).clone()
    st["v_raw"] = v[0].float().reshape(T, QKV).clone()

    g = m.forget_gate(hs)  # fp32 [1,T,H,D]
    beta = torch.sigmoid(m.b_proj(hs))  # dtype [1,T,H]
    st["gate"] = g[0].float().reshape(T, QKV).clone()
    st["beta"] = beta[0].float().clone()

    if conv_state3 is not None and T == 1:
        core, last = recurrent_kimi_delta_attention(
            q, k, v, g=g, beta=beta, initial_state=rec_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    else:
        core, last = chunk_kimi_delta_attention(
            q, k, v, g=g, beta=beta, initial_state=rec_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    st["core"] = core[0].float().reshape(T, QKV).clone()
    last = last.to(torch.float32)
    st["state"] = last[0].clone()

    out_gate = m.g_b_proj(m.g_a_proj(hs)).view(shp)
    st["out_gate"] = out_gate[0].float().reshape(T, QKV).clone()
    normed = m.o_norm(core, out_gate).reshape(1, T, -1)
    st["o_norm_out"] = normed[0].float().clone()
    final = m.o_proj(normed)
    st["final_out"] = final[0].float().clone()
    st["conv_state"] = new_conv_state
    return st, new_conv_state, last


def sample(t, stride):
    return t.flatten()[::stride].contiguous()


def ck(t):
    f = t.flatten().to(torch.float64)
    i = torch.arange(f.numel(), dtype=torch.float64)
    return float((f * (i + 1.0)).sum())


STRIDE_OF = {
    "qkv_proj": STRIDE_CONV, "conv_out": STRIDE_CONV, "conv_state": STRIDE_CONV,
    "state": STRIDE_STATE,
}


def emit(st):
    parts = []
    for name, t in st.items():
        # The prefill leg of the mixed regime is stored under a `prefill_` prefix; strip it
        # before the stride lookup or the bulky stages silently get the dense stride.
        base = name[len("prefill_"):] if name.startswith("prefill_") else name
        stride = 1 if base == "beta" else STRIDE_OF.get(base, STRIDE)
        s = t.flatten()[::stride].contiguous()
        parts.append(
            '   "%s":{"shape":%s,"n":%d,"stride":%d,"ck":%r,"data":[%s]}'
            % (name, list(t.shape), t.numel(), stride, ck(t),
               ",".join(f"{x:.9g}" for x in s.tolist()))
        )
    return "{\n" + ",\n".join(parts) + "\n  }"


probe = Lcg(0x5EED_1A70)
lcg_probe = [probe.u() for _ in range(8)]

REGIMES = [("decode1", 1), ("prefill4", 4), ("prefill7", 7), ("prefill7_decode1", 7)]
blocks = []
diag = []
for dt_name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
    mod = build(dt)
    for rname, T in REGIMES:
        # Every regime draws from a freshly seeded LCG so each is independently reproducible.
        rng = Lcg(0x5EED_1A70)
        hidden = rng.t(max(T, 8), HID)[:T].contiguous()
        conv_state0 = rng.t(CONV_DIM, KS - 1) * 0.5
        rec0 = rng.t(H, D, D).unsqueeze(0) * 0.1
        with torch.no_grad():
            if rname == "decode1":
                st, _, _ = run(mod, hidden, conv_state0, rec0, dt)
            elif rname == "prefill7_decode1":
                st_p, cs, rs = run(mod, hidden, None, None, dt)
                rng2 = Lcg(0xD3C0_DE01)
                h2 = rng2.t(8, HID)[:1].contiguous()
                st, _, _ = run(mod, h2, cs, rs, dt)
                st = {**{"prefill_" + k: v for k, v in st_p.items()}, **st}
            else:
                st, _, _ = run(mod, hidden, None, None, dt)
        blocks.append(f'  "{dt_name}__{rname}":' + emit(st))
        diag.append((dt_name, rname, float(st["final_out"].abs().max())))

body = (
    "{\n"
    f' "fixture":{{"hidden":{HID},"heads":{H},"head_dim":{D},"conv_dim":{CONV_DIM},'
    f'"qkv_dim":{QKV},"kernel":{KS},"lower_bound":{cfg.linear_lower_bound},'
    f'"rms_eps":{cfg.rms_norm_eps},"hidden_act":"{cfg.hidden_act}","o_norm_act":"sigmoid",'
    f'"hf_chunk":64,"hf_state_width":{KS - 1},"l2_eps":1e-06,"seed":"0x5EED1A70",'
    f'"seed_decode2":"0xD3C0DE01","stride":{STRIDE},"stride_conv":{STRIDE_CONV},'
    f'"stride_state":{STRIDE_STATE},'
    f'"checkpoint":"LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3","layer":0}},\n'
    f' "lcg_probe":[{",".join(f"{x:.9g}" for x in lcg_probe)}],\n'
    ' "regimes":{\n' + ",\n".join(blocks) + "\n }\n}\n"
)
with open(OUT, "w") as fh:
    fh.write(body)

print("transformers", __import__("transformers").__version__, "torch", torch.__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
for d, r, mx in diag:
    print(f"  {d:5s} {r:18s} |final_out|max {mx:.6g}")
