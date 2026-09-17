#!/usr/bin/env python3
"""Slice 9 gate 1 — mHC (Manifold-Constrained Hyper-Connections) golden, HF 5.16.1.

Runs the real `Glm5NextTextHyperConnection` on real per-layer `hc_{attn,ffn}_{fn,base,scale}`
weights and records every stage, plus the decoder layer's own residual-write expression and the
final `Glm5NextTextHyperHead` collapse.

Why this exists: Slice 2 down-graded mHC to REUSE because Atlas's `hc_mult`/`hc_sinkhorn_iters`
matched GLM's config. That is a config match, not an arithmetic one. This golden is the
arithmetic.

🪤 `hc_*_fn` is **BF16 on disk** (base/scale are F32). HF stores the parameter in the module dtype
and casts with `.float()` at use, so the mapping is computed in fp32 from bf16-rounded weights.
"""
import hashlib, json, struct, sys

import numpy as np
import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextHyperConnection,
    Glm5NextTextHyperHead,
)

LAYERS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0"])]
OUT = "/w/mhc_golden.json"
STRIDE, STRIDE_BIG = 251, 10009
_ST = {"BF16": (torch.bfloat16, np.uint16), "F32": (torch.float32, np.float32)}


def load_packet(path):
    fh = open(path, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n)); hdr.pop("__metadata__", None); base = 8 + n
    out = {}
    for k, m in hdr.items():
        a, b = m["data_offsets"]; fh.seek(base + a)
        arr = np.frombuffer(fh.read(b - a), dtype=_ST[m["dtype"]][1]).copy()
        t = torch.from_numpy(arr)
        if m["dtype"] == "BF16": t = t.view(torch.bfloat16)
        out[k] = t.reshape(m["shape"])
    return out


class Lcg:
    def __init__(self, seed): self.s = seed & 0xFFFFFFFFFFFFFFFF
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0
    def t(self, *shape):
        n = 1
        for s in shape: n *= s
        return torch.tensor([self.u() for _ in range(n)], dtype=torch.float32).reshape(shape)


raw = json.load(open("/w/config.json"))["text_config"]
cfg = Glm5NextTextConfig(**raw)
HID, HC = cfg.hidden_size, cfg.hc_mult
MIX = (2 + HC) * HC
assert cfg.hc_sinkhorn_iters == 20 and cfg.hc_eps == 1e-6, "config drifted from the pinned mHC knobs"


def build(dtype, W, site):
    m = Glm5NextTextHyperConnection(cfg).to(dtype).eval()
    sd = {
        "fn": W["hc_%s_fn" % site],
        "base": W["hc_%s_base" % site],
        "scale": W["hc_%s_scale" % site],
    }
    for k, v in sd.items():
        assert tuple(v.shape) == tuple(dict(m.named_parameters())[k].shape), (k, v.shape)
    m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    return m


def run(m, streams, block_out, dtype):
    """One mHC site: the module, then the decoder layer's residual-write expression."""
    st = {}
    hs = streams.to(dtype)
    bo = block_out.to(dtype)
    post, comb, collapsed = m(hs)
    st["post"] = post[0].float().clone()                       # [T, hc]
    st["comb"] = comb[0].float().clone()                       # [T, hc, hc]
    st["collapsed"] = collapsed[0].float().clone()             # [T, H]
    # Diagnostic: how doubly-stochastic is the reference's comb actually?
    st["comb_row_sums"] = comb[0].float().sum(-1).clone()
    st["comb_col_sums"] = comb[0].float().sum(-2).clone()
    # The decoder layer's own write-back (Glm5NextTextDecoderLayer.forward), verbatim.
    out = post.to(dtype).unsqueeze(-1) * bo.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), hs
    )
    st["site_out"] = out[0].float().reshape(-1).clone()         # [T, hc, H] flattened
    return st, out


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


# mHC is strictly per-token (no sequence mixing), so the regimes only vary T. T=1 is the decode
# shape; 2176 matches the DSA golden's sparse regime so residuals are comparable across slices.
REGIMES = [("decode1", 1), ("short7", 7), ("medium64", 64), ("long2176", 2176)]
head = Glm5NextTextHyperHead()
blocks, diag = [], []
for LAYER in LAYERS:
    W = load_packet(f"/w/mhc_layer{LAYER}.safetensors")
    per = []
    for dt_name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        mods = {s: build(dt, W, s) for s in ("attn", "ffn")}
        for rname, T in REGIMES:
            rng = Lcg(0x0E1C0DE5)
            streams = rng.t(1, T, HC, HID) * 0.5
            st_all = {}
            cur = streams
            for site in ("attn", "ffn"):
                bo = rng.t(1, T, HID) * 0.5
                with torch.no_grad():
                    st, cur = run(mods[site], cur, bo, dt)
                st_all.update({f"{site}__{k}": v for k, v in st.items()})
                st_all[f"{site}__block_out"] = bo[0].float().reshape(-1).clone()
            with torch.no_grad():
                st_all["hc_head_out"] = head(cur)[0].float().reshape(-1).clone()
            per.append(f'   "{dt_name}__{rname}":' + emit(st_all))
            diag.append((LAYER, dt_name, rname, T,
                         float(st_all["attn__comb_col_sums"].sub(1.0).abs().max()),
                         float(st_all["attn__comb_row_sums"].sub(1.0).abs().max()),
                         float(st_all["hc_head_out"].abs().max())))
            del st_all
    blocks.append(f'  "{LAYER}":{{\n' + ",\n".join(per) + "\n  }")

body = ("{\n"
        f' "fixture":{{"hidden":{HID},"hc_mult":{HC},"mix":{MIX},'
        f'"hc_sinkhorn_iters":{cfg.hc_sinkhorn_iters},"hc_eps":{cfg.hc_eps!r},'
        f'"rms_norm_eps":{cfg.rms_norm_eps!r},"seed":"0x0E1C0DE5",'
        f'"checkpoint":"LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3","layers":{LAYERS}}},\n'
        ' "by_layer":{\n' + ",\n".join(blocks) + "\n }\n}\n")
open(OUT, "w").write(body)
print("transformers", __import__("transformers").__version__, "torch", torch.__version__)
print("bytes", len(body)); print("sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"{'L':>3} {'dt':5} {'regime':9} {'T':>5} {'|colsum-1|':>11} {'|rowsum-1|':>11} {'|head|max':>10}")
for r in diag: print("%3d %-5s %-9s %5d %11.4g %11.4g %10.5g" % r)
