#!/usr/bin/env python3
"""Slice 8 Gate 5 — NoPE MLA golden over the indexer's selected tokens, HF 5.16.1.

Runs the real `Glm5NextTextAttention` (q_lora + kv_lora + NoPE + its own indexer) on real layer
weights, and records every stage. `_attn_implementation = "eager"` so the selected-token mask is
the additive `-inf` form and the whole thing is line-comparable.

NoPE: `qk_rope_head_dim = 0`, so `kv_a_proj_with_mqa` emits `kv_lora_rank + 0` and HF's `k_rot`
copy into `key_states[..., 256:]` is a zero-width no-op. There is no padded rope section.
"""
import hashlib, json, struct, sys

import numpy as np
import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention  # noqa: E402

LAYERS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["3"])]
OUT = "/w/dsa_mla_golden.json"
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
cfg._attn_implementation = "eager"
HID, NH = cfg.hidden_size, cfg.num_attention_heads
NOPE, ROPE, VD = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
QL, KVL = cfg.q_lora_rank, cfg.kv_lora_rank
assert ROPE == 0, "this golden is the NoPE path"


def build(dtype, W):
    m = Glm5NextTextAttention(cfg, 3).to(dtype).eval()
    sd = {
        "q_a_proj.weight": W["self_attn.q_a_proj.weight"],
        "q_a_layernorm.weight": W["self_attn.q_a_layernorm.weight"],
        "q_b_proj.weight": W["self_attn.q_b_proj.weight"],
        "kv_a_proj_with_mqa.weight": W["self_attn.kv_a_proj_with_mqa.weight"],
        "kv_a_layernorm.weight": W["self_attn.kv_a_layernorm.weight"],
        "kv_b_proj.weight": W["self_attn.kv_b_proj.weight"],
        "o_proj.weight": W["self_attn.o_proj.weight"],
        "indexer.wq_b.weight": W["self_attn.indexer.wq_b.weight"],
        "indexer.wk.weight": W["self_attn.indexer.wk.weight"],
        "indexer.k_norm.weight": W["self_attn.indexer.k_norm.weight"],
        "indexer.k_norm.bias": W["self_attn.indexer.k_norm.bias"],
        "indexer.weights_proj.weight": W["self_attn.indexer.weights_proj.weight"],
        "indexer.index_kpool_compress_ape": W["self_attn.indexer.index_kpool_compress_ape"],
        "indexer.index_kpool_compress_gate": W["self_attn.indexer.index_kpool_compress_gate"],
    }
    m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    return m


def run(m, hidden, mask, dtype):
    B, S = hidden.shape[:2]
    st = {}
    hs = hidden.to(dtype)
    q_resid = m.q_a_layernorm(m.q_a_proj(hs))
    q = m.q_b_proj(q_resid).view(B, S, -1, NOPE + ROPE).transpose(1, 2)
    st["q_resid"] = q_resid[0].float().clone()
    st["q_states"] = q[0].transpose(0, 1).reshape(S, -1).float().clone()

    compressed = m.kv_a_proj_with_mqa(hs)
    kv_pass, k_rot = torch.split(compressed, [KVL, ROPE], dim=-1)
    assert k_rot.shape[-1] == 0, "NoPE: the rope split must be zero-width"
    k_pass = m.kv_a_layernorm(kv_pass).view(B, 1, S, KVL)
    st["kv_c"] = k_pass[0, 0].float().clone()
    key_states, value_states = m.expand_kv(k_pass, k_rot.view(B, 1, S, ROPE))
    st["k_states"] = key_states[0].transpose(0, 1).reshape(S, -1).float().clone()
    st["v_states"] = value_states[0].transpose(0, 1).reshape(S, -1).float().clone()

    topk = m.indexer(hidden_states=hs, q_resid=q_resid, attention_mask=mask, past_key_values=None)
    st["topk_indices"] = topk[0].to(torch.int64).clone()

    add_mask = m.build_attention_mask_from_topk(topk, q, key_states.shape[2])
    st["visible_per_row"] = (add_mask[0, 0] == 0).sum(-1).to(torch.int64).clone()

    from transformers.models.glm5_next.modeling_glm5_next import eager_attention_forward
    attn_out, _ = eager_attention_forward(m, q, key_states, value_states, add_mask, m.scaling)
    st["attn_out"] = attn_out[0].reshape(S, -1).float().clone()
    out = m.o_proj(attn_out.reshape(B, S, -1).contiguous())
    st["final_out"] = out[0].float().clone()
    return st


def emit(st):
    parts = []
    for name, t in st.items():
        t = t.flatten()
        stride = 1 if t.numel() <= 4096 else (STRIDE if t.numel() <= 4_000_000 else STRIDE_BIG)
        s = t[::stride]
        if t.dtype in (torch.int64, torch.int32):
            data = ",".join(str(int(x)) for x in s.tolist())
            ck = float(sum((i + 1) * int(v) for i, v in enumerate(t.tolist())))
        else:
            data = ",".join(f"{x:.9g}" for x in s.tolist())
            f = t.to(torch.float64)
            ck = float((f * (torch.arange(f.numel(), dtype=torch.float64) + 1.0)).sum())
        parts.append('    "%s":{"n":%d,"stride":%d,"ck":%r,"data":[%s]}' % (name, t.numel(), stride, ck, data))
    return "{\n" + ",\n".join(parts) + "\n   }"


# S=2176 gives 544 pools vs select_k=512, so 32 pools are genuinely dropped and the MLA really
# runs sparse. Larger S makes eager attention's [1,64,S,S] score tensor the binding constraint.
REGIMES = [("short7", 7, 0), ("medium64", 64, 0), ("ragged13", 13, 5), ("sparse2176", 2176, 0)]
blocks, diag = [], []
for LAYER in LAYERS:
    W = load_packet(f"/w/dsa_layer{LAYER}.safetensors")
    per = []
    for dt_name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        m = build(dt, W)
        for rname, S, pad in REGIMES:
            rng = Lcg(0x0D5A_C0DE)
            hidden = rng.t(1, S, HID) * 0.5
            mask = torch.ones(1, S, dtype=torch.bool)
            if pad: mask[:, :pad] = False
            with torch.no_grad():
                st = run(m, hidden, mask, dt)
            per.append(f'   "{dt_name}__{rname}":' + emit(st))
            diag.append((LAYER, dt_name, rname, S, int(st["visible_per_row"].max()),
                         float(st["final_out"].abs().max())))
            del st
    blocks.append(f'  "{LAYER}":{{\n' + ",\n".join(per) + "\n  }")

body = ("{\n"
        f' "fixture":{{"hidden":{HID},"heads":{NH},"qk_nope_head_dim":{NOPE},'
        f'"qk_rope_head_dim":{ROPE},"v_head_dim":{VD},"q_lora_rank":{QL},"kv_lora_rank":{KVL},'
        f'"scaling":{(NOPE + ROPE) ** -0.5!r},"rms_norm_eps":{cfg.rms_norm_eps!r},'
        f'"seed":"0x0D5AC0DE","attn_impl":"eager",'
        f'"checkpoint":"LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3","layers":{LAYERS}}},\n'
        ' "by_layer":{\n' + ",\n".join(blocks) + "\n }\n}\n")
open(OUT, "w").write(body)
print("transformers", __import__("transformers").__version__, "torch", torch.__version__)
print("bytes", len(body)); print("sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"{'L':>3} {'dt':5} {'regime':11} {'S':>5} {'visible':>8} {'|out|max':>10}")
for r in diag: print("%3d %-5s %-11s %5d %8d %10.5g" % r)
