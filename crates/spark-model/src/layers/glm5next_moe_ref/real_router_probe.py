"""Slice 10 gate 5 — REAL-ACTIVATION router experiment, HF 5.16.1.

Runs the genuine prefix (embed -> layers 0,1,2 -> layer 3 attention -> ffn mHC site ->
post_attention_layernorm) on real checkpoint weights and feeds the resulting hidden
states -- the actual distribution the router sees -- through BOTH router dtype ladders.

Synthetic LCG vectors are not the model's distribution; the earlier probe could only show
the ladders are not interchangeable in principle. This shows what happens on real traffic.
"""
import json, struct, sys
import numpy as np, torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention, Glm5NextTextHyperConnection, Glm5NextTextLinearAttention,
    Glm5NextTextMLP, Glm5NextTextRMSNorm,
)

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
        out[k] = t.reshape(m["shape"]) if m["shape"] else t.reshape(())
    return out

raw = json.load(open("/w/config.json"))["text_config"]
cfg = Glm5NextTextConfig(**raw); cfg._attn_implementation = "eager"
HID, HC, DT = cfg.hidden_size, cfg.hc_mult, torch.bfloat16
T = int(sys.argv[1]) if len(sys.argv) > 1 else 256

emb = load("/w/embed.safetensors")["embed_tokens.weight"]
# Real token ids, deterministic, spread across the vocabulary (no tokenizer needed: the
# router sees hidden states, and what matters is that they are the model's own).
g = torch.Generator().manual_seed(0x6E1D)
ids = torch.randint(0, cfg.vocab_size, (T,), generator=g)
h = emb[ids].to(DT)                                        # [T, HID]
hs = h.unsqueeze(0).unsqueeze(2).expand(1, T, HC, HID).contiguous()

def sub(W, pfx):
    return {k[len(pfx):]: v for k, v in W.items() if k.startswith(pfx)}

def hc_mod(W, site):
    m = Glm5NextTextHyperConnection(cfg).to(DT).eval()
    m.load_state_dict({"fn": W[f"hc_{site}_fn"].to(DT), "base": W[f"hc_{site}_base"].to(DT),
                       "scale": W[f"hc_{site}_scale"].to(DT)}, strict=True)
    return m

def norm(W, name):
    n = Glm5NextTextRMSNorm(HID, cfg.rms_norm_eps).to(DT).eval()
    n.load_state_dict({"weight": W[f"{name}.weight"].to(DT)}, strict=True)
    return n

mask = torch.ones(1, T, dtype=torch.bool)
for L in (0, 1, 2, 3):
    W = load(f"/w/prefix_layer{L}.safetensors")
    attn_hc, ffn_hc = hc_mod(W, "attn"), hc_mod(W, "ffn")
    in_ln, post_ln = norm(W, "input_layernorm"), norm(W, "post_attention_layernorm")
    is_kda = "self_attn.q_conv1d.weight" in W
    if is_kda:
        att = Glm5NextTextLinearAttention(cfg, L).to(DT).eval()
        # 🪤 HF fuses q|k|v into ONE depthwise conv1d and nests the decay params under
        # `forget_gate.*`; the checkpoint stores three separate conv tensors at the top level.
        # Same remap the Slice-2..7 KDA generators use. Concatenation order must match
        # `mixed_qkv = cat([q, k, v])`.
        sd = {k: v for k, v in sub(W, "self_attn.").items()
              if k not in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                           "f_a_proj.weight", "f_b_proj.weight", "dt_bias", "A_log")}
        sd["conv1d.weight"] = torch.cat([W["self_attn.q_conv1d.weight"],
                                         W["self_attn.k_conv1d.weight"],
                                         W["self_attn.v_conv1d.weight"]], dim=0)
        for a, b in (("f_a_proj.weight", "forget_gate.f_a_proj.weight"),
                     ("f_b_proj.weight", "forget_gate.f_b_proj.weight"),
                     ("dt_bias", "forget_gate.dt_bias"), ("A_log", "forget_gate.A_log")):
            sd[b] = W["self_attn." + a]
        # dt_bias / A_log are F32 on disk AND F32 in the module.
        sd = {k: (v.to(torch.float32) if k.endswith(("dt_bias", "A_log")) else v.to(DT))
              for k, v in sd.items()}
    else:
        att = Glm5NextTextAttention(cfg, L).to(DT).eval()
        sd = {k: v.to(DT) for k, v in sub(W, "self_attn.").items()}
    att.load_state_dict(sd, strict=True)

    post, comb, x = attn_hc(hs)
    residual = hs
    x = in_ln(x)
    with torch.no_grad():
        if is_kda:
            x = att(hidden_states=x, cache_params=None, attention_mask=mask)
        else:
            x, _, _ = att(hidden_states=x, attention_mask=mask, position_ids=None,
                          past_key_values=None, use_cache=False, position_embeddings=None,
                          prev_topk_indices=None)
    hs = post.to(DT).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
        comb.to(DT).transpose(-1, -2), residual)

    residual = hs
    post, comb, x = ffn_hc(hs)
    router_in = post_ln(x)                                  # <- what the MLP site sees
    if L == 3:
        break
    mlp = Glm5NextTextMLP(cfg).to(DT).eval()
    mlp.load_state_dict({k: v.to(DT) for k, v in sub(W, "mlp.").items()}, strict=True)
    with torch.no_grad():
        y = mlp(router_in)
    hs = post.to(DT).unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(
        comb.to(DT).transpose(-1, -2), residual)

X = router_in[0].detach()                                   # [T, HID] REAL, bf16
print(f"real layer-3 router input: {tuple(X.shape)} {X.dtype}  "
      f"rms={X.float().pow(2).mean().sqrt():.4f}  |x|max={X.float().abs().max():.4f}")

R = load("/w/router_layer3.safetensors")
w, bias = R["gate.weight"], R["gate.e_score_correction_bias"]
TOPK, SCALE = cfg.num_experts_per_tok, cfg.routed_scaling_factor
assert cfg.n_group == 1 and cfg.topk_group == 1, "group routing is NOT a no-op here"

def route(x, dt):
    if dt == torch.float32:
        lg = torch.nn.functional.linear(x.float(), w.float())
    else:
        lg = torch.nn.functional.linear(x.to(dt), w.to(dt))
    sc = lg.sigmoid()
    idx = torch.topk(sc + bias.to(sc.dtype), k=TOPK, dim=-1, sorted=False)[1]
    wt = sc.gather(1, idx)
    wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
    return lg.float(), idx, (wt * SCALE).float()

l32, i32, w32 = route(X, torch.float32)
lbf, ibf, wbf = route(X, torch.bfloat16)
s32 = [set(r.tolist()) for r in i32]; sbf = [set(r.tolist()) for r in ibf]
inter = np.array([len(a & b) for a, b in zip(s32, sbf)])
diff = int(sum(a != b for a, b in zip(s32, sbf)))
moved = []
for r in range(T):
    common = s32[r] & sbf[r]
    moved.append(sum(float(w32[r, j]) for j in range(TOPK)
                     if int(i32[r, j]) not in common) / (float(w32[r].sum()) + 1e-20))
# Selected-weight delta on the experts BOTH ladders picked.
wd = []
for r in range(T):
    m32 = {int(i32[r, j]): float(w32[r, j]) for j in range(TOPK)}
    mbf = {int(ibf[r, j]): float(wbf[r, j]) for j in range(TOPK)}
    for e in set(m32) & set(mbf):
        wd.append(abs(m32[e] - mbf[e]))
print(f"  logit|Δ|max        {(l32 - lbf).abs().max():.4e}")
print(f"  different top-8 SET {diff}/{T} ({100*diff/T:.1f}%)   mean |∩| {inter.mean():.3f}/8")
print(f"  routed mass on non-shared experts  mean {np.mean(moved)*100:.2f}%  "
      f"max {np.max(moved)*100:.2f}%")
print(f"  |Δ weight| on shared experts  mean {np.mean(wd):.4e}  max {np.max(wd):.4e}")
torch.save({"X": X, "i32": i32, "w32": w32, "ibf": ibf, "wbf": wbf}, "/w/real_router.pt")
