#!/usr/bin/env python3
"""HF GDN layer-0 reference for Qwen3.6-35B-A3B-FP8.

Adapted from hf_gdn_ref2.py (Qwen3.6-27B). A3B's GDN config:
  linear_num_key_heads   = 16   linear_key_head_dim   = 128  -> key_dim   = 2048
  linear_num_value_heads = 32   linear_value_head_dim = 128  -> value_dim = 4096
  linear_conv_kernel_dim = 4
  conv_dim = key_dim*2 + value_dim = 2048+2048+4096 = 8192
  hidden_size = 2048 ; rms_norm_eps = 1e-6

PyTorch fast path is unavailable on dgx2 (no causal_conv1d / fla /
FusedRMSNormGated installed), so the model runs the pure-torch reference
— byte-perfect oracle.

EXACT shapes (B=1, L=31):
  in_proj_qkv  hook out         : [B, L, 8192]                          last tok [0, L-1, :]     (8192)
  conv1d       hook (RAW Conv1d): [B, 8192, L+K-1]   parent slices [:,:,:L] + silu
                                   last token  = silu(raw[0,:,:L])[:, L-1]                       (8192)
  recur (norm pre-hook input)   : [B*L*32, 128]   last token rows (L-1)*32 : L*32, flat 4096     (4096)
  norm  (norm forward-hook out) : [B*L*32, 128]   same rows, flat                                (4096)
  out_proj     hook out         : [B, L, 2048]                          last tok [0, L-1, :]     (2048)

Token IDs were captured from atlas-35b-a3b-fix via POST /tokenize for the
rendered chat-template-wrapped prompt:
  "<|im_start|>user\nWhat is 17 times 23? Reply with the number only, no prose.<|im_end|>\n"
  "<|im_start|>assistant\n<think>\n\n</think>\n\n"
"""
import json, os, pathlib, sys
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

SNAP = os.environ.get("SNAP")
OUT  = pathlib.Path(os.environ.get("OUT", "/home/claude/gdnref_a3b"))
LAYERS = [int(x) for x in os.environ.get("GDN_LAYERS", "0").split(",")]

# Token IDs captured from Atlas /tokenize for the probe prompt. 31 tokens.
TOK = [248045, 846, 198, 3710, 369, 220, 16, 22, 2942, 220, 17, 18, 30, 17308, 440, 279, 1324, 1132, 11, 874, 58655, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
L   = len(TOK)               # 31
NUM_V_HEADS = 32
HEAD_V_DIM  = 128
CONV_DIM    = 8192
HIDDEN      = 2048
CONV_K      = 4


def w(vec, path, mod, shape, note):
    """Headerless little-endian BF16 + JSON sidecar."""
    if vec is None: return
    vec = np.ascontiguousarray(vec, dtype=np.float64)
    u = torch.tensor(vec).to(torch.bfloat16).view(torch.uint16).numpy().astype("<u2")
    pathlib.Path(path).write_bytes(u.tobytes())
    pathlib.Path(str(path)[:-4] + ".json").write_text(json.dumps(
        {"module": mod, "shape": list(shape), "n": int(u.size), "note": note}))
    print(f"  wrote {pathlib.Path(path).name:32s} n={u.size:6d} shape={tuple(shape)}  {note}", flush=True)


def cap_in_proj_qkv(li, out):
    o = out.detach().float()
    assert o.dim() == 3 and o.shape[1] == L and o.shape[2] == CONV_DIM, f"in_proj_qkv {tuple(o.shape)}"
    v = o[0, L - 1, :].contiguous().cpu().numpy()
    w(v, OUT / f"gdnref_L{li}_in_proj_qkv.bin", f"L{li}.in_proj_qkv",
      tuple(o.shape), "last tok out[0,L-1,:] (q2048|k2048|v4096)")


def cap_conv1d(li, raw):
    o = raw.detach().float()
    assert o.dim() == 3 and o.shape[1] == CONV_DIM, f"conv1d raw {tuple(o.shape)}"
    sliced = o[:, :, :L]
    post   = F.silu(sliced)
    v = post[0, :, L - 1].contiguous().cpu().numpy()
    w(v, OUT / f"gdnref_L{li}_conv1d.bin", f"L{li}.conv1d_post_silu",
      (1, CONV_DIM, L), f"raw{tuple(o.shape)} -> silu(raw[:,:, :{L}])[:, :, {L-1}] (q2048|k2048|v4096)")


def _last_tok_from_flat_BL32x128(t):
    o = t.detach().float()
    assert o.dim() == 2 and o.shape[1] == HEAD_V_DIM, f"flat32x128 {tuple(o.shape)}"
    assert o.shape[0] == L * NUM_V_HEADS, f"rows {o.shape[0]} != L*32={L*NUM_V_HEADS}"
    rows = o[(L - 1) * NUM_V_HEADS: L * NUM_V_HEADS, :]
    return rows.reshape(-1).contiguous().cpu().numpy()


def cap_recur_in(li, x):
    v = _last_tok_from_flat_BL32x128(x)
    w(v, OUT / f"gdnref_L{li}_recur_in.bin", f"L{li}.recur_in(pre-norm)",
      (L * NUM_V_HEADS, HEAD_V_DIM),
      f"chunk_gated_delta_rule out reshaped[L*32,128]; rows[{(L-1)*NUM_V_HEADS}:{L*NUM_V_HEADS}] head-major 4096")


def cap_norm(li, out):
    v = _last_tok_from_flat_BL32x128(out)
    w(v, OUT / f"gdnref_L{li}_norm.bin", f"L{li}.norm(gated-rmsnorm)",
      (L * NUM_V_HEADS, HEAD_V_DIM),
      f"RMSNormGated(per-head128)*silu(z); rows[{(L-1)*NUM_V_HEADS}:{L*NUM_V_HEADS}] head-major 4096")


def cap_out_proj(li, out):
    o = out.detach().float()
    assert o.dim() == 3 and o.shape[1] == L and o.shape[2] == HIDDEN, f"out_proj {tuple(o.shape)}"
    v = o[0, L - 1, :].contiguous().cpu().numpy()
    w(v, OUT / f"gdnref_L{li}_out_proj.bin", f"L{li}.out_proj", tuple(o.shape), "last tok (2048)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"load {SNAP}", flush=True)
    print(f"is_fast_path_available={M.is_fast_path_available} "
          f"causal_conv1d_fn={M.causal_conv1d_fn} "
          f"chunk_gated_delta_rule={M.chunk_gated_delta_rule} "
          f"FusedRMSNormGated={M.FusedRMSNormGated}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True).eval()
    hooks = []
    for li in LAYERS:
        la = m.model.layers[li].linear_attn
        hooks += [
            la.in_proj_qkv.register_forward_hook(lambda mod, i, o, li=li: cap_in_proj_qkv(li, o)),
            la.conv1d.register_forward_hook(lambda mod, i, o, li=li: cap_conv1d(li, o)),
            la.norm.register_forward_pre_hook(lambda mod, i, li=li: cap_recur_in(li, i[0])),
            la.norm.register_forward_hook(lambda mod, i, o, li=li: cap_norm(li, o)),
            la.out_proj.register_forward_hook(lambda mod, i, o, li=li: cap_out_proj(li, o)),
        ]
        print(f"L{li}.linear_attn hooked", flush=True)
    with torch.no_grad():
        m(torch.tensor([TOK]), use_cache=False)
    for h in hooks: h.remove()
    print(f"DONE -> {OUT} ({len(list(OUT.glob('*.bin')))} bins)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
