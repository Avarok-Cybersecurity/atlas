#!/usr/bin/env python3
"""HF-transformers CPU oracle for the Nemotron-3-Nano per-layer divergence hunt.

CPU variant of nemotron_hf_ref.py: host torch is CPU-only (torch+cpu) and the
DGX has no spare GPU torch, so the canonical BF16 reference runs on CPU. This
is the methodology-sanctioned fallback and is memory-safe (no GPU contention).

Loads the local NVFP4 snapshot, MANUALLY dequantizes every NVFP4 weight to
BF16 (E2M1 nibble * fp8 block-scale * f32 global-scale), loads the dequantized
state_dict into a fresh NemotronH model, feeds the EXACT token IDs Atlas
prefilled (read from TOKEN_IDS json file), and captures per-block hidden
states + final norm + logits as headerless little-endian f32 .bin -- the
format the Atlas ATLAS_NEMO_DUMP hook writes -- so the comparator diffs 1:1.

The dequantized BF16 graph run through `torch_forward` is the canonical
"intended math" oracle (no fused Triton kernels, no custom CUDA).

Env: MODEL (local snapshot path), OUT, TOKEN_IDS (json file of int list).
"""
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mamba_ssm_stub  # noqa: F401  installs pure-torch mamba_ssm + forces torch path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = os.environ["MODEL"]
OUT = pathlib.Path(os.environ.get("OUT", "/out"))
TOKEN_IDS = os.environ["TOKEN_IDS"]
OUT.mkdir(parents=True, exist_ok=True)

# E2M1 FP4 code -> float value (sign-magnitude, 16 codes).
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_nvfp4(packed, wscale, wscale2, group_size=16):
    """packed uint8 [O, K/2] -> bf16 [O, K].  wscale fp8e4m3 [O, K/16].
    value = E2M1[nibble] * fp8(wscale) * f32(wscale2)."""
    O, Khalf = packed.shape
    K = Khalf * 2
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.empty(O, K, dtype=torch.long)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    vals = _E2M1.to(codes.device)[codes]                     # [O, K] f32
    s = wscale.to(torch.float32)                             # [O, K/16]
    s = s.repeat_interleave(group_size, dim=1)               # [O, K]
    vals = vals * s * float(wscale2)
    return vals.to(torch.bfloat16)


def load_dequant_state_dict():
    """Read every safetensors shard; dequant NVFP4 triples to a single
    bf16 `weight`; pass dense tensors through; drop *_scale* sidecars."""
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors")))
    raw = {}
    for f in files:
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                raw[k] = sf.get_tensor(k)
    sd = {}
    quant_bases = set()
    for k in raw:
        if k.endswith(".weight_scale"):
            quant_bases.add(k[: -len(".weight_scale")])
    for base in quant_bases:
        w = raw[base + ".weight"]
        ws = raw[base + ".weight_scale"]
        ws2 = raw[base + ".weight_scale_2"]
        sd[base + ".weight"] = dequant_nvfp4(w, ws, ws2)
    skip_suffix = (".weight_scale", ".weight_scale_2", ".input_scale")
    for k, v in raw.items():
        if any(k.endswith(s) for s in skip_suffix):
            continue
        if k.endswith(".weight") and k[: -len(".weight")] in quant_bases:
            continue  # already dequantized above
        sd[k] = v
    return sd


def save_f32(name, t):
    arr = t.detach().to(torch.float32).cpu().numpy().astype("<f4").ravel()
    (OUT / name).write_bytes(arr.tobytes())
    return arr


def main():
    ids = json.loads(pathlib.Path(TOKEN_IDS).read_text())
    ids = [int(x) for x in ids]
    print("PROMPT_TOKEN_COUNT:", len(ids))
    print("PROMPT_TOKEN_IDS:", ids)

    print("Dequantizing NVFP4 -> BF16 state_dict ...", flush=True)
    sd = load_dequant_state_dict()
    print(f"  state_dict: {len(sd)} tensors")

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    print("Building empty BF16 model ...", flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    model = model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    miss = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    print(f"  missing={len(miss)} unexpected={len(unexpected)}")
    if miss[:8]:
        print("  missing sample:", miss[:8])
    if unexpected[:8]:
        print("  unexpected sample:", unexpected[:8])
    model = model.to(device="cpu", dtype=torch.bfloat16)
    model.eval()

    input_ids = torch.tensor([ids], device="cpu")
    print("Running forward (CPU, may take a few minutes) ...", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    # NemotronHModel.forward appends `hidden_states` to all_hidden_states
    # BEFORE each of the n_layers mixer blocks runs, then appends the
    # post-norm_f tensor once at the end.  So:
    #     hs = [pre-L0, pre-L1, ..., pre-L{n-1}, POST-norm_f]
    #     len(hs) == n_layers + 1
    # hs[0] is the embedding output (== input to L0).  hs[i] is the OUTPUT
    # of layer (i-1) for i in 1 .. n_layers-1 (true outputs L0 .. L{n-2}).
    # The LAST entry hs[-1] is post-norm_f -- NOT layer (n-1)'s residual,
    # which HF never exposes.  Dump hs[-1] directly as the final-norm
    # output (applying norm_f again would double-normalize).
    hs = out.hidden_states
    n_layers = len(model.backbone.layers)
    print("NUM_HIDDEN_STATES:", len(hs),
          f"({n_layers} pre-layer states + post-norm_f)")
    assert len(hs) == n_layers + 1, f"unexpected hidden_states len {len(hs)}"
    last = -1
    emb = save_f32("hf_embed.bin", hs[0][0, last])
    print(f"hf_embed: norm={np.linalg.norm(emb):.4f}")
    for i in range(1, n_layers):
        arr = save_f32(f"hf_L{i-1}.bin", hs[i][0, last])
        if i - 1 < 4 or i - 1 >= n_layers - 5:
            print(f"hf_L{i-1}: norm={np.linalg.norm(arr):.4f}")

    fn_arr = save_f32("hf_final_norm.bin", hs[-1][0, last])
    print(f"hf_final_norm (post-norm_f): norm={np.linalg.norm(fn_arr):.4f}")

    logits = out.logits[0, last]
    save_f32("hf_logits.bin", logits)
    top = torch.topk(logits.float(), 10)
    top_list = [(int(i), float(v)) for i, v in zip(top.indices, top.values)]
    print("HF_TOP10_LOGITS:", top_list)
    (OUT / "top10.json").write_text(json.dumps(top_list))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
