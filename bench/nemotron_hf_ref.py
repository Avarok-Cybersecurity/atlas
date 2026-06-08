#!/usr/bin/env python3
"""HF-transformers GPU oracle for the Nemotron-3-Nano per-layer divergence hunt.

Loads nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4, MANUALLY dequantizes every
NVFP4 weight to BF16 (transformers 5.8 has no NVFP4 backend in this image),
loads the dequantized state_dict into a fresh NemotronH model, feeds the EXACT
chat-rendered token IDs, and captures per-block hidden states + final norm +
logits in headerless little-endian f32 .bin -- the format the Atlas
ATLAS_NEMO_DUMP hook writes -- so the comparator can diff 1:1.

The dequantized BF16 graph run through `torch_forward` is the canonical
"intended math" oracle (no fused Triton kernels, no custom CUDA).

Env: MODEL (local snapshot path), OUT, PROMPT.
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
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["MODEL"]
OUT = pathlib.Path(os.environ.get("OUT", "/out"))
PROMPT = os.environ.get("PROMPT", "Please count from 1 to 30. Output every number.")
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
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    print("PROMPT_TOKEN_COUNT:", len(ids))
    print("PROMPT_TOKEN_IDS:", ids)
    (OUT / "token_ids.json").write_text(json.dumps(ids))
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    (OUT / "rendered_prompt.txt").write_text(rendered)
    print("RENDERED_PROMPT_REPR:", repr(rendered))

    print("Dequantizing NVFP4 -> BF16 state_dict ...", flush=True)
    sd = load_dequant_state_dict()
    print(f"  state_dict: {len(sd)} tensors")

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    print("Building empty BF16 model ...", flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    model = model.to_empty(device="cpu")
    # `backbone.` prefix in checkpoint matches NemotronH module tree.
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    miss = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    print(f"  missing={len(miss)} unexpected={len(unexpected)}")
    if miss[:8]:
        print("  missing sample:", miss[:8])
    if unexpected[:8]:
        print("  unexpected sample:", unexpected[:8])
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states
    print("NUM_HIDDEN_STATES:", len(hs), "(embed + 52 layers)")
    last = -1
    emb = save_f32("hf_embed.bin", hs[0][0, last])
    print(f"hf_embed: norm={np.linalg.norm(emb):.4f}")
    for i in range(1, len(hs)):
        arr = save_f32(f"hf_L{i-1}.bin", hs[i][0, last])
        if i - 1 < 4 or i - 1 >= len(hs) - 5:
            print(f"hf_L{i-1}: norm={np.linalg.norm(arr):.4f}")

    final_hidden = hs[-1][0, last].to(torch.bfloat16)
    norm_f = model.backbone.norm_f
    fn = norm_f(final_hidden.unsqueeze(0)).squeeze(0)
    fn_arr = save_f32("hf_final_norm.bin", fn)
    print(f"hf_final_norm: norm={np.linalg.norm(fn_arr):.4f}")

    logits = out.logits[0, last]
    save_f32("hf_logits.bin", logits)
    top = torch.topk(logits.float(), 10)
    top_list = [(int(i), float(v)) for i, v in zip(top.indices, top.values)]
    print("HF_TOP10_LOGITS:", top_list)
    (OUT / "top10.json").write_text(json.dumps(top_list))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
