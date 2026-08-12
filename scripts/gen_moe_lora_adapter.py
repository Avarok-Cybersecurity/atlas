# SPDX-License-Identifier: AGPL-3.0-only

"""Generate a MINIMAL synthetic PEFT MoE LoRA adapter (routed-expert down_proj +
router) for a Qwen3.6-style MoE base — the Feature-1 GPU validation fixture.

Real unfused per-expert adapters are near-absent on the hub (fine-tunes export
the fused `gate_up_proj` via `target_parameters`, which Atlas rejects), so this
builds a tiny loadable one: a few layers × a few experts of `mlp.experts.E.
down_proj` + the `mlp.gate` router, with NONZERO B (small normal) so the delta
is non-trivial. Shapes are read from the base config (never assumed).

Keys (PEFT save_pretrained form Atlas classify_key accepts):
  base_model.model.model.layers.{L}.mlp.experts.{E}.down_proj.lora_{A,B}.weight
  base_model.model.model.layers.{L}.mlp.gate.lora_{A,B}.weight

Usage: gen_moe_lora_adapter.py <base_config_dir> <out_dir> [r=8] [n_layers=2] [n_experts=4]
"""
import json
import os
import struct
import sys

import numpy as np


def bf16_bytes(arr):
    # round-to-nearest-even f32 -> bf16
    u = arr.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16).tobytes()


def main():
    cfg_dir = sys.argv[1]
    out = sys.argv[2]
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_experts = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    alpha = 2 * r

    d = json.load(open(os.path.join(cfg_dir, "config.json")))
    t = d.get("text_config", d)
    H = t["hidden_size"]
    MI = t["moe_intermediate_size"]
    E = t["num_experts"]
    L = t["num_hidden_layers"]
    lt = t.get("layer_types") or ["full_attention"] * L
    # Layer selection: env LORA_LAYERS="3,7" for explicit; else default to the
    # full-attention layers (the currently-installable MoE path — the GDN/linear
    # layer MoE install is a follow-up), falling back to the first layers.
    env_layers = os.environ.get("LORA_LAYERS")
    if env_layers:
        layers = [int(x) for x in env_layers.split(",")][:n_layers]
    else:
        fa = [i for i, x in enumerate(lt) if x == "full_attention"]
        layers = (fa or list(range(L)))[:n_layers]
    experts = list(range(min(n_experts, E)))
    rng = np.random.default_rng(0)  # deterministic (no argless Date/rand)
    std = float(os.environ.get("LORA_STD", "0.02"))  # init magnitude (strength knob)

    tensors = {}

    def add(name, arr):
        tensors[name] = arr

    for li in layers:
        pfx = f"base_model.model.model.layers.{li}"
        # router mlp.gate: in=H -> out=E
        add(f"{pfx}.mlp.gate.lora_A.weight", rng.normal(0, std, (r, H)).astype(np.float32))
        add(f"{pfx}.mlp.gate.lora_B.weight", rng.normal(0, std, (E, r)).astype(np.float32))
        full = os.environ.get("LORA_FULL") == "1"  # also emit gate/up experts
        for e in experts:
            # expert down_proj: in=MI -> out=H
            add(f"{pfx}.mlp.experts.{e}.down_proj.lora_A.weight",
                rng.normal(0, std, (r, MI)).astype(np.float32))
            add(f"{pfx}.mlp.experts.{e}.down_proj.lora_B.weight",
                rng.normal(0, std, (H, r)).astype(np.float32))
            if full:
                # expert gate_proj + up_proj: in=H -> out=MI
                for proj in ("gate_proj", "up_proj"):
                    add(f"{pfx}.mlp.experts.{e}.{proj}.lora_A.weight",
                        rng.normal(0, std, (r, H)).astype(np.float32))
                    add(f"{pfx}.mlp.experts.{e}.{proj}.lora_B.weight",
                        rng.normal(0, std, (MI, r)).astype(np.float32))

    # serialize safetensors (bf16)
    os.makedirs(out, exist_ok=True)
    header = {}
    blob = bytearray()
    for name, arr in tensors.items():
        b = bf16_bytes(arr)
        header[name] = {"dtype": "BF16", "shape": list(arr.shape),
                        "data_offsets": [len(blob), len(blob) + len(b)]}
        blob += b
    hj = json.dumps(header).encode()
    pad = (8 - (len(hj) % 8)) % 8
    hj += b" " * pad
    with open(os.path.join(out, "adapter_model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)

    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": r, "lora_alpha": alpha, "lora_dropout": 0.0, "use_rslora": False,
        "bias": "none",
        "target_modules": ["down_proj", "gate"],
        "base_model_name_or_path": cfg_dir,
    }
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    print(f"wrote {len(tensors)} tensors to {out}")
    print(f"  layers={layers} (types={[lt[i] for i in layers]}) experts={experts} r={r}")
    print(f"  H={H} MI={MI} E={E}  down_proj[{H}x{MI}] router[{E}x{H}]")


if __name__ == "__main__":
    main()
