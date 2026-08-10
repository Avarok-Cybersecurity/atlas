#!/usr/bin/env python
"""Generate a tiny PEFT-format LoRA adapter for Hcompany/Holo-3.1-0.8B (Atlas LoRA MVP test fixture).

Builds lora_A [r, in] / lora_B [out, r] tensors (NONZERO B — small normal init, unlike PEFT's
default B=0) for k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj on ONLY the 6
full-attention layers (indices 3, 7, 11, 15, 19, 23 — read from the base config.json, asserted).
q_proj is deliberately excluded: attn_output_gate=true makes it gated/interleaved ([4096,1024] =
2x q_heads*head_dim) and it is out of scope for LoRA v0.

Per-module shapes are read from the base checkpoint's safetensors header (never assumed) and
asserted against the expected hidden=1024 / kv=512 / q-out=2048 / intermediate=3584 geometry.

Output (default /home/ms/atlas/.claude/worktrees/lora-mvp-e0877873/test_data/lora-holo-tiny):
  adapter_model.safetensors   BF16 A/B pairs, PEFT save_pretrained key format
  adapter_config.json         written by peft.LoraConfig.save_pretrained (guaranteed PEFT-valid)

Key style (--key-style):
  vlm  (default) base_model.model.model.language_model.layers.{i}....lora_A.weight
                 -- what PEFT 0.19.1 actually saves when wrapping Qwen3_5ForConditionalGeneration
                 (verified via get_peft_model_state_dict on the real class, meta device)
  text           base_model.model.model.layers.{i}....lora_A.weight
                 -- the text-tower-only form; the Atlas remapper must accept both

Run:
  /home/ms/nemotron-diffusion-playground/.venv/bin/python scripts/gen_lora_adapter.py
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

BASE_MODEL_ID = "Hcompany/Holo-3.1-0.8B"
# k_proj/v_proj/o_proj + dense FFN. q_proj EXCLUDED (attn_output_gate -> gated/interleaved).
ATTN_TARGETS = ("k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
# module -> (out_features, in_features), asserted against the real safetensors header.
EXPECTED_SHAPES = {
    "self_attn.k_proj": (512, 1024),   # kv_heads(2) * head_dim(256), hidden
    "self_attn.v_proj": (512, 1024),
    "self_attn.o_proj": (1024, 2048),  # hidden, q_heads(8) * head_dim(256)
    "mlp.gate_proj": (3584, 1024),     # intermediate, hidden
    "mlp.up_proj": (3584, 1024),
    "mlp.down_proj": (1024, 3584),     # hidden, intermediate
}


def find_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(BASE_MODEL_ID, local_files_only=True))


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/ms/atlas/.claude/worktrees/lora-mvp-e0877873/test_data/lora-holo-tiny"),
    )
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--use-rslora", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--key-style", choices=("vlm", "text"), default="vlm")
    args = ap.parse_args()

    snap = find_snapshot()
    cfg = json.loads((snap / "config.json").read_text())
    text_cfg = cfg["text_config"]

    # --- CONFIRM the locked geometry from the real config, never assume -------------------
    layer_types = text_cfg["layer_types"]
    full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    assert full_attn == EXPECTED_FULL_ATTN_LAYERS, f"full-attn layers {full_attn}"
    assert text_cfg["attn_output_gate"] is True, "q_proj-exclusion rationale changed"
    assert text_cfg["hidden_size"] == 1024 and text_cfg["intermediate_size"] == 3584
    assert text_cfg["num_attention_heads"] == 8 and text_cfg["num_key_value_heads"] == 2
    assert text_cfg["head_dim"] == 256

    # --- read real weight shapes from the single-file safetensors header ------------------
    hdr = read_safetensors_header(snap / "model-00001-of-00001.safetensors")
    base_prefix = "model.language_model.layers"
    for i in full_attn:
        for module, (out_f, in_f) in EXPECTED_SHAPES.items():
            key = f"{base_prefix}.{i}.{module}.weight"
            shape = tuple(hdr[key]["shape"])
            assert shape == (out_f, in_f), f"{key}: {shape} != {(out_f, in_f)}"
        # sanity: q_proj really is gated/interleaved (2 * q_heads * head_dim rows)
        q_shape = tuple(hdr[f"{base_prefix}.{i}.self_attn.q_proj.weight"]["shape"])
        assert q_shape == (4096, 1024), f"q_proj shape {q_shape} (attn_output_gate expectation)"

    # --- build A/B tensors ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    r = args.rank
    key_prefix = (
        "base_model.model.model.language_model.layers"
        if args.key_style == "vlm"
        else "base_model.model.model.layers"
    )
    tensors: dict[str, torch.Tensor] = {}
    for i in full_attn:
        for module, (out_f, in_f) in EXPECTED_SHAPES.items():
            stem = f"{key_prefix}.{i}.{module}"
            a = torch.randn(r, in_f, dtype=torch.float32) * args.init_std
            b = torch.randn(out_f, r, dtype=torch.float32) * args.init_std  # NONZERO B
            tensors[f"{stem}.lora_A.weight"] = a.to(torch.bfloat16)
            tensors[f"{stem}.lora_B.weight"] = b.to(torch.bfloat16)

    expected_n = len(full_attn) * len(EXPECTED_SHAPES) * 2
    assert len(tensors) == expected_n, f"{len(tensors)} != {expected_n}"

    # --- write adapter dir ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

    from peft import LoraConfig

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        use_rslora=args.use_rslora,
        target_modules=sorted(ATTN_TARGETS + MLP_TARGETS),
        layers_to_transform=full_attn,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=BASE_MODEL_ID,
        inference_mode=True,
    )
    lora_cfg.save_pretrained(str(args.out_dir))  # writes adapter_config.json (peft_type=LORA)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors ({total / 1024:.1f} KiB BF16) -> {args.out_dir}")
    print(f"  layers={full_attn} r={r} alpha={args.alpha} use_rslora={args.use_rslora} "
          f"key_style={args.key_style}")
    for k in sorted(tensors)[:4]:
        print(f"  {k} {tuple(tensors[k].shape)}")
    return None


if __name__ == "__main__":
    sys.exit(main())
