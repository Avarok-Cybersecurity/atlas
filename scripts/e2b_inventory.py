#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Gemma-4 E2B checkpoint key/shape inventory.

Reads the safetensors header of google/gemma-4-E2B-it and prints a per-layer
contract table (attention/MLP/PLE shapes, clipped-linear sub-keys, modality
tower keys). This is the CI-safe contract the weight loader is built against;
it also documents the non-obvious fact that k/v/k_norm weights exist in ALL 35
layers (the HF runtime ignores layers 15-34's own K/V via num_kv_shared_layers).

Usage:
    python3 scripts/e2b_inventory.py /path/to/model.safetensors [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys


def load_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors")
    ap.add_argument("--json", default=None, help="Also write a machine-readable table")
    args = ap.parse_args()

    header = load_header(args.safetensors)
    keys = set(header.keys())

    rows = []
    for i in range(35):
        p = f"model.language_model.layers.{i}"
        row = {
            "layer": i,
            "k_proj": f"model.language_model.layers.{i}.self_attn.k_proj.weight" in keys,
            "v_proj": f"model.language_model.layers.{i}.self_attn.v_proj.weight" in keys,
            "k_norm": f"model.language_model.layers.{i}.self_attn.k_norm.weight" in keys,
            "gate_proj": header.get(f"{p}.mlp.gate_proj.weight", {}).get("shape"),
            "per_layer_input_gate": header.get(f"{p}.per_layer_input_gate.weight", {}).get("shape"),
            "per_layer_projection": header.get(f"{p}.per_layer_projection.weight", {}).get("shape"),
        }
        rows.append(row)

    n_kv = sum(1 for r in rows if r["k_proj"])
    n_gate_double = sum(1 for r in rows if r["gate_proj"] and r["gate_proj"][0] == 12288)

    summary = {
        "total_k_proj": n_kv,
        "double_wide_mlp_layers": n_gate_double,
        "embed_tokens": header.get("model.language_model.embed_tokens.weight", {}).get("shape"),
        "embed_tokens_per_layer": header.get("model.language_model.embed_tokens_per_layer.weight", {}).get("shape"),
        "per_layer_model_projection": header.get("model.language_model.per_layer_model_projection.weight", {}).get("shape"),
        "per_layer_projection_norm": header.get("model.language_model.per_layer_projection_norm.weight", {}).get("shape"),
        "vision_input_proj": header.get("model.vision_tower.patch_embedder.input_proj.weight", {}).get("shape"),
        "vision_pos_embed": header.get("model.vision_tower.patch_embedder.position_embedding_table", {}).get("shape"),
        "vision_layers": sum(1 for k in keys if "model.vision_tower.encoder.layers." in k and ".self_attn.q_proj.linear.weight" in k),
        "audio_layers": sum(1 for k in keys if "model.audio_tower.layers." in k and ".self_attn.q_proj.linear.weight" in k),
        "audio_output_proj": header.get("model.audio_tower.output_proj.weight", {}).get("shape"),
        "embed_vision_proj": header.get("model.embed_vision.embedding_projection.weight", {}).get("shape"),
        "embed_audio_proj": header.get("model.embed_audio.embedding_projection.weight", {}).get("shape"),
        "n_clipped_linear_subsets": sum(1 for k in keys if k.endswith(".input_min")),
    }

    print(json.dumps(summary, indent=2))
    for r in rows[:3]:
        print(r)
    print("...")
    for r in rows[13:16]:
        print(r)
    print("...")
    for r in rows[33:]:
        print(r)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "layers": rows}, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
