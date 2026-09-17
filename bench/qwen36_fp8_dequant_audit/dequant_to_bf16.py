#!/usr/bin/env python3
"""Dequant Qwen3.6-35B-A3B-FP8 to plain BF16 via canonical PyTorch ops.

Phase 2a of the Atlas FP8 dequant audit. Produces a snapshot that HF
transformers can load directly (no quantization_config), enabling the
three-way cosine comparison:

    A: HF[FP8->BF16]  vs  HF[unquantized BF16]    -> inherent FP8 quant loss
    B: Atlas[FP8]     vs  HF[unquantized BF16]    -> existing Phase alpha
    C: Atlas[FP8]     vs  HF[FP8->BF16]           -> Atlas compute fidelity

Dequant per Qwen FP8 spec (block 128x128, weight_scale_inv stored as BF16):
    f32 = fp8.to(float32) * scale_inv_upsampled(N, K)
    bf16 = f32.to(bfloat16)             # PyTorch round-to-nearest-even

Reuse before write: torch / safetensors directly; compressed_tensors v0.13
does not expose a top-level FP8 dequant helper, only its sparse paths.
The dequant math is the same canonical formula that Atlas's
quant_helpers.rs uses, with PyTorch's RNE in place of Atlas's truncation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Canonical FP8 block size for Qwen3.6 (confirmed from config.json
# quantization_config.weight_block_size). See plan file.
BLOCK_N = 128
BLOCK_K = 128


def dequant_one_tensor(
    w_fp8: torch.Tensor, scale_inv_bf16: torch.Tensor
) -> torch.Tensor:
    """Block-scaled FP8 -> BF16 dequant with round-to-nearest-even.

    PyTorch's float8_e4m3fn -> float32 path is a lookup (256 entries) and
    is exact. Multiply by the upsampled BF16 scale, then cast to BF16,
    which uses RNE per IEEE-754. This is the canonical reference Atlas's
    compute path must match.
    """
    assert w_fp8.dtype == torch.float8_e4m3fn, f"expect fp8_e4m3fn, got {w_fp8.dtype}"
    assert scale_inv_bf16.dtype == torch.bfloat16, (
        f"expect bf16 scale, got {scale_inv_bf16.dtype}"
    )
    N, K = w_fp8.shape
    nb, kb = scale_inv_bf16.shape
    assert nb == (N + BLOCK_N - 1) // BLOCK_N, f"row blocks mismatch: {nb} vs {N}/{BLOCK_N}"
    assert kb == (K + BLOCK_K - 1) // BLOCK_K, f"col blocks mismatch: {kb} vs {K}/{BLOCK_K}"

    w_f32 = w_fp8.to(torch.float32)
    s_f32 = scale_inv_bf16.to(torch.float32)
    # Upsample [nb, kb] -> [N, K] via repeat_interleave; cheap CPU op
    s_full = s_f32.repeat_interleave(BLOCK_N, dim=0)[:N, :].repeat_interleave(
        BLOCK_K, dim=1
    )[:, :K]
    return (w_f32 * s_full).to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/61a5771f218894aaacf97551e24a25b866750fc2",
        help="FP8 snapshot directory",
    )
    ap.add_argument(
        "--dst",
        default="/workspace/.cache/huggingface/Qwen3.6-35B-A3B-FP8-dequanted-BF16",
        help="Output BF16 snapshot directory",
    )
    ap.add_argument("--ref-config", default="/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json", help="BF16 reference config (without quantization_config)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # --- non-weight files: copy tokenizer, generation config etc.
    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "configuration.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]:
        f = src / fname
        if f.exists():
            shutil.copy(f, dst / fname)
            print(f"copied {fname}")

    # --- config.json: use BF16 reference's (no quantization_config)
    shutil.copy(args.ref_config, dst / "config.json")
    print(f"copied BF16 config.json from {args.ref_config}")

    # --- index.json: load FP8 index, strip *.weight_scale_inv, write new
    with open(src / "model.safetensors.index.json") as f:
        idx = json.load(f)
    new_weight_map: dict[str, str] = {}
    for k, v in idx["weight_map"].items():
        if k.endswith(".weight_scale_inv"):
            continue
        new_weight_map[k] = v
    new_idx = {"metadata": idx.get("metadata", {}), "weight_map": new_weight_map}
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(new_idx, f, indent=2)
    print(
        f"index: {len(idx['weight_map'])} keys -> {len(new_weight_map)} keys "
        f"(stripped {len(idx['weight_map']) - len(new_weight_map)} scale entries)"
    )

    # --- safetensor shards: process one at a time
    shards = sorted(src.glob("*.safetensors"))
    print(f"processing {len(shards)} shards")
    total_dequant = 0
    total_passthru = 0
    for shard_i, shard in enumerate(shards):
        print(f"\n[{shard_i + 1}/{len(shards)}] {shard.name}")
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt") as f:
            keys = list(f.keys())
            # Build set of scale keys for fast lookup
            scale_keys = {k for k in keys if k.endswith(".weight_scale_inv")}
            for k in keys:
                if k in scale_keys:
                    # Don't emit scales in output (they're absorbed)
                    continue
                t = f.get_tensor(k)
                scale_name = k + "_scale_inv" if k.endswith(".weight") else None
                if scale_name and scale_name in scale_keys and t.dtype == torch.float8_e4m3fn:
                    s = f.get_tensor(scale_name)
                    out_tensors[k] = dequant_one_tensor(t, s)
                    total_dequant += 1
                else:
                    # Pass through unchanged (already bf16 norms, biases, etc.)
                    out_tensors[k] = t
                    total_passthru += 1
        # Save the shard
        out_path = dst / shard.name
        save_file(out_tensors, str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(
            f"  wrote {len(out_tensors)} tensors, {size_mb:.1f} MB "
            f"(dequant cumulative: {total_dequant}, passthru: {total_passthru})"
        )

    print(
        f"\nDONE: {total_dequant} tensors dequanted, "
        f"{total_passthru} passed through. Output: {dst}"
    )


if __name__ == "__main__":
    main()
