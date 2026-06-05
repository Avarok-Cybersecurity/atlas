#!/usr/bin/env python3
"""Preprocess Step 3.7 Flash NVFP4 checkpoint: split fused expert tensors.

Step 3.7 stores all 288 experts in single fused tensors per projection type:
    model.language_model.layers.3.moe.gate_proj.weight  [288, 1280, 2048]

Atlas EP (Expert Parallelism) filtering works by tensor name — it looks for
`.experts.N.` to decide which expert belongs to which GPU. Fused tensors
bypass this completely.

This script splits fused expert tensors into per-expert named tensors:
    model.language_model.layers.3.moe.experts.0.gate_proj.weight  [1280, 2048]
    model.language_model.layers.3.moe.experts.1.gate_proj.weight  [1280, 2048]
    ...

The output checkpoint is bit-exact with the original — same bytes, just
reorganised into per-expert tensors. Compatible with Atlas's existing EP
filtering logic (parse_expert_index).

Usage:
    python3 preprocess_step3p7_experts.py /path/to/checkpoint /path/to/output

Requirements:
    pip install safetensors
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path


# Safetensors dtype → bytes per element
DTYPE_SIZES = {
    "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

# Patterns for fused MoE expert tensors.
FUSED_MOE_RE = re.compile(
    r"(.+\.moe)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|input_scale|weight_scale_2)$"
)

# Non-expert MoE tensors that should NOT be split
MOE_SKIP_RE = [
    re.compile(r"\.moe\.gate\."),
    re.compile(r"\.moe\.router_bias"),
]


def is_fused_expert_tensor(name: str) -> bool:
    for skip in MOE_SKIP_RE:
        if skip.search(name):
            return False
    return FUSED_MOE_RE.match(name) is not None


def split_tensor_name(name: str, expert_idx: int) -> str:
    m = FUSED_MOE_RE.match(name)
    if not m:
        raise ValueError(f"Cannot split tensor name: {name}")
    prefix, proj, suffix = m.group(1), m.group(2), m.group(3)
    return f"{prefix}.experts.{expert_idx}.{proj}.{suffix}"


def parse_safetensors_header(path: Path):
    """Parse a safetensors file header, return (header_dict, data_offset)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_raw = f.read(header_size)
        data_offset = 8 + header_size
    header = json.loads(header_raw)
    metadata = header.pop("__metadata__", {})
    return header, metadata, data_offset


def get_safetensor_shards(model_dir: Path) -> list[Path]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        seen = set()
        shards = []
        for shard_name in index["weight_map"].values():
            if shard_name not in seen:
                seen.add(shard_name)
                shards.append(model_dir / shard_name)
        return shards
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def tensor_byte_size(shape, dtype_str):
    """Compute total bytes for a tensor given shape and dtype string."""
    numel = 1
    for d in shape:
        numel *= d
    return numel * DTYPE_SIZES.get(dtype_str, 2)


def write_safetensors(path: Path, tensors: list[tuple[str, bytes, list[int], str]]):
    """Write a safetensors file from a list of (name, data_bytes, shape, dtype)."""
    header = {}
    offset = 0
    for name, data, shape, dtype in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)]
        }
        offset += len(data)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad to 8-byte alignment
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for _, data, _, _ in tensors:
            f.write(data)


def process_checkpoint(model_dir: Path, output_dir: Path, max_shard_bytes: int = 10 * 1024**3):
    shards = get_safetensor_shards(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate tensors for output shards
    # Each entry: (name, data_bytes, shape, dtype)
    current_shard: list[tuple[str, bytes, list[int], str]] = []
    current_bytes = 0
    shard_idx = 1
    weight_map = {}
    fused_split = 0
    passed_through = 0
    total_bytes = 0
    total_experts = None

    def flush_shard():
        nonlocal current_shard, current_bytes, shard_idx
        if not current_shard:
            return
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        shard_path = output_dir / shard_name
        print(f"  Writing shard {shard_idx}: {len(current_shard)} tensors, "
              f"{current_bytes / 1024**3:.2f} GB")
        write_safetensors(shard_path, current_shard)
        for name, _, _, _ in current_shard:
            weight_map[name] = shard_name
        current_shard = []
        current_bytes = 0
        shard_idx += 1

    for shard_path in shards:
        print(f"\nProcessing: {shard_path.name}")
        header, metadata, data_offset = parse_safetensors_header(shard_path)

        with open(shard_path, "rb") as f:
            for name, info in sorted(header.items()):
                dtype = info["dtype"]
                shape = info["shape"]
                start, end = info["data_offsets"]
                nbytes = end - start

                f.seek(data_offset + start)
                data = f.read(nbytes)

                if is_fused_expert_tensor(name) and len(shape) >= 1 and shape[0] > 1:
                    # Split along dimension 0 (expert dimension)
                    num_experts = shape[0]
                    if total_experts is None:
                        total_experts = num_experts
                        print(f"  Detected {num_experts} experts")

                    expert_shape = shape[1:]  # remove expert dim
                    bytes_per_expert = nbytes // num_experts

                    for e in range(num_experts):
                        expert_name = split_tensor_name(name, e)
                        expert_data = data[e * bytes_per_expert : (e + 1) * bytes_per_expert]
                        current_shard.append((expert_name, expert_data, expert_shape, dtype))
                        current_bytes += bytes_per_expert
                        total_bytes += bytes_per_expert

                        if current_bytes >= max_shard_bytes:
                            flush_shard()

                    fused_split += 1
                    if fused_split % 20 == 0:
                        print(f"    Split {fused_split} fused tensors...")
                else:
                    # Pass through unchanged
                    current_shard.append((name, data, shape, dtype))
                    current_bytes += nbytes
                    total_bytes += nbytes
                    passed_through += 1

                    if current_bytes >= max_shard_bytes:
                        flush_shard()

    flush_shard()
    total_shards = shard_idx - 1

    # Fix shard filenames
    final_weight_map = {}
    for name, shard_name in weight_map.items():
        final_name = shard_name.replace("PLACEHOLDER", f"{total_shards:05d}")
        final_weight_map[name] = final_name

    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)

    # Write index
    index = {
        "metadata": {
            "total_size": total_bytes,
            "format": "step3p7-nvfp4-split-experts",
            "original_format": "step3p7-nvfp4-fused-experts",
            "num_experts": total_experts or 0,
        },
        "weight_map": final_weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)

    # Copy config files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "generation_config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")

    print(f"\n{'='*60}")
    print(f"Done! {total_shards} shards written to {output_dir}")
    print(f"  Split {fused_split} fused tensors → "
          f"{fused_split * (total_experts or 0)} per-expert tensors")
    print(f"  Passed through {passed_through} non-fused tensors unchanged")
    print(f"  Total: {len(final_weight_map)} tensors in index")
    print(f"{'='*60}")


def dry_run(model_dir: Path):
    shards = get_safetensor_shards(model_dir)
    fused_count = 0
    fused_bytes = 0
    pass_count = 0
    pass_bytes = 0
    total_experts = None

    for shard_path in shards:
        header, _, _ = parse_safetensors_header(shard_path)
        for name, info in header.items():
            shape = info["shape"]
            dtype = info["dtype"]
            nbytes = info["data_offsets"][1] - info["data_offsets"][0]
            if is_fused_expert_tensor(name) and len(shape) >= 1 and shape[0] > 1:
                fused_count += 1
                fused_bytes += nbytes
                if total_experts is None:
                    total_experts = shape[0]
            else:
                pass_count += 1
                pass_bytes += nbytes

    total_experts = total_experts or 0
    print(f"Fused expert tensors: {fused_count}")
    print(f"Pass-through tensors: {pass_count}")
    print(f"Total experts: {total_experts}")
    print(f"Output tensors: {pass_count + fused_count * total_experts}")
    print(f"Fused data: {fused_bytes / 1024**3:.2f} GB")
    print(f"Pass-through data: {pass_bytes / 1024**3:.2f} GB")
    print(f"Total: {(fused_bytes + pass_bytes) / 1024**3:.2f} GB (same after split)")


def main():
    parser = argparse.ArgumentParser(
        description="Split Step 3.7 Flash NVFP4 fused expert tensors for Atlas EP support"
    )
    parser.add_argument("input", type=Path, help="Input model directory (HF checkpoint)")
    parser.add_argument("output", type=Path, help="Output directory for split checkpoint")
    parser.add_argument("--max-shard-gb", type=float, default=10.0,
                        help="Maximum shard size in GB (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only estimate sizes, don't write")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        dry_run(args.input)
        return

    max_shard_bytes = int(args.max_shard_gb * 1024**3)
    process_checkpoint(args.input, args.output, max_shard_bytes)


if __name__ == "__main__":
    main()
