# SPDX-License-Identifier: AGPL-3.0-only

"""Convert a sharded Hugging Face safetensors checkpoint to BF16 tensors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def shard_names(src: Path) -> list[str]:
    index = src / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        return sorted(set(data["weight_map"].values()))
    single = src / "model.safetensors"
    if single.exists():
        return [single.name]
    return sorted(p.name for p in src.glob("*.safetensors"))


def copy_sidecars(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.suffix == ".safetensors":
            continue
        target = dst / path.name
        if path.is_file():
            shutil.copy2(path, target)


def convert_shard(src_file: Path, dst_file: Path) -> None:
    tensors = load_file(src_file, device="cpu")
    converted = {}
    for name, tensor in tensors.items():
        if tensor.is_floating_point():
            converted[name] = tensor.to(torch.bfloat16)
        else:
            converted[name] = tensor
    tmp = dst_file.with_suffix(dst_file.suffix + ".tmp")
    save_file(converted, tmp, metadata={"format": "pt"})
    tmp.replace(dst_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not (src / "config.json").exists():
        raise SystemExit(f"missing config.json in {src}")

    names = shard_names(src)
    if not names:
        raise SystemExit(f"no safetensors shards found in {src}")
    copy_sidecars(src, dst)
    for i, name in enumerate(names, start=1):
        print(f"[{i}/{len(names)}] {name} -> BF16")
        convert_shard(src / name, dst / name)
    print(f"wrote BF16 checkpoint to {dst}")


if __name__ == "__main__":
    main()
