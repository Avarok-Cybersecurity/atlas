#!/usr/bin/env python3
"""Fetch one EXL3 linear's tensors (trellis/suh/svh/mul1) from the HF
checkpoint via HTTP range requests, using a locally saved safetensors
header for the byte offsets. Writes raw .bin files + meta.txt for the
exl3_reconstruct_parity example's EXL3_REAL_DIR leg.

Usage: fetch_exl3_tensor.py <header.json> <shard_url> <tensor_prefix> <out_dir>
"""
import json
import struct
import subprocess
import sys


def main() -> None:
    header_path, url, prefix, out_dir = sys.argv[1:5]
    h = json.load(open(header_path))
    # data section starts after the 8-byte length + header JSON. The header
    # length is the ORIGINAL on-disk length; recover it from the max end
    # offset alignment instead of re-reading the shard: safetensors data
    # offsets are relative to the data section, so we need the absolute
    # base = 8 + original_header_len. Take it from the file we saved: the
    # saved JSON length equals the on-disk header length (we saved the
    # exact byte range).
    import os

    header_len = os.path.getsize(header_path)
    base = 8 + header_len

    metas = {}
    for suffix in ["trellis", "suh", "svh", "mul1"]:
        name = f"{prefix}.{suffix}"
        t = h[name]
        start, end = t["data_offsets"]
        out = f"{out_dir}/{suffix}.bin"
        subprocess.run(
            [
                "curl", "-sL", "-r", f"{base + start}-{base + end - 1}",
                url, "-o", out,
            ],
            check=True,
        )
        got = os.path.getsize(out)
        want = end - start
        assert got == want, f"{name}: got {got} bytes, want {want}"
        metas[suffix] = t
        print(f"{name}: {t['dtype']} {t['shape']} -> {out} ({got} bytes)")

    tr = metas["trellis"]["shape"]  # [in/16, out/16, 16*K]
    in_dim = tr[0] * 16
    out_dim = tr[1] * 16
    k = tr[2] // 16
    mul1 = struct.unpack("<i", open(f"{out_dir}/mul1.bin", "rb").read(4))[0]
    # cb: mul1 flag set -> 2, else mcg... the checkpoint carries an explicit
    # flag per tensor; nonzero means the mul1 codebook.
    cb = 2 if mul1 else 1
    with open(f"{out_dir}/meta.txt", "w") as f:
        f.write(f"in={in_dim}\nout={out_dim}\nk={k}\ncb={cb}\n")
    print(f"meta: in={in_dim} out={out_dim} K={k} mul1_flag={mul1} -> cb={cb}")


if __name__ == "__main__":
    main()
