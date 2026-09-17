# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Inspect a llama.cpp GGUF control vector (`general.architecture=controlvector`).

Reproduces the numbers quoted in docs/design/qwen4exp-control-vectors.md.

    uv run .research/inspect_control_vector.py <file.gguf> [--range A B]

Prints the metadata, per-layer norms, and the cross-layer cosine structure that
says whether the directions are one coherent feature across depth or 40+
independent per-layer fits. Optionally compares two vectors layer-by-layer
(`--against other.gguf`), which is the cross-quantization check: a direction
derived from an NVFP4 serve should have high cosine with one derived from a
GGUF serve if it is a property of the model rather than of the quantization.
"""
import argparse
import struct

import numpy as np

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
       10: "<Q", 11: "<q", 12: "<d"}


def load(path):
    """Return (metadata dict, directions [n_layer, hidden] f32; row 0 is zeros)."""
    d = open(path, "rb").read()
    if d[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    ntensor, = struct.unpack_from("<Q", d, 8)
    nkv, = struct.unpack_from("<Q", d, 16)
    off = 24

    def rd_str(o):
        n, = struct.unpack_from("<Q", d, o)
        return d[o + 8:o + 8 + n].decode(), o + 8 + n

    def rd_val(o, t):
        if t == 8:
            return rd_str(o)
        if t == 9:  # array
            et, = struct.unpack_from("<I", d, o)
            n, = struct.unpack_from("<Q", d, o + 4)
            o2 = o + 12
            for _ in range(n):
                _, o2 = rd_val(o2, et)
            return f"<array[{et}] x{n}>", o2
        v, = struct.unpack_from(FMT[t], d, o)
        return v, o + SZ[t]

    meta = {}
    for _ in range(nkv):
        k, off = rd_str(off)
        t, = struct.unpack_from("<I", d, off)
        off += 4
        meta[k], off = rd_val(off, t)

    tensors = []
    for _ in range(ntensor):
        nm, off = rd_str(off)
        nd, = struct.unpack_from("<I", d, off)
        off += 4
        dims = struct.unpack_from("<" + "Q" * nd, d, off)
        off += 8 * nd
        dt, = struct.unpack_from("<I", d, off)
        off += 4
        o, = struct.unpack_from("<Q", d, off)
        off += 8
        if dt != 0:
            raise SystemExit(f"{nm}: expected F32 (ggml type 0), got {dt}")
        if not nm.startswith("direction."):
            raise SystemExit(f"{nm}: unexpected tensor name")
        tensors.append((int(nm.split(".")[1]), dims[0], o))

    base = (off + 31) // 32 * 32  # gguf default alignment
    hidden = tensors[0][1]
    out = np.zeros((max(t[0] for t in tensors) + 1, hidden), dtype=np.float32)
    for il, n, o in tensors:
        if n != hidden:
            raise SystemExit(f"direction.{il}: dim {n} != {hidden}")
        out[il] = np.frombuffer(d, dtype=np.float32, count=n, offset=base + o)
    return meta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--range", nargs=2, type=int, metavar=("A", "B"), default=[4, 44],
                    help="active layer range, inclusive-exclusive of B as llama.cpp uses it")
    ap.add_argument("--against", help="a second control vector to compare layer-by-layer")
    args = ap.parse_args()

    meta, v = load(args.gguf)
    n_layer, hidden = v.shape
    print(f"{args.gguf}")
    for k, val in meta.items():
        print(f"  {k} = {val}")
    print(f"  directions: [{n_layer}, {hidden}]  (row 0 unused)")

    norms = np.linalg.norm(v[1:], axis=1)
    unit = np.allclose(norms, 1.0, atol=1e-5)
    print(f"  |v| min={norms.min():.6f} max={norms.max():.6f}  unit-normalized={unit}")

    a, b = args.range
    act = v[a:b]
    act = act / np.linalg.norm(act, axis=1, keepdims=True)
    cos = act @ act.T
    n = cos.shape[0]
    print(f"  layers {a}..{b - 1}: adjacent cos mean={np.mean(np.diag(cos, 1)):.4f}  "
          f"all-pairs off-diagonal mean={np.mean(cos[np.triu_indices(n, 1)]):.4f}")

    if args.against:
        _, w = load(args.against)
        if w.shape != v.shape:
            raise SystemExit(f"shape mismatch: {v.shape} vs {w.shape}")
        vn = v[a:b] / np.linalg.norm(v[a:b], axis=1, keepdims=True)
        wn = w[a:b] / np.linalg.norm(w[a:b], axis=1, keepdims=True)
        per = np.sum(vn * wn, axis=1)
        print(f"\n  vs {args.against}: per-layer cos min={per.min():.4f} "
              f"mean={per.mean():.4f} max={per.max():.4f}")
        for i, c in enumerate(per, start=a):
            print(f"    layer {i:2d}  cos={c:+.4f}")


if __name__ == "__main__":
    main()
