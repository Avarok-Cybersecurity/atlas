#!/usr/bin/env python3
"""Decode the fetched REAL ngram rows with the same math as the Rust CPU
reference (independent third implementation) and print statistics — a
sanity check that the row framing (offset, 61-word stride, scale word,
ring order) matches the real checkpoint."""
import struct

ROW_DIM, K, MUL1 = 160, 6, 0x83DCD12D
WORDS = 1 + ROW_DIM * K // 16


def f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def decode_mul1(state: int) -> float:
    x = (state * MUL1) & 0xFFFFFFFF
    s = 0x6400 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    # f16 window value * k_inv + k_bias, single f16 rounding (approximated in
    # f64 here; sanity-stats only)
    return f16(s) * f16(0x1EEE) + f16(0xC931)


rows = open("ngram_rows.bin", "rb").read()
bias = open("ngram_head_bias.bin", "rb").read()
n_rows = len(rows) // (WORDS * 2)
all_vals = []
for r in range(n_rows):
    w = struct.unpack(f"<{WORDS}H", rows[r * WORDS * 2 : (r + 1) * WORDS * 2])
    scale = f16(w[0])
    stream = w[1:]
    vals = []
    for i in range(ROW_DIM):
        state = 0
        for m in range(16):
            pos = (i - m // K) % ROW_DIM
            bit = pos * K + m % K
            state |= ((stream[bit // 16] >> (bit % 16)) & 1) << m
        vals.append(decode_mul1(state) * scale)
    all_vals.extend(vals)
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / len(vals)
    print(f"row {r}: scale={scale:.6f} mean={mean:+.6f} min={mn:+.5f} max={mx:+.5f}")

import math

m = sum(all_vals) / len(all_vals)
var = sum((v - m) ** 2 for v in all_vals) / len(all_vals)
finite = all(math.isfinite(v) for v in all_vals)
print(f"ALL: mean={m:+.6f} std={math.sqrt(var):.6f} finite={finite}")
bvals = [f16(struct.unpack('<H', bias[i*2:i*2+2])[0]) for i in range(len(bias)//2)]
print(f"head_bias: mean={sum(bvals)/len(bvals):+.6f} absmax={max(abs(v) for v in bvals):.5f}")
