#!/usr/bin/env python3
"""Simulate the GPU path's BF16 accumulation chain over the compacted test
data — separates BF16 rounding noise from logic bugs when the GPU parity
test's elementwise max_rel looks high."""
import json

import numpy as np

from ngram_parity import Cfg, precompute_vocab_mods, ngram_ids_for_table


def to_bf16(x):
    b = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    return (((b + 0x8000) >> 16) << 16).astype(np.uint32).view(np.float32)


d = '/tank/atlas-testdata/ngram_longcat_lite'
meta = json.load(open(f'{d}/meta.json'))
toks = np.array(meta['tokens'])
h = meta['hidden_size']
m = len(toks)
gold = np.fromfile(f'{d}/golden_f32.bin', dtype=np.float32).reshape(m, h)

cfg = Cfg()
mods = precompute_vocab_mods(cfg)


def bf16_rows(path, dim):
    a = np.fromfile(path, dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32).reshape(-1, dim)


word = bf16_rows(f'{d}/word_rows.bin', h)
wmap = {v: i for i, v in enumerate(meta['word_ids'])}
inv = np.float32(1.0 / 13.0)

out = np.zeros((m, h), dtype=np.float32)
base = word[[wmap[int(t)] for t in toks]]
out = to_bf16(out + to_bf16(base * inv))
for t in meta['tables']:
    index = t['index']
    ids = ngram_ids_for_table(toks, cfg, 2 + index // 4, index % 4, mods)
    tmap = {v: i for i, v in enumerate(t['ids'])}
    rows = bf16_rows(f'{d}/table{index}_rows.bin', 256)[
        [tmap[int(v)] for v in ids]]
    proj = bf16_rows(f'{d}/proj{index}.bin', 256)
    contrib = to_bf16(
        (rows.astype(np.float32) @ proj.T.astype(np.float32)) * inv)
    out = to_bf16(out + contrib)

rel = np.abs(out - gold) / np.maximum(np.abs(gold), 1e-3)
fro = np.linalg.norm(out - gold, axis=1) / np.linalg.norm(gold, axis=1)
print('simulated BF16-chain: max_rel(elementwise) =', float(rel.max()))
print('per-row Frobenius rel: max =', float(fro.max()),
      'mean =', float(fro.mean()))
