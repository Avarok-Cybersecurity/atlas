#!/usr/bin/env python3
"""Emit COMPACTED n-gram test data for the Atlas GPU unit test.

The real tables are ~5.2 GB each; the GPU parity test only needs the rows
the fixture tokens actually touch. For each table we emit just those rows
(in ngram-id order) plus the id list, so the Rust test can remap its own
computed ids to compact row indices. Everything lands in an output dir
OUTSIDE git (~20 MB, dominated by the 12 full projection matrices).

Usage: python3 make_gpu_testdata.py SNAPSHOT_DIR OUT_DIR
"""
import json
import os
import struct
import sys

import numpy as np

from ngram_parity import Cfg, precompute_vocab_mods, ngram_ids_for_table

snap, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

idx = json.load(open(os.path.join(snap, 'model.safetensors.index.json')))
cfgj = json.load(open(os.path.join(snap, 'config.json')))
cfg = Cfg(vocab_size=cfgj['vocab_size'], hidden_size=cfgj['hidden_size'],
          ngram_vocab_size_ratio=cfgj['ngram_vocab_size_ratio'],
          emb_neighbor_num=cfgj['emb_neighbor_num'],
          emb_split_num=cfgj['emb_split_num'],
          eos_token_id=cfgj.get('eos_token_id', 2))

handles = {}


def open_for(name):
    shard = os.path.join(snap, idx['weight_map'][name])
    if shard not in handles:
        f = open(shard, 'rb')
        n = struct.unpack('<Q', f.read(8))[0]
        handles[shard] = (f, json.loads(f.read(n)), 8 + n)
    return handles[shard]


def raw_rows(name, rows):
    """Raw BF16 bytes of the given rows, concatenated in the given order."""
    f, hdr, base = open_for(name)
    info = hdr[name]
    assert info['dtype'] == 'BF16'
    dim = info['shape'][1]
    out = bytearray()
    for r in rows:
        f.seek(base + info['data_offsets'][0] + r * dim * 2)
        out += f.read(dim * 2)
    return bytes(out), dim


def raw_full(name):
    f, hdr, base = open_for(name)
    info = hdr[name]
    f.seek(base + info['data_offsets'][0])
    return f.read(info['data_offsets'][1] - info['data_offsets'][0]), info['shape']


# Same fixed token sequence as the golden dump (seed 7).
rng = np.random.default_rng(7)
toks = rng.integers(3, cfg.vocab_size, 16).astype(np.int64)
mods = precompute_vocab_mods(cfg)

meta = {
    'tokens': [int(t) for t in toks],
    'hidden_size': cfg.hidden_size,
    'vocab_size': cfg.vocab_size,
    'ngram_vocab_size_ratio': cfg.ngram_vocab_size_ratio,
    'emb_neighbor_num': cfg.emb_neighbor_num,
    'emb_split_num': cfg.emb_split_num,
    'eos_token_id': cfg.eos_token_id,
    'tables': [],
}

# Compacted word rows (unique tokens, sorted).
uniq_toks = sorted(set(int(t) for t in toks))
data, dim = raw_rows('model.embed_tokens.weight', uniq_toks)
open(os.path.join(out_dir, 'word_rows.bin'), 'wb').write(data)
meta['word_ids'] = uniq_toks
meta['word_dim'] = dim

num_tables = cfg.emb_split_num * (cfg.emb_neighbor_num - 1)
for i in range(2, cfg.emb_neighbor_num + 1):
    for j in range(cfg.emb_split_num):
        index = (i - 2) * cfg.emb_split_num + j
        ids = ngram_ids_for_table(toks, cfg, i, j, mods)
        uniq = sorted(set(int(v) for v in ids))
        name = f'model.ngram_embeddings.embedders.{index}.weight'
        data, dim = raw_rows(name, uniq)
        open(os.path.join(out_dir, f'table{index}_rows.bin'), 'wb').write(data)
        pdata, pshape = raw_full(
            f'model.ngram_embeddings.post_projs.{index}.weight')
        open(os.path.join(out_dir, f'proj{index}.bin'), 'wb').write(pdata)
        meta['tables'].append({'index': index, 'ids': uniq, 'dim': dim,
                               'proj_shape': pshape})

# Golden fused embeddings, raw f32 (recomputed here so meta+golden stay in
# lockstep even if the npz fixture regenerates separately).
gold = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ngram_golden_longcat_lite.npz'))
assert (gold['tokens'] == toks).all(), 'token draw drifted from the fixture'
gold['fused'].astype(np.float32).tofile(os.path.join(out_dir, 'golden_f32.bin'))
meta['golden_shape'] = list(gold['fused'].shape)

json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'))
total = sum(os.path.getsize(os.path.join(out_dir, f))
            for f in os.listdir(out_dir))
print(f'wrote {out_dir}: {total/1e6:.1f} MB, {num_tables} tables')
