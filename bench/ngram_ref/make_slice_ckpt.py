#!/usr/bin/env python3
"""Synthesize a SLICED LongCat validation checkpoint for backbone parity.

Validating the backbone port does not need the 129 GB model: the committed
golden (`longcat_forward_golden.npz`) records every layer's output, so ONE
real checkpoint layer is enough to test every new code path — MLA with the
rope de-interleave + LoRA-scale folds, the dual-sublayer split, the
softmax+bias router, the zero-expert fold and the shortcut-MoE carry.

The trick that avoids touching the engine: the 16 fixture tokens are
DISTINCT, so a synthetic `embed_tokens` whose row `tokens[i]` holds the
golden fused n-gram embedding for position `i` makes an ORDINARY embedding
lookup reproduce the golden input exactly. The n-gram keys are dropped from
the config so the plain embedding path runs.

Emits ~7 GB: embed (805 MB) + lm_head (805 MB) + layer 0 (~5.2 GB).

Usage: make_slice_ckpt.py SNAPSHOT_DIR OUT_DIR [NUM_LAYERS]
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

snap, out = sys.argv[1], sys.argv[2]
n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))

cfg = json.load(open(f'{snap}/config.json'))
idx = json.load(open(f'{snap}/model.safetensors.index.json'))
gold = np.load(f'{here}/longcat_forward_golden.npz')
toks = [int(t) for t in gold['tokens']]
fused = gold['input'].astype(np.float32)  # [16, hidden]
assert len(set(toks)) == len(toks), 'fixture tokens must be distinct'

H = cfg['hidden_size']
V = cfg['vocab_size']
handles = {}


def open_shard(name):
    shard = idx['weight_map'][name]
    if shard not in handles:
        f = open(f'{snap}/{shard}', 'rb')
        n = struct.unpack('<Q', f.read(8))[0]
        handles[shard] = (f, json.loads(f.read(n)), 8 + n)
    return handles[shard]


def raw(name):
    f, hdr, base = open_shard(name)
    info = hdr[name]
    f.seek(base + info['data_offsets'][0])
    return f.read(info['data_offsets'][1] - info['data_offsets'][0]), info


def f32_to_bf16(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


# ── tensors to emit ──
wanted = ['model.norm.weight', 'lm_head.weight']
for layer in range(n_layers):
    pfx = f'model.layers.{layer}.'
    wanted += [n for n in idx['weight_map'] if n.startswith(pfx)]

# Synthetic embedding: zeros except the fixture rows.
embed = np.zeros((V, H), dtype=np.uint16)
embed[toks] = f32_to_bf16(fused)

out_hdr = {}
blobs = []
off = 0


def add(name, data, dtype, shape):
    global off
    out_hdr[name] = {'dtype': dtype, 'shape': shape,
                     'data_offsets': [off, off + len(data)]}
    blobs.append(data)
    off += len(data)


add('model.embed_tokens.weight', embed.tobytes(), 'BF16', [V, H])
for name in wanted:
    data, info = raw(name)
    add(name, data, info['dtype'], info['shape'])

hj = json.dumps(out_hdr).encode()
hj += b' ' * ((8 - len(hj) % 8) % 8)
with open(f'{out}/model.safetensors', 'wb') as fo:
    fo.write(struct.pack('<Q', len(hj)))
    fo.write(hj)
    for b in blobs:
        fo.write(b)

# Config: sliced layer count, NO n-gram keys (plain embedding path).
c = dict(cfg)
c['num_layers'] = n_layers
for k in ('ngram_vocab_size_ratio', 'emb_neighbor_num', 'emb_split_num'):
    c.pop(k, None)
json.dump(c, open(f'{out}/config.json', 'w'), indent=1)
for extra in ('generation_config.json', 'tokenizer.json', 'tokenizer_config.json'):
    if os.path.exists(f'{snap}/{extra}'):
        shutil.copy(f'{snap}/{extra}', f'{out}/{extra}')

total = os.path.getsize(f'{out}/model.safetensors')
print(f'wrote {out}: {total/1e9:.2f} GB, {len(out_hdr)} tensors, '
      f'{n_layers} checkpoint layer(s) -> {2*n_layers} engine sublayers')
print('fixture tokens:', toks)
