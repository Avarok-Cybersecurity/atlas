#!/usr/bin/env python3
"""Expected logits for the SLICED (N-layer) LongCat validation checkpoint.

`lm_head(final_norm(layerN-1_out))` computed from the committed golden — the
target the Atlas serve must reproduce when handed the 16 fixture token ids.
Prints the last token's top-8 ids + logits (what the server reports back via
logprobs) and writes them as JSON for the comparison script.

Usage: slice_expected.py SNAPSHOT_DIR [NUM_LAYERS] [OUT_JSON]
"""
import json
import struct
import sys

import numpy as np

snap = sys.argv[1]
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
out_json = sys.argv[3] if len(sys.argv) > 3 else '/tmp/slice_expected.json'

import os
here = os.path.dirname(os.path.abspath(__file__))
gold = np.load(f'{here}/longcat_forward_golden.npz')
cfg = json.load(open(f'{snap}/config.json'))
idx = json.load(open(f'{snap}/model.safetensors.index.json'))
eps = cfg['rms_norm_eps']

handles = {}


def tensor(name):
    shard = idx['weight_map'][name]
    if shard not in handles:
        f = open(f'{snap}/{shard}', 'rb')
        n = struct.unpack('<Q', f.read(8))[0]
        handles[shard] = (f, json.loads(f.read(n)), 8 + n)
    f, hdr, base = handles[shard]
    info = hdr[name]
    f.seek(base + info['data_offsets'][0])
    buf = f.read(info['data_offsets'][1] - info['data_offsets'][0])
    if info['dtype'] == 'BF16':
        a = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
        return (a << 16).view(np.float32).reshape(info['shape'])
    return np.frombuffer(buf, dtype=np.float32).reshape(info['shape']).copy()


h = gold[f'layer{n_layers - 1}_out'].astype(np.float32)
v = np.mean(h.astype(np.float64) ** 2, axis=-1, keepdims=True)
h = (tensor('model.norm.weight') * (h / np.sqrt(v + eps))).astype(np.float32)
logits = h @ tensor('lm_head.weight').T
last = logits[-1]
top = np.argsort(-last)[:8]

report = {
    'num_layers': n_layers,
    'tokens': [int(t) for t in gold['tokens']],
    'top_ids': [int(t) for t in top],
    'top_logits': [float(last[t]) for t in top],
    'argmax': int(top[0]),
    'logit_spread': float(last[top[0]] - last[top[1]]),
}
json.dump(report, open(out_json, 'w'), indent=1)
print(json.dumps(report, indent=1))
