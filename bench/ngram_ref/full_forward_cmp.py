"""Full-model LongCat validation against the committed golden.

The 1-layer slice deliberately CANNOT test the n-gram embedding: it bakes the
golden fused embedding into a synthetic `embed_tokens` and drops the n-gram
config keys, so a plain gather reproduces the golden input. That isolates the
backbone — and leaves the n-gram path untested until the real checkpoint runs.

This script closes that gap on the full model:

  gold['input']       is the FUSED n-gram embedding -> engine sublayer 0's input
  gold['layerK_out']  is checkpoint layer K's output

Each CHECKPOINT layer is TWO engine sublayers, so checkpoint layer K's output
is the input of engine sublayer 2K+2:

  gold['input']      <-> atlas_op_L0_input_norm_in
  gold['layer0_out'] <-> atlas_op_L2_input_norm_in
  gold['layerK_out'] <-> atlas_op_L{2K+2}_input_norm_in

A high cosine on the FIRST row (the embedding) is the n-gram wiring working;
the later rows say whether that stays true through all 28 sublayers.

Usage: full_forward_cmp.py <port> [dump_dir]
"""
import json
import os
import sys

import numpy as np
import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else '8897'
D = sys.argv[2] if len(sys.argv) > 2 else '/home/ms/.claude/jobs/5a7bd33d/tmp/opdump_full'
HERE = os.path.dirname(os.path.abspath(__file__))
gold = np.load(f'{HERE}/longcat_forward_golden.npz')
tokens = [int(t) for t in gold['tokens']]

os.makedirs(D, exist_ok=True)
for f in os.listdir(D):
    if f.endswith('.bin'):
        os.remove(os.path.join(D, f))

r = requests.post(f'http://127.0.0.1:{PORT}/v1/completions',
                  json={'model': 'longcat-full', 'prompt': tokens,
                        'max_tokens': 1, 'temperature': 0}, timeout=1800).json()
if 'choices' in r:
    print('served text:', repr(r['choices'][0]['text']))
else:
    print('server error:', json.dumps(r)[:400])

n_ckpt = 14
pairs = [('input', 0)]
for k in range(n_ckpt - 1):
    pairs.append((f'layer{k}_out', 2 * k + 2))

print(f'\n{"stage":16s} {"sublayer":>9s} {"|ref|":>9s} {"|atlas|":>9s} {"cos":>8s} {"relerr":>8s}')
worst = (1.0, '')
for key, sub in pairs:
    path = f'{D}/atlas_op_L{sub}_input_norm_in.bin'
    if not os.path.exists(path):
        print(f'{key:16s} {sub:9d}   MISSING  (dump layer {sub} not enabled?)')
        continue
    a = np.fromfile(path, dtype=np.float32)
    ref = gold[key][-1].astype(np.float32)     # dumps are the LAST token
    n = min(len(a), len(ref))
    a, ref = a[:n], ref[:n]
    cos = float(ref @ a / (np.linalg.norm(ref) * np.linalg.norm(a) + 1e-9))
    rel = float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-9))
    print(f'{key:16s} {sub:9d} {np.linalg.norm(ref):9.3f} {np.linalg.norm(a):9.3f} '
          f'{cos:8.4f} {rel:8.4f}')
    if cos < worst[0]:
        worst = (cos, key)

print(f'\nworst stage: {worst[1]} at cos {worst[0]:.4f}')
