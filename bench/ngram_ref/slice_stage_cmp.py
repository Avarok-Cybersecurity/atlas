"""Stage-by-stage compare, reading the op dumps as FP32.

`dump_bf16` reads `n_elements` BF16 values off the device and writes them
WIDENED to f32, so one record is `n_elements` f32 — the whole last-token
hidden row (12288 bytes at hidden=3072), not a half of one.
"""
import json
import os

import numpy as np
import requests

D = '/home/ms/.claude/jobs/5a7bd33d/tmp/opdump'
ref = np.load('/home/ms/.claude/jobs/5a7bd33d/tmp/stage_ref.npz')
exp = json.load(open('/home/ms/.claude/jobs/5a7bd33d/tmp/slice_expected.json'))

r = requests.post('http://127.0.0.1:8896/v1/completions',
                  json={'model': 'longcat-slice', 'prompt': exp['tokens'],
                        'max_tokens': 1, 'temperature': 0}, timeout=600).json()
print('served text:', repr(r['choices'][0]['text']))

# The engine's "moe_out" is the DENSE FFN delta; the shortcut MoE is dumped
# separately as "shortcut_moe_out". Pair each against the stage of the same
# meaning — conflating them compares different tensors and reads as a bug.
pairs = [('sub0_input_norm_in', 'atlas_op_L0_input_norm_in.bin'),
         ('sub0_input_norm_out', 'atlas_op_L0_input_norm_out.bin'),
         ('sub0_post_attn_norm_out', 'atlas_op_L0_post_attn_norm_out.bin'),
         ('sub0_shortcut_moe', 'atlas_op_L0_shortcut_moe_out.bin'),
         ('sub0_dense_ffn', 'atlas_op_L0_moe_out.bin'),
         ('sub1_input_norm_in', 'atlas_op_L1_input_norm_in.bin'),
         ('sub1_input_norm_out', 'atlas_op_L1_input_norm_out.bin'),
         ('sub1_post_attn_norm_out', 'atlas_op_L1_post_attn_norm_out.bin'),
         ('sub1_dense_ffn', 'atlas_op_L1_moe_out.bin')]

print(f'{"stage":26s} {"|ref|":>9s} {"|atlas|":>9s} {"cos":>8s} {"relerr":>8s}')
for rk, fn in pairs:
    path = f'{D}/{fn}'
    if not os.path.exists(path):
        print(f'{rk:26s} {"MISSING":>9s}  {fn}')
        continue
    # op_dump widens BF16 -> f32, so a record is `hidden` f32 (12288 bytes at
    # hidden=3072) — the WHOLE last-token row, not a half of one.
    raw = np.fromfile(path, dtype=np.float32)
    n = min(len(raw), len(ref[rk]))
    a = raw[:n]
    rv = ref[rk][:n]
    cos = float(rv @ a / (np.linalg.norm(rv) * np.linalg.norm(a) + 1e-9))
    rel = float(np.linalg.norm(a - rv) / (np.linalg.norm(rv) + 1e-9))
    print(f'{rk:26s} {np.linalg.norm(rv):9.3f} {np.linalg.norm(a):9.3f} '
          f'{cos:8.4f} {rel:8.4f}')
