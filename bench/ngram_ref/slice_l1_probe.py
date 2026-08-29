"""Single-token MLA isolation.

At sequence length 1 the attention softmax is over ONE key, so it is exactly
1.0 regardless of q/k values: the attention output reduces to `v0 @ o_proj`.
That makes the result INDEPENDENT of rope and of the q-side LoRA scale, and
dependent ONLY on the kv path (kv_a_proj -> kv_a_layernorm(+kv scale) ->
kv_b_proj -> v) and o_proj.

  match at L=1  => kv/v/o path is right, the bug is rope or q-side scaling
  mismatch      => the bug is in the kv path (e.g. the kv_a_layernorm fold)
"""
import glob
import json
import os
import struct
import sys

import numpy as np
import requests

SNAP = ('/tank/hf/hub/models--meituan-longcat--LongCat-Flash-Lite/snapshots/'
        'b62b68827ead0b7fef3ba98b57f18484acaaec06')
D = '/home/ms/.claude/jobs/5a7bd33d/tmp/opdump'
cfg = json.load(open(f'{SNAP}/config.json'))
idx = json.load(open(f'{SNAP}/model.safetensors.index.json'))
gold = np.load('/home/ms/atlas/.claude/worktrees/nemo-behavior/bench/ngram_ref/'
               'longcat_forward_golden.npz')

H, NH = cfg['hidden_size'], cfg['num_attention_heads']
QK_NOPE, QK_ROPE = cfg['qk_nope_head_dim'], cfg['qk_rope_head_dim']
V_HEAD, KV_LORA = cfg['v_head_dim'], cfg['kv_lora_rank']
EPS = cfg['rms_norm_eps']
handles = {}


def tensor(name):
    shard = idx['weight_map'][name]
    if shard not in handles:
        f = open(f'{SNAP}/{shard}', 'rb')
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


def rms(x, w):
    v = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (w * (x / np.sqrt(v + EPS))).astype(np.float32)


lp, ap = 'model.layers.0', 'model.layers.0.self_attn.0'
h = gold['input'].astype(np.float32)[:1]           # first token only
x = rms(h, tensor(f'{lp}.input_layernorm.0.weight'))

# kv path only (softmax over one key == 1.0)
ckv = x @ tensor(f'{ap}.kv_a_proj_with_mqa.weight').T
k_pass = rms(ckv[:, :KV_LORA], tensor(f'{ap}.kv_a_layernorm.weight'))
sk = (H / KV_LORA) ** 0.5 if cfg.get('mla_scale_kv_lora') else 1.0
kv = (k_pass * sk) @ tensor(f'{ap}.kv_b_proj.weight').T
kv = kv.reshape(1, NH, QK_NOPE + V_HEAD)
v = kv[..., QK_NOPE:].reshape(1, NH * V_HEAD)
attn_out = v @ tensor(f'{ap}.o_proj.weight').T
hidden_after = h + attn_out
post = rms(hidden_after, tensor(f'{lp}.post_attention_layernorm.0.weight'))

for f in glob.glob(f'{D}/*.bin'):
    os.remove(f)
tok = int(gold['tokens'][0])
requests.post('http://127.0.0.1:8895/v1/completions',
              json={'model': 'longcat-slice', 'prompt': [tok],
                    'max_tokens': 1, 'temperature': 0}, timeout=600)

n = 1536
for label, refv, fn in [
    ('input_norm_in', h[0], 'atlas_op_L0_input_norm_in.bin'),
    ('input_norm_out', x[0], 'atlas_op_L0_input_norm_out.bin'),
    ('post_attn_norm_out', post[0], 'atlas_op_L0_post_attn_norm_out.bin'),
]:
    a = np.fromfile(f'{D}/{fn}', dtype=np.float32)[:n]
    rv = refv[:n]
    cos = float(rv @ a / (np.linalg.norm(rv) * np.linalg.norm(a) + 1e-9))
    rel = float(np.linalg.norm(a - rv) / (np.linalg.norm(rv) + 1e-9))
    print(f'{label:22s} |ref|={np.linalg.norm(rv):8.3f} |atlas|={np.linalg.norm(a):8.3f} '
          f'cos={cos:7.4f} relerr={rel:7.4f}')
