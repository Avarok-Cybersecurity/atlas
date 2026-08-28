#!/usr/bin/env python3
"""Per-SUBLAYER reference intermediates for layer 0, to bisect the Atlas
LongCat backbone against `ATLAS_OP_DUMP` output.

Emits, for the LAST token of the 16-token fixture (the slice `dump_bf16`
hooks record exactly that row), the four stage vectors of each of the two
sublayers of checkpoint layer 0:

  input_norm_in    hidden entering the sublayer
  input_norm_out   input_layernorm(hidden)   -> the MLA input
  post_attn_norm_out  post_attention_layernorm(hidden + attn_out)
  out              hidden leaving the sublayer

Compare with:  slice_stage_cmp.py <dump_dir>

Usage: slice_stage_ref.py SNAPSHOT_DIR [OUT_NPZ]
"""
import json
import os
import struct
import sys

import numpy as np

snap = sys.argv[1]
out_npz = sys.argv[2] if len(sys.argv) > 2 else '/tmp/slice_stage_ref.npz'
here = os.path.dirname(os.path.abspath(__file__))

cfg = json.load(open(f'{snap}/config.json'))
idx = json.load(open(f'{snap}/model.safetensors.index.json'))
gold = np.load(f'{here}/longcat_forward_golden.npz')

H = cfg['hidden_size']
NH = cfg['num_attention_heads']
QK_NOPE, QK_ROPE = cfg['qk_nope_head_dim'], cfg['qk_rope_head_dim']
QK_HEAD = QK_NOPE + QK_ROPE
V_HEAD = cfg['v_head_dim']
Q_LORA, KV_LORA = cfg['q_lora_rank'], cfg['kv_lora_rank']
N_ROUTED, N_ZERO = cfg['n_routed_experts'], cfg['zero_expert_num']
TOPK, SCALE_ROUTED = cfg['moe_topk'], cfg['routed_scaling_factor']
EPS, THETA = cfg['rms_norm_eps'], cfg['rope_theta']

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


def rms_norm(x, w):
    v = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (w * (x / np.sqrt(v + EPS))).astype(np.float32)


def mlp(x, g, u, d):
    a = x @ g.T
    return ((a / (1 + np.exp(-a))) * (x @ u.T)) @ d.T


def rope_tables(seq, dim):
    inv = 1.0 / (THETA ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = np.outer(np.arange(seq, dtype=np.float64), inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rope_interleave(x, cos, sin):
    hds, s, d = x.shape
    x = x.reshape(hds, s, d // 2, 2).transpose(0, 1, 3, 2).reshape(hds, s, d)
    return x * cos + rotate_half(x) * sin


def mla(h, pfx, cos, sin):
    seq = h.shape[0]
    q = rms_norm(h @ tensor(f'{pfx}.q_a_proj.weight').T,
                 tensor(f'{pfx}.q_a_layernorm.weight'))
    q = (q @ tensor(f'{pfx}.q_b_proj.weight').T).reshape(
        seq, NH, QK_HEAD).transpose(1, 0, 2)
    q_pass, q_rot = q[..., :QK_NOPE], q[..., QK_NOPE:]
    ckv = h @ tensor(f'{pfx}.kv_a_proj_with_mqa.weight').T
    k_pass, k_rot = ckv[:, :KV_LORA], ckv[:, KV_LORA:]
    k_pass = rms_norm(k_pass, tensor(f'{pfx}.kv_a_layernorm.weight'))
    sq = (H / Q_LORA) ** 0.5 if cfg.get('mla_scale_q_lora') else 1.0
    sk = (H / KV_LORA) ** 0.5 if cfg.get('mla_scale_kv_lora') else 1.0
    q_pass, q_rot, k_pass = q_pass * sq, q_rot * sq, k_pass * sk
    kv = (k_pass @ tensor(f'{pfx}.kv_b_proj.weight').T).reshape(
        seq, NH, QK_NOPE + V_HEAD).transpose(1, 0, 2)
    k_pass, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
    k_rot = np.broadcast_to(rope_interleave(k_rot.reshape(1, seq, QK_ROPE), cos, sin),
                            (NH, seq, QK_ROPE))
    qs = np.concatenate([q_pass, rope_interleave(q_rot, cos, sin)], axis=-1)
    ks = np.concatenate([k_pass, k_rot], axis=-1)
    aw = (qs @ ks.transpose(0, 2, 1)) * (QK_HEAD ** -0.5)
    aw = aw + np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), 1)
    aw = aw - aw.max(axis=-1, keepdims=True)
    aw = np.exp(aw)
    aw /= aw.sum(axis=-1, keepdims=True)
    o = (aw @ v).transpose(1, 0, 2).reshape(seq, NH * V_HEAD)
    return o @ tensor(f'{pfx}.o_proj.weight').T


def moe(h, lp):
    logits = h @ tensor(f'{lp}.mlp.router.classifier.weight').T
    logits = logits - logits.max(axis=-1, keepdims=True)
    scores = np.exp(logits)
    scores /= scores.sum(axis=-1, keepdims=True)
    choice = scores + tensor(f'{lp}.mlp.router.e_score_correction_bias')
    topk = np.argsort(-choice, axis=-1)[:, :TOPK]
    out = np.zeros_like(h)
    for t in range(h.shape[0]):
        for e in topk[t]:
            w = scores[t, e] * SCALE_ROUTED
            if e >= N_ROUTED:
                out[t] += w * h[t]
            else:
                ep = f'{lp}.mlp.experts.{e}'
                out[t] += w * mlp(h[t:t + 1], tensor(f'{ep}.gate_proj.weight'),
                                  tensor(f'{ep}.up_proj.weight'),
                                  tensor(f'{ep}.down_proj.weight'))[0]
    return out


h = gold['input'].astype(np.float32)
seq = h.shape[0]
cos, sin = rope_tables(seq, QK_ROPE)
lp = 'model.layers.0'
dump = {}

for s in range(2):
    dump[f'sub{s}_input_norm_in'] = h[-1].copy()
    x = rms_norm(h, tensor(f'{lp}.input_layernorm.{s}.weight'))
    dump[f'sub{s}_input_norm_out'] = x[-1].copy()
    h = h + mla(x, f'{lp}.self_attn.{s}', cos, sin)
    x = rms_norm(h, tensor(f'{lp}.post_attention_layernorm.{s}.weight'))
    dump[f'sub{s}_post_attn_norm_out'] = x[-1].copy()
    # The engine dumps the DENSE FFN delta as "moe_out" and the shortcut MoE
    # separately as "shortcut_moe_out", so record both here under matching
    # names rather than one conflated stage.
    dense = mlp(x, tensor(f'{lp}.mlps.{s}.gate_proj.weight'),
                tensor(f'{lp}.mlps.{s}.up_proj.weight'),
                tensor(f'{lp}.mlps.{s}.down_proj.weight'))
    dump[f'sub{s}_dense_ffn'] = dense[-1].copy()
    if s == 0:
        shortcut = moe(x, lp)
        dump['sub0_shortcut_moe'] = shortcut[-1].copy()
        dump['sub0_moe_out'] = shortcut[-1].copy()  # back-compat alias
        h = h + dense
    else:
        h = h + dense + shortcut
    dump[f'sub{s}_out'] = h[-1].copy()

assert np.allclose(h, gold['layer0_out'], atol=1e-3), 'stage ref drifted from golden'
np.savez(out_npz, **dump)
print('stage reference matches the committed golden layer0_out')
for k in sorted(dump):
    print(f'  {k:28s} |v|={np.linalg.norm(dump[k]):.3f}')
print('wrote', out_npz)
