#!/usr/bin/env python3
"""Aligned engine-vs-reference logit comparison over the greedy continuation.

Alignment protocol (the naive single-position compare was wrong twice —
the engine's raw-row dump never captures the OPENER token's distribution,
so row k is P(.|prompt + k+1 generated tokens) and a sampled context
diverges):

  1. greedy_capture.py: engine chat request at temperature 0, max_tokens=8,
     with ATLAS_DUMP_LOGITS_PATH armed -> greedy token ids (greedy_ids.txt)
     + 7 raw rows (greedy_rows.npy), row k = P(.|prompt + greedy[0..=k]).
  2. forward_ref.py with QWEN4EXP_EXTRA_IDS=<those ids> -> golden logits at
     positions 0..P+7 where position P+k = P(.|prompt + greedy[0..=k]).
  3. This script: KL / p(top1) / top-20 overlap at the 7 aligned positions.

Usage: logit_quality_aligned.py [golden.npz] [rows.npy] [ids.txt]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = '/home/ms/.claude/jobs/5a7bd33d/tmp'
GOLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, 'qwen4exp_forward_golden.npz')
ROWS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    TMP, 'greedy_rows.npy')
IDS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    TMP, 'greedy_ids.txt')
ENGINE_V = 248077

gold = np.load(GOLD)
tokens = [int(t) for t in gold['tokens']]
logits = gold['logits'].astype(np.float64)
rows = np.load(ROWS).astype(np.float64)
greedy = [int(t) for t in open(IDS).read().split(',')]
n_extra = len(greedy)
P = len(tokens) - n_extra
assert tokens[P:] == greedy, 'golden fixture tail != greedy ids'
assert len(rows) == n_extra - 1, (len(rows), n_extra)

from transformers import AutoTokenizer  # noqa: E402

SNAP = ('/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
        '129972269565f7f4f664fdf8dd42268d3bbda9fd')
tok = AutoTokenizer.from_pretrained(SNAP)


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


print(f'prompt {P} tokens + {n_extra} greedy; {len(rows)} aligned positions')
print(f'{"pos":>4} {"ctx-tok":>12} {"KL(r||e)":>9} {"p1_ref":>7} {"p1_eng":>7} '
      f'{"ov20":>5}  ref_top1 / eng_top1')
for k in range(len(rows)):
    ref = logits[P + k][:ENGINE_V]
    eng = rows[k][:ENGINE_V]
    p_r, p_e = softmax(ref), softmax(eng)
    eps = 1e-12
    kl = float(np.sum(p_r * np.log((p_r + eps) / (p_e + eps))))
    tr = np.argsort(ref)[::-1][:20]
    te = np.argsort(eng)[::-1][:20]
    ov = len(set(tr.tolist()) & set(te.tolist()))
    print(f'{P + k:>4} {tok.decode([greedy[k]])!r:>12} {kl:9.4f} '
          f'{p_r.max():7.4f} {p_e.max():7.4f} {ov:>5}  '
          f'{tok.decode([int(tr[0])])!r} / {tok.decode([int(te[0])])!r}')
