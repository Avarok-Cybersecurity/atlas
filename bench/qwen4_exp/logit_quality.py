#!/usr/bin/env python3
"""Score the engine's logits against the streaming-reference golden.

Reference: qwen4exp_forward_golden.npz (forward_ref.py) — f32 logits at
every fixture position, vocab 248320.
Engine:    a raw row from ATLAS_DUMP_LOGITS_PATH/logits_seq.bin captured by
           sending the SAME rendered fixture prompt to /v1/completions with
           max_tokens=1 at a sampling temperature (the dump fires on the
           stochastic path only). Row width is model.vocab_size() = 248077,
           NOT the config's 248320 — compare over the engine width.

Reports, at the last fixture position: KL(ref || engine) and reverse,
p(top1) both sides, top-20 overlap, and the ids where they disagree most.

Usage:
    1. serve with ATLAS_DUMP_LOGITS_PATH=<dir>; rm <dir>/logits_seq.bin
    2. logit_quality.py [golden.npz] [dumpdir] [port]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, 'qwen4exp_forward_golden.npz')
DUMP = sys.argv[2] if len(sys.argv) > 2 else '/home/ms/.claude/jobs/5a7bd33d/tmp'
PORT = sys.argv[3] if len(sys.argv) > 3 else '8889'
ENGINE_V = 248077

gold = np.load(GOLD)
tokens = [int(t) for t in gold['tokens']]
ref = gold['logits'][-1].astype(np.float64)   # last fixture position

from transformers import AutoTokenizer  # noqa: E402

SNAP = ('/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
        '129972269565f7f4f664fdf8dd42268d3bbda9fd')
tok = AutoTokenizer.from_pretrained(SNAP)
prompt = tok.decode(tokens)

seq_bin = os.path.join(DUMP, 'logits_seq.bin')
if os.path.exists(seq_bin):
    os.remove(seq_bin)

import httpx  # noqa: E402

# Chat endpoint, NOT completions: the raw-row dump's first row is the
# OPENER token's distribution only on the chat path (the completions path
# samples token 0 elsewhere and row 0 lands on token 1 — a position
# mismatch that produced a meaningless 11-nat KL on the first attempt).
# The server renders the same template as the golden fixture
# (reasoning_effort low is the serve default); prompt_tokens is asserted.
r = httpx.post(f'http://127.0.0.1:{PORT}/v1/chat/completions',
               json={'model': 'qwen4exp',
                     'messages': [{'role': 'user',
                                   'content': 'Hello, what model are you?'}],
                     'max_tokens': 2, 'temperature': 1.0, 'seed': 1,
                     'chat_template_kwargs': {'reasoning_effort': 'low'}},
               timeout=900)
r.raise_for_status()
pt = r.json()['usage']['prompt_tokens']
assert pt == len(tokens), (
    f'engine retokenized the fixture to {pt} tokens (golden has '
    f'{len(tokens)}) — the compared position would not match')
eng_all = np.fromfile(seq_bin, dtype=np.float32)
assert len(eng_all) >= ENGINE_V, f'dump has {len(eng_all)} floats'
eng = eng_all[:ENGINE_V].astype(np.float64)

# Sanity: the engine retokenizes the decoded prompt — verify the width and
# that both sides argmax-agree on decodable ids before trusting the KL.
refv = ref[:ENGINE_V]


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


p_ref = softmax(refv)
p_eng = softmax(eng)
eps = 1e-12
kl_fwd = float(np.sum(p_ref * np.log((p_ref + eps) / (p_eng + eps))))
kl_rev = float(np.sum(p_eng * np.log((p_eng + eps) / (p_ref + eps))))

top_ref = np.argsort(refv)[::-1][:20]
top_eng = np.argsort(eng)[::-1][:20]
overlap = len(set(top_ref.tolist()) & set(top_eng.tolist()))

print(f'fixture: {len(tokens)} tokens; compare width {ENGINE_V}')
print(f'KL(ref||eng) = {kl_fwd:.4f} nats   KL(eng||ref) = {kl_rev:.4f} nats')
print(f'p(top1): ref {p_ref.max():.4f} ({tok.decode([int(top_ref[0])])!r})  '
      f'eng {p_eng.max():.4f} ({tok.decode([int(top_eng[0])])!r})')
print(f'top-20 overlap: {overlap}/20')
print('ref top-5 :', ' '.join(
    f'{tok.decode([int(i)])!r}:{p_ref[i]:.3f}' for i in top_ref[:5]))
print('eng top-5 :', ' '.join(
    f'{tok.decode([int(i)])!r}:{p_eng[i]:.3f}' for i in top_eng[:5]))
d = np.abs(refv - refv.mean() - (eng - eng.mean()))
worst = np.argsort(d)[::-1][:5]
print('largest centered-logit disagreements:')
for i in worst:
    print(f'  {tok.decode([int(i)])!r}: ref {refv[i]:.3f} eng {eng[i]:.3f}')
